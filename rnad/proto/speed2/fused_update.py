"""F5: pass 1 の online 順伝播を pass 2 へ融合した更新の試作（docs/rl-learning-speed.md「さらに速くする」の S3。
**数式は同じ**。本物の RNaDLearner.update と同じ学習状態・同じ batch で突き合わせる。学習には使っていない）。

    # CPU（どこでも。--games 4 なら数十秒）
    python rnad/proto/speed2/fused_update.py --games 4 --steps 7000,2500
    # ゲーム用 PC（768局の batch を一度保存して、α=1 と補間中の両方で時間も測る）
    python rnad/proto/speed2/fused_update.py --games 768 --save-batch batch768.pt --device cuda --timing
    python rnad/proto/speed2/fused_update.py --batch batch768.pt --device cuda --timing --out fused768.json

batch の決定点は時刻の昇順（collect が手番ごとに足す）なので、チャンクを**後ろから**回せば、v-trace の
後ろ向き再帰（時刻 t の出力は t 以降の入力だけで決まる）をチャンクごとに進められる。各チャンクで
  online の順伝播（勾配つき）→ 正則化・target の順伝播（勾配なし）→ そのチャンクの時刻の v-trace → 損失 → backward
を1回で行う。従来の pass 1 の online 順伝播（1決定点あたり順伝播 1回分）が消える。
局ごとの持ち越し（carry）は [B] で持ち、チャンクに入っている局だけ更新する（局どうしは独立）。v-trace は CPU に置き、
チャンクごとに π_a・ent・v_tgt だけを CPU へ戻す（GPU へ移すと小さい演算が約3.3万回入る）。
自己対局（各局の有効な手番が 0 から連続）・c_ply 既定・報酬の縮めなし・補助ヘッドなしに限る（assert する）。

判定（S3 を `--fused` として入れるか）: 勾配の相対誤差が 1e-6 未満（CPU の実測 128局で 8.4e-8、4局で 4.4e-8）で、
ゲーム用 PC の `sec_fused` が `sec_update` より α=1 と補間中の両方で小さいこと（CPU・128局・交互に2回で、α=1 は
従来 24.9〜25.1s 対 融合 22.6〜23.0s、補間中は 28.0〜30.2s 対 27.8〜28.4s。GPU で効く割合は未計測）
"""

import argparse
import copy
import json
import time

import torch

from proto_common import DEFAULT_WEIGHTS, MODELS, load_or_collect, make_learner
from rnad import forward, legal_log_policy, loss_terms  # noqa: E402


def fused_update(L, batch):
    cfg, dev = L.cfg, L.device
    assert tuple(cfg.c_ply) == (0, 1.0) and cfg.aux_truth == 0, "c_ply 既定・補助ヘッドなしに限る"
    alpha, update_reg = L.schedule(L.step_count)
    n = len(batch["action"])
    t_idx, b_idx = batch["t"], batch["b"]
    T, B = batch["valid"].shape
    valid, player = batch["valid"], batch["player"]
    assert bool((t_idx[1:] >= t_idx[:-1]).all()), "決定点が時刻順でない"
    nv = valid.sum(0)
    assert bool((valid == (torch.arange(T)[:, None] < nv[None, :])).all()), "有効な手番が連続していない局がある"
    player_flat = player[t_idx, b_idx].long()
    counts = [int((player_flat == 0).sum()), int((player_flat == 1).sum())]
    rew = batch["reward"]
    mu_flat = batch["mu_a"]
    zeros, ones = torch.zeros(B), torch.ones(B)
    carry = {k: [zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone(), ones.clone()] for k in (0, 1)}
    c = cfg.c_vtrace
    L.opt.zero_grad(set_to_none=True)
    tot_v = tot_nerd = 0.0
    top_sum = ent_sum = 0.0
    vt_all = torch.zeros(n)
    s_all = torch.zeros(n)
    for sl in reversed(list(L._chunks(n))):
        obs, legal, act, sel = L._chunk_view(batch, sl)
        logits, v = forward(L.net, obs)
        logits = sel(logits)
        with torch.no_grad():
            log_pi = legal_log_policy(logits, legal)
            pi = log_pi.exp() * legal
            mix = L._lpr_mix(obs, legal, alpha, sel)
            lpr = log_pi - mix
            pi_a = pi.gather(1, act[:, None]).squeeze(1).cpu()
            ent = (pi * lpr).sum(-1).cpu()
            top = pi.max(1).values
            top_sum += float(top.sum())
            ent_sum += float(-(pi * log_pi).sum())
            v_tgt = forward(L.target, obs)[1].cpu()
            # このチャンクの時刻を後ろから（v_trace_scalars の1ステップを、このチャンクに入っている局だけに）
            ct, cb = t_idx[sl], b_idx[sl]
            cs_all = pi_a / mu_flat[sl]
            inv_mu_all = 1.0 / mu_flat[sl]
            if cfg.inv_mu_cap > 0:
                inv_mu_all = inv_mu_all.clamp(max=cfg.inv_mu_cap)
            vt_c = torch.zeros(len(ct))
            s_c = torch.zeros(len(ct))
            for t in range(int(ct.max()), int(ct.min()) - 1, -1):
                rows = (ct == t).nonzero().squeeze(1)
                bs = cb[rows]
                pl = player[t, bs]
                for k in (0, 1):
                    c_reward, c_reward_unc, c_next_value, c_next_vt, c_is = (x[bs] for x in carry[k])
                    ours = pl == k
                    po = torch.where(ours, 1.0, -1.0)
                    eta_ent = -cfg.eta * ent[rows] * po
                    vv = v_tgt[rows]
                    cs = cs_all[rows]
                    r = rew[t, bs, k]
                    r_unc = r + c_reward_unc + eta_ent
                    disc = r + c_reward
                    w = cs * c_is
                    w_c = torch.clamp(w, max=c)
                    our_vt = vv + torch.clamp(w, max=float("inf")) * (r_unc + c_next_value - vv) + w_c * (c_next_vt - c_next_value)
                    our_s = inv_mu_all[rows] * (disc + c_is * c_next_vt - vv)
                    z = torch.zeros(len(bs))
                    o = torch.ones(len(bs))
                    vt_c[rows] = torch.where(ours, our_vt, vt_c[rows])
                    s_c[rows] = torch.where(ours, our_s, s_c[rows])
                    new = (torch.where(ours, z, eta_ent + cs * disc), torch.where(ours, z, r_unc),
                           torch.where(ours, vv, c_next_value), torch.where(ours, our_vt, c_next_vt),
                           torch.where(ours, o, cs * c_is))
                    for x, y in zip(carry[k], new):
                        x[bs] = y
        lv, ln = loss_terms(logits, v, legal, act, player_flat[sl].to(dev), lpr, vt_c.to(dev), s_c.to(dev), counts,
                            cfg.eta, cfg.beta, cfg.clip, cfg.policy_loss)
        (lv + ln).backward()
        tot_v += lv.item()
        tot_nerd += ln.item()
        vt_all[sl] = vt_c
        s_all[sl] = s_c
    L.opt.step()
    with torch.no_grad():
        for pt, p in zip(L.target.parameters(), L.net.parameters()):
            pt.add_(cfg.target_avg * (p - pt))
        for bt, b in zip(L.target.buffers(), L.net.buffers()):
            bt.copy_(b)
        if update_reg:
            L.prev_.load_state_dict(L.prev.state_dict())
            L.prev.load_state_dict(L.target.state_dict())
    L.step_count += 1
    return {"loss_v": tot_v, "loss_nerd": tot_nerd, "pi_top_mean": top_sum / n, "entropy_mean": ent_sum / n,
            "v_target_mean": float(vt_all.mean()), "s_abs_mean": float(s_all.abs().mean())}


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", default="", help="保存した batch（torch.save。無ければ --games 局を集める）")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-batch", default="", help="集めた batch をここへ保存する（次から --batch で使う）")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="batch を集めるネット")
    ap.add_argument("--models", default=str(MODELS), help="rl_deduce_v5〜v7.bin のディレクトリ（学習状態）")
    ap.add_argument("--steps", default="7000,2500", help="段階の中の位置（α=1 と補間中）。カンマ区切り")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--timing", action="store_true",
                    help="1回ずつの秒も出す（最初に両方を1回ずつ空回しして、CUDA の初期化を計時から外す）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    batch = load_or_collect(args.batch, args.games, args.seed, args.weights, dev)
    if args.save_batch:
        torch.save(batch, args.save_batch)
    print(f"batch: 決定点 {len(batch['action'])}、T {batch['valid'].shape[0]}", flush=True)
    res = {}
    ok = True
    steps = [int(s) for s in args.steps.split(",")]
    if args.timing:
        make_learner(steps[0], chunk=args.chunk, models=args.models, device=dev).update(copy.copy(batch))
        fused_update(make_learner(steps[0], chunk=args.chunk, models=args.models, device=dev), copy.copy(batch))
    for step in steps:
        A = make_learner(step, chunk=args.chunk, models=args.models, device=dev)
        Bl = make_learner(step, chunk=args.chunk, models=args.models, device=dev)
        p0 = torch.nn.utils.parameters_to_vector(A.net.parameters()).detach().clone()
        sync(dev)
        t0 = time.perf_counter()
        ia = A.update(copy.copy(batch))
        sync(dev)
        t1 = time.perf_counter()
        ib = fused_update(Bl, copy.copy(batch))
        sync(dev)
        t2 = time.perf_counter()
        ga = torch.cat([p.grad.flatten() for p in A.net.parameters()]).double()
        gb = torch.cat([p.grad.flatten() for p in Bl.net.parameters()]).double()
        da = torch.nn.utils.parameters_to_vector(A.net.parameters()).detach() - p0
        db = torch.nn.utils.parameters_to_vector(Bl.net.parameters()).detach() - p0
        r = {"alpha": A.schedule(step)[0], "grad_rel_err": float((ga - gb).norm() / ga.norm()),
             "grad_cos": float(torch.nn.functional.cosine_similarity(ga, gb, dim=0)),
             "delta_rel_err": float((da - db).norm() / da.norm())}
        for k in ("loss_v", "loss_nerd", "v_target_mean", "s_abs_mean", "entropy_mean"):
            r[k] = [ia[k], ib[k]]
        if args.timing:
            r["sec_update"] = t1 - t0
            r["sec_fused"] = t2 - t1
        ok &= r["grad_rel_err"] < 1e-6
        res[str(step)] = r
        print(f"step {step}（α {r['alpha']:.2f}）: 勾配の相対誤差 {r['grad_rel_err']:.2e}、コサイン {r['grad_cos']:.12f}、"
              f"更新量の相対誤差 {r['delta_rel_err']:.2e}、loss_v {ia['loss_v']:.6f} / {ib['loss_v']:.6f}"
              + (f"、秒 従来 {r['sec_update']:.2f} / 融合 {r['sec_fused']:.2f}" if args.timing else ""), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print("融合した更新の勾配は従来と一致（相対誤差 < 1e-6）" if ok else "勾配が一致しない")


if __name__ == "__main__":
    main()
