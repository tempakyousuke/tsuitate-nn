"""F4: S2-lite（batch を更新の頭で1回だけデバイスへ送り、更新中の同期を除いた update の簡約版）が、本物の
RNaDLearner.update と**ビット単位で同じ**結果になることを CPU で確かめる（docs/rl-learning-speed.md「さらに速くする」の S2。
学習には使っていない）。

    python rnad/proto/speed2/s2lite_equiv.py --games 8
    python rnad/proto/speed2/s2lite_equiv.py --batch batch768.pt --games 24   # 保存した batch の先頭 24局で
    # ゲーム用 PC: 同期と H2D の回数を数える（予想は従来 約1,500回 → 数十回。768局の batch なら --games 768）
    python rnad/proto/speed2/s2lite_equiv.py --batch batch768.pt --games 768 --chunk 1024 --device cuda --profile

変更点（RNaDLearner.update の写しに対して）:
  1. 観測・行動・CSR を update の頭で1回だけデバイスへ送る（fp16 のまま。チャンクで .to(float32) = デバイス上の変換）
  2. pack_csr の K を CPU の legal_counts から決め、repeat_interleave に output_size を渡す（同期なし）。
     pass 1 の pack を pass 2 で使い回す。マスク外の手の検査は最後にまとめて1回
  3. π_a・ent・v_tgt・統計・損失はデバイス上に貯め、1回だけ CPU へ返す（統計の和は Python の double で同じ順に足す）
  4. vt・s・player は pass 2 の前に1回だけ送る
比べるもの: 更新後の net / target の全パラメータ（ビット）、Adam の状態、ログの値。全部同じなら終了コード 0。
CUDA でのビット一致は確かめていない（`--device cuda` で見る）。
"""
import argparse
import sys

import torch

from proto_common import DEFAULT_WEIGHTS, MODELS, load_or_collect, make_learner
from rnad import forward, legal_log_policy, loss_terms, v_trace_scalars  # noqa: E402


def pack_csr_nosync(cols, counts_dev, k, action, out_size):
    n = len(counts_dev)
    counts = counts_dev.long()
    rows = torch.repeat_interleave(torch.arange(n, device=cols.device), counts, output_size=out_size)
    pos = torch.arange(len(rows), device=cols.device) - (torch.cumsum(counts, 0) - counts)[rows]
    idx = torch.zeros(n, k, dtype=torch.long, device=cols.device)
    idx[rows, pos] = cols.long()
    pmask = torch.zeros(n, k, dtype=torch.bool, device=cols.device)
    pmask[rows, pos] = True
    apos = ((idx == action[:, None]) & pmask).to(torch.uint8).argmax(1)
    ok = (idx.gather(1, apos[:, None]).squeeze(1) == action).all()
    return idx, pmask, apos, ok


def update_lite(self, batch):
    cfg, dev = self.cfg, self.device
    alpha, update_reg = self.schedule(self.step_count)
    n = len(batch["action"])
    t_idx, b_idx = batch["t"], batch["b"]
    T, B = batch["valid"].shape
    # 1回だけ送る
    obs_d = batch["obs"].to(dev)
    act_d = batch["action"].to(dev)
    cols_d = batch["legal_cols"].to(dev)
    cnt_cpu = batch["legal_counts"]
    cnt_d = cnt_cpu.to(dev)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(cnt_cpu.long(), 0)])
    packs, oks = [], []
    pi_a_d = torch.zeros(n, device=dev)
    ent_d = torch.zeros(n, device=dev)
    v_tgt_d = torch.zeros(n, device=dev)
    tops, his, ents = [], [], []
    mix_cache = []
    chunks = list(self._chunks(n))
    with torch.no_grad():
        for sl in chunks:
            obs = obs_d[sl].to(torch.float32)
            a0, a1 = int(ptr[sl.start]), int(ptr[sl.stop])
            k = int(cnt_cpu[sl].max())
            idx, pmask, apos, ok = pack_csr_nosync(cols_d[a0:a1], cnt_d[sl], k, act_d[sl], a1 - a0)
            packs.append((idx, pmask, apos))
            oks.append(ok)
            sel = (lambda x, idx=idx: x.gather(1, idx))
            legal, act = pmask, apos
            logits = sel(forward(self.net, obs)[0])
            log_pi = legal_log_policy(logits, legal)
            pi = log_pi.exp() * legal
            mix = self._lpr_mix(obs, legal, alpha, sel)
            mix_cache.append(mix)
            lpr = log_pi - mix
            pi_a_d[sl] = pi.gather(1, act[:, None]).squeeze(1)
            ent_d[sl] = (pi * lpr).sum(-1)
            top = pi.max(1).values
            tops.append(top.sum())
            his.append((top > 0.999).sum())
            ents.append(-(pi * log_pi).sum())
            v_tgt_d[sl] = forward(self.target, obs)[1]
        # 1回だけ CPU へ
        pi_a, ent, v_tgt = pi_a_d.cpu(), ent_d.cpu(), v_tgt_d.cpu()
        top_sum = sum(float(x) for x in torch.stack(tops).cpu())
        top_hi = sum(float(x) for x in torch.stack(his).to(torch.float64).cpu())
        ent_sum = sum(float(x) for x in torch.stack(ents).cpu())
        if not bool(torch.stack(oks).all()):
            raise ValueError("マスク外の手が batch にある")

    def grid(x):
        g = torch.zeros(T, B)
        g[t_idx, b_idx] = x
        return g

    valid = batch["valid"]
    player = batch["player"]
    mu_g = torch.where(valid, grid(batch["mu_a"]), torch.ones(T, B))
    vt = torch.zeros(n)
    s = torch.zeros(n)
    for kk in (0, 1):
        vt_k, s_k = v_trace_scalars(grid(v_tgt), valid, player, grid(pi_a), mu_g, grid(ent),
                                    batch["reward"][:, :, kk], kk, cfg.eta, c=cfg.c_vtrace,
                                    inv_mu_cap=cfg.inv_mu_cap)
        mine = player[t_idx, b_idx] == kk
        vt = torch.where(mine, vt_k[t_idx, b_idx], vt)
        s = torch.where(mine, s_k[t_idx, b_idx], s)
    player_flat = player[t_idx, b_idx].long()
    counts = [int((player_flat == 0).sum()), int((player_flat == 1).sum())]
    vt_d, s_d, pl_d = vt.to(dev), s.to(dev), player_flat.to(dev)

    self.opt.zero_grad(set_to_none=True)
    lvs, lns = [], []
    for ci, sl in enumerate(chunks):
        obs = obs_d[sl].to(torch.float32)
        idx, pmask, apos = packs[ci]
        logits, v = forward(self.net, obs)
        logits = logits.gather(1, idx)
        with torch.no_grad():
            lpr = legal_log_policy(logits, pmask) - mix_cache[ci]
        lv, ln = loss_terms(logits, v, pmask, apos, pl_d[sl], lpr, vt_d[sl], s_d[sl], counts,
                            cfg.eta, cfg.beta, cfg.clip, cfg.policy_loss)
        (lv + ln).backward()
        lvs.append(lv.detach())
        lns.append(ln.detach())
    self.opt.step()
    with torch.no_grad():
        for pt, p in zip(self.target.parameters(), self.net.parameters()):
            pt.add_(cfg.target_avg * (p - pt))
        for bt, b in zip(self.target.buffers(), self.net.buffers()):
            bt.copy_(b)
        if update_reg:
            self.prev_.load_state_dict(self.prev.state_dict())
            self.prev.load_state_dict(self.target.state_dict())
    self.step_count += 1
    tot_v = 0.0
    for x in torch.stack(lvs).cpu():
        tot_v += x.item()
    tot_n = 0.0
    for x in torch.stack(lns).cpu():
        tot_n += x.item()
    ratio = pi_a / batch["mu_a"]
    return {"loss_v": tot_v, "loss_nerd": tot_n, "pi_top_mean": top_sum / n, "pi_top_gt_0999": top_hi / n,
            "entropy_mean": ent_sum / n, "ratio_dev_mean": float((ratio - 1).abs().mean()),
            "v_target_mean": float(vt.mean()), "s_abs_mean": float(s.abs().mean())}


def subset(batch, games):
    keep = batch["b"] < games
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(batch["legal_counts"].long(), 0)])
    rows = torch.nonzero(keep).squeeze(1)
    cols = torch.cat([batch["legal_cols"][int(ptr[r]):int(ptr[r + 1])] for r in rows.tolist()])
    out = {k: batch[k][keep] for k in ("obs", "action", "mu_a", "t", "b", "legal_counts")}
    out["legal_cols"] = cols
    T = int(out["t"].max()) + 1
    out["valid"] = batch["valid"][:T, :games]
    out["player"] = batch["player"][:T, :games]
    out["reward"] = batch["reward"][:T, :games]
    return out


PROFILE_KEYS = ("cudaMemcpyAsync", "Memcpy HtoD", "Memcpy DtoH", "cudaStreamSynchronize", "cudaDeviceSynchronize")


def count_events(fn):
    """fn() を torch.profiler の下で回し、転送と同期の呼び出し回数を数える（CUDA でだけ意味がある）"""
    acts = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=acts) as prof:
        out = fn()
    cnt = {k: 0 for k in PROFILE_KEYS}
    for e in prof.key_averages():
        for k in PROFILE_KEYS:
            if k in e.key:
                cnt[k] += e.count
    return out, cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", default="", help="保存した batch（torch.save。無ければ --games 局を集める）")
    ap.add_argument("--games", type=int, default=24, help="使う局数（保存した batch ならその先頭の局）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="batch を集めるネット")
    ap.add_argument("--models", default=str(MODELS), help="rl_deduce_v5〜v7.bin のディレクトリ（学習状態）")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--profile", action="store_true", help="転送と同期の回数を torch.profiler で数える")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    full = load_or_collect(args.batch, args.games, args.seed, args.weights, dev)
    batch = subset(full, args.games) if args.batch else full
    print("rows", len(batch["action"]), "T", batch["valid"].shape[0])
    ok = True
    for step in (2500, 7000):  # α=0.5（傾斜中）と α=1
        A = make_learner(step, packed=True, chunk=args.chunk, models=args.models, device=dev)
        Bl = make_learner(step, packed=True, chunk=args.chunk, models=args.models, device=dev)
        if args.profile:
            ia, ca = count_events(lambda: A.update(dict(batch)))
            ib, cb = count_events(lambda: update_lite(Bl, dict(batch)))
            print(f"step {step} 回数 従来 / S2-lite: " + "、".join(f"{k} {ca[k]} / {cb[k]}" for k in PROFILE_KEYS))
        else:
            ia = A.update(dict(batch))
            ib = update_lite(Bl, dict(batch))
        same_net = all(torch.equal(p.view(torch.int32), q.view(torch.int32))
                       for p, q in zip(A.net.parameters(), Bl.net.parameters()))
        same_tgt = all(torch.equal(p.view(torch.int32), q.view(torch.int32))
                       for p, q in zip(A.target.parameters(), Bl.target.parameters()))
        sa, sb = A.opt.state_dict()["state"], Bl.opt.state_dict()["state"]
        same_opt = all(torch.equal(sa[k]["exp_avg_sq"], sb[k]["exp_avg_sq"]) for k in sa)
        logs = {k: (ia[k], ib[k], ia[k] == ib[k]) for k in ib}
        ok &= same_net and same_tgt and same_opt and all(v[2] for v in logs.values())
        print(f"step {step} alpha {ia['alpha']}: net bit-equal {same_net}, target bit-equal {same_tgt}, "
              f"adam v bit-equal {same_opt}")
        for k, v in logs.items():
            print("   ", k, v)
    print("S2-lite は本物の update とビット単位で同じ（net・target・Adam・ログ）" if ok else "ビット一致しない")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
