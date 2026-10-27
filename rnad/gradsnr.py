"""勾配の信号対雑音比で「1標本あたりの学習の効き」を学習なしに比べる（tsuitate-bot の docs/rl-learning-speed.md）。

    python rnad/gradsnr.py --weights ../tsuitate-bot/models/rl_deduce_v4.bin --games 256 \\
        --variant ref:clip=10000 --variant clip1000:clip=1000 --variant clip100:clip=100 --variant mu50:mucap=50

本線の後半は**勾配の雑音が律速**（1局ごとの勾配の分散 trΣ に対し、期待勾配の大きさ |G|² が小さい。
雑音のスケール B_noise = trΣ / |G|² が 1更新の局数よりずっと大きい）。この領域では1標本あたりの進みは
およそ |G|² / trΣ に比例するので、**分散を減らす手（クリップ・1/μ の上限）の効き目は学習を回さずに測れる**。
ただし偏りと引き換えなので、クリップなしの期待勾配（ref）との向きのずれも測る。

手順: 重みのネットで自己対局した局を、そのネットを現在・ターゲット・正則化方策の全部に使って（= 段階の頭で
lpr = 0。η の項は消え、s の項だけが残る）R-NaD の損失にかけ、**局ごとの勾配**を変種ごとに求める。
変種ごとに出すもの:
- trΣ: 局ごとの勾配の分散の和（1局あたり）
- |G|²: 期待勾配の大きさの2乗（不偏推定 = 平均の2乗 − trΣ / 局数）
- B_noise = trΣ / |G|²（局。小さいほど少ない局で同じ向きが出る）
- G·G_ref と cos: ref の期待勾配との内積（局を半分に分け、片方の変種ともう片方の ref の平均で不偏に推定）
- 有効 SNR = (G·G_ref)² / (|G_ref|² · trΣ)（ref の向きへの進みの信号対雑音比。ref 自身なら |G_ref|² / trΣ_ref）

変種の書き方: `<名前>:clip=<NeuRD のクリップ>,mucap=<1/μ の上限>,loss=<neurd|pg>`（省いた項は参照の値）

**MMD 学習器の変種**（`mmd.py`。docs/rl-learning-speed.md 10節の (a)）: `<名前>:algo=mmd,lam=<λ_A>,filter=<none|pps0.25|q0.75|topk0.25>,
vrows=<auto|filtered|all>,part=<pol|val|mag|all>,magnet=<uniform|mix0.03>,temp=<α>`。収集した直後（π=μ）なので PPO の項の
勾配は −w·A·∇log π(a)、近い方策への KL の勾配は 0。A は `mmd.player_lambda_returns`（価値はこのネット）。行は全局の |A| で
1回だけ選び、局ごとの勾配は「その局の残した行の重み付き損失 ÷ 全体の残した行の数 × 局数」（平均が全体の勾配になる尺度）。
磁石の mix は「このネット自身の方策と一様の混合」（始めた時点の磁石と同じ）。MMD の変種は行ごとの係数だけが違うので、
局ごとに1回の逆伝播を変種の数だけ束ねて回す（`is_grads_batched`）

**変種どうしの期待勾配のコサイン**（全変種の組）: 同じ局の雑音は変種の間で相関するので、標本平均どうしの内積は使わず、
U 統計量 G_a·G_b ≈ (Σ_i g_a,i · Σ_j g_b,j − Σ_i g_a,i·g_b,i) / (n(n−1)) で推定する（同じ局どうしの項を除く）
"""

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tsuitate_rl  # noqa: E402
from export_weights import read as read_weights  # noqa: E402
from model import NUM_PLANES  # noqa: E402
from mmd import parse_config, player_lambda_returns, select_rows  # noqa: E402
from mmd import make_config as make_mmd_config  # noqa: E402
from rnad import legal_log_policy, loss_terms, pack_legal, sample_actions, v_trace_scalars  # noqa: E402

DEV = torch.device("cpu")


MMD_KEYS = ("algo", "lam", "filter", "vrows", "part", "magnet", "temp")


def parse_variant(spec):
    name, _, rest = spec.partition(":")
    v = {"name": name, "clip": 10_000.0, "mucap": 0.0, "loss": "neurd", "algo": "rnad"}
    for kv in filter(None, rest.split(",")):
        k, _, x = kv.partition("=")
        if k not in ("clip", "mucap", "loss") + MMD_KEYS:
            raise SystemExit(f"変種の項 {k} は clip / mucap / loss / {' / '.join(MMD_KEYS)} のどれか")
        v[k] = float(x) if k in ("clip", "mucap", "lam", "temp") else x
    if v["algo"] == "mmd":
        v.setdefault("lam", 0.8)
        v.setdefault("filter", "none")
        v.setdefault("vrows", "auto")
        v.setdefault("part", "pol")
        v.setdefault("magnet", "uniform")
        v.setdefault("temp", 1.0)
        f = v["filter"]
        v["filter_spec"] = ("none" if f == "none" else f"pps:{f[3:]},floor=0.05" if f.startswith("pps")
                            else f"topk:{f[4:]}" if f.startswith("topk") else f"quantile:{f[1:]},min=0.01")
        if v["part"] not in ("pol", "val", "mag", "all") or v["magnet"] not in ("uniform",) and not v["magnet"].startswith("mix"):
            raise SystemExit(f"{spec}: part は pol / val / mag / all、magnet は uniform / mix<ε>")
    elif any(k in v for k in MMD_KEYS[1:]):
        raise SystemExit(f"{spec}: MMD の項は algo=mmd のときだけ")
    return v


def mmd_rows(var, batch, v_net, n_games, seed):
    """MMD の変種の行ごとの係数の材料: (A, G_td, 方策の行の重み × 局数 / 方策の行の数, 価値の行の重み × 局数 / 価値の行の数)"""
    cfg = make_mmd_config("tsuitate", filter=var["filter_spec"], value_rows=var["vrows"], gae_lambda=var["lam"])
    parsed = parse_config(cfg)
    adv = player_lambda_returns(batch, v_net, var["lam"]) - v_net
    g_td = player_lambda_returns(batch, v_net, 0.8)
    gen = torch.Generator().manual_seed(seed)
    keep_p, wp, keep_v, wv, _ = select_rows(adv, parsed["filter"], parsed["value_rows"], gen)
    return (adv, g_td, wp * n_games / max(int(keep_p.sum()), 1), wv * n_games / max(int(keep_v.sum()), 1),
            float(keep_p.float().mean()))


def u_stat_jackknife(Sblk, cross, nk):
    """変種どうしの期待勾配の内積の U 統計量と、その標準誤差（局を K 組に分けた1組抜きのジャックナイフ）。

    Sblk[k] = 組 k の局ごとの勾配の和（変種 × パラメータ）、cross[k] = 組 k の Σ_i g_a,i·g_b,i（変種 × 変種）、
    nk[k] = 組 k の局数。U = (S·Sᵀ − Σ_i g_i·g_iᵀ) / (N(N−1))（同じ局どうしの項を除く）。
    組ごとの U のばらつき / √K は、雑音が勝つとき（B_noise が局数と同程度以上）に U の分散の 1/N² の項
    （2·tr(Σ²)/(N(N−1))）を K 倍に数えて過大になる（模擬で K 4 のとき 1.0〜1.6倍、K 32 で 2〜5倍）。
    ジャックナイフは模擬で真の値の 1.0〜1.4倍（やや保守的）"""
    K = len(nk)
    N = float(nk.sum())
    S = Sblk.sum(0)
    C = cross.sum(0)
    U = (S @ S.T - C) / (N * (N - 1))
    M = N - nk
    U_del = torch.stack([((S - Sblk[k]) @ (S - Sblk[k]).T - (C - cross[k])) / (M[k] * (M[k] - 1)) for k in range(K)])
    U_se = torch.sqrt((K - 1) / K * ((U_del - U_del.mean(0)) ** 2).sum(0))
    return U, U_se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--eta", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=16.0)
    ap.add_argument("--variant", action="append", required=True,
                    help="最初の変種が ref（向きの基準）。例 ref:clip=10000")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=32,
                    help="U 統計量の標準誤差（1組抜きのジャックナイフ）の局の組の数。16 以上を勧める。組ごとの勾配の和を持つので"
                         "メモリは 組数 × 変種の数 × パラメータ数 × 8 バイト（64ch×6・14変種・32組で約2GB）")
    ap.add_argument("--check-batched", type=int, default=0,
                    help="最初の N 局で、MMD の変種の束ねた逆伝播を変種ごとの素直な逆伝播と突き合わせる")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    variants = [parse_variant(s) for s in args.variant]
    mmd_vars = [v for v in variants if v["algo"] == "mmd"]
    rnad_vars = [v for v in variants if v["algo"] == "rnad"]
    net = read_weights(args.weights).eval()
    planes = net.config["planes"]
    kw = {"planes": planes} if planes != NUM_PLANES else {}
    torch.manual_seed(args.seed)
    n_games = args.games
    t0 = time.time()

    # 自己対局（collect と同じ記録。観測は fp16 に丸める）
    env = tsuitate_rl.VecEnv(n_games, seed=args.seed, auto_reset=False, **kw)
    env.reset()
    obs_l, legal_l, act_l, mu_l, t_l, b_l = [], [], [], [], [], []
    T_MAX = 260
    valid = torch.zeros(T_MAX, n_games, dtype=torch.bool)
    player = torch.zeros(T_MAX, n_games, dtype=torch.long)
    reward = torch.zeros(T_MAX, n_games, 2)
    for t in range(T_MAX):
        alive = env.alive()
        if not alive.any():
            break
        obs, mask, pl = env.observe()
        idx = np.flatnonzero(alive)
        a, mu = sample_actions(net, obs[idx], mask[idx], DEV)
        actions = np.full(n_games, -1, dtype=np.int64)
        actions[idx] = a.numpy()
        rew, _ = env.step(actions)
        valid[t, idx] = True
        player[t, idx] = torch.from_numpy(pl[idx].astype(np.int64))
        reward[t] += torch.from_numpy(rew)
        obs_l.append(torch.from_numpy(obs[idx]).half())
        legal_l.append(torch.from_numpy(mask[idx]))
        act_l.append(a)
        mu_l.append(mu)
        t_l.append(torch.full((len(idx),), t, dtype=torch.long))
        b_l.append(torch.from_numpy(idx))
    T = t + 1
    valid, player, reward = valid[:T], player[:T], reward[:T]
    obs_all, legal_all, act_all = torch.cat(obs_l), torch.cat(legal_l), torch.cat(act_l)
    mu_all, t_idx, b_idx = torch.cat(mu_l), torch.cat(t_l), torch.cat(b_l)
    n = len(act_all)
    t1 = time.time()
    print(f"自己対局 {n_games}局・{n}決定点（{t1 - t0:.0f}s）", flush=True)

    # v-trace（現在 = ターゲット = 正則化方策 = このネット → π = μ、ent = 0）
    with torch.no_grad():
        v_tgt = torch.cat([net(obs_all[i:i + 1024].float())[1] for i in range(0, n, 1024)])

    def grid(x):
        g = torch.zeros(T, n_games)
        g[t_idx, b_idx] = x
        return g

    mu_g = torch.where(valid, grid(mu_all), torch.ones(T, n_games))
    pflat = player[t_idx, b_idx]
    counts = [int((pflat == 0).sum()), int((pflat == 1).sum())]
    s_by = {}
    for v in rnad_vars:
        key = v["mucap"]
        if key in s_by:
            continue
        s = torch.zeros(n)
        vt = torch.zeros(n)
        for k in (0, 1):
            vt_k, s_k = v_trace_scalars(grid(v_tgt), valid, player, mu_g, mu_g, torch.zeros(T, n_games),
                                        reward[:, :, k], k, args.eta, inv_mu_cap=key)
            mine = pflat == k
            vt = torch.where(mine, vt_k[t_idx, b_idx], vt)
            s = torch.where(mine, s_k[t_idx, b_idx], s)
        s_by[key] = (vt, s)
    s_ref = s_by[0.0][1] if 0.0 in s_by else (s_by[rnad_vars[0]["mucap"]][1] if rnad_vars else torch.zeros(n))
    sa = s_ref.abs()
    top = torch.sort(s_ref ** 2, descending=True).values
    s_stats = {"s_abs_median": float(sa.median()), "s_abs_p99": float(sa.quantile(0.99)),
               "s_abs_max": float(sa.max()), "top0.1pct_share_of_s2": float(top[: max(1, n // 1000)].sum() / top.sum()),
               "mu_p1": float(mu_all.quantile(0.01)), "mu_median": float(mu_all.median())}
    print(json.dumps(s_stats), flush=True)
    batch_g = {"valid": valid, "player": player, "reward": reward, "t": t_idx, "b": b_idx}
    mrows = {v["name"]: mmd_rows(v, batch_g, v_tgt, n_games, args.seed + i) for i, v in enumerate(mmd_vars)}
    for v in mmd_vars:
        print(f"  {v['name']}: 方策の行 {mrows[v['name']][4]:.1%}", flush=True)

    # 局ごとの勾配（変種ごとに逆伝播）。統計は流しながら積む: 局を偶数番 A / 奇数番 B に分けた和と、2乗ノルムの和
    params = [p for p in net.parameters()]
    for p in params:
        p.requires_grad_(True)
    P = sum(p.numel() for p in params)
    acc = {v["name"]: {"A": torch.zeros(P, dtype=torch.float64), "B": torch.zeros(P, dtype=torch.float64),
                       "sq": 0.0, "nA": 0, "nB": 0} for v in variants}
    names = [v["name"] for v in variants]
    K = args.blocks
    cross = torch.zeros(K, len(variants), len(variants), dtype=torch.float64)  # 組ごとの Σ_i g_a,i·g_b,i
    Sblk = torch.zeros(K, len(variants), P, dtype=torch.float64)  # 組ごとの Σ_i g_a,i
    for g in range(n_games):
        rows = torch.nonzero(b_idx == g).squeeze(1)
        if len(rows) == 0:
            continue
        obs = obs_all[rows].float()
        legal = legal_all[rows]
        act = act_all[rows]
        idx, pmask, apos = pack_legal(legal, act)
        logits, v = net(obs)
        lg = logits.gather(1, idx)
        hs = {}
        for vi, var in enumerate(rnad_vars):
            vt, s = s_by[var["mucap"]]
            lv, ln = loss_terms(lg, v, pmask, apos, pflat[rows], torch.zeros_like(lg), vt[rows], s[rows],
                                counts, args.eta, args.beta, var["clip"], var["loss"])
            grads = torch.autograd.grad(lv + ln, params, retain_graph=vi + 1 < len(rnad_vars) or bool(mmd_vars))
            hs[var["name"]] = torch.cat([x.flatten() for x in grads]).double() * n_games  # 平均が全体の勾配になる尺度
        if mmd_vars:
            # 行ごとの出力 [log π(a), v, KL(π‖一様), KL(π‖混合)] に、変種ごとの係数を掛けて1回の逆伝播で束ねる
            lp = legal_log_policy(lg, pmask)
            pi_r = lp.exp() * pmask
            lp_a = lp.gather(1, apos[:, None]).squeeze(1)
            cnt = pmask.sum(1, keepdim=True).float()
            kl_u = (pi_r * (lp + cnt.log())).sum(1)
            outs = [lp_a, v, kl_u]
            mixes = sorted({var["magnet"] for var in mmd_vars if var["magnet"].startswith("mix")})
            for mname in mixes:
                eps = float(mname[3:])
                rho = torch.where(pmask, (1 - eps) * lp.detach().exp() + eps / cnt, torch.ones_like(lp))
                outs.append((pi_r * (lp - rho.log())).sum(1))
            Y = torch.cat(outs)
            m = len(rows)
            coefs = []
            for var in mmd_vars:
                adv, g_td, sp, sv, _ = mrows[var["name"]]
                c = torch.zeros(len(outs), m)
                part = var["part"]
                if part in ("pol", "all"):
                    c[0] = -sp[rows] * adv[rows]
                if part in ("val", "all"):
                    c[1] = 2 * sv[rows] * (v.detach() - g_td[rows])
                if part in ("mag", "all"):
                    k = 2 if var["magnet"] == "uniform" else 3 + mixes.index(var["magnet"])
                    c[k] = var["temp"] * sp[rows]
                coefs.append(c.reshape(-1))
            grads = torch.autograd.grad(Y, params, grad_outputs=torch.stack(coefs), is_grads_batched=True,
                                        retain_graph=g < args.check_batched)
            flat = torch.cat([x.reshape(len(mmd_vars), -1) for x in grads], 1).double()
            for k, var in enumerate(mmd_vars):
                hs[var["name"]] = flat[k]
            if g < args.check_batched:
                # 束ねた逆伝播の検査: 変種ごとに素直な損失を組んで逆伝播したものと比べる
                for k, var in enumerate(mmd_vars):
                    adv, g_td, sp, sv, _ = mrows[var["name"]]
                    part = var["part"]
                    loss = torch.zeros(())
                    if part in ("pol", "all"):
                        loss = loss - (sp[rows] * adv[rows] * lp_a).sum()
                    if part in ("val", "all"):
                        loss = loss + (sv[rows] * (v - g_td[rows]) ** 2).sum()
                    if part in ("mag", "all"):
                        kk = 2 if var["magnet"] == "uniform" else 3 + mixes.index(var["magnet"])
                        loss = loss + var["temp"] * (sp[rows] * outs[kk]).sum()
                    gx = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    ref_h = torch.cat([(x if x is not None else torch.zeros_like(q)).flatten()
                                       for x, q in zip(gx, params)]).double()
                    err = float((ref_h - flat[k]).norm() / max(float(ref_h.norm()), 1e-30))
                    assert err < 1e-4, f"束ねた逆伝播が素直な逆伝播と違う（{var['name']}、相対差 {err:.2e}）"
                    print(f"  検査 局 {g} {var['name']}: 束ねた逆伝播 = 素直な逆伝播（相対差 {err:.1e}）", flush=True)
        H = torch.stack([hs[nm] for nm in names])
        cross[g % K] += H @ H.T
        Sblk[g % K] += H
        for var in variants:
            h = hs[var["name"]]
            a = acc[var["name"]]
            half = "A" if g % 2 == 0 else "B"
            a[half] += h
            a["n" + half] += 1
            a["sq"] += float((h * h).sum())
        if (g + 1) % 32 == 0:
            print(f"  勾配 {g + 1}/{n_games}局（{time.time() - t1:.0f}s）", flush=True)

    ref = variants[0]["name"]
    out = {"weights": args.weights, "games": n_games, "samples": n, "s_stats": s_stats, "variants": {}}
    N_g = acc[ref]["nA"] + acc[ref]["nB"]
    nk = torch.bincount(torch.arange(N_g) % K, minlength=K).double()
    U, U_se = u_stat_jackknife(Sblk, cross, nk)
    out["u_stat_se"] = U_se.tolist()
    d = U.diag()
    cos_u = U / torch.sqrt(torch.clamp(d[:, None] * d[None, :], min=1e-300))
    cos_u[(d[:, None] <= 0) | (d[None, :] <= 0)] = float("nan")
    out["u_stat"] = {"names": names, "G_dot": U.tolist(), "cos": cos_u.tolist()}
    print("変種どうしの期待勾配のコサイン（U 統計量。|G|² ≤ 0 の変種は nan）:", flush=True)
    for i, a in enumerate(names):
        print(f"  {a:12s} |G|² {float(d[i]):10.4g}±{float(U_se[i, i]):.2g} "
              + " ".join(f"{float(cos_u[i, j]):6.3f}" for j in range(len(names))), flush=True)

    def summary(name):
        a = acc[name]
        N = a["nA"] + a["nB"]
        G = (a["A"] + a["B"]) / N
        tr = (a["sq"] - N * float((G * G).sum())) / (N - 1)  # 局ごとの分散の和
        g2 = float((G * G).sum()) - tr / N
        return G, tr, g2, a

    Gr, trr, g2r, ar = summary(ref)
    for var in variants:
        G, tr, g2, a = summary(var["name"])
        # 内積は「この変種の A と ref の B」「この変種の B と ref の A」の平均（局が重ならないので不偏）
        dot = 0.5 * (float((a["A"] / a["nA"] * ar["B"] / ar["nB"]).sum())
                     + float((a["B"] / a["nB"] * ar["A"] / ar["nA"]).sum()))
        rec = {"algo": var["algo"], **({k: var[k] for k in MMD_KEYS[1:]} if var["algo"] == "mmd" else {}),
               **({"keep_rate": mrows[var["name"]][4]} if var["algo"] == "mmd" else {}),
               "clip": var["clip"], "mucap": var["mucap"], "loss": var["loss"], "trSigma": tr, "G2": g2,
               "B_noise": tr / g2 if g2 > 0 else None, "dot_ref": dot,
               "cos_ref": dot / math.sqrt(g2 * g2r) if g2 > 0 and g2r > 0 else None,
               "eff_snr": dot * dot / (g2r * tr) if g2r > 0 else None,
               "trSigma_ratio_to_ref": tr / trr}
        out["variants"][var["name"]] = rec
        print(f"{var['name']:10s} trΣ {tr:10.4g}（ref の {tr / trr:.3g}倍） |G|² {g2:10.4g} "
              f"B_noise {rec['B_noise'] if rec['B_noise'] is None else round(rec['B_noise'])} "
              f"G·G_ref {dot:10.4g} cos {rec['cos_ref']} 有効SNR {rec['eff_snr']}", flush=True)
    out["sec"] = time.time() - t0
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
