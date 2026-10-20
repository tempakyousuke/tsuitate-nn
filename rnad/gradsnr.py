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
from rnad import loss_terms, pack_legal, sample_actions, v_trace_scalars  # noqa: E402

DEV = torch.device("cpu")


def parse_variant(spec):
    name, _, rest = spec.partition(":")
    v = {"name": name, "clip": 10_000.0, "mucap": 0.0, "loss": "neurd"}
    for kv in filter(None, rest.split(",")):
        k, _, x = kv.partition("=")
        if k not in ("clip", "mucap", "loss"):
            raise SystemExit(f"変種の項 {k} は clip / mucap / loss のどれか")
        v[k] = x if k == "loss" else float(x)
    return v


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
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    variants = [parse_variant(s) for s in args.variant]
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
    for v in variants:
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
    s_ref = s_by[0.0][1] if 0.0 in s_by else s_by[variants[0]["mucap"]][1]
    sa = s_ref.abs()
    top = torch.sort(s_ref ** 2, descending=True).values
    s_stats = {"s_abs_median": float(sa.median()), "s_abs_p99": float(sa.quantile(0.99)),
               "s_abs_max": float(sa.max()), "top0.1pct_share_of_s2": float(top[: max(1, n // 1000)].sum() / top.sum()),
               "mu_p1": float(mu_all.quantile(0.01)), "mu_median": float(mu_all.median())}
    print(json.dumps(s_stats), flush=True)

    # 局ごとの勾配（変種ごとに逆伝播）。統計は流しながら積む: 局を偶数番 A / 奇数番 B に分けた和と、2乗ノルムの和
    params = [p for p in net.parameters()]
    for p in params:
        p.requires_grad_(True)
    P = sum(p.numel() for p in params)
    acc = {v["name"]: {"A": torch.zeros(P, dtype=torch.float64), "B": torch.zeros(P, dtype=torch.float64),
                       "sq": 0.0, "nA": 0, "nB": 0} for v in variants}
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
        for vi, var in enumerate(variants):
            vt, s = s_by[var["mucap"]]
            lv, ln = loss_terms(lg, v, pmask, apos, pflat[rows], torch.zeros_like(lg), vt[rows], s[rows],
                                counts, args.eta, args.beta, var["clip"], var["loss"])
            grads = torch.autograd.grad(lv + ln, params, retain_graph=vi + 1 < len(variants))
            h = torch.cat([x.flatten() for x in grads]).double() * n_games  # 平均が全体の勾配になる尺度
            a = acc[var["name"]]
            half = "A" if g % 2 == 0 else "B"
            a[half] += h
            a["n" + half] += 1
            a["sq"] += float((h * h).sum())
        if (g + 1) % 32 == 0:
            print(f"  勾配 {g + 1}/{n_games}局（{time.time() - t1:.0f}s）", flush=True)

    ref = variants[0]["name"]
    out = {"weights": args.weights, "games": n_games, "samples": n, "s_stats": s_stats, "variants": {}}

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
        rec = {"clip": var["clip"], "mucap": var["mucap"], "loss": var["loss"], "trSigma": tr, "G2": g2,
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
