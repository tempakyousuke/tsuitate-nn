"""MMD 学習器の価値の行の選び方が、価値の固定点をどれだけ偏らせるかを玩具で見る（docs/rl-learning-speed.md 10節の (a2)）。

    python rnad/toy_value_rows.py --out /tmp/value_rows.json

- 状態は二項格子の酔歩（時刻 t・位置 x、上下に 1/2 ずつ）。長さ L の後、終わりの位置 x_L から勝つ確率 sigmoid(c·(x_L − L/2))
  で結果 z = ±1。真の価値 V*(t, x) は後ろ向きに厳密に出す。序盤はほぼ 0、終盤は ±1 に近い（中盤と決着の付いた局面が混ざる）
- 価値は表（状態ごとに1つ）。目標は TD(λ)（λ = 0.8）、アドバンテージは GAE(λ_A = 0.8)。どちらも今の価値で作る
  （本物の `mmd.player_lambda_returns` と `mmd.select_rows` を通す）。真の価値から始めて、価値の行の選び方ごとにどこへ流れるかを見る
- 選び方: all（全行）/ uniform:0.25（無作為の 25%）/ pps（|A| に比例した確率で残し重み s̄/p。不偏）/
  filtered（|A| の 75%点以上だけ。Ataraxos）
- 出すもの: 決着の付いた状態（|V*| > 0.6）の mean|V| ÷ mean|V*|（訪問の重み）、V* を V へ回帰した全体の傾き、
  V の帯ごとの mean(G − V)、真の価値との平均の差。判定（docs）: pps と uniform:0.25 が all と ±0.02 で同じ値に着けば、
  価値は方策と同じ pps の行・重みで学んでよい
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mmd import V_BANDS, player_lambda_returns, select_rows  # noqa: E402

MODES = {"all": (("none",), ("all",)), "uniform0.25": (("none",), ("uniform", 0.25)),
         "pps0.25": (("pps", 0.25, 0.05), ("filtered",)), "filtered0.75": (("quantile", 0.75, 0.01), ("filtered",))}


def true_values(L, c):
    """V*[t, x]（t = 0..L、x = 0..t）。V*[L, x] = 2·sigmoid(c·(x − L/2)) − 1"""
    V = np.zeros((L + 1, L + 1))
    x = np.arange(L + 1)
    V[L] = 2 / (1 + np.exp(-c * (x - L / 2))) - 1
    for t in range(L - 1, -1, -1):
        V[t, : t + 1] = 0.5 * (V[t + 1, : t + 1] + V[t + 1, 1: t + 2])
    return V


def run(mode, args, seed):
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    L, N = args.length, args.episodes
    Vs = true_values(L, args.sharp)
    V = Vs[:L].copy()  # 学習する価値（決定点は t = 0..L−1）。真の価値から始める
    filt, vrows = MODES[mode]
    valid = torch.ones(L, N, dtype=torch.bool)
    player = torch.zeros(L, N, dtype=torch.long)
    t_idx, b_idx = valid.nonzero(as_tuple=True)
    visits = np.zeros_like(V)
    for it in range(args.iters):
        steps = rng.integers(0, 2, (N, L))
        pos = np.concatenate([np.zeros((N, 1), dtype=np.int64), np.cumsum(steps, 1)], 1)  # [N, L+1]
        p_win = (Vs[L, pos[:, L]] + 1) / 2
        z = np.where(rng.random(N) < p_win, 1.0, -1.0)
        reward = torch.zeros(L, N, 2)
        reward[L - 1, :, 0] = torch.from_numpy(z).float()
        reward[L - 1, :, 1] = -reward[L - 1, :, 0]
        batch = {"valid": valid, "player": player, "reward": reward, "t": t_idx, "b": b_idx}
        xs = pos[b_idx.numpy(), t_idx.numpy()]
        v = torch.from_numpy(V[t_idx.numpy(), xs]).float()
        g = player_lambda_returns(batch, v, args.lam)
        adv = g - v
        _, _, keep_v, wv, _ = select_rows(adv, filt, vrows, gen)
        m = max(int(keep_v.sum()), 1)
        grad = np.zeros_like(V)
        np.add.at(grad, (t_idx.numpy(), xs), (wv * (v - g)).numpy() / m)
        V -= args.lr * grad * (L * (L + 1) / 2)  # 状態の数を掛けて、状態あたりの歩幅を全行のときとそろえる
        V = np.clip(V, -1, 1)
        if it >= args.iters // 2:
            np.add.at(visits, (t_idx.numpy(), xs), 1)
    w = visits / visits.sum()
    Vt = Vs[:L]
    decided = (np.abs(Vt) > 0.6) & (w > 0)
    sw = w[decided]
    ratio = float((np.abs(V[decided]) * sw).sum() / (np.abs(Vt[decided]) * sw).sum())
    mask = w > 0
    vm, tm, wm = V[mask], Vt[mask], w[mask]
    vbar, tbar = (vm * wm).sum(), (tm * wm).sum()
    slope = float(((vm - vbar) * (tm - tbar) * wm).sum() / ((vm - vbar) ** 2 * wm).sum())
    bias = float(((V - Vt)[mask] * wm).sum())
    # 帯ごとの mean(G − V)（最後の反復の batch で）
    res = []
    for lo, hi in zip(V_BANDS[:-1], V_BANDS[1:]):
        sel = (v >= lo) & (v < hi)
        res.append(None if not bool(sel.any()) else round(float((g - v)[sel].mean()), 4))
    return {"decided_abs_ratio": ratio, "slope_true_on_V": slope, "mean_V_minus_true": bias,
            "mean_abs_err": float((np.abs(V - Vt)[mask] * wm).sum()), "resid_band": res,
            "keep_share": float(keep_v.float().mean()), "decided_share": float(w[np.abs(Vt) > 0.6].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--length", type=int, default=24)
    ap.add_argument("--sharp", type=float, default=1.5, help="終わりの位置から勝率への傾き（大きいほど終盤が決着している）")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = {"args": vars(args), "modes": {}}
    for mode in args.modes.split(","):
        rs = [run(mode, args, s) for s in range(args.seeds)]
        agg = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]) / math.sqrt(len(rs))))
               for k in ("decided_abs_ratio", "slope_true_on_V", "mean_V_minus_true", "mean_abs_err", "keep_share")}
        out["modes"][mode] = {"mean_se": agg, "runs": rs}
        print(f"{mode:13s} 決着の帯の |V|/|V*| {agg['decided_abs_ratio'][0]:.3f}±{agg['decided_abs_ratio'][1]:.3f} "
              f"傾き {agg['slope_true_on_V'][0]:.3f} 平均の差 {agg['mean_V_minus_true'][0]:+.4f} "
              f"|V−V*| {agg['mean_abs_err'][0]:.4f} 残した行 {agg['keep_share'][0]:.2f} "
              f"帯の残差 {rs[0]['resid_band']}（決着の帯の訪問 {rs[0]['decided_share']:.0%}）", flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
