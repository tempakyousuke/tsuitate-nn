"""信念ネット（NN段階②）ゲート1の分解: 現行の粒子フィルタと部分集合ごとに比べる。

使い方:
    python analyze_belief.py --probe data/belief_probe_human.csv --model out_belief_seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from belief_model import BeliefNet
from train_belief import FEATURE_NAMES, binary_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.probe)
    stats = json.loads((args.model / "norm_stats.json").read_text())
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    model = BeliefNet()
    model.load_state_dict(torch.load(args.model / "model.pt", map_location="cpu"))
    model.eval()
    x = (df[FEATURE_NAMES].to_numpy(np.float32) - mean) / std
    with torch.no_grad():
        p_net = torch.sigmoid(model(torch.from_numpy(x.copy()))[0]).numpy().astype(np.float64)
    df["p_net"] = p_net
    y = df["occ"].to_numpy(np.float64)

    def row(name: str, mask) -> None:
        mask = np.asarray(mask)
        if mask.sum() == 0:
            print(f"{name:30s} n=0")
            return
        yy = y[mask]
        a = binary_metrics(df["p_all"].to_numpy(np.float64)[mask], yy)
        b = binary_metrics(p_net[mask], yy)
        print(
            f"{name:30s} n={a['n']:6d} 占有={yy.mean():.3f} | "
            f"粒子 {a['logloss']:.4f}/{a['auc']:.3f} | ネット {b['logloss']:.4f}/{b['auc']:.3f}"
        )

    print("=== 部分集合ごとの 対数損失/AUC ===")
    row("全体", np.ones(len(df), dtype=bool))
    row("厳密粒子ゼロ（ブラインド）", df["n_strict"].to_numpy() == 0)
    row("厳密粒子あり", df["n_strict"].to_numpy() > 0)
    row("序盤（相手手数<20）", df["opp_moves_n"].to_numpy() < 20 / 60)
    row("終盤（相手手数>=40）", df["opp_moves_n"].to_numpy() >= 40 / 60)
    row("未接触マス", df["untouched"].to_numpy() == 1)
    row("接触済みマス", df["untouched"].to_numpy() == 0)
    row("王手中の決定点", df["in_check"].to_numpy() == 1)

    print("\n=== 王手駒ビリーフ（王手中・自玉へ利く幾何のマス）===")
    geo = (df["king_line"].to_numpy() == 1) | (df["king_step"].to_numpy() == 1)
    ck = (df["in_check"].to_numpy() == 1) & geo
    true_ck = ck & (df["occ"].to_numpy() == 1)
    print(
        f"候補マス全体 n={ck.sum()} 実占有率={y[ck].mean():.4f} "
        f"粒子={df['p_all'].to_numpy()[ck].mean():.4f} ネット={p_net[ck].mean():.4f}"
    )
    print(
        f"真に相手駒が居るマス n={true_ck.sum()} "
        f"粒子の平均信念={df['p_all'].to_numpy()[true_ck].mean():.4f} "
        f"ネット={p_net[true_ck].mean():.4f}"
    )

    print("\n=== 打ちマス反則の直接証拠（drop_foul>0）===")
    d = df["drop_foul"].to_numpy() > 0
    print(
        f"n={d.sum()} 実占有率={y[d].mean():.4f} "
        f"粒子={df['p_all'].to_numpy()[d].mean():.4f} ネット={p_net[d].mean():.4f}"
    )

    print("\n=== 粒子とネットの混合 ===")
    for w in (0.25, 0.5, 0.75):
        mix = (1 - w) * df["p_all"].to_numpy(np.float64) + w * p_net
        m = binary_metrics(mix, y)
        print(f"  ネット重み{w:.2f}: 対数損失={m['logloss']:.4f} AUC={m['auc']:.4f}")


if __name__ == "__main__":
    main()
