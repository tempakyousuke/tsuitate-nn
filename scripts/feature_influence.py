"""各特徴量が出力にどれだけ効いているかを、検証データ上の実測で測る。

第1層の重みだけを見ても意味がない（正規化後スケール・後段の打ち消しがある）ので、
**その列だけを平均値に潰したときの出力変化**（permutation importance の
ablation 版）を使う。値は「その特徴量を殺したときの平均絶対スコア変化」。

接近特徴量を足したのに重みがゼロなら、勝敗ラベルからの credit assignment が
効いていないということ（NN方向フェーズ1で transition 特徴量が踏んだ問題と同じ）。

使い方: python scripts/feature_influence.py --model out_approach_seed0 --data data/value_data_approach.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import ValueNet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=20000)
    args = ap.parse_args()

    with (args.model / "norm_stats.json").open() as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    names = stats["feature_names"]

    df = pd.read_csv(args.data).head(args.rows)
    x = (df[names].to_numpy(dtype=np.float32) - mean) / std

    model = ValueNet()
    model.load_state_dict(torch.load(args.model / "model.pt"))
    model.eval()

    with torch.no_grad():
        base = model(torch.from_numpy(x)).numpy()
        rows = []
        for i, n in enumerate(names):
            killed = x.copy()
            killed[:, i] = 0.0  # 正規化後の0 = その特徴量の平均値
            score = model(torch.from_numpy(killed)).numpy()
            rows.append((n, float(np.abs(score - base).mean())))

    rows.sort(key=lambda t: -t[1])
    print(f"{'特徴量':<28} {'殺したときの平均|Δスコア|':>12}")
    for n, v in rows:
        mark = " ←接近" if n in {
            "my_prey_proximity",
            "opp_prey_proximity",
            "my_king_siege",
            "opp_king_siege",
            "king_approach",
            "prey_approach",
            "king_closeness_after",
        } else ""
        print(f"{n:<28} {v:>12.5f}{mark}")


if __name__ == "__main__":
    main()
