"""オフライン検証: eval_candidates（tsuitate-bot側）が出したCSVを読み、
学習済みモデルで各候補手をスコアリングして表示する。

使い方:
    cd tsuitate-bot && cargo run --release --bin eval_candidates -- \\
        <moves.json> <ply> <sente|gote> <USI手...> > /tmp/candidates.csv
    cd tsuitate-nn && python eval_scenario.py --candidates /tmp/candidates.csv --model out/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import ValueNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path("out"))
    args = ap.parse_args()

    with (args.model / "norm_stats.json").open() as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    feature_names = stats["feature_names"]

    df = pd.read_csv(args.candidates)
    if list(df.columns[1:]) != feature_names:
        raise ValueError(
            f"特徴量の並びが学習時と不一致: CSV={list(df.columns[1:])} "
            f"学習時={feature_names}"
        )

    model = ValueNet()
    model.load_state_dict(torch.load(args.model / "model.pt"))
    model.eval()

    x = (df[feature_names].to_numpy(dtype=np.float32) - mean) / std
    with torch.no_grad():
        scores = model(torch.from_numpy(x)).numpy()

    ranked = sorted(zip(df["usi"], scores), key=lambda t: -t[1])
    print("候補の評価（値が高いほど良い、手番側視点の勝率相当）:")
    for usi, score in ranked:
        print(f"  {usi:>8}  {score:.4f}")


if __name__ == "__main__":
    main()
