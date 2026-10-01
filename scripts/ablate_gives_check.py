"""gives_check 特徴量のアブレーション: 同じ26列CSVから gives_check を落とした
25次元で学習し、26次元との val NLL 差を測る（2026-08-21、gives_check 追加の寄与確認）。

    ./.venv/bin/python scripts/ablate_gives_check.py --data data/opp_move_data_givescheck.csv --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opp_move_model import OppMoveNet
from train_opp_move import FEATURE_NAMES, build_groups, load_data, train

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = load_data(args.data)
    feats = [f for f in FEATURE_NAMES if f != "gives_check"]
    print(f"アブレーション: {len(feats)}次元（gives_check なし） / {df['game_id'].nunique()}局")

    groups_col = df["game_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    train_row_idx, _ = next(splitter.split(df, groups=groups_col))
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_row_idx] = True

    x_all = df[feats].to_numpy(dtype=np.float32)
    chosen_all = df["chosen"].to_numpy(dtype=np.float32)
    mean = x_all[train_mask].mean(axis=0)
    std = x_all[train_mask].std(axis=0)
    std[std == 0] = 1.0
    x_t = torch.from_numpy((x_all - mean) / std)
    chosen_t = torch.from_numpy(chosen_all)
    all_groups = build_groups(df)
    train_groups = [g for g in all_groups if train_mask[g[0]]]
    val_groups = [g for g in all_groups if not train_mask[g[0]]]

    model = OppMoveNet(input_dim=len(feats), hidden=16)
    model, best = train(
        model, x_t, chosen_t, train_groups, x_t, chosen_t, val_groups,
        200, 1e-2, 1e-4, 64, 20,
    )
    print(f"25次元（gives_checkなし） 最良val: {best}")

if __name__ == "__main__":
    main()
