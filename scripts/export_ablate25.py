"""切り分け用: 新データ×旧25特徴量（gives_check なし）で学習した重みを
26次元の Rust 定数として書き出す（gives_check の行は 0、MEAN/STD はダミー安全値）。
「m020/m026/m028 の悪化はデータ由来か特徴量由来か」を同じエンジンで測るための
一時的な重みで、採用対象ではない。

    ./.venv/bin/python scripts/export_ablate25.py --data data/opp_move_data_givescheck.csv --seed 2 --out ../tsuitate-bot/src/opp_move_nn.rs
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opp_move_model import OppMoveNet
from train_opp_move import FEATURE_NAMES, build_groups, load_data, train

BEGIN = "// AUTO-GENERATED BEGIN (export_opp_move_weights.py)"
END = "// AUTO-GENERATED END"


def arr1(name, v):
    body = ", ".join(f"{x:.8e}" for x in v.tolist())
    return f"pub const {name}: [f64; {len(v)}] = [{body}];"


def arr2(name, v):
    rows = "\n".join("    [" + ", ".join(f"{x:.8e}" for x in row) + "]," for row in v.tolist())
    return f"pub const {name}: [[f64; {v.shape[1]}]; {v.shape[0]}] = [\n{rows}\n];"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = load_data(args.data)
    feats = [f for f in FEATURE_NAMES if f != "gives_check"]
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
    model, best = train(model, x_t, chosen_t, train_groups, x_t, chosen_t, val_groups,
                        200, 1e-2, 1e-4, 64, 20)
    print(f"最良val: {best}")

    w1 = model.net[0].weight.detach().numpy().astype(np.float64)  # (16,25)
    b1 = model.net[0].bias.detach().numpy().astype(np.float64)
    w2 = model.net[2].weight.detach().numpy().astype(np.float64).reshape(-1)
    b2 = float(model.net[2].bias.detach().numpy()[0])

    # 26次元へパディング（gives_check 列 = 重み0・mean 0・std 1）
    mean26 = np.append(mean.astype(np.float64), 0.0)
    std26 = np.append(std.astype(np.float64), 1.0)
    w1_26 = np.concatenate([w1, np.zeros((16, 1))], axis=1)

    gen = "\n".join([
        BEGIN,
        "// 【切り分け用・一時重み】新データ×旧25特徴量（gives_check の重み0パディング）",
        f"// 学習: seed={args.seed} hidden=16 val_nll={best['nll']:.4f} val_top1={best['top1']:.3f}",
        arr1("OPP_MOVE_NN_MEAN", mean26),
        arr1("OPP_MOVE_NN_STD", std26),
        arr2("OPP_MOVE_NN_W1", w1_26),
        arr1("OPP_MOVE_NN_B1", b1),
        arr1("OPP_MOVE_NN_W2", w2),
        f"pub const OPP_MOVE_NN_B2: f64 = {b2:.8e};",
        END,
    ])
    src = args.out.read_text()
    pattern = re.compile(re.escape(BEGIN) + ".*?" + re.escape(END), re.S)
    args.out.write_text(pattern.sub(lambda _: gen, src))
    print(f"更新: {args.out}（切り分け用25次元パディング重み）")


if __name__ == "__main__":
    main()
