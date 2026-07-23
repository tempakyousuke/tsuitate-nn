"""学習済み粒子尤度MLPの重みを、tsuitate-bot側の手書きforward passが読む
Rust定数配列として書き出す（export_opp_move_weights.py と同じ方式）。

使い方:
    python export_particle_weights.py --data data/particle_data.csv \
        --out ../tsuitate-bot/src/particle_nn.rs

出力ファイルの AUTO-GENERATED 区間だけがこのスクリプトの管轄。
forward pass 関数は tsuitate-bot 側で手書き済み（particle_nn.rs）。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from opp_move_model import OppMoveNet
from train_particle import FEATURE_NAMES, build_groups, load_data, train

BEGIN_MARKER = "// AUTO-GENERATED BEGIN (export_particle_weights.py)"
END_MARKER = "// AUTO-GENERATED END"


def rust_array_1d(name: str, values: np.ndarray) -> str:
    body = ", ".join(f"{v:.8e}" for v in values.tolist())
    return f"pub const {name}: [f64; {len(values)}] = [{body}];"


def rust_array_2d(name: str, values: np.ndarray) -> str:
    rows = []
    for row in values.tolist():
        body = ", ".join(f"{v:.8e}" for v in row)
        rows.append(f"    [{body}],")
    rows_str = "\n".join(rows)
    return f"pub const {name}: [[f64; {values.shape[1]}]; {values.shape[0]}] = [\n{rows_str}\n];"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-groups", type=int, default=64)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = load_data(args.data)
    groups_col = df["game_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_row_idx, _ = next(splitter.split(df, groups=groups_col))
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_row_idx] = True

    x_all = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    offset_all = df["offset"].to_numpy(dtype=np.float32)
    chosen_all = df["chosen"].to_numpy(dtype=np.float32)
    mean = x_all[train_mask].mean(axis=0)
    std = x_all[train_mask].std(axis=0)
    std[std == 0] = 1.0
    x_norm = (x_all - mean) / std

    x_t = torch.from_numpy(x_norm)
    offset_t = torch.from_numpy(offset_all)
    chosen_t = torch.from_numpy(chosen_all)
    all_groups = build_groups(df)
    train_groups = [g for g in all_groups if train_mask[g[0]]]
    val_groups = [g for g in all_groups if not train_mask[g[0]]]

    model = OppMoveNet(input_dim=len(FEATURE_NAMES), hidden=args.hidden)
    model, best_metrics = train(
        model,
        x_t,
        offset_t,
        chosen_t,
        train_groups,
        x_t,
        offset_t,
        chosen_t,
        val_groups,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.batch_groups,
        args.patience,
    )
    print(f"最良val: {best_metrics}")

    state = model.state_dict()
    w1 = state["net.0.weight"].numpy()
    b1 = state["net.0.bias"].numpy()
    w2 = state["net.2.weight"].numpy().reshape(-1)
    b2 = float(state["net.2.bias"].numpy()[0])

    gen_lines = [
        BEGIN_MARKER,
        f"// 学習: seed={args.seed} hidden={args.hidden} val_nll={best_metrics['nll']:.4f} "
        f"実効候補数={best_metrics['eff_candidates']:.1f} "
        f"top_half={best_metrics['top_half']:.3f} ({best_metrics['n_groups']}グループ)",
        f"// 再生成: tsuitate-nn/export_particle_weights.py --data {args.data} "
        "--out ../tsuitate-bot/src/particle_nn.rs",
        rust_array_1d("PARTICLE_NN_MEAN", mean),
        rust_array_1d("PARTICLE_NN_STD", std),
        rust_array_2d("PARTICLE_NN_W1", w1),
        rust_array_1d("PARTICLE_NN_B1", b1),
        rust_array_1d("PARTICLE_NN_W2", w2),
        f"pub const PARTICLE_NN_B2: f64 = {b2:.8e};",
        END_MARKER,
    ]
    gen_block = "\n".join(gen_lines)

    text = args.out.read_text()
    pattern = re.compile(re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(
            f"{args.out} に AUTO-GENERATED マーカーが見つからない。手で確認してください"
        )
    args.out.write_text(pattern.sub(gen_block, text))
    print(f"更新: {args.out}（AUTO-GENERATED区間のみ置き換え）")


if __name__ == "__main__":
    main()
