"""学習済み opp_move MLP の重みを、tsuitate-bot側の手書きforward passが読む
Rust定数配列として書き出す（ONNX/推論クレートは使わない設計、詳細はplan参照）。

呼び出し頻度が1手の意思決定あたり最大10万回超というオーダーのため、
ONNXランタイム経由の推論は避け、Linear->ReLU->Linear の小さいforward passを
Rust側に手書きし、その入力になる重み・正規化統計だけをここで生成する。

使い方:
    python export_opp_move_weights.py --data data/opp_move_data.csv \
        --out ../tsuitate-bot/src/opp_move_nn.rs

出力ファイルの上半分（AUTO-GENERATED区間）だけがこのスクリプトの管轄。
下半分の forward pass 関数は tsuitate-bot側で手書きするので、既存ファイルが
あれば区間だけ置き換える（無ければ新規作成しforward pass雛型も添える）。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from opp_move_model import OppMoveNet
from train_opp_move import FEATURE_NAMES, build_groups, load_data, train

BEGIN_MARKER = "// AUTO-GENERATED BEGIN (export_opp_move_weights.py)"
END_MARKER = "// AUTO-GENERATED END"

FORWARD_PASS_TEMPLATE = '''//! opp_move_weight のNN版（NN段階①-a）。tsuitate-nn/train_opp_move.pyで
//! 線形モデルに対する頑健な改善を確認済み（4シードでNLL/top-1とも一貫して
//! 改善）。重み配列はexport_opp_move_weights.pyで生成、forward passは
//! ここに手書き（呼び出し頻度が1手あたり最大10万回超のオーダーのため、
//! ONNX/推論クレートは使わずLinear->ReLU->Linearを直接計算する）。

{gen_block}

/// 学習時と同じ正規化 + Linear({input_dim}→{hidden}) → ReLU → Linear({hidden}→1) のforward pass。
/// 出力はlogit（Sigmoidではない）。呼び出し側は `exp(logit)` として使う
pub fn opp_move_nn_forward(features: &[f64; {input_dim}]) -> f64 {{
    let mut x = [0.0f64; {input_dim}];
    for i in 0..{input_dim} {{
        x[i] = (features[i] - OPP_MOVE_NN_MEAN[i]) / OPP_MOVE_NN_STD[i];
    }}
    let mut h = [0.0f64; {hidden}];
    for j in 0..{hidden} {{
        let mut s = OPP_MOVE_NN_B1[j];
        for i in 0..{input_dim} {{
            s += OPP_MOVE_NN_W1[j][i] * x[i];
        }}
        h[j] = s.max(0.0); // ReLU
    }}
    let mut out = OPP_MOVE_NN_B2;
    for j in 0..{hidden} {{
        out += OPP_MOVE_NN_W2[j] * h[j];
    }}
    out
}}
'''


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
    train_row_idx, _val_row_idx = next(splitter.split(df, groups=groups_col))
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_row_idx] = True

    x_all = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    chosen_all = df["chosen"].to_numpy(dtype=np.float32)
    mean = x_all[train_mask].mean(axis=0)
    std = x_all[train_mask].std(axis=0)
    std[std == 0] = 1.0
    x_norm = (x_all - mean) / std

    x_t = torch.from_numpy(x_norm)
    chosen_t = torch.from_numpy(chosen_all)
    all_groups = build_groups(df)
    train_groups = [g for g in all_groups if train_mask[g[0]]]
    val_groups = [g for g in all_groups if not train_mask[g[0]]]

    model = OppMoveNet(hidden=args.hidden)
    model, best_metrics = train(
        model,
        x_t,
        chosen_t,
        train_groups,
        x_t,
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
    w1 = state["net.0.weight"].numpy()  # (hidden, input)
    b1 = state["net.0.bias"].numpy()  # (hidden,)
    w2 = state["net.2.weight"].numpy().reshape(-1)  # (hidden,)  net.2: Linear(hidden,1)
    b2 = float(state["net.2.bias"].numpy()[0])

    gen_lines = [
        BEGIN_MARKER,
        f"// 学習: seed={args.seed} hidden={args.hidden} val_nll={best_metrics['nll']:.4f} "
        f"val_top1={best_metrics['top1']:.3f} ({best_metrics['n_groups']}決定点)",
        "// 再生成: tsuitate-nn/export_opp_move_weights.py --data data/opp_move_data.csv "
        "--out ../tsuitate-bot/src/opp_move_nn.rs",
        rust_array_1d("OPP_MOVE_NN_MEAN", mean),
        rust_array_1d("OPP_MOVE_NN_STD", std),
        rust_array_2d("OPP_MOVE_NN_W1", w1),
        rust_array_1d("OPP_MOVE_NN_B1", b1),
        rust_array_1d("OPP_MOVE_NN_W2", w2),
        f"pub const OPP_MOVE_NN_B2: f64 = {b2:.8e};",
        END_MARKER,
    ]
    gen_block = "\n".join(gen_lines)

    if args.out.exists():
        text = args.out.read_text()
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL
        )
        if pattern.search(text):
            new_text = pattern.sub(gen_block, text)
            args.out.write_text(new_text)
            print(f"更新: {args.out}（AUTO-GENERATED区間のみ置き換え）")
            return
        raise ValueError(
            f"{args.out} に AUTO-GENERATED マーカーが見つからない。手で確認してください"
        )

    args.out.write_text(
        FORWARD_PASS_TEMPLATE.format(
            gen_block=gen_block, input_dim=len(FEATURE_NAMES), hidden=args.hidden
        )
    )
    print(f"新規作成: {args.out}")


if __name__ == "__main__":
    main()
