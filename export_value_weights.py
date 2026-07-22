"""学習済み value ネット（train.py の out/）の重みを、tsuitate-bot側の
手書きforward passが読むRust定数配列として書き出す。

export_opp_move_weights.py と同じ方式（AUTO-GENERATED区間だけを管轄）だが、
こちらは学習を内包せず train.py が保存した out/model.pt + norm_stats.json を
読むだけ（value ネットは pairwise 補助loss等の学習設定が多く、学習は train.py に
一本化しておきたいため）。

使い方:
    python train.py --data data/value_data.csv --pairwise-data data/pairwise.csv \
        --pairwise-weight 20.0 --pairwise-margin 0.1 --out out/
    python export_value_weights.py --model-dir out/ \
        --out ../tsuitate-bot/src/value_nn.rs --note "data=value_data.csv seed=0"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from model import INPUT_DIM, ValueNet

BEGIN_MARKER = "// AUTO-GENERATED BEGIN (export_value_weights.py)"
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
    ap.add_argument("--model-dir", type=Path, default=Path("out"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    stats = json.loads((args.model_dir / "norm_stats.json").read_text())
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    if len(mean) != INPUT_DIM:
        raise ValueError(f"norm_statsの次元 {len(mean)} が INPUT_DIM={INPUT_DIM} と不一致")

    model = ValueNet()
    model.load_state_dict(torch.load(args.model_dir / "model.pt", map_location="cpu"))
    model.eval()
    state = model.state_dict()
    # model.py の Sequential: [0]Linear [1]ReLU [2]Dropout [3]Linear [4]ReLU [5]Dropout [6]Linear [7]Sigmoid
    w1 = state["net.0.weight"].numpy().astype(np.float64)  # (hidden, input)
    b1 = state["net.0.bias"].numpy().astype(np.float64)
    w2 = state["net.3.weight"].numpy().astype(np.float64)  # (hidden//2, hidden)
    b2 = state["net.3.bias"].numpy().astype(np.float64)
    w3 = state["net.6.weight"].numpy().astype(np.float64).reshape(-1)  # (hidden//2,)
    b3 = float(state["net.6.bias"].numpy()[0])
    h1, h2 = w1.shape[0], w2.shape[0]

    gen_lines = [
        BEGIN_MARKER,
        f"// 学習メタ: {args.note}" if args.note else "// 学習メタ: (--note 未指定)",
        "// 再生成: tsuitate-nn/export_value_weights.py --model-dir out/ "
        "--out ../tsuitate-bot/src/value_nn.rs",
        rust_array_1d("VALUE_NN_MEAN", mean),
        rust_array_1d("VALUE_NN_STD", std),
        rust_array_2d("VALUE_NN_W1", w1),
        rust_array_1d("VALUE_NN_B1", b1),
        rust_array_2d("VALUE_NN_W2", w2),
        rust_array_1d("VALUE_NN_B2", b2),
        rust_array_1d("VALUE_NN_W3", w3),
        f"pub const VALUE_NN_B3: f64 = {b3:.8e};",
        END_MARKER,
    ]
    gen_block = "\n".join(gen_lines)

    if args.out.exists():
        text = args.out.read_text()
        pattern = re.compile(re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
        if pattern.search(text):
            args.out.write_text(pattern.sub(gen_block, text))
            print(f"更新: {args.out}（AUTO-GENERATED区間のみ置き換え）")
            return
        raise ValueError(f"{args.out} に AUTO-GENERATED マーカーが見つからない。手で確認してください")

    template = f'''//! 粒子上のvalueネット（NN段階③、フェーズ2統合）。学習はtsuitate-nnの
//! train.py（勝敗回帰 + pairwise補助loss）、重み配列はexport_value_weights.pyで
//! 生成、forward passはここに手書き（evaluate()の粒子ループから
//! 候補×粒子のオーダーで呼ばれるホットパスのため、ONNX/推論クレートは使わない）。

{gen_block}

/// 学習時と同じ正規化 + Linear({INPUT_DIM}→{h1}) → ReLU → Linear({h1}→{h2}) → ReLU
/// → Linear({h2}→1) → Sigmoid のforward pass（Dropoutは推論時無効なので存在しない）。
/// 出力は [0,1]（手番側の勝率相当）
pub fn value_nn_forward(features: &[f64; {INPUT_DIM}]) -> f64 {{
    let mut x = [0.0f64; {INPUT_DIM}];
    for i in 0..{INPUT_DIM} {{
        x[i] = (features[i] - VALUE_NN_MEAN[i]) / VALUE_NN_STD[i];
    }}
    let mut h1 = [0.0f64; {h1}];
    for j in 0..{h1} {{
        let mut s = VALUE_NN_B1[j];
        for i in 0..{INPUT_DIM} {{
            s += VALUE_NN_W1[j][i] * x[i];
        }}
        h1[j] = s.max(0.0); // ReLU
    }}
    let mut h2 = [0.0f64; {h2}];
    for j in 0..{h2} {{
        let mut s = VALUE_NN_B2[j];
        for i in 0..{h1} {{
            s += VALUE_NN_W2[j][i] * h1[i];
        }}
        h2[j] = s.max(0.0); // ReLU
    }}
    let mut z = VALUE_NN_B3;
    for j in 0..{h2} {{
        z += VALUE_NN_W3[j] * h2[j];
    }}
    1.0 / (1.0 + (-z).exp()) // Sigmoid
}}
'''
    args.out.write_text(template)
    print(f"新規作成: {args.out}")


if __name__ == "__main__":
    main()
