"""学習済み信念ネット（train_belief.py の out/）の重みを、tsuitate-bot 側の
手書き forward pass が読む Rust 定数配列として書き出す。

export_value_weights.py と同じ方式（AUTO-GENERATED 区間だけを管轄）。
書き出すのは**占有ヘッド**だけ（駒種ヘッドは学習時の補助loss として trunk を
整えるためのもので、Rust 側の用途はまだ無い）。

使い方:
    python train_belief.py --data data/belief_arena570.csv --out out_belief570_seed3/
    python export_belief_weights.py --model-dir out_belief570_seed3/ \
        --out ../tsuitate-bot/src/belief_nn.rs --note "arena570 seed3"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from belief_model import INPUT_DIM, BeliefNet

BEGIN_MARKER = "// AUTO-GENERATED BEGIN (export_belief_weights.py)"
END_MARKER = "// AUTO-GENERATED END"


def rust_array_1d(name: str, values: np.ndarray) -> str:
    body = ", ".join(f"{v:.8e}" for v in values.tolist())
    return f"pub const {name}: [f64; {len(values)}] = [{body}];"


def rust_array_2d(name: str, values: np.ndarray) -> str:
    """2次元の重みは `static` で置く（`const` だと使用箇所ごとに実体が作られるため
    clippy::large_const_arrays が出る。信念ネットの W1 は 35×64 でその閾値を超える）"""
    rows = []
    for row in values.tolist():
        body = ", ".join(f"{v:.8e}" for v in row)
        rows.append(f"    [{body}],")
    rows_str = "\n".join(rows)
    return f"pub static {name}: [[f64; {values.shape[1]}]; {values.shape[0]}] = [\n{rows_str}\n];"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    stats = json.loads((args.model_dir / "norm_stats.json").read_text())
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    if len(mean) != INPUT_DIM:
        raise ValueError(f"norm_stats の次元 {len(mean)} が INPUT_DIM={INPUT_DIM} と不一致")

    model = BeliefNet()
    model.load_state_dict(torch.load(args.model_dir / "model.pt", map_location="cpu"))
    model.eval()
    s = model.state_dict()
    w1 = s["trunk.0.weight"].numpy().astype(np.float64)
    b1 = s["trunk.0.bias"].numpy().astype(np.float64)
    w2 = s["trunk.2.weight"].numpy().astype(np.float64)
    b2 = s["trunk.2.bias"].numpy().astype(np.float64)
    w3 = s["occ_head.weight"].numpy().astype(np.float64).reshape(-1)
    b3 = float(s["occ_head.bias"].numpy()[0])
    h1, h2 = w1.shape[0], w2.shape[0]

    gen_lines = [
        BEGIN_MARKER,
        f"// 学習メタ: {args.note}" if args.note else "// 学習メタ: (--note 未指定)",
        "// 再生成: tsuitate-nn/export_belief_weights.py --model-dir out_belief.../ "
        "--out ../tsuitate-bot/src/belief_nn.rs",
        rust_array_1d("BELIEF_NN_MEAN", mean),
        rust_array_1d("BELIEF_NN_STD", std),
        rust_array_2d("BELIEF_NN_W1", w1),
        rust_array_1d("BELIEF_NN_B1", b1),
        rust_array_2d("BELIEF_NN_W2", w2),
        rust_array_1d("BELIEF_NN_B2", b2),
        rust_array_1d("BELIEF_NN_W3", w3),
        f"pub const BELIEF_NN_B3: f64 = {b3:.8e};",
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
        raise ValueError(f"{args.out} に AUTO-GENERATED マーカーが無い。手で確認してください")

    template = f'''//! 信念ネット（NN段階②）の推論。学習は tsuitate-nn の train_belief.py、
//! 重み配列は export_belief_weights.py で生成、forward pass はここに手書き
//! （value_nn.rs / opp_move_nn.rs と同じ規約。推論クレートは持ち込まない）。
//!
//! 入力は `belief_features.rs` の35特徴量（マス1つぶん）、出力はそのマスに
//! **相手の駒が居る確率**。1決定点あたり81回しか呼ばないので、
//! リプレイの内側から呼ぶのではなく decision ごとに1度だけ盤面ぶん計算して
//! 使い回す（`board_occupancy`）。

use crate::belief_features::{{BELIEF_FEATURES, BeliefContext, all_squares, sq_index}};

{gen_block}

/// 学習時と同じ正規化 + Linear({INPUT_DIM}→{h1}) → ReLU → Linear({h1}→{h2}) → ReLU
/// → Linear({h2}→1) の forward pass。返すのは**ロジット**
pub fn occupancy_logit(features: &[f64; BELIEF_FEATURES]) -> f64 {{
    let mut x = [0.0f64; BELIEF_FEATURES];
    for i in 0..BELIEF_FEATURES {{
        x[i] = (features[i] - BELIEF_NN_MEAN[i]) / BELIEF_NN_STD[i];
    }}
    let mut h1 = [0.0f64; {h1}];
    for j in 0..{h1} {{
        let mut s = BELIEF_NN_B1[j];
        for i in 0..BELIEF_FEATURES {{
            s += BELIEF_NN_W1[j][i] * x[i];
        }}
        h1[j] = s.max(0.0);
    }}
    let mut h2 = [0.0f64; {h2}];
    for j in 0..{h2} {{
        let mut s = BELIEF_NN_B2[j];
        for i in 0..{h1} {{
            s += BELIEF_NN_W2[j][i] * h1[i];
        }}
        h2[j] = s.max(0.0);
    }}
    let mut out = BELIEF_NN_B3;
    for i in 0..{h2} {{
        out += BELIEF_NN_W3[i] * h2[i];
    }}
    // 分布外入力への安全弁（opp_move_nn の clamp と同じ趣旨）。
    // ロジット ±12 は確率 6e-6 〜 1-6e-6 に相当し、実運用の範囲は覆う
    out.clamp(-12.0, 12.0)
}}

/// 占有確率（[0,1]）
pub fn occupancy(features: &[f64; BELIEF_FEATURES]) -> f64 {{
    1.0 / (1.0 + (-occupancy_logit(features)).exp())
}}

/// 盤面ぶんの占有確率。自駒のマスは 0（相手の駒は居ない）
pub fn board_occupancy(ctx: &BeliefContext) -> [f64; 81] {{
    let mut out = [0.0f64; 81];
    for sq in all_squares() {{
        if ctx.is_mine(sq) {{
            continue;
        }}
        out[sq_index(sq)] = occupancy(&ctx.features(sq));
    }}
    out
}}
'''
    args.out.write_text(template)
    print(f"作成: {args.out}")


if __name__ == "__main__":
    main()
