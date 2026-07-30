"""学習済み玉位置ネット（train_king_belief.py の out/）の重みを、tsuitate-bot 側の
手書き forward pass が読む Rust 定数配列として書き出す。

export_belief_weights.py と同じ方式（AUTO-GENERATED 区間だけを管轄）。

使い方:
    python train_king_belief.py --data data/king_belief_data.csv --out out_king_seed0/
    python export_king_belief_weights.py --model-dir out_king_seed0/ \
        --out ../tsuitate-bot/src/king_belief_nn.rs --note "arena2400 seed0"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from king_belief_model import INPUT_DIM, KingBeliefNet

BEGIN_MARKER = "// AUTO-GENERATED BEGIN (export_king_belief_weights.py)"
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

    model = KingBeliefNet()
    model.load_state_dict(torch.load(args.model_dir / "model.pt", map_location="cpu"))
    model.eval()
    s = model.state_dict()
    w1 = s["trunk.0.weight"].numpy().astype(np.float64)
    b1 = s["trunk.0.bias"].numpy().astype(np.float64)
    w2 = s["trunk.2.weight"].numpy().astype(np.float64)
    b2 = s["trunk.2.bias"].numpy().astype(np.float64)
    w3 = s["score_head.weight"].numpy().astype(np.float64).reshape(-1)
    b3 = float(s["score_head.bias"].numpy()[0])
    h1, h2 = w1.shape[0], w2.shape[0]

    gen_lines = [
        BEGIN_MARKER,
        f"// 学習メタ: {args.note}" if args.note else "// 学習メタ: (--note 未指定)",
        "// 再生成: tsuitate-nn/export_king_belief_weights.py --model-dir out_king.../ "
        "--out ../tsuitate-bot/src/king_belief_nn.rs",
        rust_array_1d("KING_NN_MEAN", mean),
        rust_array_1d("KING_NN_STD", std),
        rust_array_2d("KING_NN_W1", w1),
        rust_array_1d("KING_NN_B1", b1),
        rust_array_2d("KING_NN_W2", w2),
        rust_array_1d("KING_NN_B2", b2),
        rust_array_1d("KING_NN_W3", w3),
        f"pub const KING_NN_B3: f64 = {b3:.8e};",
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

    template = f'''//! 玉位置ネット（king belief net）の推論。学習は tsuitate-nn の
//! train_king_belief.py、重み配列は export_king_belief_weights.py で生成、
//! forward pass はここに手書き（belief_nn.rs / value_nn.rs と同じ規約）。
//!
//! 段階②の pointwise 信念ネットが表現できない「玉1枚の同時分布」を、
//! **候補マス集合（deduce::opp_king_candidates − 自駒マス）上の softmax**
//! として出す。入力は belief_features の35特徴量 + KING_EXTRA = 36次元。
//! 1決定点あたり候補マス数（平均33）回の forward pass なので安い。

use crate::belief_features::{{BELIEF_FEATURES, BeliefContext, KING_EXTRA_FEATURES}};
use crate::board::Coord;
use std::collections::BTreeSet;

pub const KING_NN_INPUT: usize = BELIEF_FEATURES + KING_EXTRA_FEATURES;

{gen_block}

/// 学習時と同じ正規化 + Linear({INPUT_DIM}→{h1}) → ReLU → Linear({h1}→{h2}) → ReLU
/// → Linear({h2}→1) の forward pass。返すのは softmax 前の**スコア**。
/// clamp は分布外入力への安全弁（opp_move_nn と同じ趣旨）
pub fn king_score(features: &[f64; KING_NN_INPUT]) -> f64 {{
    let mut x = [0.0f64; KING_NN_INPUT];
    for i in 0..KING_NN_INPUT {{
        x[i] = (features[i] - KING_NN_MEAN[i]) / KING_NN_STD[i];
    }}
    let mut h1 = [0.0f64; {h1}];
    for j in 0..{h1} {{
        let mut s = KING_NN_B1[j];
        for i in 0..KING_NN_INPUT {{
            s += KING_NN_W1[j][i] * x[i];
        }}
        h1[j] = s.max(0.0);
    }}
    let mut h2 = [0.0f64; {h2}];
    for j in 0..{h2} {{
        let mut s = KING_NN_B2[j];
        for i in 0..{h1} {{
            s += KING_NN_W2[j][i] * h1[i];
        }}
        h2[j] = s.max(0.0);
    }}
    let mut out = KING_NN_B3;
    for i in 0..{h2} {{
        out += KING_NN_W3[i] * h2[i];
    }}
    out.clamp(-15.0, 15.0)
}}

/// 候補マス集合上の相手玉位置分布（正規化済み）。
/// 学習時と同じマスク（deduce の候補集合 − 自駒マス）を適用する。
/// マスクで空になったら（deduce の健全性違反時のみ起きうる）候補集合そのままの
/// 一様分布に落とす（玉位置の質量を潰す事故を避ける。ansatsu 回帰の教訓）
pub fn king_distribution(ctx: &BeliefContext, cands: &BTreeSet<Coord>) -> Vec<(Coord, f64)> {{
    let usable: Vec<Coord> = cands.iter().copied().filter(|&c| !ctx.is_mine(c)).collect();
    if usable.is_empty() {{
        let n = cands.len().max(1) as f64;
        return cands.iter().map(|&c| (c, 1.0 / n)).collect();
    }}
    let scores: Vec<f64> = usable
        .iter()
        .map(|&c| {{
            let base = ctx.features(c);
            let extra = ctx.king_extra(c);
            let mut x = [0.0f64; KING_NN_INPUT];
            x[..BELIEF_FEATURES].copy_from_slice(&base);
            x[BELIEF_FEATURES..].copy_from_slice(&extra);
            king_score(&x)
        }})
        .collect();
    let max = scores.iter().copied().fold(f64::MIN, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
    let z: f64 = exps.iter().sum();
    usable
        .into_iter()
        .zip(exps)
        .map(|(c, e)| (c, e / z))
        .collect()
}}
'''
    args.out.write_text(template)
    print(f"作成: {args.out}")


if __name__ == "__main__":
    main()
