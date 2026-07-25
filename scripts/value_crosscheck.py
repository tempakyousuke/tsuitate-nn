"""value_nn.rs のクロスチェック期待値を再生成する。

**Rust ソースの AUTO-GENERATED 区間から定数をパースして** numpy で forward を
計算する（model.pt から計算すると「書き出しが正しいか」の検査にならないため）。
行列の転置・インデックス取り違えを検出するのが目的。

重みを差し替えたら必ず実行して、value_nn.rs の matches_numpy_cross_check の
期待値を貼り替えること。

使い方: python scripts/value_crosscheck.py [../tsuitate-bot/src/value_nn.rs]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(
    sys.argv[1] if len(sys.argv) > 1 else Path.home() / "Develop/tsuitate-bot/src/value_nn.rs"
)


def parse_array(text: str, name: str) -> np.ndarray:
    """`pub const NAME: ... = [...];` の数値をネスト構造ごと取り出す"""
    m = re.search(rf"pub const {name}:[^=]*=\s*(.*?);", text, re.S)
    if not m:
        raise SystemExit(f"{name} が見つからない")
    body = m.group(1)
    rows = re.findall(r"\[([^\[\]]*)\]", body)
    parsed = [[float(v) for v in re.findall(r"-?\d+\.?\d*e?[-+]?\d*", r)] for r in rows]
    arr = np.array(parsed, dtype=np.float64)
    return arr[0] if arr.shape[0] == 1 and name.endswith(("MEAN", "STD", "B1", "B2", "W3")) else arr


def main() -> None:
    text = SRC.read_text()
    mean = parse_array(text, "VALUE_NN_MEAN")
    std = parse_array(text, "VALUE_NN_STD")
    w1 = parse_array(text, "VALUE_NN_W1")
    b1 = parse_array(text, "VALUE_NN_B1")
    w2 = parse_array(text, "VALUE_NN_W2")
    b2 = parse_array(text, "VALUE_NN_B2")
    w3 = parse_array(text, "VALUE_NN_W3")
    b3 = float(re.search(r"pub const VALUE_NN_B3: f64 = (-?[\d.e+-]+)", text).group(1))
    n = len(mean)
    print(f"入力次元 {n} / W1 {w1.shape} / W2 {w2.shape} / W3 {w3.shape}")

    def forward(f: np.ndarray) -> float:
        x = (f - mean) / std
        h1 = np.maximum(w1 @ x + b1, 0.0)
        h2 = np.maximum(w2 @ h1 + b2, 0.0)
        z = float(w3 @ h2 + b3)
        return 1.0 / (1.0 + np.exp(-z))

    # value_nn.rs のテストと同じ4ケース（インデックスは
    # VALUE_FEATURE_NAMES + TRANSITION_FEATURE_NAMES の並び）
    state = 20  # VALUE_FEATURES
    quiet = np.zeros(n)
    quiet[15] = 0.3  # ply_progress
    quiet[state + 0] = 1.0  # moved_piece_value（歩）

    attack = np.zeros(n)
    attack[0] = 6.0  # material_diff
    attack[1] = 6.0  # my_hand_value
    attack[4] = 3.0  # king_pressure_on_opp
    attack[8] = 1.0  # opp_in_check
    attack[15] = 0.8  # ply_progress
    attack[state + 0] = 9.0  # moved_piece_value（飛）
    attack[state + 3] = 5.5  # captured_value
    attack[state + 4] = 5.5  # net_capture_then_recapture
    attack[state + 5] = 1.0  # gives_check

    blunder = np.zeros(n)
    blunder[11] = 6.0  # my_max_hanging
    blunder[13] = 4.0  # my_max_exchange_loss
    blunder[15] = 0.5  # ply_progress
    blunder[state + 0] = 6.0  # moved_piece_value（金）
    blunder[state + 1] = 6.0  # moved_piece_hanging_value
    blunder[state + 2] = 5.75  # moved_piece_exchange_loss
    blunder[state + 4] = -5.75  # net_capture_then_recapture

    for label, f in [("zeros", np.zeros(n)), ("quiet", quiet), ("attack", attack), ("blunder", blunder)]:
        print(f"{label:8} {forward(f)!r}")


if __name__ == "__main__":
    main()
