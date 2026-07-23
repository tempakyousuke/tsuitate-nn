"""particle_nn.rs の numpy クロスチェック期待値を再生成する。

Rustソースから AUTO-GENERATED 定数をパースし、同じ forward pass を numpy で
独立に再計算して、Rust側テスト（matches_numpy_cross_check）に貼る期待値を
出力する（opp_move_nn のクロスチェックと同じ手法。行列の転置・インデックス
取り違えの検出用）。重みを再学習したら実行して期待値を貼り替えること。

使い方: python scripts/crosscheck_particle.py ../tsuitate-bot/src/particle_nn.rs
"""

from __future__ import annotations

import re
import sys

import numpy as np


def parse_const(text: str, name: str) -> np.ndarray:
    m = re.search(rf"pub const {name}: \[f64; \d+\] = \[(.*?)\];", text, re.DOTALL)
    if m:
        return np.array([float(v) for v in m.group(1).split(",") if v.strip()])
    m = re.search(rf"pub const {name}: \[\[f64; \d+\]; \d+\] = \[(.*?)\n\];", text, re.DOTALL)
    if m:
        rows = re.findall(r"\[(.*?)\]", m.group(1))
        return np.array([[float(v) for v in row.split(",") if v.strip()] for row in rows])
    m = re.search(rf"pub const {name}: f64 = (.*?);", text)
    if m:
        return np.array(float(m.group(1)))
    raise ValueError(f"{name} が見つからない")


def forward(text: str, features: np.ndarray) -> float:
    mean = parse_const(text, "PARTICLE_NN_MEAN")
    std = parse_const(text, "PARTICLE_NN_STD")
    w1 = parse_const(text, "PARTICLE_NN_W1")
    b1 = parse_const(text, "PARTICLE_NN_B1")
    w2 = parse_const(text, "PARTICLE_NN_W2")
    b2 = parse_const(text, "PARTICLE_NN_B2")
    x = (features - mean) / std
    h = np.maximum(w1 @ x + b1, 0.0)
    return float(w2 @ h + b2)


def main() -> None:
    path = sys.argv[1]
    text = open(path).read()
    dim = len(parse_const(text, "PARTICLE_NN_MEAN"))

    zeros = np.zeros(dim)
    # 初期局面の真実に近いベクトル
    initial_like = np.zeros(dim)
    initial_like[4] = 1.0  # pieces_home
    initial_like[8:13] = 1.0  # pawns..golds home
    initial_like[13] = 1.0  # bishop_home
    initial_like[14] = 1.0  # rook_home
    initial_like[21] = 0.6  # defended_frac
    initial_like[23] = 0.02  # ply
    # 展開の進んだ粒子
    developed = np.zeros(dim)
    developed[0] = 1.0  # king_moved
    developed[1] = 2.0  # king_advance
    developed[3] = 1.5  # pawn_advance
    developed[4] = 0.5  # pieces_home
    developed[8] = 0.5
    developed[15] = 3.0  # pawn_advance_max
    developed[16] = 2.0  # nonpawn_advance_max
    developed[17] = 0.4  # promoted_count
    developed[18] = 0.4  # opp_hand_count
    developed[19] = 0.4  # attacked_by_me
    developed[22] = 0.2  # king_zone_attackers
    developed[23] = 0.8  # ply
    developed[24] = 0.4  # my_dead
    developed[25] = 1.0  # you_in_check

    for name, v in [("zeros", zeros), ("initial_like", initial_like), ("developed", developed)]:
        print(f"{name}: {forward(text, v)!r}")


if __name__ == "__main__":
    main()
