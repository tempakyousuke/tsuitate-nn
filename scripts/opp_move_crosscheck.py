"""src/opp_move_nn.rs の定数を numpy で読み直し、`matches_numpy_cross_check` の5ケースの期待値を出す。

    python3 scripts/opp_move_crosscheck.py ../tsuitate-bot/src/opp_move_nn.rs
"""

import sys

import numpy as np

from opp_move_eval import parse


def case(**kw):
    x = np.zeros(26)
    for i, v in kw.items():
        x[int(i[1:])] = v
    return x


CASES = [
    ("[0.0; 26]", np.zeros(26)),
    ("pawn_push", case(f0=1.0, f13=1.0, f21=1.0, f22=1.0)),
    ("bishop_probe", case(f0=3.0, f4=1.0, f12=2.0, f18=1.0, f21=6.0, f22=1.0, f23=2.0)),
    ("en_prise_escape", case(f0=1.0, f18=1.0, f21=3.0, f24=1.0)),
    ("check_knight_drop", case(f3=1.0, f15=1.0, f25=1.0)),
]


def main():
    params = parse(sys.argv[1])
    mean, std, w1, b1, w2, b2 = params
    for name, x in CASES:
        z = (x - mean) / std
        h = np.maximum(w1 @ z + b1, 0.0)
        out = float(h @ w2 + b2)  # Rust の forward はクランプ前の値を返す
        print(f"{name:20s} {out!r}")


if __name__ == "__main__":
    main()
