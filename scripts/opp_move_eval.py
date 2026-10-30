"""opp_move NN の重み（Rust の AUTO-GENERATED 定数）を numpy で読み、CSV の候補集合で当てはまりを測る。

    python3 scripts/opp_move_eval.py NAME=../tsuitate-bot/src/opp_move_nn.rs [NAME=a.rs+b.rs@0.5 ...] -- data.csv [data.csv ...]

`a.rs+b.rs@α` は2つのモデルの候補集合内の確率を α : 1−α で混ぜたもの（配合の当たりを付けるとき用）。
入力 CSV は tsuitate-bot の scripts/opp_move/export_by_style.py の出力（style 列で系統ごとに分けて出す）。

系統（style 列）ごとに、正解手の平均 log p（候補集合内 softmax）・1位率・5位以内の率を出す。
Rust の推論と同じく、ロジットは clamp(−15, 15) してから温度 1 で softmax する。
"""

import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


def parse(path):
    src = open(path).read()

    def arr(name):
        m = re.search(rf"pub const {name}: [^=]+= (.*?);\n", src, re.S)
        nums = re.findall(r"[-+]?\d+\.\d+(?:e[-+]?\d+)?|[-+]?\d+e[-+]?\d+", m.group(1))
        return np.array([float(x) for x in nums])

    mean, std = arr("OPP_MOVE_NN_MEAN"), arr("OPP_MOVE_NN_STD")
    d = len(mean)
    w1 = arr("OPP_MOVE_NN_W1").reshape(-1, d)
    b1, w2 = arr("OPP_MOVE_NN_B1"), arr("OPP_MOVE_NN_W2")
    b2 = float(re.search(r"pub const OPP_MOVE_NN_B2: f64 = ([^;]+);", src).group(1))
    return mean, std, w1, b1, w2, b2


def forward(params, x):
    mean, std, w1, b1, w2, b2 = params
    z = (x - mean) / std
    h = np.maximum(z @ w1.T + b1, 0.0)
    return np.clip(h @ w2 + b2, -15.0, 15.0)


def score(params, df, feats):
    x = df[feats].to_numpy(dtype=np.float64)
    if isinstance(params[0], str):
        # 2つのモデルの候補集合内の確率を α : 1−α で混ぜる（log の世界で扱うため、混合後の確率の log を _s に入れる）
        _, pa, pb, alpha = params
        df = df.assign(_a=forward(pa, x), _b=forward(pb, x))
        out = defaultdict(lambda: [0, 0.0, 0, 0])
        for (_, _), g in df.groupby(["game_id", "decision_id"], sort=False):
            a, b = g["_a"].to_numpy(), g["_b"].to_numpy()
            pa_ = np.exp(a - a.max()); pa_ /= pa_.sum()
            pb_ = np.exp(b - b.max()); pb_ /= pb_.sum()
            pm = alpha * pa_ + (1 - alpha) * pb_
            ch = g["chosen"].to_numpy() > 0.5
            if ch.sum() != 1:
                continue
            lp = float(np.log(pm[ch][0]))
            rank = int((pm > pm[ch][0]).sum())
            st = g["style"].iloc[0] if "style" in g else "all"
            for key in (st, "_全体"):
                o = out[key]
                o[0] += 1; o[1] += lp; o[2] += rank == 0; o[3] += rank < 5
        return out
    s = forward(params, x)
    df = df.assign(_s=s)
    out = defaultdict(lambda: [0, 0.0, 0, 0])
    for (_, _), g in df.groupby(["game_id", "decision_id"], sort=False):
        sc = g["_s"].to_numpy()
        ch = g["chosen"].to_numpy() > 0.5
        if ch.sum() != 1:
            continue
        lse = np.log(np.exp(sc - sc.max()).sum()) + sc.max()
        lp = sc[ch][0] - lse
        rank = int((sc > sc[ch][0]).sum())
        st = g["style"].iloc[0] if "style" in g else "all"
        for key in (st, "_全体"):
            a = out[key]
            a[0] += 1
            a[1] += lp
            a[2] += rank == 0
            a[3] += rank < 5
    return out


def main():
    i = sys.argv.index("--")
    models = [a.partition("=") for a in sys.argv[1:i]]
    csvs = sys.argv[i + 1:]
    params = {}
    for name, _, path in models:
        if "+" in path:
            # a.rs+b.rs@α = α·p_a + (1−α)·p_b
            ab, _, alpha = path.partition("@")
            a, _, b = ab.partition("+")
            params[name] = ("mix", parse(a), parse(b), float(alpha))
        else:
            params[name] = parse(path)
    for path in csvs:
        df = pd.read_csv(path)
        feats = [c for c in df.columns if c not in ("game_id", "decision_id", "chosen", "style")]
        res = {name: score(p, df, feats) for name, p in params.items()}
        print(f"== {path}（{df['game_id'].nunique()} 局）")
        keys = sorted(next(iter(res.values())).keys())
        for k in keys:
            cells = []
            for name in params:
                n, lp, t1, t5 = res[name][k]
                cells.append(f"{name}: log p {lp / n:+.3f} / 1位 {100 * t1 / n:4.1f}% / 5位 {100 * t5 / n:4.1f}%")
            print(f"  {k:20s} n={res[next(iter(params))][k][0]:6d}  " + "  |  ".join(cells))


if __name__ == "__main__":
    main()
