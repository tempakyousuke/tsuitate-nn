"""ゲート2: 信念 prior を提案分布へ入れたとき、**live 粒子の占有信念**が
実際に良くなるかを対照と並べる。

belief_probe の出力（対照と条件つき）を (game_id, decision_id, sq) で突き合わせ、
p_all / p_strict の対数損失・Brier と、厳密粒子ゼロの決定点の割合を比べる。

使い方:
    python compare_probe.py --control data/belief_probe_human.csv \
        --runs live1=/tmp/probe_live1.csv guide1=/tmp/probe_guide1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_belief import binary_metrics

KEY = ["game_id", "decision_id", "sq"]


def summarize(name: str, df: pd.DataFrame) -> None:
    y = df["occ"].to_numpy(np.float64)
    a = binary_metrics(df["p_all"].to_numpy(np.float64), y)
    s = binary_metrics(df["p_strict"].to_numpy(np.float64), y)
    blind = df["n_strict"].to_numpy() == 0
    n_dec = df.groupby(["game_id", "decision_id"]).ngroups
    blind_dec = df[blind].groupby(["game_id", "decision_id"]).ngroups
    line = (
        f"{name:10s} n={a['n']:6d} 決定点={n_dec:4d} ブラインド={blind_dec:3d}"
        f" ({100 * blind_dec / max(n_dec, 1):4.1f}%) | "
        f"p_all {a['logloss']:.4f}/{a['auc']:.3f} | p_strict {s['logloss']:.4f}/{s['auc']:.3f}"
    )
    if blind.sum():
        b = binary_metrics(df["p_all"].to_numpy(np.float64)[blind], y[blind])
        line += f" | ブラインド時 p_all {b['logloss']:.4f}"
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="name=path の並び")
    args = ap.parse_args()

    control = pd.read_csv(args.control)
    runs = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        runs[name] = pd.read_csv(path)

    # 対照は多めの対局を含むことがあるので、全条件に共通の決定点だけで比べる
    keys = control[KEY]
    for df in runs.values():
        keys = keys.merge(df[KEY], on=KEY, how="inner")
    print(f"共通の行数: {len(keys)}\n")

    summarize("control", control.merge(keys, on=KEY, how="inner"))
    for name, df in runs.items():
        summarize(name, df.merge(keys, on=KEY, how="inner"))


if __name__ == "__main__":
    main()
