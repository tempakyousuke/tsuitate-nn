"""学習ログ（<out>/log.jsonl）の見張りの値を、step の窓ごとに平均して並べる（tsuitate-bot の docs/rl-learning-speed.md 10.7）。

    # MMD の腕と対照の R-NaD を、同じ窓で並べる（締まり方が MMD だけのものか、区切りの直後の R-NaD にもあるか）
    .venv/bin/python rnad/log_windows.py out_rnad/ab_mmd_s1/log.jsonl out_rnad/deduce_s10000/log.jsonl \\
        --from-step 108000 --to-step 108600 --width 100
    # 本線の前の区切りの直後とも比べる
    .venv/bin/python rnad/log_windows.py out_rnad/deduce_s10000/log.jsonl --from-step 98000 --to-step 98600 --width 100

窓は (from + i·width, from + (i+1)·width]（更新の行の step は更新の後の番号）。無いキーは空欄。
"""

import argparse
import json
import math
import pathlib

DEFAULT_KEYS = "entropy_mean,pi_top_gt_0999,plies,fouls_per_side,magnet_kl,grad_norm_p50,clip_rate"


def windows(rows, keys, start, stop, width):
    """[(窓の終わりの step, 行の数, {キー: 平均 or None})]"""
    out = []
    a = start
    while a < stop:
        b = min(a + width, stop)
        w = [r for r in rows if a < r["step"] <= b]
        means = {}
        for k in keys:
            xs = [r[k] for r in w if isinstance(r.get(k), (int, float)) and math.isfinite(r[k])]
            means[k] = sum(xs) / len(xs) if xs else None
        out.append((b, len(w), means))
        a = b
    return out


def read_rows(path):
    """更新の行だけ（評価の行・パネルの行は除く）"""
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        r = json.loads(line)
        if "eval" in r or "collect_sec" not in r or "step" not in r:
            continue
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--from-step", type=int, required=True)
    ap.add_argument("--to-step", type=int, required=True)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--keys", default=DEFAULT_KEYS, help="カンマ区切り")
    args = ap.parse_args()
    keys = [k for k in args.keys.split(",") if k]
    for path in args.logs:
        print(f"## {path}")
        print("| 窓の終わり | 行 | " + " | ".join(keys) + " |")
        print("| --- | --- |" + " --- |" * len(keys))
        for b, n, means in windows(read_rows(path), keys, args.from_step, args.to_step, args.width):
            cells = ["" if means[k] is None else f"{means[k]:.4g}" for k in keys]
            print(f"| {b} | {n} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
