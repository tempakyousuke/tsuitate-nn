"""局の信号の監視値（train_rnad.py の log.jsonl の `signals`）から、学習の進み方と「勝ちを先取りする量」を読む。

    .venv/bin/python rnad/signal_report.py out_rnad/fx_base out_rnad/fx_mat ...          # 学習曲線（run ごと）
    .venv/bin/python rnad/signal_report.py out_rnad/fx_base --split --from 200 --to 400  # 勝ち局と負け局を分ける信号

tsuitate-bot の docs/rl-fixed-opponent.md。torch も tsuitate_rl も使わない（ログを読むだけ）。

- **学習曲線**（既定）: 更新の窓ごとに、得点（勝ち 1・引き分け 0.5）と、学習側の主な信号の平均。
  「勝ちが出る前に何が先に動くか」を並べて見る
- **`--split`**: 指定した更新の範囲で、信号ごとに勝ち局の平均・負け局の平均・標準化した差
  d =（勝ち局の平均 − 負け局の平均）/ 合わせた標準偏差 を、終局時と途中（両者の受理手の合計が 20 / 40 / 60 に届いた時点）で出す。
  終局時の値は詰み・反則負けそのものを映す（負け局の反則数は必ず多い）ので、**途中の時点の d が大きい信号**が
  「勝ちを先取りする量」の候補になる。相関であって因果ではない（その量を報酬にして学習が速くなるかは A/B で確かめる）
"""

import argparse
import json
import math
import pathlib

GROUPS = ("win", "loss", "draw")
# 学習曲線に並べる列（鍵, 見出し）。鍵は <名前>（自分）/ opp_<名前>（相手）/ d_<名前>（自分 − 相手）
CURVE_COLS = (("moves", "手数/側"), ("fouls", "反則"), ("opp_fouls", "相手の反則"), ("d_capture_value", "駒得"),
              ("checks", "王手"), ("opp_checks", "被王手"), ("d_material", "駒割り"), ("d_king_attack", "玉への利き"),
              ("promotions", "成り"))


class Moments:
    """n・平均・標準偏差の組を足し合わせる（更新ごとの集計 → 窓の集計）"""

    def __init__(self, dim):
        self.n = 0
        self.s1 = [0.0] * dim
        self.s2 = [0.0] * dim

    def add(self, m):
        n = m.get("n", 0)
        if not n:
            return
        self.n += n
        for i, (mu, sd) in enumerate(zip(m["mean"], m["sd"])):
            self.s1[i] += n * mu
            self.s2[i] += n * (sd * sd + mu * mu)

    def merge(self, other):
        self.n += other.n
        self.s1 = [a + b for a, b in zip(self.s1, other.s1)]
        self.s2 = [a + b for a, b in zip(self.s2, other.s2)]

    def mean(self, i):
        return self.s1[i] / self.n if self.n else float("nan")

    def var(self, i):
        if not self.n:
            return float("nan")
        mu = self.s1[i] / self.n
        return max(self.s2[i] / self.n - mu * mu, 0.0)


def load(run: pathlib.Path):
    cfg_path = run / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{cfg_path} が無い")
    names = json.loads(cfg_path.read_text()).get("signal_names")
    if not names:
        raise SystemExit(f"{run} は局の信号を記録していない（--shape / --lose-len / --watch-signals を付けた run が要る）")
    rows = []
    for line in open(run / "log.jsonl"):
        if not line.strip():
            continue
        x = json.loads(line)
        if "signals" in x and "step" in x:
            rows.append(x)
    # 再開で同じ step が2回あれば後のものを使う
    by_step = {x["step"]: x for x in rows}
    return names, [by_step[k] for k in sorted(by_step)]


def column(names, key):
    """鍵 → 並び [自分 K, 相手 K, 自分 − 相手 K] の添字"""
    K = len(names)
    for prefix, off in (("opp_", K), ("d_", 2 * K)):
        if key.startswith(prefix) and key[len(prefix):] in names:
            return off + names.index(key[len(prefix):])
    if key in names:
        return names.index(key)
    raise SystemExit(f"信号 {key} を知らない（{' / '.join(names)}。相手は opp_<名前>、差は d_<名前>）")


def window_stats(rows, dim, where="final"):
    """行の集まり → {勝ち・負け・引き分けごとの Moments}"""
    out = {g: Moments(dim) for g in GROUPS}
    for x in rows:
        for g in GROUPS:
            m = x["signals"].get(g, {})
            m = m.get("final", {}) if where == "final" else m.get("at", {}).get(where, {})
            out[g].add(m)
    return out


def score_of(st):
    n = sum(st[g].n for g in GROUPS)
    return ((st["win"].n + 0.5 * st["draw"].n) / n if n else float("nan")), n


def cmd_curve(args):
    for run in args.runs:
        names, rows = load(run)
        dim = 3 * len(names)
        rows = [x for x in rows if args.from_ <= x["step"] <= (args.to or 10 ** 12)]
        if not rows:
            print(f"## {run}: 範囲に行が無い")
            continue
        cols = [(column(names, k), h) for k, h in CURVE_COLS if k.split("_", 1)[-1] in names or k in names]
        print(f"\n## {run}（{rows[0]['step']}〜{rows[-1]['step']} 更新、窓 {args.window}）\n")
        print("| 更新 | 局 | 得点 | 勝 / 負 / 分 | " + " | ".join(h for _, h in cols) + " | 整形の合計 | 倍率 |")
        print("| --- | --- | --- | --- | " + " | ".join("---" for _ in cols) + " | --- | --- |")
        for i in range(0, len(rows), args.window):
            chunk = rows[i:i + args.window]
            st = window_stats(chunk, dim)
            allm = Moments(dim)
            for g in GROUPS:
                allm.merge(st[g])
            score, n = score_of(st)
            shaped = [x["signals"]["shaped_total"]["mean"] for x in chunk if "shaped_total" in x["signals"]]
            mult = [x["shape_mult"] for x in chunk if "shape_mult" in x]
            print(f"| {chunk[0]['step']}〜{chunk[-1]['step']} | {n} | {score:.3f} | "
                  f"{st['win'].n} / {st['loss'].n} / {st['draw'].n} | "
                  + " | ".join(f"{allm.mean(c):.2f}" for c, _ in cols)
                  + f" | {(sum(shaped) / len(shaped)) if shaped else float('nan'):+.3f}"
                  + f" | {(sum(mult) / len(mult)) if mult else float('nan'):.2f} |")


def cmd_split(args):
    for run in args.runs:
        names, rows = load(run)
        K = len(names)
        dim = 3 * K
        rows = [x for x in rows if args.from_ <= x["step"] <= (args.to or 10 ** 12)]
        if not rows:
            print(f"## {run}: 範囲に行が無い")
            continue
        wheres = [w for w in args.at.split(",") if w]
        stats = {w: window_stats(rows, dim, w) for w in wheres}
        score, n = score_of(stats["final"] if "final" in stats else window_stats(rows, dim))
        print(f"\n## {run}（{rows[0]['step']}〜{rows[-1]['step']} 更新、{n} 局、得点 {score:.3f}）\n")
        for w in wheres:
            print(f"- {w}: 勝ち局 {stats[w]['win'].n}・負け局 {stats[w]['loss'].n}")
        keys = ([f"d_{k}" for k in names] if not args.all else
                [p + k for k in names for p in ("", "opp_", "d_")])

        def dval(w, c):
            a, b = stats[w]["win"], stats[w]["loss"]
            if a.n < args.min_n or b.n < args.min_n:
                return float("nan")
            sd = math.sqrt((a.n * a.var(c) + b.n * b.var(c)) / (a.n + b.n))
            return (a.mean(c) - b.mean(c)) / sd if sd > 1e-9 else float("nan")

        sort_w = args.sort or wheres[0]
        table = []
        for k in keys:
            c = column(names, k)
            table.append((k, c, [dval(w, c) for w in wheres]))
        j = wheres.index(sort_w)
        table.sort(key=lambda r: -abs(r[2][j]) if not math.isnan(r[2][j]) else 1.0)
        print("\n| 信号 | " + " | ".join(f"{w}: 勝ち局 / 負け局 / d" for w in wheres) + " |")
        print("| --- | " + " | ".join("---" for _ in wheres) + " |")
        for k, c, ds in table:
            cells = []
            for w, d in zip(wheres, ds):
                a, b = stats[w]["win"], stats[w]["loss"]
                cells.append(f"{a.mean(c):.2f} / {b.mean(c):.2f} / {d:+.2f}" if not math.isnan(d) else "—")
            print(f"| {k} | " + " | ".join(cells) + " |")
        print(f"\nd =（勝ち局の平均 − 負け局の平均）/ 合わせた標準偏差。{sort_w} の |d| の大きい順。"
              f"勝ち局か負け局が {args.min_n} 局未満の時点は —")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=pathlib.Path, help="train_rnad.py の --out（log.jsonl と config.json）")
    ap.add_argument("--window", type=int, default=50, help="学習曲線の窓（更新回数）")
    ap.add_argument("--from", dest="from_", type=int, default=0, help="この更新から")
    ap.add_argument("--to", type=int, default=0, help="この更新まで（0 = 最後まで）")
    ap.add_argument("--split", action="store_true", help="勝ち局と負け局を分ける信号の表を出す")
    ap.add_argument("--at", default="20,40,60,final", help="--split で見る時点（両者の受理手の合計。final = 終局時）")
    ap.add_argument("--sort", default="", help="--split の並べ替えに使う時点（既定は --at の先頭）")
    ap.add_argument("--all", action="store_true", help="--split で自分・相手・差の全部を出す（既定は差だけ）")
    ap.add_argument("--min-n", type=int, default=20, help="--split で d を出す最少の局数（勝ち局・負け局それぞれ）")
    args = ap.parse_args()
    (cmd_split if args.split else cmd_curve)(args)


if __name__ == "__main__":
    main()
