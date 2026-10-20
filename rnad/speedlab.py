"""学習の速さの比較（tsuitate-bot の docs/rl-learning-speed.md）。

同じ物差し（**固定パネル**）で各 run の書き出しを測り、Elo を「更新回数・局数・壁時計」の関数として並べる。
学習中の錨（train_rnad.py の eval_vs_anchor）は run ごとに相手が違い、錨が強くなると分解能も無くなるので、
方式どうしの比較には使わない。

    # 1) パネルを作る（相手どうしの総当たりで Elo を決める。乱数方策 = 0 に固定）
    python rnad/speedlab.py panel --out lab/panel.json \\
        --member random --member heuristic --member ref75=lab/runs/ref/current_75.bin ...
    # 2) run の書き出し（policy_<step>.bin = ターゲット、current_<step>.bin = 現在のネット）をパネルで測る
    python rnad/speedlab.py eval --panel lab/panel.json lab/runs/base_s0 lab/runs/base_s1 ...
    # 3) 腕ごとにまとめる（run のディレクトリ名の最後の `_s<seed>` を落とした名前を腕とみなす）
    python rnad/speedlab.py report lab/runs/* --threshold 300 --out lab/report.md

**同じ対局条件（共通乱数）**: パネルの相手ごとに評価の env の seed と手の乱数の seed を固定するので、
どの書き出しも同じ「相手 × 局番号」の組で測られる（ペアで比べられる）。

パネルの相手の書き方:
- `random`        … マスク内一様（方策の頭を 0 にしたネット。合法手から一様に選ぶ）
- `heuristic` など … tsuitate-bot の `strategy::make` の名前（Rust の評価モードで指す）
- `<名前>=<重み>`  … 方策ネットの重み（TSRLPV01、`policy_<step>.bin` や凍結版の `models/rl_*.bin`）。
  PyTorch でまとめて推論する
"""

import argparse
import json
import math
import pathlib
import re
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tsuitate_rl  # noqa: E402
from export_weights import read as read_weights  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import play_match, sample_actions  # noqa: E402

DEV = torch.device("cpu")
EVAL_SEED = 777_000  # パネル評価の env の seed の基準（相手ごとに足す）


# ---------------------------------------------------------------- 対戦


def uniform_net(planes):
    """マスク内一様に指すネット（方策の頭のロジットが常に 0）"""
    net = PolicyValueNet(8, 1, planes)
    with torch.no_grad():
        net.p2.weight.zero_()
        net.p2.bias.zero_()
    return net.eval()


class Member:
    """パネルの相手（または測られる側）。`net` があれば PyTorch で指し、無ければ Rust の戦略名"""

    def __init__(self, spec, planes):
        name, _, path = spec.partition("=")
        self.name = name
        self.spec = spec
        self.net = None
        self.rust = None
        if path:
            self.net = read_weights(path).eval()
        elif name == "random":
            self.net = uniform_net(planes)
        else:
            self.rust = name


def stable_seed(*parts) -> int:
    h = 0
    for p in parts:
        for ch in str(p):
            h = (h * 131 + ord(ch)) % 1_000_003
    return h


@torch.no_grad()
def play_vs(a_net, b: Member, games: int, seed: int, planes: int):
    """a_net（PyTorch）と b を games 局（先後半々）。a から見た (勝, 負, 分)"""
    gen = torch.Generator().manual_seed(seed)
    if b.net is not None:
        env = tsuitate_rl.VecEnv(games, seed=seed, auto_reset=False, **planes_kw(planes))
        return play_match(env, a_net, b.net, DEV, gen)
    # Rust の戦略が相手: 評価モード（偶数番の局で学習側が先手。相手の手番は env の中で進む）
    env = tsuitate_rl.VecEnv(games, opponent=b.rust, seed=seed, auto_reset=False, **planes_kw(planes))
    env.reset()
    w = l = d = 0
    while True:
        alive = env.alive()
        if not alive.any():
            break
        obs, mask, _ = env.observe()
        learner = env.learner_colors()
        actions = np.full(games, -1, dtype=np.int64)
        idx = np.flatnonzero(alive)
        actions[idx] = sample_actions(a_net, obs[idx], mask[idx], DEV, gen)[0].numpy()
        rew, done = env.step(actions)
        for i in np.flatnonzero(done):
            r = rew[i, learner[i]]
            w += r > 0
            l += r < 0
            d += r == 0
    return int(w), int(l), int(d)


def planes_kw(planes):
    return {"planes": planes} if planes != 86 else {}


# ---------------------------------------------------------------- Elo


def fit_fixed(results, ratings, iters=100):
    """パネルの Elo を固定して、測られる側1体の Elo を最尤で求める。
    results = [(相手の名前, 勝, 負, 分)]。(Elo, 標準誤差) を返す。全勝・全敗でも発散しないよう、
    各相手に 0.5勝0.5敗の弱い事前を足す"""
    r = 0.0
    for _ in range(iters):
        g = h = 0.0
        for name, w, l, d in results:
            n = w + l + d + 1.0
            s = w + 0.5 * d + 0.5
            p = 1.0 / (1.0 + 10 ** ((ratings[name] - r) / 400))
            k = math.log(10) / 400
            g += k * (s - n * p)
            h += k * k * n * p * (1 - p)
        step = g / max(h, 1e-12)
        r += step
        if abs(step) < 1e-6:
            break
    return r, 1.0 / math.sqrt(max(h, 1e-12))


def fit_round_robin(names, pairs, fixed="random", iters=500):
    """総当たりの結果 {(a, b): (a の勝, 負, 分)} から全員の Elo（`fixed` を 0 に固定）"""
    r = {n: 0.0 for n in names}
    k = math.log(10) / 400
    for _ in range(iters):
        delta = 0.0
        for x in names:
            if x == fixed:
                continue
            g = h = 0.0
            for (a, b), (w, l, d) in pairs.items():
                if x not in (a, b):
                    continue
                y = b if x == a else a
                sx = (w + 0.5 * d if x == a else l + 0.5 * d) + 0.5
                n = w + l + d + 1.0
                p = 1.0 / (1.0 + 10 ** ((r[y] - r[x]) / 400))
                g += k * (sx - n * p)
                h += k * k * n * p * (1 - p)
            st = g / max(h, 1e-12)
            r[x] += st
            delta = max(delta, abs(st))
        if delta < 1e-6:
            break
    return r


# ---------------------------------------------------------------- コマンド


def cmd_panel(args):
    members = [Member(s, args.planes) for s in args.member]
    names = [m.name for m in members]
    if "random" not in names:
        raise SystemExit("パネルには random（Elo 0 の基準）を入れる")
    pairs = {}
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            if a.net is None and b.net is None:
                continue  # Rust どうしは測らない（評価モードは片側がネットの前提）
            x, y = (a, b) if a.net is not None else (b, a)
            t0 = time.time()
            w, l, d = play_vs(x.net, y, args.games, stable_seed("panel", x.name, y.name), args.planes)
            pairs[(x.name, y.name)] = (w, l, d)
            print(f"{x.name} 対 {y.name}: {w}勝{l}敗{d}分（{time.time() - t0:.0f}s）", flush=True)
    ratings = fit_round_robin(names, pairs)
    out = {"planes": args.planes, "games": args.games, "members": args.member,
           "ratings": ratings, "pairs": {f"{a}|{b}": v for (a, b), v in pairs.items()}}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for n in sorted(names, key=lambda n: ratings[n]):
        print(f"  {n:20s} {ratings[n]:7.1f}")


def step_files(run: pathlib.Path):
    """run の書き出し [(step, which, path)]（which = target / current）"""
    out = []
    for p in run.glob("*.bin"):
        m = re.fullmatch(r"(policy|current)_(\d+)\.bin", p.name)
        if m:
            out.append((int(m.group(2)), "target" if m.group(1) == "policy" else "current", p))
    return sorted(out)


def cmd_eval(args):
    panel = json.loads(pathlib.Path(args.panel).read_text())
    planes = panel["planes"]
    members = [Member(s, planes) for s in panel["members"]]
    members = [m for m in members if args.only is None or m.name in args.only]
    ratings = panel["ratings"]
    for run in map(pathlib.Path, args.runs):
        path = run / "panel_eval.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                done.add((r["step"], r["which"], r["opponent"]))
        with open(path, "a") as f:
            for step, which, wpath in step_files(run):
                if args.which and which not in args.which:
                    continue
                if args.every and step % args.every:
                    continue
                net = None
                for m in members:
                    if (step, which, m.name) in done:
                        continue
                    if net is None:
                        net = read_weights(wpath).eval()
                    t0 = time.time()
                    # 相手ごとに seed を固定 = どの書き出しも同じ対局条件（共通乱数）
                    w, l, d = play_vs(net, m, args.games, EVAL_SEED + stable_seed(m.name), planes)
                    rec = {"step": step, "which": which, "opponent": m.name, "w": w, "l": l, "d": d,
                           "games": args.games, "sec": round(time.time() - t0, 2)}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                print(f"{run.name} {which} {step}: 済", flush=True)


def load_eval(run: pathlib.Path, ratings):
    """{(step, which): (Elo, SE, 平均得点, 局数)}"""
    by = {}
    path = run / "panel_eval.jsonl"
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["opponent"] not in ratings:
            continue
        by.setdefault((r["step"], r["which"]), []).append((r["opponent"], r["w"], r["l"], r["d"]))
    out = {}
    for key, res in by.items():
        elo, se = fit_fixed(res, ratings)
        n = sum(w + l + d for _, w, l, d in res)
        score = sum(w + 0.5 * d for _, w, _, d in res) / max(n, 1)
        out[key] = (elo, se, score, n)
    return out


def load_log(run: pathlib.Path):
    """step → (累積局数, 累積の壁時計秒（自己対局＋更新）)"""
    path = run / "log.jsonl"
    cum = {0: (0, 0.0)}
    games = 0
    sec = 0.0
    if not path.exists():
        return cum
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "eval" in r or "collect_sec" not in r:
            continue
        games += r.get("games", 0) + r.get("league_games_n", 0)
        sec += r["collect_sec"] + r["update_sec"]
        cum[r["step"]] = (games, sec)
    return cum


def arm_of(run: pathlib.Path):
    return re.sub(r"_s\d+$", "", run.name)


def cmd_report(args):
    panel = json.loads(pathlib.Path(args.panel).read_text())
    ratings = panel["ratings"]
    runs = [pathlib.Path(r) for r in args.runs if (pathlib.Path(r) / "log.jsonl").exists()]
    arms = {}
    for run in runs:
        ev = load_eval(run, ratings)
        cum = load_log(run)
        arms.setdefault(arm_of(run), []).append((run, ev, cum))
    lines = [f"パネル: " + ", ".join(f"{n} {r:.0f}" for n, r in sorted(ratings.items(), key=lambda x: x[1])), ""]
    which = args.which
    steps = sorted({s for runs_ in arms.values() for _, ev, _ in runs_ for (s, w) in ev if w == which})
    # 曲線（腕ごと: seed 平均 ± seed 間の標準偏差 / seed 数の平方根）
    lines.append(f"## Elo（{which}、seed 平均 ± 標準誤差。括弧は seed 数）")
    lines.append("")
    lines.append("| 腕 | " + " | ".join(str(s) for s in steps) + " |")
    lines.append("| --- |" + " --- |" * len(steps))
    summary = {}
    for arm, rs in sorted(arms.items()):
        cells = []
        curve = []
        for s in steps:
            vals = [ev[(s, which)][0] for _, ev, _ in rs if (s, which) in ev]
            if not vals:
                cells.append("")
                continue
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")
            cells.append(f"{m:.0f}±{se:.0f}({len(vals)})" if len(vals) > 1 else f"{m:.0f}(1)")
            curve.append((s, m))
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        summary[arm] = curve
    lines.append("")
    # 要約: 予算の最後の Elo・面積（更新回数で平均した Elo）・しきい値に届いた更新回数・壁時計
    lines.append("## 要約")
    lines.append("")
    lines.append(f"| 腕 | seed | 最後の Elo | 平均 Elo（曲線下の面積 / 更新回数） | Elo {args.threshold} に届いた更新回数 | "
                 "同・局数 | 同・壁時計（分） | 1更新の秒 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm, rs in sorted(arms.items()):
        finals, aucs, hits, hit_games, hit_secs, per = [], [], [], [], [], []
        for run, ev, cum in rs:
            pts = sorted((s, ev[(s, w)][0]) for (s, w) in ev if w == which)
            if not pts:
                continue
            finals.append(pts[-1][1])
            xs = [0] + [s for s, _ in pts] if pts[0][0] != 0 else [s for s, _ in pts]
            ys = ([pts[0][1]] if pts[0][0] != 0 else []) + [e for _, e in pts]
            aucs.append(float(np.trapezoid(ys, xs) / max(xs[-1] - xs[0], 1)))
            hit = next((s for s, e in pts if e >= args.threshold), None)
            hits.append(hit)
            if hit is not None and hit in cum:
                hit_games.append(cum[hit][0])
                hit_secs.append(cum[hit][1] / 60)
            last = max(cum)
            if last:
                per.append(cum[last][1] / last)

        def fmt(xs, f="{:.0f}"):
            xs = [x for x in xs if x is not None]
            if not xs:
                return "—"
            if len(xs) == 1:
                return f.format(xs[0])
            return f.format(np.mean(xs)) + "±" + f.format(np.std(xs, ddof=1) / math.sqrt(len(xs)))
        n_hit = sum(h is not None for h in hits)
        lines.append(f"| {arm} | {len(finals)} | {fmt(finals)} | {fmt(aucs)} | {fmt(hits)}（{n_hit}/{len(hits)}） | "
                     f"{fmt(hit_games)} | {fmt(hit_secs, '{:.1f}')} | {fmt(per, '{:.1f}')} |")
    lines.append("")
    # 対照とのペア差（同じ seed・同じ書き出しの step どうし。共通乱数の評価なので局ごとの差は取らず Elo の差）
    if args.control and args.control in arms:
        lines.append(f"## 対照（{args.control}）との差（同じ seed どうしの Elo の差の平均 ± 標準誤差、step ごと）")
        lines.append("")
        lines.append("| 腕 | " + " | ".join(str(s) for s in steps) + " | 全 step 平均 |")
        lines.append("| --- |" + " --- |" * (len(steps) + 1))
        ctrl = {re.search(r"_s(\d+)$", r.name).group(1) if re.search(r"_s(\d+)$", r.name) else "0": ev
                for r, ev, _ in arms[args.control]}
        for arm, rs in sorted(arms.items()):
            if arm == args.control:
                continue
            cells = []
            allds = []
            for s in steps:
                ds = []
                for run, ev, _ in rs:
                    m = re.search(r"_s(\d+)$", run.name)
                    seed = m.group(1) if m else "0"
                    if seed in ctrl and (s, which) in ev and (s, which) in ctrl[seed]:
                        ds.append(ev[(s, which)][0] - ctrl[seed][(s, which)][0])
                allds.append(ds)
                if not ds:
                    cells.append("")
                elif len(ds) == 1:
                    cells.append(f"{ds[0]:+.0f}")
                else:
                    cells.append(f"{np.mean(ds):+.0f}±{np.std(ds, ddof=1) / math.sqrt(len(ds)):.0f}")
            # seed ごとに step 平均の差を取ってから seed で平均（step どうしは相関するので独立に数えない）
            per_seed = {}
            for run, ev, _ in rs:
                m = re.search(r"_s(\d+)$", run.name)
                seed = m.group(1) if m else "0"
                if seed not in ctrl:
                    continue
                ds = [ev[(s, which)][0] - ctrl[seed][(s, which)][0] for s in steps
                      if (s, which) in ev and (s, which) in ctrl[seed]]
                if ds:
                    per_seed[seed] = float(np.mean(ds))
            v = list(per_seed.values())
            tot = ("—" if not v else f"{np.mean(v):+.0f}" if len(v) == 1
                   else f"{np.mean(v):+.0f}±{np.std(v, ddof=1) / math.sqrt(len(v)):.0f}（{len(v)} seed）")
            lines.append(f"| {arm} | " + " | ".join(cells) + f" | {tot} |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(
            {arm: [{"run": str(r), "eval": {f"{s}|{w}": v for (s, w), v in ev.items()},
                    "cum": {str(k): v for k, v in cum.items()}} for r, ev, cum in rs]
             for arm, rs in arms.items()}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("panel", help="パネルを作る（総当たりで Elo を決める）")
    p.add_argument("--member", action="append", required=True)
    p.add_argument("--games", type=int, default=128)
    p.add_argument("--planes", type=int, default=91)
    p.add_argument("--out", required=True)
    p = sub.add_parser("eval", help="run の書き出しをパネルで測る（済んだものは飛ばす）")
    p.add_argument("runs", nargs="+")
    p.add_argument("--panel", required=True)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--which", action="append", choices=["target", "current"])
    p.add_argument("--every", type=int, default=0, help="この倍数の step だけ測る（0 = 全部）")
    p.add_argument("--only", action="append", help="この名前のパネルの相手だけ")
    p = sub.add_parser("report", help="腕ごとにまとめる")
    p.add_argument("runs", nargs="+")
    p.add_argument("--panel", required=True)
    p.add_argument("--which", default="current", choices=["target", "current"])
    p.add_argument("--threshold", type=float, default=300.0)
    p.add_argument("--control", default="")
    p.add_argument("--out", default="")
    p.add_argument("--json", default="")
    args = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "1")))
    {"panel": cmd_panel, "eval": cmd_eval, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
