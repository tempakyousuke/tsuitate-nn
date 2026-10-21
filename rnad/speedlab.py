"""学習の速さの比較（tsuitate-bot の docs/rl-learning-speed.md）。

同じ物差し（**固定パネル**）で各 run の書き出しを測り、Elo を「更新回数・局数・壁時計」の関数として並べる。
学習中の錨（train_rnad.py の eval_vs_anchor）は run ごとに相手が違い、錨が強くなると分解能も無くなるので、
方式どうしの比較には使わない。

    # 1) パネルを作る（相手どうしの総当たりで Elo を決める。--fix の相手を固定。既定は乱数方策 = 0）
    python rnad/speedlab.py panel --out lab/panel.json \\
        --member random --member heuristic --member ref75=lab/runs/ref/current_75.bin ...
    # 2) run の書き出し（policy_<step>.bin = ターゲット、current_<step>.bin = 現在のネット）をパネルで測る
    python rnad/speedlab.py eval --panel lab/panel.json lab/runs/base_s0 lab/runs/base_s1 ...
    # 3) 腕ごとにまとめる（run のディレクトリ名の最後の `_s<seed>` を落とした名前を腕とみなす）
    python rnad/speedlab.py report lab/runs/* --panel lab/panel.json --control base --gain 50 --out lab/report.md

**同じ対局条件（共通乱数）**: 手の乱数を (相手ごとの seed, 局番号, その局の何番目の手番か) だけで決める
（`keyed_actions`）ので、どの書き出しも同じ「相手 × 局番号」の組で、方策が同じ手を選ぶ限り同じ局面を指す
（ペアで比べたときの分散が減る）。

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
from rnad import pack_legal, play_match  # noqa: E402

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

    def __init__(self, spec, planes, device=DEV):
        name, _, path = spec.partition("=")
        self.name = name
        self.spec = spec
        self.net = None
        self.rust = None
        if path:
            self.net = read_weights(path).eval().to(device)
        elif name == "random":
            self.net = uniform_net(planes).to(device)
        else:
            self.rust = name


def stable_seed(*parts) -> int:
    h = 0
    for p in parts:
        for ch in str(p):
            h = (h * 131 + ord(ch)) % 1_000_003
    return h


_M64 = (1 << 64) - 1


def _i64(x):
    """64 ビットの符号なし定数を torch の int64（2の補数）で表す"""
    x &= _M64
    return x - (1 << 64) if x >= (1 << 63) else x


def _shr(z, k):
    """int64 の論理右シフト（torch の >> は算術シフトなので上位を落とす）"""
    return (z >> k) & ((1 << (64 - k)) - 1)


def _splitmix(z):
    z = z + _i64(0x9E3779B97F4A7C15)
    z = (z ^ _shr(z, 30)) * _i64(0xBF58476D1CE4E5B9)
    z = (z ^ _shr(z, 27)) * _i64(0x94D049BB133111EB)
    return z ^ _shr(z, 31)


def keyed_uniform(salt, games, decs, actions, device):
    """(salt, 局, その局の何番目の手番か, 行動) だけで決まる (0, 1) の一様乱数（splitmix64 のハッシュ。
    CPU でも GPU でも同じ値）。`actions` は行動の番号 [行, K]（合法手だけ渡せば速い）か、行動の総数（全部）"""
    g = torch.as_tensor(games, dtype=torch.int64, device=device)[:, None]
    k = torch.as_tensor(decs, dtype=torch.int64, device=device)[:, None]
    if isinstance(actions, int):
        a = torch.arange(actions, dtype=torch.int64, device=device)[None, :]
    else:
        a = actions.to(device, torch.int64)
    z = _splitmix(_splitmix(_splitmix(torch.full_like(g, int(salt)) ^ g) ^ k) ^ a)
    return (_shr(z, 11).double() + 0.5) / float(1 << 53)


@torch.no_grad()
def keyed_actions(net, obs, mask, games, decs, salt, device=DEV):
    """**局ごとに鍵を決めた** Gumbel-max のサンプリング（共通乱数）。

    一様乱数は (salt, 局, 手番) だけで決まり、バッチの並びや他の局の進み方に依らない。同じ局を2つの書き出しで
    測ると、両者が同じ手を選ぶ限り同じ局面が続き、違いが出るのは方策が実際に違う手を選んだところからになる
    （`sample_actions` は1バッチ分の乱数をまとめて引くので、どこかの局で反則が起きるだけで全局の乱数がずれる）"""
    x = torch.from_numpy(obs).to(device).half().float()
    logits, _ = net(x)
    legal = torch.from_numpy(mask).to(device)
    # 合法手だけを詰めて乱数を作る（乱数は行動の番号で決まるので、全行動に作ったときと同じ手を選ぶ）
    idx, pmask, _ = pack_legal(legal, legal.to(torch.uint8).argmax(1))
    u = keyed_uniform(salt, games, decs, idx, device)
    gumbel = (-torch.log(-torch.log(u))).float()
    masked = logits.float().gather(1, idx).masked_fill(~pmask, float("-inf"))
    return idx.gather(1, (masked + gumbel).argmax(-1, keepdim=True)).squeeze(1).cpu().numpy()


@torch.no_grad()
def play_vs(a_net, b: Member, games: int, seed: int, planes: int, per_game: bool = False, device=DEV):
    """a_net（PyTorch）と b を games 局（偶数番の局で a が先手）。a から見た (勝, 負, 分)（per_game なら局番号ごとの
    結果 1 / 0 / −1 の配列）。
    手は `keyed_actions`（局ごとの共通乱数）で選ぶので、同じ (b, games, seed) で測った書き出しどうしは局ごとに対になる"""
    salt = seed
    dec = np.zeros(games, dtype=np.int64)  # 局ごとの手番の数（反則で指し直した手番も数える）
    if b.net is not None:
        env = tsuitate_rl.VecEnv(games, seed=seed, auto_reset=False, **planes_kw(planes))
        a_color = np.arange(games) % 2
    else:
        # Rust の戦略が相手: 評価モード（偶数番の局で学習側が先手。相手の手番は env の中で進む）
        env = tsuitate_rl.VecEnv(games, opponent=b.rust, seed=seed, auto_reset=False, **planes_kw(planes))
        a_color = None
    env.reset()
    outcome = np.zeros(games, dtype=np.int64)  # 局番号ごとの a から見た結果（1 / 0 / −1）
    while True:
        alive = env.alive()
        if not alive.any():
            break
        obs, mask, pl = env.observe()
        actions = np.full(games, -1, dtype=np.int64)
        if a_color is None:
            learner = env.learner_colors()
            groups = [(a_net, np.flatnonzero(alive))]
        else:
            learner = a_color
            groups = [(a_net, np.flatnonzero(alive & (pl == a_color))),
                      (b.net, np.flatnonzero(alive & (pl != a_color)))]
        for net, idx in groups:
            if len(idx):
                actions[idx] = keyed_actions(net, obs[idx], mask[idx], idx, dec[idx], salt, device)
        dec[alive] += 1
        rew, done = env.step(actions)
        for i in np.flatnonzero(done):
            outcome[i] = int(np.sign(rew[i, learner[i]]))
    if per_game:
        return outcome
    return int((outcome > 0).sum()), int((outcome < 0).sum()), int((outcome == 0).sum())


def planes_kw(planes):
    return {"planes": planes} if planes != 86 else {}


# ---------------------------------------------------------------- Elo


def fit_fixed(results, ratings, iters=100):
    """パネルの Elo を固定して、測られる側1体の Elo を最尤で求める。
    results = [(相手の名前, 勝, 負, 分)]。(Elo, 標準誤差) を返す。全勝・全敗でも発散しないよう、
    各相手に 0.5勝0.5敗の弱い事前を足す"""
    r = sum(ratings[name] for name, *_ in results) / len(results)  # 相手の平均から（0 からだと遠すぎて発散する）
    for _ in range(iters):
        g = h = 0.0
        for name, w, l, d in results:
            n = w + l + d + 1.0
            s = w + 0.5 * d + 0.5
            p = 1.0 / (1.0 + 10 ** ((ratings[name] - r) / 400))
            k = math.log(10) / 400
            g += k * (s - n * p)
            h += k * k * n * p * (1 - p)
        step = max(-200.0, min(200.0, g / max(h, 1e-12)))
        r += step
        if abs(step) < 1e-6:
            break
    return r, 1.0 / math.sqrt(max(h, 1e-12))


def fit_round_robin(names, pairs, fixed="random", iters=500, fixed_value=0.0):
    """総当たりの結果 {(a, b): (a の勝, 負, 分)} から全員の Elo（`fixed` を `fixed_value` に固定）"""
    r = {n: fixed_value for n in names}
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
    fixed, _, value = args.fix.partition("=")
    if fixed not in names:
        raise SystemExit(f"--fix の {fixed} がパネルにいない")
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
    ratings = fit_round_robin(names, pairs, fixed=fixed, fixed_value=float(value or 0))
    out = {"planes": args.planes, "games": args.games, "members": args.member, "fix": args.fix,
           "ratings": ratings, "pairs": {f"{a}|{b}": v for (a, b), v in pairs.items()}}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for n in sorted(names, key=lambda n: ratings[n]):
        print(f"  {n:20s} {ratings[n]:7.1f}")


def step_files(run: pathlib.Path):
    """run の書き出し [(step, which, path)]（which = target / current）"""
    out = []
    for p in run.glob("*.bin"):
        m = re.fullmatch(r"(policy|current|tail|ema[0-9.e-]+)_(\d+)\.bin", p.name)
        if m:
            which = {"policy": "target", "current": "current"}.get(m.group(1), m.group(1))
            out.append((int(m.group(2)), which, p))
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
                    o = play_vs(net, m, args.games, EVAL_SEED + stable_seed(m.name), planes, per_game=True)
                    w, l, d = int((o > 0).sum()), int((o < 0).sum()), int((o == 0).sum())
                    rec = {"step": step, "which": which, "opponent": m.name, "w": w, "l": l, "d": d,
                           "games": args.games, "outcomes": o.tolist(), "sec": round(time.time() - t0, 2)}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                print(f"{run.name} {which} {step}: 済", flush=True)


def load_eval(run: pathlib.Path, ratings):
    """{(step, which): (Elo, SE, 平均得点, 局数)}"""
    by = {}
    path = run / "panel_eval.jsonl"
    if not path.exists():
        return {}
    seen = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["opponent"] not in ratings:
            continue
        # 同じ (step, ネット, 相手) が2回書かれていたら（eval を並べて回したときなど）後のほうだけ使う
        # （両方数えるとその相手だけ重みが倍になり Elo がずれる）
        seen[(r["step"], r["which"], r["opponent"])] = (r["opponent"], r["w"], r["l"], r["d"])
    for (step, which, _), res in seen.items():
        by.setdefault((step, which), []).append(res)
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
        league = r.get("league_games") or {}
        games += r.get("games", 0) + sum(sum(v) if isinstance(v, (list, tuple)) else 0 for v in league.values())
        sec += r["collect_sec"] + r["update_sec"]
        cum[r["step"]] = (games, sec)
    return cum


def load_outcomes(run: pathlib.Path):
    """{(step, which, 相手): 局番号ごとの結果の配列}（局ごとの結果を残した評価だけ）"""
    out = {}
    path = run / "panel_eval.jsonl"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "outcomes" in r:
            out[(r["step"], r["which"], r["opponent"])] = np.array(r["outcomes"], dtype=float)
    return out


def seed_of(run: pathlib.Path):
    m = re.search(r"_s(\d+)$", run.name)
    return m.group(1) if m else "0"


def paired_section(arms, control, which, steps):
    """対照との**局ごとのペア差**（同じ seed・step・相手・局番号。評価は共通乱数なので局どうしが対になる）。
    得点（勝 1・分 0.5・負 0）の差を pt で。標準誤差は seed ごとに step・相手・局をまとめた平均の、seed 間の
    ばらつき（seed が1つなら局ごとの差の標準誤差。step どうしの相関は無視するので小さめに出る）"""
    lines = [f"## 対照（{control}）との局ごとのペア差（得点の差、pt。{which}）", "",
             "| 腕 | " + " | ".join(str(s) for s in steps if s) + " | step > 0 の全体 | 局ごとの相関 |",
             "| --- |" + " --- |" * (len([s for s in steps if s]) + 2)]
    ctrl = {seed_of(r): load_outcomes(r) for r, _, _ in arms[control]}
    skipped = set()
    for arm, rs in sorted(arms.items()):
        if arm == control:
            continue
        per_step = {s: [] for s in steps if s}
        per_step_seed = {}  # (step, seed) → 局ごとの差
        per_seed = {}
        corr_a, corr_b = [], []
        for run, _, _ in rs:
            sd = seed_of(run)
            if sd not in ctrl:
                continue
            mine = load_outcomes(run)
            for (st, w, opp), o in mine.items():
                if w != which or st == 0 or (st, w, opp) not in ctrl[sd]:
                    continue
                c = ctrl[sd][(st, w, opp)]
                if len(c) != len(o):
                    # 局数が違うと局番号が対にならない（相手の seed や先後が局番号で決まる）
                    skipped.add((arm, st, opp))
                    continue
                diff = ((o + 1) / 2 - (c + 1) / 2) * 100
                per_step.setdefault(st, []).extend(diff.tolist())
                per_step_seed.setdefault((st, sd), []).extend(diff.tolist())
                per_seed.setdefault(sd, []).extend(diff.tolist())
                corr_a.extend(o.tolist())
                corr_b.extend(c.tolist())
        cells = []
        for s in steps:
            if not s:
                continue
            x = per_step.get(s, [])
            seed_means = [np.mean(v) for (st, _), v in per_step_seed.items() if st == s]
            if len(seed_means) > 1:
                # seed が複数あれば seed ごとの平均のばらつきから（局ごとの誤差だけでは seed の間の違いを見落とす）
                cells.append(f"{np.mean(seed_means):+.1f}±{np.std(seed_means, ddof=1) / math.sqrt(len(seed_means)):.1f}")
            else:
                cells.append(f"{np.mean(x):+.1f}±{np.std(x, ddof=1) / math.sqrt(len(x)):.1f}" if len(x) > 1 else "")
        allx = [v for vs in per_seed.values() for v in vs]
        if len(per_seed) > 1:
            means = [np.mean(v) for v in per_seed.values()]
            tot = f"{np.mean(allx):+.1f}±{np.std(means, ddof=1) / math.sqrt(len(means)):.1f}（seed 間、{len(means)} seed）"
        elif allx:
            tot = f"{np.mean(allx):+.1f}±{np.std(allx, ddof=1) / math.sqrt(len(allx)):.1f}（局ごと、1 seed）"
        else:
            tot = "—"
        rho = f"{np.corrcoef(corr_a, corr_b)[0, 1]:.2f}" if len(corr_a) > 2 else "—"
        lines.append(f"| {arm} | " + " | ".join(cells) + f" | {tot} | {rho} |")
    lines.append("")
    lines.append("step ごとの ± は seed が2本以上なら seed ごとの平均の標準誤差、1本なら局ごとの標準誤差（その書き出しを固定したときの誤差）。")
    if skipped:
        lines.append(f"局数が対照と違うため比べなかった組: {len(skipped)}（腕・step・相手）")
    lines.append("")
    return lines


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
    lines.append(f"| 腕 | seed | 最後の Elo | 平均 Elo（曲線下の面積 / 更新回数） | 起点 +{args.gain:g} に届いた更新回数 | "
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
            # しきい値は起点（step 0）からの伸び（パネルの絶対値はパネル込みでしか意味を持たないため）
            hit = next((s for s, e in pts if e >= pts[0][1] + args.gain), None)
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
    if args.control and args.control in arms:
        lines += paired_section(arms, args.control, which, steps)
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
    p.add_argument("--fix", default="random=0",
                   help="Elo を固定する相手と値（例 rl_v15=1338.9 で凍結版のレート表 ratings/ratings.json の尺度に合わせる）")
    p.add_argument("--out", required=True)
    p = sub.add_parser("eval", help="run の書き出しをパネルで測る（済んだものは飛ばす）")
    p.add_argument("runs", nargs="+")
    p.add_argument("--panel", required=True)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--which", action="append", help="target / current / ema<係数>（省くと全部）")
    p.add_argument("--every", type=int, default=0, help="この倍数の step だけ測る（0 = 全部）")
    p.add_argument("--only", action="append", help="この名前のパネルの相手だけ")
    p = sub.add_parser("report", help="腕ごとにまとめる")
    p.add_argument("runs", nargs="+")
    p.add_argument("--panel", required=True)
    p.add_argument("--which", default="current", help="target / current / ema<係数>")
    p.add_argument("--gain", type=float, default=50.0, help="起点（step 0）からこの Elo だけ伸びた更新回数を数える")
    p.add_argument("--control", default="")
    p.add_argument("--out", default="")
    p.add_argument("--json", default="")
    args = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "1")))
    {"panel": cmd_panel, "eval": cmd_eval, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
