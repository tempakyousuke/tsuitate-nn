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
    # 締まりの効果を除いて比べる: 測られる側を greedy（または温度 T）で指させ、`<which>@greedy` の名前で記録する。
    # 途中から始めた run の起点（--init-step）は checkpoint から書き出しておく（export-ck）。起点を何度で指すと
    # 締まった腕のエントロピー・π>0.999 の割合にそろうかは temp-match で測る
    python rnad/speedlab.py export-ck out/ab_base_108000.pt out/ab_mmd_s1
    python rnad/speedlab.py eval out/ab_mmd_s1 --panel panel_v2.json --which current --decode greedy --steps 108000,108500
    python rnad/speedlab.py temp-match out/ab_mmd_s1/current_108000.bin --temps 1,0.8,0.7,0.6 --like out/ab_mmd_s1/current_108500.bin
    # 途中の checkpoint から分けた2つの学習法を「同じ壁時計」で比べる（MMD の A/B。docs/rl-learning-speed.md 10節）
    python rnad/speedlab.py report out/ab_rnad_s1 out/ab_mmd_s1 --panel panel_v2.json --control ab_rnad --which target \
        --pair-by wallclock --sec-per-update ab_rnad=5.5,ab_mmd=3.9 --from-step 108000 --exclude-step 108000 \
        --control-any-seed --groups lineage.json

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
def keyed_actions(net, obs, mask, games, decs, salt, device=DEV, temp=1.0):
    """**局ごとに鍵を決めた** Gumbel-max のサンプリング（共通乱数）。

    一様乱数は (salt, 局, 手番) だけで決まり、バッチの並びや他の局の進み方に依らない。同じ局を2つの書き出しで
    測ると、両者が同じ手を選ぶ限り同じ局面が続き、違いが出るのは方策が実際に違う手を選んだところからになる
    （`sample_actions` は1バッチ分の乱数をまとめて引くので、どこかの局で反則が起きるだけで全局の乱数がずれる）。
    `temp` は方策の温度（softmax(logits / temp) から引く。同じ乱数のまま温度だけを変える）。0 なら最大の手（greedy）"""
    x = torch.from_numpy(obs).to(device).half().float()
    logits, _ = net(x)
    legal = torch.from_numpy(mask).to(device)
    # 合法手だけを詰めて乱数を作る（乱数は行動の番号で決まるので、全行動に作ったときと同じ手を選ぶ）
    idx, pmask, _ = pack_legal(legal, legal.to(torch.uint8).argmax(1))
    u = keyed_uniform(salt, games, decs, idx, device)
    gumbel = (-torch.log(-torch.log(u))).float()
    masked = logits.float().gather(1, idx).masked_fill(~pmask, float("-inf"))
    if temp == 0:
        score = masked
    elif temp == 1:
        score = masked + gumbel
    else:
        score = masked / temp + gumbel
    return idx.gather(1, score.argmax(-1, keepdim=True)).squeeze(1).cpu().numpy()


@torch.no_grad()
def play_vs(a_net, b: Member, games: int, seed: int, planes: int, per_game: bool = False, device=DEV, a_temp=1.0):
    """a_net（PyTorch）と b を games 局（偶数番の局で a が先手）。a から見た (勝, 負, 分)（per_game なら局番号ごとの
    結果 1 / 0 / −1 の配列）。
    手は `keyed_actions`（局ごとの共通乱数）で選ぶので、同じ (b, games, seed) で測った書き出しどうしは局ごとに対になる。
    `a_temp` は a だけの温度（0 = greedy）。相手はいつも温度 1"""
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
                actions[idx] = keyed_actions(net, obs[idx], mask[idx], idx, dec[idx], salt, device,
                                             temp=a_temp if net is a_net else 1.0)
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


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_decode(spec):
    """`--decode` の指定 → (温度, 記録する which の後ろに付ける名前)。sample = 温度 1（名前は付けない）"""
    if spec in ("", "sample"):
        return 1.0, ""
    if spec == "greedy":
        return 0.0, "@greedy"
    if spec.startswith("temp:"):
        t = float(spec[5:])
        if not t > 0:
            raise SystemExit("--decode temp:<T> の T は正（0 なら greedy）")
        return t, f"@t{t:g}"
    raise SystemExit(f"--decode {spec}（sample / greedy / temp:<T>）")


def cmd_eval(args):
    panel = json.loads(pathlib.Path(args.panel).read_text())
    temp, suffix = parse_decode(args.decode)
    planes = panel["planes"]
    device = pick_device(args.device)
    members = [Member(s, planes, device) for s in panel["members"]]
    members = [m for m in members if args.only is None or m.name in args.only]
    only_steps = {int(x) for x in args.steps.split(",")} if args.steps else None
    for run in map(pathlib.Path, args.runs):
        path = run / args.out_file
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                done.add((r["step"], r["which"], r["opponent"]))
        with open(path, "a") as f:
            for step, base_which, wpath in step_files(run):
                if args.which and base_which not in args.which:
                    continue
                which = base_which + suffix
                if args.every and step % args.every:
                    continue
                if only_steps is not None and step not in only_steps:
                    continue
                net = None
                for m in members:
                    if (step, which, m.name) in done:
                        continue
                    if net is None:
                        net = read_weights(wpath).eval().to(device)
                    t0 = time.time()
                    # 相手ごとに seed を固定 = どの書き出しも同じ対局条件（共通乱数）
                    o = play_vs(net, m, args.games, EVAL_SEED + stable_seed(m.name), planes, per_game=True,
                                device=device, a_temp=temp)
                    w, l, d = int((o > 0).sum()), int((o < 0).sum()), int((o == 0).sum())
                    rec = {"step": step, "which": which, "opponent": m.name, "w": w, "l": l, "d": d,
                           "games": args.games, "outcomes": o.tolist(), "sec": round(time.time() - t0, 2)}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                print(f"{run.name} {which} {step}: 済", flush=True)


def cmd_export_ck(args):
    """R-NaD / MMD の checkpoint の現在のネットとターゲットを、run の書き出しの形（current_<step>.bin・policy_<step>.bin）で
    書く。--init-step で途中から始めた run の起点を、同じ指し方（--decode）で測り直すため"""
    from export_weights import write as write_weights
    ck = torch.load(args.checkpoint, map_location="cpu")
    if "net" not in ck or "target" not in ck:
        raise SystemExit(f"{args.checkpoint} は R-NaD / MMD の checkpoint でない（net / target が無い）")
    ck_step = int(ck["step"])
    step = args.step if args.step is not None else ck_step
    if step != ck_step:
        # checkpoint.pt は --checkpoint-every ごとに書き直されるので、写した時点の step がラベルとずれていることがある
        print(f"注意: checkpoint の step は {ck_step} だが、--step {step} の名前で書く", file=sys.stderr)
    cfg = ck["config"]
    out = pathlib.Path(args.run)
    pairs = [(key, out / f"{name}_{step}.bin") for key, name in (("net", "current"), ("target", "policy"))]
    # 片方だけ書いてから止まらないよう、書く前に両方を確かめる
    taken = [str(path) for _, path in pairs if path.exists()]
    if taken and not args.force:
        raise SystemExit(f"{'・'.join(taken)} がもうある（上書きするなら --force）")
    out.mkdir(parents=True, exist_ok=True)
    for key, path in pairs:
        net = PolicyValueNet(cfg["channels"], cfg["blocks"], cfg["planes"])
        net.load_state_dict(ck[key])
        write_weights(net.eval(), path)
        print(f"書き出し: {path}（checkpoint の {key}、checkpoint の step {ck_step}）")


@torch.no_grad()
def self_play_stats(net, games, seed, planes, temp, device=DEV):
    """net を温度 temp で自分自身と games 局指させ、その局面での方策（同じ温度）のエントロピーの平均と、
    最大の手の確率が 0.999 を超える局面の割合（train_rnad.py のログの entropy_mean / pi_top_gt_0999 と同じ定義。
    反則で指し直した手番も数える）"""
    env = tsuitate_rl.VecEnv(games, seed=seed, auto_reset=False, **planes_kw(planes))
    env.reset()
    dec = np.zeros(games, dtype=np.int64)
    ent_sum, hi, n = 0.0, 0, 0
    while True:
        alive = env.alive()
        if not alive.any():
            break
        obs, mask, _ = env.observe()
        idx = np.flatnonzero(alive)
        x = torch.from_numpy(obs[idx]).to(device).half().float()
        logits, _ = net(x)
        legal = torch.from_numpy(mask[idx]).to(device)
        lg = logits.float().masked_fill(~legal, float("-inf"))
        if temp > 0:
            lp = torch.log_softmax(lg / temp, -1)
            p = lp.exp()
            ent_sum += float(-(p * lp.masked_fill(~legal, 0.0)).sum(-1).sum())
            hi += int((p.max(-1).values > 0.999).sum())
        else:
            hi += len(idx)
        n += len(idx)
        actions = np.full(games, -1, dtype=np.int64)
        actions[idx] = keyed_actions(net, obs[idx], mask[idx], idx, dec[idx], seed, device, temp=temp)
        dec[alive] += 1
        env.step(actions)
    return ent_sum / max(n, 1), hi / max(n, 1), n


def cmd_temp_match(args):
    """起点のネットを温度 T で指したときの自己対局の entropy_mean・pi_top_gt_0999 を並べ、--like のネット（温度 1）の値と
    比べる。締まった腕と「同じだけ締めた起点」を比べる対照（`eval --decode temp:<T>`）の T を決めるため"""
    device = pick_device(args.device)
    ref = read_weights(args.ref).eval().to(device)
    planes = ref.config["planes"]
    rows = []
    for t in [float(x) for x in args.temps.split(",")]:
        e, h, n = self_play_stats(ref, args.games, args.seed, planes, t, device)
        rows.append((f"{pathlib.Path(args.ref).name} @T{t:g}", e, h, n))
    for path in args.like or []:
        net = read_weights(path).eval().to(device)
        e, h, n = self_play_stats(net, args.games, args.seed, planes, 1.0, device)
        rows.append((f"{pathlib.Path(path).name}（温度 1）", e, h, n))
    print(f"自己対局 {args.games}局（seed {args.seed}）の局面での方策の entropy_mean / pi_top_gt_0999:")
    for name, e, h, n in rows:
        print(f"  {name:40s} entropy {e:.3f}  π>0.999 {h * 100:5.1f}%  （決定点 {n}）")


def load_eval(run: pathlib.Path, ratings, fname="panel_eval.jsonl"):
    """{(step, which): (Elo, SE, 平均得点, 局数)}"""
    by = {}
    path = run / fname
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


def load_outcomes(run: pathlib.Path, fname="panel_eval.jsonl"):
    """{(step, which, 相手): 局番号ごとの結果の配列}（局ごとの結果を残した評価だけ）"""
    out = {}
    path = run / fname
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


def median_sec_per_update(run: pathlib.Path, from_step=0):
    """log.jsonl の更新の行の (collect_sec + update_sec) の中央値（WSL の一時停止のような外れ値を入れない）"""
    path = run / "log.jsonl"
    xs = []
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "eval" in r or "collect_sec" not in r or r["step"] <= from_step:
                continue
            xs.append(r["collect_sec"] + r["update_sec"])
    return float(np.median(xs)) if xs else None


class Pairing:
    """対照の run の選び方（seed ごと / `--control-any-seed`）と、step の対応（同じ step / 同じ壁時計）"""

    def __init__(self, arms, control, args):
        self.args = args
        self.ctrl_runs = {seed_of(r): r for r, _, _ in arms.get(control, [])}
        self.any_seed = getattr(args, "control_any_seed", False) and len(self.ctrl_runs) == 1
        self.base = getattr(args, "from_step", 0) or 0
        self.by = getattr(args, "pair_by", "step") or "step"
        self.spu = {}
        for item in (getattr(args, "sec_per_update", None) or []):
            for part in filter(None, item.split(",")):
                k, _, v = part.partition("=")
                self.spu[k] = float(v)
        for arm, rs in arms.items():
            if arm not in self.spu and self.by == "wallclock":
                xs = [median_sec_per_update(r, self.base) for r, _, _ in rs]
                xs = [x for x in xs if x is not None]
                if not xs:
                    raise SystemExit(f"--pair-by wallclock: 腕 {arm} の秒/更新が分からない（--sec-per-update {arm}=秒）")
                self.spu[arm] = float(np.median(xs))
        self.control = control

    def control_run(self, run):
        sd = seed_of(run)
        if sd in self.ctrl_runs:
            return self.ctrl_runs[sd]
        return next(iter(self.ctrl_runs.values())) if self.any_seed else None

    def control_step(self, arm, step, ctrl_steps):
        """arm の step に対応する対照の step（無ければ None）"""
        if self.by == "step":
            return step if step in ctrl_steps else None
        if not ctrl_steps:
            return None
        t = (step - self.base) * self.spu[arm]
        return min(ctrl_steps, key=lambda c: (abs((c - self.base) * self.spu[self.control] - t), c))


def step_ok(st, args):
    """集計に入れる step か（0・--from-step より前・--exclude-step を除く）"""
    return st != 0 and st >= (getattr(args, "from_step", 0) or 0) and st not in set(getattr(args, "exclude_step", None) or [])


def _fmt_diff(xs, seed_means=None):
    if seed_means is not None and len(seed_means) > 1:
        return f"{np.mean(seed_means):+.1f}±{np.std(seed_means, ddof=1) / math.sqrt(len(seed_means)):.1f}"
    return f"{np.mean(xs):+.1f}±{np.std(xs, ddof=1) / math.sqrt(len(xs)):.1f}" if len(xs) > 1 else ""


def paired_section(arms, control, which, steps, args=None):
    """対照との**局ごとのペア差**（同じ seed・step・相手・局番号。評価は共通乱数なので局どうしが対になる）。
    得点（勝 1・分 0.5・負 0）の差を pt で。標準誤差は seed ごとに step・相手・局をまとめた平均の、seed 間の
    ばらつき（seed が1つなら局ごとの差の標準誤差。step どうしの相関は無視するので小さめに出る）。

    `--pair-by wallclock` なら、腕の step を「(step − --from-step) × 秒/更新」の壁時計へ写し、対照の書き出しのうち
    同じ壁時計に最も近いものと対にする（列は腕の step。対応は表の下に書く）。`--control-any-seed` なら対照の seed が
    1つのとき、どの seed の run とも対にする。`--groups` があれば相手の組ごとの小計も出す"""
    args = args or argparse.Namespace()
    fname = getattr(args, "eval_file", "panel_eval.jsonl")
    pr = Pairing(arms, control, args)
    groups = json.loads(pathlib.Path(args.groups).read_text()) if getattr(args, "groups", "") else {}
    title = "同じ壁時計" if pr.by == "wallclock" else "同じ step"
    lines = [f"## 対照（{control}）との局ごとのペア差（得点の差、pt。{which}、{title}）", "",
             "| 腕 | " + " | ".join(str(s) for s in steps if s) + " | step > 0 の全体 | 局ごとの相関 |",
             "| --- |" + " --- |" * (len([s for s in steps if s]) + 2)]
    ctrl_cache = {}
    skipped = set()
    mapping = {}
    group_rows = []
    for arm, rs in sorted(arms.items()):
        if arm == control:
            continue
        per_step = {s: [] for s in steps if s}
        per_step_seed = {}  # (step, seed) → 局ごとの差
        per_seed = {}
        per_group = {g: {} for g in groups}  # 組 → seed → 局ごとの差
        corr_a, corr_b = [], []
        for run, _, _ in rs:
            crun = pr.control_run(run)
            if crun is None:
                continue
            if crun not in ctrl_cache:
                ctrl_cache[crun] = load_outcomes(crun, fname)
            ctrl = ctrl_cache[crun]
            ctrl_steps = sorted({st for (st, w, _) in ctrl if w == which and step_ok(st, args)})
            sd = seed_of(run)
            mine = load_outcomes(run, fname)
            for (st, w, opp), o in mine.items():
                if w != which or not step_ok(st, args):
                    continue
                cst = pr.control_step(arm, st, ctrl_steps)
                if cst is None or (cst, w, opp) not in ctrl:
                    continue
                mapping[(arm, st)] = cst
                c = ctrl[(cst, w, opp)]
                if len(c) != len(o):
                    # 局数が違うと局番号が対にならない（相手の seed や先後が局番号で決まる）
                    skipped.add((arm, st, opp))
                    continue
                diff = ((o + 1) / 2 - (c + 1) / 2) * 100
                per_step.setdefault(st, []).extend(diff.tolist())
                per_step_seed.setdefault((st, sd), []).extend(diff.tolist())
                per_seed.setdefault(sd, []).extend(diff.tolist())
                for gname, members in groups.items():
                    if opp in members:
                        per_group[gname].setdefault(sd, []).extend(diff.tolist())
                corr_a.extend(o.tolist())
                corr_b.extend(c.tolist())
        cells = []
        for s in steps:
            if not s:
                continue
            seed_means = [np.mean(v) for (st, _), v in per_step_seed.items() if st == s]
            cells.append(_fmt_diff(per_step.get(s, []), seed_means))
        allx = [v for vs in per_seed.values() for v in vs]
        if len(per_seed) > 1:
            means = [np.mean(v) for v in per_seed.values()]
            tot = f"{np.mean(allx):+.1f}±{np.std(means, ddof=1) / math.sqrt(len(means)):.1f}（seed 間、{len(means)} seed）"
        elif allx:
            tot = f"{np.mean(allx):+.1f}±{np.std(allx, ddof=1) / math.sqrt(len(allx)):.1f}（局ごと、1 seed）"
        else:
            tot = "—"
        rho = (f"{np.corrcoef(corr_a, corr_b)[0, 1]:.2f}"
               if len(corr_a) > 2 and np.std(corr_a) > 0 and np.std(corr_b) > 0 else "—")
        lines.append(f"| {arm} | " + " | ".join(cells) + f" | {tot} | {rho} |")
        if groups:
            gcells = []
            for gname in groups:
                xs = [v for vs in per_group[gname].values() for v in vs]
                means = [np.mean(v) for v in per_group[gname].values()]
                gcells.append((_fmt_diff(xs, means) or "—") + f"（{len(xs)}組）")
            group_rows.append(f"| {arm} | " + " | ".join(gcells) + " |")
    lines.append("")
    lines.append("step ごとの ± は seed が2本以上なら seed ごとの平均の標準誤差、1本なら局ごとの標準誤差（その書き出しを固定したときの誤差）。")
    if pr.by == "wallclock":
        lines.append("壁時計の対応（腕の step → 対照の step、秒/更新 " + ", ".join(f"{a} {v:.2f}" for a, v in sorted(pr.spu.items()))
                     + f"、起点 {pr.base}）: " + ", ".join(f"{a} {s}→{c}" for (a, s), c in sorted(mapping.items())))
    if pr.any_seed:
        lines.append(f"対照の seed は1つ（{next(iter(pr.ctrl_runs))}）なので、どの seed の run とも対にした（--control-any-seed）。")
    if skipped:
        lines.append(f"局数が対照と違うため比べなかった組: {len(skipped)}（腕・step・相手）")
    lines.append("")
    if groups:
        lines += ["### 相手の組ごとの小計（step > 0 の全体）", "", "| 腕 | " + " | ".join(groups) + " |",
                  "| --- |" + " --- |" * len(groups), *group_rows, ""]
    return lines


def vs_step_section(arms, which, steps, args):
    """同じ run の中で、各 step を step N（--vs-step）と局ごとに対にした得点の差（pt）。起点からの伸びを、
    起点の書き出しの評価の揺れごと打ち消して測る。`--groups` があれば相手の組ごとの表も出す（自分の系列の相手にだけ
    伸びているのか、系列の外の相手にも伸びているのかを分ける）。`--vs-which` で起点の側だけ別の記録名と対にできる
    （例: 腕は current、起点は締まりをそろえた current@t0.8。`eval --decode temp:0.8 --steps <起点> --games <学習中の
    パネルと同じ局数（--panel-games）>` で測っておく。局数が違う組は対にできないので数だけ出して飛ばす）"""
    fname = getattr(args, "eval_file", "panel_eval.jsonl")
    groups = json.loads(pathlib.Path(args.groups).read_text()) if getattr(args, "groups", "") else {}
    n0 = args.vs_step
    base_which = getattr(args, "vs_which", "") or which
    cols = [s for s in steps if (s != n0 or base_which != which) and s >= (getattr(args, "from_step", 0) or 0)]
    label = which if base_which == which else f"{which}、起点は {base_which}"
    lines = [f"## step {n0} との局ごとのペア差（同じ run、得点の差、pt。{label}）", "",
             "| 腕 | " + " | ".join(str(s) for s in cols) + " |", "| --- |" + " --- |" * len(cols)]
    group_rows = []
    skipped = []  # 局数が起点の側と違うので対にしなかった (腕, step, 相手)
    for arm, rs in sorted(arms.items()):
        per = {s: {} for s in cols}
        per_group = {g: {s: {} for s in cols} for g in groups}  # 組 → step → seed → 局ごとの差
        for run, _, _ in rs:
            out = load_outcomes(run, fname)
            sd = seed_of(run)
            for (st, w, opp), o in out.items():
                if w != which or st not in per or (n0, base_which, opp) not in out:
                    continue
                c = out[(n0, base_which, opp)]
                if len(c) != len(o):
                    skipped.append((arm, st, opp, len(o), len(c)))
                    continue
                diff = ((o + 1) / 2 - (c + 1) / 2) * 100
                per[st].setdefault(sd, []).extend(diff.tolist())
                for gname, members in groups.items():
                    if opp in members:
                        per_group[gname][st].setdefault(sd, []).extend(diff.tolist())
        cells = []
        for s in cols:
            xs = [v for vs in per[s].values() for v in vs]
            cells.append(_fmt_diff(xs, [np.mean(v) for v in per[s].values()]) or "")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        for gname in groups:
            gcells = []
            for s in cols:
                xs = [v for vs in per_group[gname][s].values() for v in vs]
                gcells.append(_fmt_diff(xs, [np.mean(v) for v in per_group[gname][s].values()]) or "")
            group_rows.append(f"| {arm}（{gname}） | " + " | ".join(gcells) + " |")
    if skipped:
        ex = skipped[0]
        lines += ["", f"局数が step {n0}（{base_which}）と違うため比べなかった組: {len(skipped)}（腕・step・相手。"
                      f"例: {ex[0]} の {ex[1]}・{ex[2]} は {ex[3]} 局、起点は {ex[4]} 局。起点の側を同じ --games で測り直す）"]
    lines.append("")
    if groups:
        lines += [f"### 相手の組ごと（step {n0} とのペア差）", "", "| 腕（組） | " + " | ".join(str(s) for s in cols) + " |",
                  "| --- |" + " --- |" * len(cols), *group_rows, ""]
    return lines


def arm_of(run: pathlib.Path):
    return re.sub(r"_s\d+$", "", run.name)


def cmd_report(args):
    panel = json.loads(pathlib.Path(args.panel).read_text())
    ratings = panel["ratings"]
    # log.jsonl が無くても評価の記録があれば残す（export-ck で起点だけ書き出したフォルダ。局数・壁時計の列は空になる）
    runs = []
    for r in map(pathlib.Path, args.runs):
        if (r / "log.jsonl").exists() or (r / args.eval_file).exists():
            runs.append(r)
        else:
            print(f"飛ばす: {r} に log.jsonl も {args.eval_file} も無い", file=sys.stderr)
    arms = {}
    base = args.from_step or 0
    origins = {}  # run → 起点（--from-step、無ければ step 0）の Elo。--exclude-step で表から外しても伸びの基準に使う
    for run in runs:
        ev = load_eval(run, ratings, args.eval_file)
        if (base, args.which) in ev:
            origins[run] = ev[(base, args.which)][0]
        # --from-step より前と --exclude-step は表から外す（--from-step が無ければ step 0 は起点として曲線と要約に残す）
        ev = {k: v for k, v in ev.items()
              if (k[0] == 0 and base == 0) or (k[0] >= base and k[0] not in set(args.exclude_step or []))}
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

    def cum_at(cum, st):
        """st の時点の累積（その step の行が無ければ手前の最後の行。途中から分けた run は起点で (0, 0)）"""
        return cum[max(k for k in cum if k <= st)]

    for arm, rs in sorted(arms.items()):
        finals, aucs, hits, hit_games, hit_secs, per = [], [], [], [], [], []
        for run, ev, cum in rs:
            pts = sorted((s, ev[(s, w)][0]) for (s, w) in ev if w == which)
            if not pts:
                continue
            finals.append(pts[-1][1])
            # 曲線は起点（--from-step、無ければ 0）から。起点の Elo が無ければ最初の点の値で埋める
            y0 = origins.get(run, pts[0][1])
            xs = [base] + [s for s, _ in pts] if pts[0][0] != base else [s for s, _ in pts]
            ys = ([y0] if pts[0][0] != base else []) + [e for _, e in pts]
            # 点が起点だけ（始めたばかりの run）なら面積は起点の Elo そのもの
            aucs.append(float(np.trapezoid(ys, xs) / (xs[-1] - xs[0])) if len(xs) > 1 else float(ys[0]))
            # しきい値は起点からの伸び（パネルの絶対値はパネル込みでしか意味を持たないため）。更新回数・局数・壁時計も起点から
            hit = next((s for s, e in pts if e >= y0 + args.gain), None)
            hits.append(None if hit is None else hit - base)
            c0 = cum_at(cum, base)
            if hit is not None and hit in cum:
                hit_games.append(cum[hit][0] - c0[0])
                hit_secs.append((cum[hit][1] - c0[1]) / 60)
            last = max(cum)
            if last > base:
                per.append((cum[last][1] - c0[1]) / (last - base))

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
    if base:
        lines += ["", f"（--from-step {base}: 更新回数・局数・壁時計・1更新の秒は起点 {base} からの差。"
                      f"起点の Elo は --exclude-step で表から外していても基準に使う）"]
    lines.append("")
    # 対照とのペア差（同じ seed・同じ書き出しの step どうし。共通乱数の評価なので局ごとの差は取らず Elo の差）
    if args.control and args.control in arms:
        pr = Pairing(arms, args.control, args)
        title = "同じ壁時計の書き出しと" if pr.by == "wallclock" else "step ごと"
        lines.append(f"## 対照（{args.control}）との差（同じ seed どうしの Elo の差の平均 ± 標準誤差、{title}）")
        lines.append("")
        lines.append("| 腕 | " + " | ".join(str(s) for s in steps) + " | 全 step 平均 |")
        lines.append("| --- |" + " --- |" * (len(steps) + 1))
        ctrl = {seed_of(r): ev for r, ev, _ in arms[args.control]}
        if args.control_any_seed and len(ctrl) == 1:
            only = next(iter(ctrl.values()))
            ctrl = {seed_of(r): only for rs_ in arms.values() for r, _, _ in rs_}

        def ctrl_at(arm, seed, st):
            """対照の Elo（--pair-by wallclock なら同じ壁時計に最も近い書き出し）。無ければ None。
            step 0（出発点）は step 0 とだけ対にする（下の局ごとのペア差と同じく、対照の候補から外す）"""
            c = ctrl.get(seed)
            if c is None:
                return None
            if st == 0:
                return c[(0, which)][0] if (0, which) in c else None
            cs = pr.control_step(arm, st, sorted(s_ for s_, w in c if w == which and step_ok(s_, args)))
            return c[(cs, which)][0] if cs is not None else None
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
                    cv = ctrl_at(arm, seed, s) if (s, which) in ev else None
                    if cv is not None:
                        ds.append(ev[(s, which)][0] - cv)
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
                ds = [ev[(s, which)][0] - cv for s in steps
                      if (s, which) in ev and (cv := ctrl_at(arm, seed, s)) is not None]
                if ds:
                    per_seed[seed] = float(np.mean(ds))
            v = list(per_seed.values())
            tot = ("—" if not v else f"{np.mean(v):+.0f}" if len(v) == 1
                   else f"{np.mean(v):+.0f}±{np.std(v, ddof=1) / math.sqrt(len(v)):.0f}（{len(v)} seed）")
            lines.append(f"| {arm} | " + " | ".join(cells) + f" | {tot} |")
        lines.append("")
    if args.control and args.control in arms:
        lines += paired_section(arms, args.control, which, steps, args)
    if args.vs_step is not None:
        lines += vs_step_section(arms, which, steps, args)
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
    p.add_argument("--device", default="cpu", help="cpu / cuda / auto（学習中の書き出しを GPU でまとめて測る）")
    p.add_argument("--out-file", default="panel_eval.jsonl",
                   help="結果を書く run の中のファイル名（局数の違う測り直しはループ内の 1,024局と混ぜない）")
    p.add_argument("--steps", default="", help="この step だけ測る（カンマ区切り）")
    p.add_argument("--decode", default="sample",
                   help="測られる側の指し方: sample（温度 1、既定）/ greedy / temp:<T>。sample 以外は which を "
                        "<which>@greedy / <which>@t<T> の名前で記録する（report --which current@greedy で読む）")
    p = sub.add_parser("export-ck", help="checkpoint の現在のネットとターゲットを run の書き出しの形で書く（起点の測り直し用）")
    p.add_argument("checkpoint")
    p.add_argument("run")
    p.add_argument("--step", type=int, default=None,
                   help="書き出しの番号（省くと checkpoint の step。違う番号を与えると注意を出す）")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("temp-match", help="起点を温度 T で指したときのエントロピー・π>0.999 の割合を、締まった腕と比べる")
    p.add_argument("ref", help="起点の重み（TSRLPV01）")
    p.add_argument("--temps", default="1,0.9,0.8,0.7,0.6,0.5")
    p.add_argument("--like", action="append", help="比べる腕の重み（温度 1 で測る）")
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("report", help="腕ごとにまとめる")
    p.add_argument("runs", nargs="+")
    p.add_argument("--panel", required=True)
    p.add_argument("--which", default="current", help="target / current / ema<係数>")
    p.add_argument("--gain", type=float, default=50.0, help="起点（--from-step、無ければ step 0）の Elo からこの値だけ伸びるまでの更新回数を数える")
    p.add_argument("--control", default="")
    p.add_argument("--out", default="")
    p.add_argument("--json", default="")
    p.add_argument("--eval-file", default="panel_eval.jsonl", help="run の中の評価のファイル名")
    p.add_argument("--from-step", type=int, default=0, help="この step より前の書き出しを読まない（途中から分けた A/B の起点）")
    p.add_argument("--exclude-step", type=int, action="append", help="この step を集計から外す（両腕で同じネットの起点など）")
    p.add_argument("--pair-by", choices=["step", "wallclock"], default="step",
                   help="対照との対応: 同じ step / 同じ壁時計（(step − --from-step) × 秒/更新 が最も近い書き出し）")
    p.add_argument("--sec-per-update", action="append",
                   help="腕=秒（カンマ区切り可）。省いた腕は log.jsonl の中央値（--pair-by wallclock）")
    p.add_argument("--control-any-seed", action="store_true",
                   help="対照の seed が1つなら、どの seed の run とも対にする（パネルの乱数は相手の名前だけで決まる）")
    p.add_argument("--vs-step", type=int, default=None, help="同じ run の中で各 step をこの step と局ごとに対にする")
    p.add_argument("--vs-which", default="",
                   help="--vs-step の起点の側の記録名（省くと --which と同じ。締まりをそろえた起点 current@t<T> と対にするとき）")
    p.add_argument("--groups", default="", help="相手の組 {組名: [相手の名前, ...]} の JSON（ペア差の小計を出す）")
    args = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "1")))
    {"panel": cmd_panel, "eval": cmd_eval, "report": cmd_report, "export-ck": cmd_export_ck,
     "temp-match": cmd_temp_match}[args.cmd](args)


if __name__ == "__main__":
    main()
