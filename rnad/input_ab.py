"""入力プレーンの組の効果を、小さい規模の自己対局で比べる（CI の CPU で回る大きさ）。

    # 腕（入力の版）× seed をまとめて回す。途中で止まった run は checkpoint から続ける
    .venv/bin/python rnad/input_ab.py run --root out_rnad/iab --arms v4,v5 --seeds 1,2,3,4 --steps 800 --jobs 4 \\
        --book ../tsuitate-bot/rl_openings.json [--time-limit 18000] [--arm v6k='<v5 の引数> --新しい組のフラグ']
    # 進み具合の表（回している途中でも読める）
    .venv/bin/python rnad/input_ab.py status out_rnad/iab
    # 書き出した重みが、入力のどの組をどれだけ読んでいるか（組を 0 にして渡したときの方策の変化）
    .venv/bin/python rnad/input_ab.py depend out_rnad/iab --book ../tsuitate-bot/rl_openings.json --out depend.json

tsuitate-bot の `.github/workflows/input-ab.yml` が使う（対局での比較と集計はあちらの `scripts/ci/input_ab.py`）。

**何を確かめる道具か**: 本番の大きさ（64ch×6・1更新256局・数万更新）で入力を足した効果を見るには GPU で数日かかる。
同じ学習の型（乱数初期化からの自己対局・序盤は定跡を 100% 強制・MMD の一様な磁石・駒得と玉への利きの整形）を
32ch×3・1更新64局・1本1スレッドに縮め、入力だけを変えた腕を seed を変えて何本かずつ回す。腕どうしの強さは
tsuitate-bot の arena で総当たりにして比べる。**この大きさで入力の差が出るかどうか自体が未検証**なので、まず答えの
分かっている組（v4 と v5。本番の大きさでは v5 が強い）で、差が再現するかを見るのに使う。

- 腕の名前 `v1`〜`v6` は入力の版（下の `ARMS`）。`--arm 名前='引数'` で好きな組み合わせを足せる
- 出力は `<root>/<腕>_s<seed>/`（train_rnad.py の出力そのまま。`current_<更新>.bin` が比べる重み）
- `--time-limit` 秒を過ぎたら回している run を止めて終わる（CI のジョブの上限の手前で止め、続きは次のジョブで回す）
"""

import argparse
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

V3 = "--opp-hand-planes --opp-foul-planes"
V4 = f"{V3} --king-foul-planes"
V5 = f"{V4} --lost-fresh-planes --lost-role-planes --cap-role-planes --pawn-file-planes"
V6 = (f"{V5} --king-tight-planes --check-clock-planes --checker-memo-planes --own-attack-planes "
      "--king-line-planes --opp-count-planes")
# 入力の版ごとの引数（v3 以降は組のフラグ。プレーン数は選んだ組が入る最小の版になる）
ARMS = {"v1": "--planes 86", "v2": "--planes 91", "v3": f"--planes 91 {V3}", "v4": f"--planes 91 {V4}",
        "v5": f"--planes 91 {V5}", "v6": f"--planes 91 {V6}"}

# 縮めた学習の設定。型は scratch の系列（tsuitate-bot の docs/frozen-versions.md の rl_scratch_v1 / rl_scratch5_v1）、
# 大きさと MMD の歩数は相手を固定した小さい学習（takeoff.py の COMMON）と同じ。整形は最後まで切らない（短いので）
COMMON = ("--algo mmd --init none --anchor none --eval-every 0 --channels 32 --blocks 3 --batch 64 "
          "--compact-legal --device cpu --threads 1 "
          "--mmd-magnet uniform --mmd-temp 0.01,0,0.01,0.01 --mmd-lr 3e-4,0,3e-4,3e-4 "
          "--mmd-steps 8 --mmd-epochs 2 --mmd-adv-norm 0.001 "
          "--open-p 1.0 --shape d_capture_value=0.02,king_attack=0.03 --shape-cap 1.0 "
          "--export-current --checkpoint-every 25")


def read_log(run: pathlib.Path):
    """log.jsonl の更新の行（step の昇順。再開で step が戻ったら後の行を使う）"""
    path = run / "log.jsonl"
    rows = {}
    if path.exists():
        for line in open(path):
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "update_sec" in x and "step" in x:
                rows[x["step"]] = x
    return [rows[k] for k in sorted(rows)]


def last_step(run: pathlib.Path) -> int:
    rows = read_log(run)
    return rows[-1]["step"] if rows else 0


def exported_steps(run: pathlib.Path):
    """書き出し済みの `current_<更新>.bin` の更新（昇順）"""
    out = []
    for p in run.glob("current_*.bin"):
        try:
            out.append(int(p.stem.split("_")[1]))
        except ValueError:
            pass
    return sorted(out)


def runs_in(root: pathlib.Path):
    """[(腕, seed, ディレクトリ)]（`<腕>_s<seed>`）"""
    out = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        name, sep, seed = d.name.rpartition("_s")
        if d.is_dir() and sep and seed.isdigit():
            out.append((name, int(seed), d))
    return out


def parse_arms(args):
    """[(腕の名前, train_rnad.py の引数)]（--arms の版の名前と --arm の追加分）"""
    arms = []
    for name in [a for a in args.arms.split(",") if a]:
        if name not in ARMS:
            raise SystemExit(f"--arms の '{name}' は {sorted(ARMS)} のどれか（ほかの組み合わせは --arm 名前='引数'）")
        arms.append((name, ARMS[name]))
    for spec in args.arm:
        name, sep, extra = spec.partition("=")
        if not sep or not name:
            raise SystemExit(f"--arm は '名前=引数'（{spec}）")
        arms.append((name, extra))
    if not arms:
        raise SystemExit("腕が無い（--arms か --arm）")
    if len({n for n, _ in arms}) != len(arms):
        raise SystemExit("腕の名前が重複している")
    if any("_s" in n or ":" in n or "," in n for n, _ in arms):
        raise SystemExit("腕の名前に '_s' ':' ',' は使えない（出力のディレクトリ名 <腕>_s<seed> や --only と紛れる）")
    return arms


def plan_queue(args, arms):
    """回す順の [(腕, 引数, seed)]。seed を外側に回す（途中で止めても腕ごとの本数がそろう）。
    `--only 腕:seed,...` があればその組だけ（CI で 1ランナーに 4本ずつ配る用）"""
    seeds = [int(s) for s in args.seeds.split(",")]
    queue = [(name, extra, s) for s in seeds for name, extra in arms]
    if args.only:
        want = set()
        for spec in args.only.split(","):
            name, sep, seed = spec.partition(":")
            if not sep or not seed.isdigit():
                raise SystemExit(f"--only は '腕:seed' のカンマ区切り（{spec}）")
            want.add((name, int(seed)))
        missing = want - {(n, s) for n, _, s in queue}
        if missing:
            raise SystemExit(f"--only の {sorted(missing)} は --arms / --arm / --seeds に無い")
        queue = [q for q in queue if (q[0], q[2]) in want]
    return queue


def cmd_pending(args):
    """まだ `--steps` に届いていない run の数を出す（CI が「このランナーはもう回すものが無い」を知る用）"""
    root = pathlib.Path(args.root)
    queue = plan_queue(args, parse_arms(args))
    print(sum(1 for name, _, seed in queue if last_step(root / f"{name}_s{seed}") < args.steps))


def cmd_run(args):
    root = pathlib.Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    arms = parse_arms(args)
    book = str(pathlib.Path(args.book).resolve())
    meta_path = root / "arms.json"
    known = json.loads(meta_path.read_text()).get("arms", {}) if meta_path.exists() else {}
    for name, extra in arms:
        if known.get(name, extra) != extra:
            raise SystemExit(f"腕 {name} の引数が前と違う（前: {known[name]} / 今: {extra}）。別の名前にする")
    meta_path.write_text(json.dumps({"common": args.common, "export_every": args.export_every,
                                     "arms": {**known, **dict(arms)}}, ensure_ascii=False, indent=2))
    deadline = time.time() + args.time_limit if args.time_limit else None
    queue = plan_queue(args, arms)
    running = {}  # Popen → run

    def launch(name, extra, seed):
        run = root / f"{name}_s{seed}"
        done = last_step(run)
        if done >= args.steps:
            print(f"[skip] {run.name}: 済み（{done} 更新）", flush=True)
            return
        cmd = [sys.executable, "-u", str(HERE / "train_rnad.py"), *shlex.split(args.common), "--open-book", book,
               "--export-every", str(args.export_every), *shlex.split(extra), "--seed", str(seed),
               "--steps", str(args.steps), "--out", str(run)]
        run.mkdir(parents=True, exist_ok=True)
        log = open(run / "stdout.log", "a")
        env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1")
        running[subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)] = run
        print(f"[start] {run.name}（{done} 更新から）", flush=True)

    failed = 0
    last_report = 0.0
    while queue or running:
        timed_out = deadline is not None and time.time() >= deadline
        while queue and len(running) < args.jobs and not timed_out:
            launch(*queue.pop(0))
        if timed_out:
            if queue:
                print(f"[time] 時間切れ: {len(queue)} 本は始めていない", flush=True)
                queue = []
            for p, run in running.items():
                print(f"[time] {run.name} を止める（{last_step(run)} 更新）", flush=True)
                p.send_signal(signal.SIGTERM)
            for p in running:
                try:
                    p.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    p.kill()
            running.clear()
            break
        time.sleep(args.poll)
        for p in list(running):
            run = running[p]
            if p.poll() is not None:
                del running[p]
                tail = ""
                if p.returncode != 0:
                    failed += 1
                    lines = (run / "stdout.log").read_text().strip().splitlines()
                    tail = " / " + (lines[-1][:300] if lines else "")
                print(f"[done] {run.name}: {last_step(run)} 更新（終了コード {p.returncode}{tail}）", flush=True)
        if time.time() - last_report >= args.report_every and running:
            last_report = time.time()
            print("[進み] " + "  ".join(f"{run.name} {last_step(run)}" for run in running.values()), flush=True)
    print("終わった", flush=True)
    if failed:
        raise SystemExit(f"{failed} 本が異常終了した")


def cmd_status(args):
    root = pathlib.Path(args.root)
    print("| run | 更新 | 書き出し済み | 1更新の秒 | 手数 | 1局の反則/側 | 方策のエントロピー |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, seed, d in runs_in(root):
        rows = read_log(d)
        if not rows:
            print(f"| {d.name} | 0 | – | – | – | – | – |")
            continue
        tail = rows[-args.window:]

        def mean(key):
            xs = [r[key] for r in tail if isinstance(r.get(key), (int, float))]
            return sum(xs) / len(xs) if xs else float("nan")

        sec = mean("collect_sec") + mean("update_sec")
        ex = exported_steps(d)
        print(f"| {d.name} | {rows[-1]['step']} | {ex[-1] if ex else '–'} | {sec:.1f} | {mean('plies'):.0f} | "
              f"{mean('fouls_per_side'):.2f} | {mean('entropy_mean'):.2f} |")


def common_step(root: pathlib.Path):
    """全部の run が書き出している最大の更新（無ければ None）"""
    sets = [set(exported_steps(d)) for _, _, d in runs_in(root)]
    both = set.intersection(*sets) if sets else set()
    return max(both) if both else None


def cmd_depend(args):
    """入力の組を 0 にして渡したときの方策の変化（全変動距離）と、最善手が変わる割合。
    局面はその重みの自己対局（序盤は定跡を強制）の、`--skip-plies` 手目以降"""
    import numpy as np
    import torch

    import openings
    import tsuitate_rl
    from export_weights import read as read_weights
    from rnad import sample_actions

    torch.set_num_threads(args.threads)
    dev = torch.device("cpu")
    planes = tsuitate_rl.NUM_PLANES_V6  # いちばん広い版で観測を書き、ネットは自分の枚数ぶんだけ読む
    extra_groups = dict(tsuitate_rl.EXTRA_PLANE_GROUPS)
    root = pathlib.Path(args.root)
    step = args.step if args.step is not None else common_step(root)
    if step is None:
        raise SystemExit("全部の run に共通の書き出しが無い")
    kw = openings.env_kwargs(openings.load_book(args.book, "category"), 1.0, None)
    rows = []
    for arm, seed, d in runs_in(root):
        path = d / f"current_{step}.bin"
        if not path.exists():
            continue
        net = read_weights(path).eval()
        used = net.config.get("extras", []) or []
        groups = {}
        if net.config["planes"] >= tsuitate_rl.NUM_PLANES_DEDUCED:
            groups["deduced"] = (tsuitate_rl.NUM_PLANES, tsuitate_rl.NUM_PLANES_DEDUCED - tsuitate_rl.NUM_PLANES)
        groups.update({g: extra_groups[g] for g in extra_groups if g in used})
        env = tsuitate_rl.VecEnv(args.games, seed=1000 + seed, auto_reset=False, planes=planes, **kw)
        env.reset()
        obs_l, mask_l = [], []
        with torch.no_grad():
            for _ in range(260):
                alive = env.alive()
                if not alive.any():
                    break
                obs, mask, _ = env.observe()
                idx = np.flatnonzero(alive)
                a, _ = sample_actions(net, obs[idx], mask[idx], dev)
                actions = np.full(args.games, -1, dtype=np.int64)
                actions[idx] = a.numpy()
                env.step(actions)
                obs_l.append(obs[idx].reshape(len(idx), planes, 9, 9).astype(np.float32))
                mask_l.append(mask[idx].astype(bool))
        obs, mask = np.concatenate(obs_l), np.concatenate(mask_l)
        # 手数のプレーン（84番。値は 手数 ÷ 200）
        keep = obs[:, 84, 0, 0] * 200 >= args.skip_plies
        obs, mask = obs[keep], mask[keep]

        @torch.no_grad()
        def policy(x):
            out = []
            for i in range(0, len(x), 512):
                logits, _ = net(torch.from_numpy(x[i:i + 512]))
                out.append(logits.masked_fill(~torch.from_numpy(mask[i:i + 512]), float("-inf")).softmax(-1))
            return torch.cat(out)

        p0 = policy(obs)
        top0 = p0.argmax(-1)
        for g, (start, n) in groups.items():
            x = obs.copy()
            nz = np.abs(x[:, start:start + n]).reshape(len(x), -1).max(1) > 0
            x[:, start:start + n] = 0
            p1 = policy(x)
            tv = 0.5 * (p0 - p1).abs().sum(-1).numpy()
            flip = (p1.argmax(-1) != top0).numpy()
            rows.append({"run": d.name, "arm": arm, "seed": seed, "step": step, "group": g, "positions": int(len(obs)),
                         "nonzero": float(nz.mean()), "tv": float(tv.mean()),
                         "tv_nonzero": float(tv[nz].mean()) if nz.any() else None,
                         "flip_nonzero": float(flip[nz].mean()) if nz.any() else None})
        print(f"[depend] {d.name} 更新 {step}: " + "  ".join(
            f"{r['group']} {r['tv']:.3f}" for r in rows if r["run"] == d.name), flush=True)
    pathlib.Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="腕 × seed を回す")
    pe = sub.add_parser("pending", help="まだ --steps に届いていない run の数")
    for x in (r, pe):
        x.add_argument("--root", required=True)
        x.add_argument("--arms", default="", help=f"入力の版（カンマ区切り。{sorted(ARMS)}）")
        x.add_argument("--arm", action="append", default=[], help="追加の腕 '名前=train_rnad.py の引数'（何度でも）")
        x.add_argument("--seeds", required=True, help="カンマ区切り")
        x.add_argument("--only", default="", help="この組だけ回す（'腕:seed' のカンマ区切り。既定は全部）")
        x.add_argument("--steps", type=int, required=True, help="通しの更新回数（1更新 64局）")
    pe.set_defaults(func=cmd_pending)
    r.add_argument("--book", required=True, help="序盤の定跡（tsuitate-bot の rl_openings.json）")
    r.add_argument("--jobs", type=int, default=4, help="同時に回す本数（1本 1スレッド）")
    r.add_argument("--export-every", type=int, default=100, help="重みを書き出す間隔（更新）")
    r.add_argument("--time-limit", type=float, default=0, help="この秒数で止める（0 = 止めない）")
    r.add_argument("--poll", type=float, default=30)
    r.add_argument("--report-every", type=float, default=600, help="進み具合を出す間隔（秒）")
    r.add_argument("--common", default=COMMON, help="全部の腕に共通の引数")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("status", help="進み具合の表")
    s.add_argument("root")
    s.add_argument("--window", type=int, default=50, help="平均する直近の更新の数")
    s.set_defaults(func=cmd_status)
    d = sub.add_parser("depend", help="入力の組をどれだけ読んでいるか")
    d.add_argument("root")
    d.add_argument("--book", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--step", type=int, default=None, help="測る更新（既定は全部の run に共通の最大の書き出し）")
    d.add_argument("--games", type=int, default=32)
    d.add_argument("--skip-plies", type=int, default=20)
    d.add_argument("--threads", type=int, default=2)
    d.set_defaults(func=cmd_depend)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
