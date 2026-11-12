"""離陸（相手に勝ち始める時点）の率と局数を、報酬の整形の腕ごとに測る。

    # 腕 × seed をまとめて回す（並列 --jobs 本。離陸した run は --after 更新だけ続けてから止める）
    .venv/bin/python rnad/takeoff.py run --root out_rnad/tk --opponent rl_mmdopen_v15 --seeds 1,2,3,4 --steps 400 --jobs 4 \\
        --arm mat='--shape d_capture_value=0.02 --shape-cap 1.0' \\
        --arm mat+chk='--shape d_capture_value=0.02,checks=0.05 --shape-cap 1.0'
    # 表にする（回している途中でも読める）
    .venv/bin/python rnad/takeoff.py report out_rnad/tk [--extra mat=out_rnad/fx_mat,out_rnad/fx_mat_s2]

tsuitate-bot の docs/rl-fixed-opponent.md（8節）。乱数初期化から固定した相手に当てる学習は、最初は1局も勝てず、
ある時点で勝ち方を1つ見つけて勝率が急に立ち上がる（離陸）。離陸は確率的な出来事なので、整形の腕の良し悪しは
「同じ局数で何割の seed が離陸するか」「離陸までの局数」で比べる。

- **離陸** = 直近 `--window` 更新（既定 50）の勝率が `--threshold`（既定 0.10）以上になった最初の更新。
  勝率は学習中の対局（現在のネットのサンプリング）の勝ち数 ÷ 局数（log.jsonl の `league_games`）
- `run` は腕ごとの引数に共通の引数（`COMMON`。パイロットで離陸した PPO 型の設定）を足して train_rnad.py を起動する。
  出力は `<root>/<腕>_s<seed>/`。終わった run・離陸で止めた run は飛ばす（途中で止まった run は checkpoint から続ける）
- `report` は腕ごとに seed 別の離陸の局数（離陸しなければ「> 打ち切りの局数」）、離陸の率、離陸までの局数の中央値、
  打ち切りを上限とした平均（離陸しなかった run は打ち切りの局数で数える。小さいほど速い）を出す
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
from stage_log import drop_rolled_back  # noqa: E402

# パイロットで離陸した設定（32ch×3・64局/更新・相手固定・PPO 型。docs/rl-fixed-opponent.md 4.1節）
COMMON = ("--algo mmd --init none --anchor none --eval-every 0 --channels 32 --blocks 3 --planes 91 --batch 64 "
          "--compact-legal --device cpu --threads 1 --league-frac 1 --league-device gpu "
          "--mmd-magnet uniform --mmd-temp 0.01,0,0.01,0.01 --mmd-lr 3e-4,0,3e-4,3e-4 "
          "--mmd-steps 8 --mmd-epochs 2 --mmd-adv-norm 0.001 "
          "--export-every 50 --export-current --checkpoint-every 25 --watch-signals")


def _log_kind(x):
    if "eval" in x:
        return "eval"
    return "update" if "league_games" in x or "update_sec" in x else None


def read_log(run: pathlib.Path):
    """log.jsonl → [(step, 勝ち, 局数)]（step の昇順）。checkpoint から再開して step が戻ったら、そこから先の前の run の
    行を捨てて後の行を使う（stage_log.drop_rolled_back と同じ規約。同じ step を後の行で上書きするだけでは、まだ書き直されて
    いない前の run の後続の step が残り、離陸を誤って見つける）"""
    path = run / "log.jsonl"
    rows = []
    if path.exists():
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue  # 書きかけの行
            if isinstance(x, dict) and "step" in x:
                rows.append(x)
    kept, _, _ = drop_rolled_back(rows, _log_kind)
    out = []
    for kind, x in kept:
        lg = x.get("league_games")
        if kind == "update" and lg is not None:
            out.append((x["step"], sum(v[0] for v in lg.values()), sum(sum(v) for v in lg.values())))
    return out


def find_takeoff(rows, window=50, threshold=0.10):
    """離陸した更新（直近 window 更新の勝率が threshold 以上になった最初の更新）。無ければ None。
    学習の頭の window 更新に満たない区間では判定しない"""
    wins = games = 0
    q = []
    for step, w, n in rows:
        q.append((w, n))
        wins += w
        games += n
        if len(q) > window:
            w0, n0 = q.pop(0)
            wins -= w0
            games -= n0
        if len(q) == window and games and wins / games >= threshold:
            return step
    return None


def run_status(run: pathlib.Path, window, threshold):
    rows = read_log(run)
    last = rows[-1][0] if rows else 0
    t = find_takeoff(rows, window, threshold)
    games = sum(n for _, _, n in rows)
    per = games / len(rows) if rows else 0.0
    return {"rows": rows, "last": last, "takeoff": t, "games_per_update": per}


def cmd_run(args):
    root = pathlib.Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    arms = []
    for spec in args.arm:
        name, sep, extra = spec.partition("=")
        if not sep or not name:
            raise SystemExit(f"--arm は '名前=引数'（{spec}）")
        arms.append((name, extra))
    if len({n for n, _ in arms}) != len(arms):
        raise SystemExit("腕の名前が重複している")
    seeds = [int(s) for s in args.seeds.split(",")]
    # 腕の引数は足していく（腕を変えて回し直しても、前に回した腕の引数を残す）。同じ名前で引数が違えば止める
    meta_path = root / "arms.json"
    known = json.loads(meta_path.read_text()).get("arms", {}) if meta_path.exists() else {}
    for name, extra in arms:
        if known.get(name, extra) != extra:
            raise SystemExit(f"腕 {name} の引数が前と違う（前: {known[name]} / 今: {extra}）。別の名前にする")
    meta_path.write_text(json.dumps({"opponent": args.opponent, "common": args.common, "steps": args.steps,
                                     "arms": {**known, **dict(arms)}, "window": args.window,
                                     "threshold": args.threshold}, ensure_ascii=False, indent=2))
    # seed を外側に回す（途中で止めても腕ごとの本数がそろう）。--queue があればその順（'腕:seed,...'）で、書いたものだけ回す
    queue = [(name, extra, s) for s in seeds for name, extra in arms]
    if args.queue:
        by_name = dict(arms)
        queue = []
        for item in args.queue.split(","):
            name, sep, seed = item.strip().rpartition(":")
            if not sep or name not in by_name:
                raise SystemExit(f"--queue の '{item}' は '腕:seed'（腕は --arm で渡したもの）")
            queue.append((name, by_name[name], int(seed)))
    running = {}  # Popen → (名前, run, 離陸を見つけた時の更新)

    def launch(name, extra, seed):
        run = root / f"{name}_s{seed}"
        st = run_status(run, args.window, args.threshold)
        stop_at = (st["takeoff"] + args.after) if st["takeoff"] is not None else None
        if st["last"] >= args.steps or (stop_at is not None and st["last"] >= stop_at):
            print(f"[skip] {run.name}: 済み（{st['last']} 更新、離陸 {st['takeoff']}）", flush=True)
            return
        cmd = [sys.executable, "-u", str(HERE / "train_rnad.py"), *shlex.split(args.common),
               "--league", args.opponent, *shlex.split(extra), "--seed", str(seed), "--steps", str(args.steps),
               "--out", str(run)]
        run.mkdir(parents=True, exist_ok=True)
        log = open(run / "stdout.log", "a")
        env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        running[p] = (name, run, None)
        print(f"[start] {run.name}（{st['last']} 更新から）", flush=True)

    while queue or running:
        while queue and len(running) < args.jobs:
            launch(*queue.pop(0))
        time.sleep(args.poll)
        for p in list(running):
            name, run, seen = running[p]
            st = run_status(run, args.window, args.threshold)
            if p.poll() is not None:
                del running[p]
                tail = ""
                if p.returncode != 0:
                    lines = (run / "stdout.log").read_text().strip().splitlines()
                    tail = " / " + (lines[-1][:200] if lines else "")
                print(f"[done] {run.name}: {st['last']} 更新、離陸 {st['takeoff']}（終了コード {p.returncode}{tail}）", flush=True)
                continue
            if st["takeoff"] is not None:
                if seen is None:
                    print(f"[takeoff] {run.name}: 更新 {st['takeoff']}（約 {st['takeoff'] * st['games_per_update']:,.0f} 局）",
                          flush=True)
                    running[p] = (name, run, st["takeoff"])
                if st["last"] >= st["takeoff"] + args.after:
                    p.send_signal(signal.SIGTERM)
    print("全部終わった", flush=True)


def fmt_games(x):
    return f"{x / 1000:.1f}k"


def cmd_report(args):
    groups = {}  # 腕 → [(ラベル, run)]
    meta = {}
    for root in args.roots:
        root = pathlib.Path(root)
        if (root / "arms.json").exists():
            meta = json.loads((root / "arms.json").read_text())
        for run in sorted(root.iterdir()):
            if run.is_dir() and "_s" in run.name and (run / "log.jsonl").exists():
                arm, _, seed = run.name.rpartition("_s")
                groups.setdefault(arm, []).append((f"s{seed}", run))
    for spec in args.extra:
        arm, sep, dirs = spec.partition("=")
        if not sep:
            raise SystemExit(f"--extra は '腕=dir,dir,...'（{spec}）")
        for i, d in enumerate(dirs.split(",")):
            groups.setdefault(arm, []).append((pathlib.Path(d).name, pathlib.Path(d)))
    window, thr = args.window, args.threshold
    horizon = args.horizon
    print(f"離陸 = 直近 {window} 更新の勝率が {thr:.0%} 以上になった最初の更新。"
          + (f"打ち切り {horizon} 更新。" if horizon else "") + "局数は学習の頭からの合計\n")
    print("| 腕 | 離陸した run | 離陸までの局数（seed ごと） | 中央値（離陸した run） | 打ち切りを上限にした平均 | 離陸の前の勝ち（最初の 200更新） |")
    print("| --- | --- | --- | --- | --- | --- |")
    out = []
    for arm, runs in groups.items():
        cells, took, restricted, pre = [], [], [], []
        n_valid = 0
        for label, run in runs:
            st = run_status(run, window, thr)
            rows = st["rows"]
            if not rows:
                continue
            per = st["games_per_update"]
            t = st["takeoff"]
            if horizon and t is not None and t > horizon:
                t = None
            limit = min(st["last"], horizon) if horizon else st["last"]
            pre.append(sum(w for s, w, _ in rows if s <= 200))
            if t is not None:
                took.append(t * per)
                restricted.append(t * per)
                cells.append(f"{label}: **{fmt_games(t * per)}**")
                n_valid += 1
            else:
                # 打ち切りより前で止まっている run は「まだ分からない」ので率に数えない
                short = horizon and st["last"] < horizon
                cells.append(f"{label}: > {fmt_games(limit * per)}" + ("（途中）" if short else ""))
                if not short:
                    restricted.append(limit * per)
                    n_valid += 1
        if not cells:
            continue
        med = sorted(took)[len(took) // 2] if took else None
        out.append((arm, len(took), n_valid, med, restricted))
        print(f"| `{arm}` | {len(took)} / {n_valid} | " + " ・ ".join(cells) + " | "
              + (fmt_games(med) if med else "—") + " | "
              + (fmt_games(sum(restricted) / len(restricted)) if restricted else "—") + " | "
              + " / ".join(str(x) for x in pre) + " |")
    if meta.get("arms"):
        print("\n腕の引数:")
        for name, extra in meta["arms"].items():
            print(f"- `{name}`: `{extra}`")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="腕 × seed をまとめて回す")
    r.add_argument("--root", required=True, help="出力の親（<root>/<腕>_s<seed>/）")
    r.add_argument("--opponent", required=True, help="学習の相手（train_rnad.py の --league）")
    r.add_argument("--arm", action="append", default=[], help="'名前=train_rnad.py の追加の引数'（何個でも）")
    r.add_argument("--seeds", default="1,2")
    r.add_argument("--queue", default="", help="回す順を '腕:seed,腕:seed,...' で指定する（--seeds の代わり。書いたものだけ回す）")
    r.add_argument("--steps", type=int, default=400, help="打ち切りの更新回数")
    r.add_argument("--jobs", type=int, default=4, help="同時に回す本数")
    r.add_argument("--common", default=COMMON, help="全部の腕に共通の引数")
    r.add_argument("--window", type=int, default=50)
    r.add_argument("--threshold", type=float, default=0.10)
    r.add_argument("--after", type=int, default=30, help="離陸を見つけてから続ける更新の数（その後で止める）")
    r.add_argument("--poll", type=float, default=20.0, help="ログを見る間隔（秒）")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report", help="離陸の表")
    p.add_argument("roots", nargs="*", help="run の --root（何個でも）")
    p.add_argument("--extra", action="append", default=[], help="'腕=dir,dir,...'（ほかの場所の run を腕に足す）")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--horizon", type=int, default=0, help="この更新で打ち切ったものとして数える（0 = 各 run の最後まで）")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
