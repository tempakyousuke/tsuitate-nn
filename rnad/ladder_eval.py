"""ハンデの梯子で進み具合を測る: 学習中のネットを、同じ相手を雑音で弱めた段（`<相手>@<ε>`）と対戦させる。

    .venv/bin/python rnad/ladder_eval.py out_rnad/fx_base --opponent rl_v15 --eps 1,0.5,0.3,0.1,0 --games 64
    .venv/bin/python rnad/ladder_eval.py out_rnad/fx_base out_rnad/fx_mat --report        # 測った結果を並べる

tsuitate-bot の docs/rl-fixed-opponent.md。強い相手にゼロから当たると得点は長く 0 のままで、学習が進んでいるかが見えない。
`<相手>@<ε>` は相手の手番ごとに確率 ε で合法手の乱数手を指す版（tsuitate_rl のハンデ）で、ε = 1 は全部が乱数手、ε = 0 が本物。
段ごとの得点から **ε50（得点が 0.5 になる ε。小さいほど強い。本物に五分なら 0）** を出すと、勝ちが出る前から
同じ相手への近さを1つの数で追える。

- run の書き出し（`current_<step>.bin`。無ければ `policy_<step>.bin`）を、段ごとに `--games` 局（先後同数）サンプリングで指させる
- 結果は `<run>/ladder_eval.jsonl` に1行ずつ足す（同じ step・同じ段は、局数・seed・書き出しの重みの指紋が前の評価と
  同じなら測り直さない。違えば測り直して行を足し、集計は後の行を使う。`--redo` で同じ条件でも測り直す）。手の乱数は torch の乱数を
  step と段によらない固定の seed から作るので、書き出しどうしの差は同じ乱数列の上で比べられる（共通乱数ではない）
- ε50 は段の得点を ε の小さい順に見て、0.5 をまたぐ最初の区間の線形補間。全段で 0.5 未満なら「> 最大の ε」、
  全段で 0.5 以上なら「0」
"""

import argparse
import json
import pathlib
import re
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent


def step_files(run: pathlib.Path, which: str):
    out = {}
    for p in run.glob(f"{which}_*.bin"):
        m = re.fullmatch(rf"{which}_(\d+)\.bin", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


@torch.no_grad()
def play(net, opponent, games, seed, planes, device):
    """net（サンプリング）対 Rust の相手 `opponent`。学習側から見た局ごとの結果（+1 / 0 / −1）と、局の記録"""
    import tsuitate_rl
    from rnad import sample_actions
    kw = {"planes": planes} if planes != tsuitate_rl.NUM_PLANES else {}
    env = tsuitate_rl.VecEnv(games, opponent=opponent, seed=seed, auto_reset=False, **kw)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    out = np.zeros(games, dtype=np.int8)
    while True:
        alive = env.alive()
        if not alive.any():
            break
        obs, mask, _ = env.observe()
        idx = np.flatnonzero(alive)
        actions = np.full(games, -1, dtype=np.int64)
        actions[idx] = sample_actions(net, obs[idx], mask[idx], device, gen)[0].numpy()
        learner = env.learner_colors()
        rew, done = env.step(actions)
        d = np.flatnonzero(done)
        out[d] = np.sign(rew[d, learner[d]]).astype(np.int8)
    return out, env.pop_finished()


def eps50(points):
    """[(ε, 得点)] → ε50（得点が 0.5 になる ε）。(値, 印) を返す。印は '' / '>'（全段で 0.5 未満）"""
    pts = sorted(points)
    if not pts:
        return float("nan"), ""
    if pts[0][1] >= 0.5:
        return pts[0][0], ""
    for (e0, s0), (e1, s1) in zip(pts, pts[1:]):
        if s1 >= 0.5:
            return e0 + (e1 - e0) * (0.5 - s0) / (s1 - s0), ""
    return pts[-1][0], ">"


def load_rows(run: pathlib.Path):
    path = run / "ladder_eval.jsonl"
    rows = {}
    if path.exists():
        for line in open(path):
            if line.strip():
                x = json.loads(line)
                rows[(x["which"], x["step"], x["opponent"])] = x
    return rows


def cmd_eval(args):
    from export_weights import fingerprint, stale_conditions
    from export_weights import read as read_weights
    from model import NUM_PLANES
    device = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    eps = [float(e) for e in args.eps.split(",")]
    if args.games % 2:
        raise SystemExit("--games は偶数（先後を同数にするため）")
    for run in args.runs:
        files = step_files(run, args.which)
        if not files:
            raise SystemExit(f"{run} に {args.which}_<step>.bin が無い（train_rnad.py の --export-every / --export-current）")
        steps = sorted(files)
        if args.steps:
            want = [int(s) for s in args.steps.split(",")]
            steps = [s for s in steps if s in want]
        elif args.every:
            steps = [s for s in steps if s % args.every == 0 or s == steps[-1]]
        done = load_rows(run)
        with open(run / "ladder_eval.jsonl", "a") as f:
            for step in steps:
                net = None
                w_fp = fingerprint(files[step])
                for e in eps:
                    opp = args.opponent if e == 0 else f"{args.opponent}@{e:g}"
                    cond = {"games": args.games, "seed": args.seed, "weights": w_fp}
                    prev = done.get((args.which, step, opp))
                    if prev is not None and not getattr(args, "redo", False):
                        diff = stale_conditions(prev, cond)
                        if not diff:
                            continue
                        print(f"{run.name} {args.which}_{step} 対 {opp}: 前の評価と条件が違うので測り直す（"
                              + "、".join(f"{k} {a} → {b}" for k, a, b in diff) + "）", flush=True)
                    if net is None:
                        net = read_weights(files[step]).to(device).eval()
                    t0 = time.time()
                    planes = net.config["planes"] if hasattr(net, "config") else NUM_PLANES
                    res, fin = play(net, opp, args.games, args.seed, planes, device)
                    w, l, d = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
                    lc = [x["learner"] for x in fin]
                    rec = {"which": args.which, "step": step, "opponent": opp, "base": args.opponent, "eps": e,
                           "w": w, "l": l, "d": d, **cond,
                           "plies": float(np.mean([x["plies"] for x in fin])),
                           "fouls": float(np.mean([x["fouls"][c] for x, c in zip(fin, lc)])),
                           "opp_fouls": float(np.mean([x["fouls"][1 - c] for x, c in zip(fin, lc)])),
                           "reasons": {r: sum(x["reason"] == r for x in fin) for r in sorted({x["reason"] for x in fin})},
                           "sec": time.time() - t0}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done[(args.which, step, opp)] = rec
                    print(f"{run.name} {args.which}_{step} 対 {opp}: {w}勝{l}敗{d}分 = {(w + 0.5 * d) / args.games:.3f} "
                          f"（手数 {rec['plies']:.0f}・反則 {rec['fouls']:.1f}、{rec['sec']:.0f}s）", flush=True)


def cmd_report(args):
    for run in args.runs:
        rows = [x for x in load_rows(run).values() if x["which"] == args.which
                and (not args.opponent or x["base"] == args.opponent)]
        if not rows:
            print(f"## {run}: 測った結果が無い")
            continue
        for base in sorted({x["base"] for x in rows}):
            sub = [x for x in rows if x["base"] == base]
            eps = sorted({x["eps"] for x in sub}, reverse=True)
            print(f"\n## {run}（{args.which}、相手 {base}。得点 = 勝ち 1・引き分け 0.5）\n")
            print("| 更新 | " + " | ".join(f"ε={e:g}" for e in eps) + " | ε50 | 反則（ε 最小の段） |")
            print("| --- | " + " | ".join("---" for _ in eps) + " | --- | --- |")
            for step in sorted({x["step"] for x in sub}):
                by = {x["eps"]: x for x in sub if x["step"] == step}
                pts = [(e, (by[e]["w"] + 0.5 * by[e]["d"]) / by[e]["games"]) for e in eps if e in by]
                v, mark = eps50(pts)
                cells = [f"{dict(pts)[e]:.3f}" if e in by else "—" for e in eps]
                low = by[min(by)]
                print(f"| {step} | " + " | ".join(cells) + f" | {mark}{v:.2f} | {low['fouls']:.1f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=pathlib.Path, help="train_rnad.py の --out")
    ap.add_argument("--report", action="store_true", help="測らずに、ladder_eval.jsonl の結果を並べる")
    ap.add_argument("--opponent", default="", help="梯子の相手（tsuitate_rl の相手の名前。例 rl_v15）")
    ap.add_argument("--eps", default="1,0.5,0.3,0.1,0", help="ハンデの段（カンマ区切りの ε。0 = 本物）")
    ap.add_argument("--games", type=int, default=64, help="段ごとの局数（偶数）")
    ap.add_argument("--which", default="current", help="測る書き出し（current / policy / ema<係数> など）")
    ap.add_argument("--steps", default="", help="測る step（カンマ区切り。空 = 全部）")
    ap.add_argument("--every", type=int, default=0, help="この倍数の step だけ測る（最後の書き出しは必ず測る）")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--redo", action="store_true",
                    help="測った step・段も測り直す（行を足す。集計は後の行を使う）。--redo なしでも、局数・seed・"
                         "書き出しの重みの指紋が前の評価と違えば測り直す")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.report:
        cmd_report(args)
    else:
        if not args.opponent:
            raise SystemExit("--opponent が要る")
        import sys
        sys.path.insert(0, str(HERE))
        cmd_eval(args)


if __name__ == "__main__":
    main()
