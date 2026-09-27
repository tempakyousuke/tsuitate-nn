"""M2b: R-NaD の自己対局学習（tsuitate-bot の docs/rl-deepnash-design.md）。

    .venv/bin/python rnad/train_rnad.py --init out_rnad/bc/checkpoint.pt --out out_rnad/rnad_bc
    .venv/bin/python rnad/train_rnad.py --init none --out out_rnad/rnad_scratch   # ゼロから

1回の更新 = バッチ数ぶんの局を最後まで自己対局（tsuitate_rl.VecEnv、auto_reset=False）→
R-NaD の損失で1ステップ。学習の進み具合は**錨（既定は模倣学習のネット）との直接対戦の勝率**で
測る（凍結版 estimator は1手1秒以上かかり学習ループに入れられないため）。錨に勝ち越していけば
学習が進んでいる。区切りでターゲットネットの重みを Rust 用に書き出すので、CI の arena
（`rl_policy:release:...`）でも測れる。

記録: <out>/log.jsonl（更新ごと・評価ごとに1行）、<out>/checkpoint.pt、<out>/policy_<step>.bin
"""

import argparse
import json
import pathlib
import time
from collections import Counter

import numpy as np
import torch

import tsuitate_rl
from export_weights import write as write_weights
from model import PolicyValueNet
from rnad import RNaDConfig, RNaDLearner, collect, play_match

ROOT = pathlib.Path(__file__).resolve().parent.parent


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_net(path: str, channels: int, blocks: int) -> PolicyValueNet:
    if path == "none":
        return PolicyValueNet(channels, blocks)
    ck = torch.load(path, map_location="cpu")
    net = PolicyValueNet(ck["config"]["channels"], ck["config"]["blocks"], ck["config"]["planes"])
    net.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck["net"])
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "out_rnad" / "bc" / "checkpoint.pt"),
                    help="初期化に使う checkpoint（none = 乱数初期化）")
    ap.add_argument("--anchor", default=str(ROOT / "out_rnad" / "bc" / "checkpoint.pt"),
                    help="進み具合を測る対戦相手（固定）")
    ap.add_argument("--out", default=str(ROOT / "out_rnad" / "rnad"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64, help="1回の更新で自己対局する局数")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-games", type=int, default=128)
    ap.add_argument("--export-every", type=int, default=100)
    ap.add_argument("--schedule", type=int, default=100, help="正則化方策を差し替える間隔（更新回数）")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eta", type=float, default=0.2)
    # NeuRD のしきい値（合法手の平均を引いたロジットが ±β を超える方向へは押さない）。
    # 参照の既定 2.0 は小さいゲーム用で、行動数の多い本ゲームの模倣ネットは最善手が +6.6
    # （中央値、99% 点で +14.9）にあるので、β=2 だと方策が平らへ潰される（2026-09-28 実測。
    # 300 更新で現在のネットが錨に 23% まで弱くなった）。模倣ネットの広がりより上に置く
    ap.add_argument("--beta", type=float, default=16.0)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = RNaDConfig(eta=args.eta, beta=args.beta, lr=args.lr, schedule_sizes=(args.schedule,),
                     schedule_repeats=(1,))
    learner = RNaDLearner(load_net(args.init, args.channels, args.blocks), cfg, device)
    anchor = load_net(args.anchor, args.channels, args.blocks).to(device).eval()
    ck_path = out / "checkpoint.pt"
    if ck_path.exists():
        learner.load_state_dict(torch.load(ck_path, map_location=device))
        print(f"再開: step {learner.step_count}")
    (out / "config.json").write_text(json.dumps({"args": vars(args), "rnad": cfg.__dict__,
                                                 "net": learner.net.config}, indent=2))

    env = tsuitate_rl.VecEnv(args.batch, seed=args.seed, auto_reset=False)
    eval_env = tsuitate_rl.VecEnv(args.eval_games, seed=args.seed + 10_000, auto_reset=False)
    log = open(out / "log.jsonl", "a")

    def evaluate(step):
        # ターゲットは現在のネットにゆっくり追従するだけなので、両方を測る
        # （ターゲットだけだと現在のネットの劣化が隠れる。2026-09-28 の β=2 で実際に隠れた）
        for which, net in (("target", learner.target), ("current", learner.net)):
            t0 = time.time()
            w, l, d = play_match(eval_env, net, anchor, device)
            eval_env.pop_finished()
            n = w + l + d
            rate = (w + 0.5 * d) / n
            rec = {"step": step, "eval": which, "eval_vs_anchor": rate, "w": w, "l": l, "d": d,
                   "ci95": 1.96 * (rate * (1 - rate) / n) ** 0.5, "sec": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[eval] step {step} {which}: 錨に {w}勝{l}敗{d}分 = {rate:.3f}±{rec['ci95']:.3f}",
                  flush=True)

    if learner.step_count == 0:
        evaluate(0)
    while learner.step_count < args.steps:
        t0 = time.time()
        batch = collect(env, learner.net, device)
        t1 = time.time()
        finished = env.pop_finished()
        info = learner.update(batch)
        t2 = time.time()
        step = learner.step_count
        reasons = Counter(f["reason"] for f in finished)
        fouls = [sum(f["fouls"]) / 2 for f in finished]
        rec = {"step": step, **info, "games": len(finished),
               "plies": float(np.mean([f["plies"] for f in finished])),
               "fouls_per_side": float(np.mean(fouls)), "reasons": dict(reasons),
               "collect_sec": t1 - t0, "update_sec": t2 - t1}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"step {step}: 標本 {info['samples']} 手数 {rec['plies']:.0f} 反則/側 "
              f"{rec['fouls_per_side']:.2f} loss_v {info['loss_v']:.3f} nerd {info['loss_nerd']:.3f} "
              f"α {info['alpha']:.2f}{' [正則化更新]' if info['reg_update'] else ''} "
              f"({t1 - t0:.1f}s + {t2 - t1:.1f}s)", flush=True)
        if step % args.eval_every == 0:
            evaluate(step)
        if step % args.export_every == 0 or step == args.steps:
            torch.save(learner.state_dict(), ck_path)
            write_weights(learner.target, out / f"policy_{step}.bin")
    torch.save(learner.state_dict(), ck_path)


if __name__ == "__main__":
    main()
