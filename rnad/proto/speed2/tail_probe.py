"""F9: 収集の末尾（終局した局が増えて 768局の枠が空く手番）の量を数える（学習には使わない）。

    python rnad/proto/speed2/tail_probe.py [--weights ../tsuitate-bot/models/rl_mmdopen_v10.bin] \\
        [--book ../tsuitate-bot/rl_openings.json] [--open-p 0.25] [--games 768] [--box box_bench.txt]

本線と同じ形（`observe_alive_f16_csr`・Gumbel の抽出・序盤の強制）で全局を最後まで指し、手番ごとの生存局数を記録する。
数えるのは手番の数だけで、壁時計は使わない（CPU でも 3070 でも同じ数になる）:
- T（逐次の手番の数）、決定点の数 N、平均の生存率 N/(T·B)、自動リセットなら T がいくつになるか（N/B）
- 生存率が 50% / 25% / 10% 未満の手番が T と N に占める割合、手数の分布
`--box` に `box_bench.py` の出力（3. の「n=… … ms」の行）を渡すと、生存局数ごとの1手番の秒を補間して、
今の収集と自動リセット（毎手番 B 局）の GPU 側の秒を見積もる（CPU 側の env の秒と、更新との取り合いは入らない）。
tsuitate-bot の docs/rl-learning-speed.md 11.3（rl_mmdopen_v10 で T 185・N/B 80.5）
"""

import argparse
import json
import re

import numpy as np
import torch

from proto_common import MODELS, tsuitate_rl
from export_weights import read as read_weights  # noqa: E402
from openings import env_kwargs, load_book  # noqa: E402
from rnad import NUM_ACTIONS, _gumbel_sample, forward  # noqa: E402


def box_times(path):
    """box_bench.py の出力の 3. から {n: 1手番の ms}（本線の収集と同じ「fp16 + CSR で送って…」の行）"""
    out = {}
    for line in open(path, encoding="utf-8"):
        if "fp16 + CSR" not in line:
            continue
        m = re.search(r"n=\s*(\d+)\s.*?([0-9.]+) ms", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def interp(times, n):
    ks = sorted(times)
    return float(np.interp(n, ks, [times[k] for k in ks]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(MODELS / "rl_mmdopen_v10.bin"))
    ap.add_argument("--book", default=str(MODELS.parent / "rl_openings.json"))
    ap.add_argument("--open-p", type=float, default=0.25)
    ap.add_argument("--games", type=int, default=768)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--box", default="", help="box_bench.py の出力を保存したファイル")
    ap.add_argument("--counts-out", default="", help="手番ごとの生存局数を JSON で書く")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    net = read_weights(args.weights)
    B = args.games
    kw = env_kwargs(load_book(args.book), args.open_p) if args.open_p > 0 else {}
    env = tsuitate_rl.VecEnv(B, seed=args.seed, auto_reset=False, planes=net.config["planes"], **kw)
    if kw:
        env.set_opening_seed(args.seed)
    env.reset()
    gen = torch.Generator().manual_seed(args.seed)
    counts_per_step = []
    with torch.no_grad():
        for _ in range(400):
            alive = env.alive()
            if not alive.any():
                break
            idx = np.flatnonzero(alive)
            n = len(idx)
            bits, cols, counts, _pl = env.observe_alive_f16_csr()
            oh = torch.from_numpy(bits.view(np.float16).reshape(n, -1, 9, 9))
            logits, _v = forward(net, oh.float())
            rows = torch.repeat_interleave(torch.arange(n), torch.from_numpy(counts).long())
            legal = torch.zeros((n, NUM_ACTIONS), dtype=torch.bool)
            legal[rows, torch.from_numpy(cols).long()] = True
            a, _mu = _gumbel_sample(logits, legal, gen)
            actions = np.full(B, -1, dtype=np.int64)
            actions[idx] = a.numpy()
            env.step(actions)
            counts_per_step.append(n)
    fin = env.pop_finished()
    c = np.array(counts_per_step)
    T, N = len(c), int(c.sum())
    out = {"weights": args.weights, "games": B, "open_p": args.open_p, "T": T, "N": N, "decisions_per_game": N / B,
           "mean_alive_frac": N / (T * B), "autoreset_T": N / B, "steps_ratio": T / (N / B)}
    for x in (0.5, 0.25, 0.1):
        m = c / B < x
        out[f"steps_share_alive_lt_{x}"] = float(m.mean())
        out[f"decisions_share_alive_lt_{x}"] = float(c[m].sum() / N)
    plies = np.array([f["plies"] for f in fin if f.get("plies") is not None])
    if len(plies):
        out.update(plies_mean=float(plies.mean()), plies_p50=float(np.median(plies)),
                   plies_p90=float(np.percentile(plies, 90)), plies_max=int(plies.max()))
    if args.box:
        t = box_times(args.box)
        if len(t) >= 2:
            now = sum(interp(t, n) for n in c) / 1000
            auto = (N / B) * interp(t, B) / 1000
            out.update(gpu_side_sec_now=now, gpu_side_sec_autoreset=auto, gpu_side_ratio=now / auto)
        else:
            out["box_error"] = f"{args.box} から n ごとの ms を読めなかった"
    if args.counts_out:
        with open(args.counts_out, "w") as f:
            json.dump(counts_per_step, f)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
