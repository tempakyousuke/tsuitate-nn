"""1つの段階の中で書き出し（`policy_<step>.bin` = ターゲット）がどう動くかを測り、段階の中で方策が落ち着くか
（＝段階の後半が無駄になっていないか）を見る（tsuitate-bot の docs/rl-learning-speed.md「さらに速くする」）。
学習は止めずに、run の書き出しだけを読む。

    # ゲーム用 PC（段階の途中の書き出しが残っている run で）
    python rnad/stage_profile.py --dir out_rnad/deduce_s10000 --start 78000 --end 88000 --dm 10000 \\
        --games 128 --device cuda --out profile_78000.json
    # CPU なら --every 500 などで書き出しを間引く（1点 約1万決定点の順伝播）

局面は `--start` の書き出しの自己対局（`--games` 局）から作り、全部の書き出しを同じ局面の上で比べる。
出すもの:
- K(n) = KL(書き出し[start] ‖ 書き出し[start + n])（共通の局面の上の決定点平均）と、隣どうしの KL
- 最後の 20% の傾き ÷ 段階全体の平均の傾き（1 に近い = 線形に伸び続ける = 拡散か未収束、0.2 未満 = 落ち着いた）
- 模型のあてはめ: K(n) = A·c(n)² + B·(1 − e^{−τn}) + C·(n − (1 − e^{−τn})/τ)
    c(n) = 補間（[0, R] で固定点が線形に動く。R は既定 Δm/2）を「時定数 T_r の現在のネット」と「τ のターゲット」の
           1次遅れ2段に通した被覆率（A = 段階の固定点の動き²）
    B = ターゲットに残る速い雑音（2V）、C = ターゲットに見える遅い拡散（1更新あたり）
  T_r を格子で探して、(A, B, C) は非負の最小二乗。T_r が段階内の収束時間（stage_log.py の g と同じ問いの、
  方策の動きの側の証拠。T_r が数百なら段階数比例、C が大きく T_r が決まらなければ更新数比例か拡散）
"""

import argparse
import itertools
import json
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
from model import NUM_PLANES  # noqa: E402
from rnad import PRECISIONS, collect, pack_csr, set_precision  # noqa: E402

TR_GRID = [0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]


def coverage_curve(N, R, Tr, tau):
    """被覆率 c(n)（n = 0..N）。補間 [0, R] の線形の動きを、時定数 Tr（0 = 遅れなし）と τ の1次遅れ2段に通す"""
    o = t = 0.0
    out = np.zeros(N + 1)
    for n in range(N):
        p = 1.0 if R <= 0 else min(1.0, (n + 1) / R)
        o = p if Tr <= 0 else o + (p - o) / Tr
        t += tau * (o - t)
        out[n + 1] = t
    return out


def nnls_small(X, y):
    """列が少ない非負の最小二乗（全部の部分集合を試す）→ (SSE, 係数)"""
    best = (np.inf, np.zeros(X.shape[1]))
    for k in range(1, X.shape[1] + 1):
        for S in itertools.combinations(range(X.shape[1]), k):
            coef, *_ = np.linalg.lstsq(X[:, S], y, rcond=None)
            if (coef < 0).any():
                continue
            full = np.zeros(X.shape[1])
            full[list(S)] = coef
            sse = float(((X @ full - y) ** 2).sum())
            if sse < best[0]:
                best = (sse, full)
    return best


def fit(ns, K, R, tau):
    N = int(max(ns))
    rows = []
    for Tr in TR_GRID:
        c = coverage_curve(N, R, Tr, tau)[ns]
        X = np.stack([c ** 2, 1 - np.exp(-tau * ns), ns - (1 - np.exp(-tau * ns)) / tau], 1)
        sse, coef = nnls_small(X, K)
        rows.append({"T_r": Tr, "sse": sse, "A": coef[0], "B": coef[1], "C": coef[2]})
    return sorted(rows, key=lambda r: r["sse"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="run の出力ディレクトリ（policy_<step>.bin がある）")
    ap.add_argument("--prefix", default="policy", help="書き出しの名前（policy / current / tail / ema0.001 など）")
    ap.add_argument("--start", type=int, required=True, help="段階の始まりの step（この書き出しが基準と局面の元）")
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--every", type=int, default=0, help="start からこの間隔の倍数の書き出しだけ使う（0 = 全部）")
    ap.add_argument("--dm", type=int, required=True, help="段階の長さ（補間は Δm/2。--ramp で上書き）")
    ap.add_argument("--ramp", type=int, default=-1, help="補間の長さ R（-1 = Δm/2）")
    ap.add_argument("--tau", type=float, default=0.001, help="ターゲットの平均化係数（--target-avg）")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", choices=PRECISIONS, default="fp32",
                    help="順伝播の精度（既定 fp32。隣どうしの KL は 1e-5 程度なので TF32 の丸めを入れない）")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    set_precision(args.precision)
    dev = torch.device(args.device)
    files = {}
    for p in pathlib.Path(args.dir).glob(f"{args.prefix}_*.bin"):
        m = re.fullmatch(rf"{re.escape(args.prefix)}_(\d+)\.bin", p.name)
        if m and args.start <= int(m.group(1)) <= args.end:
            s = int(m.group(1))
            if not args.every or (s - args.start) % args.every == 0:
                files[s] = p
    steps = sorted(files)
    if not steps or steps[0] != args.start:
        raise SystemExit(f"{args.prefix}_{args.start}.bin が無い（見つかった: {steps[:5]}…）")
    if len(steps) < 2:
        raise SystemExit(f"{args.start}〜{args.end} の書き出しが1つしか無い")
    head = read_weights(files[args.start]).to(dev).eval()
    planes = head.config["planes"]
    kw = {"planes": planes} if planes != NUM_PLANES else {}
    env = tsuitate_rl.VecEnv(args.games, seed=args.seed, auto_reset=False, **kw)
    t0 = time.time()
    b = collect(env, head, dev, compact_legal=True)
    n = len(b["action"])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(b["legal_counts"].long(), 0)])
    print(f"局面 {n}（{args.games} 局、{time.time() - t0:.0f}s）、書き出し {len(steps)} 個", flush=True)
    chunks = []
    for i in range(0, n, 2048):
        sl = slice(i, min(i + 2048, n))
        cols = b["legal_cols"][int(ptr[sl.start]):int(ptr[sl.stop])]
        idx, pmask, _ = pack_csr(cols, b["legal_counts"][sl], b["action"][sl])
        chunks.append((sl, idx.to(dev), pmask.to(dev)))

    @torch.no_grad()
    def logp(net):
        out = []
        for sl, idx, pmask in chunks:
            logits = net(b["obs"][sl].to(dev).float())[0].gather(1, idx).masked_fill(~pmask, float("-inf"))
            out.append(logits.log_softmax(-1))
        return out

    def kl(la, lb):
        tot = 0.0
        for (sl, idx, pmask), a, c in zip(chunks, la, lb):
            p = a.exp()
            tot += float(torch.where(pmask, p * (a - c), torch.zeros_like(a)).sum())
        return tot / n

    lp0 = logp(head)
    prev = lp0
    res = []
    for s in steps[1:]:
        lp = logp(read_weights(files[s]).to(dev).eval())
        res.append({"step": s, "n": s - args.start, "kl_head": kl(lp0, lp), "kl_prev": kl(prev, lp)})
        prev = lp
        print(f"{s}: KL(頭‖今) {res[-1]['kl_head']:.5f}  KL(前‖今) {res[-1]['kl_prev']:.6f}", flush=True)
    ns = np.array([r["n"] for r in res], dtype=float)
    K = np.array([r["kl_head"] for r in res])
    out = {"args": vars(args), "positions": n, "profile": res}
    if len(ns) >= 4:
        last = ns >= 0.8 * ns.max()
        if last.sum() >= 2:
            sl_last = np.polyfit(ns[last], K[last], 1)[0]
            sl_avg = K[-1] / ns[-1]
            out["slope_last20_over_avg"] = float(sl_last / sl_avg)
            print(f"最後の 20% の傾き ÷ 平均の傾き = {sl_last / sl_avg:.2f}（1 ≈ 線形に伸び続ける、< 0.2 ≈ 落ち着いた）")
    if not args.no_fit and len(ns) >= 4:
        R = args.dm // 2 if args.ramp < 0 else args.ramp
        rows = fit(ns.astype(int), K, R, args.tau)
        out["fit"] = rows
        for r in rows[:4]:
            print(f"T_r={r['T_r']:5d} SSE {r['sse']:.3e}  固定点の動き A={r['A']:.4f}  速い雑音 B={r['B']:.4f}  "
                  f"拡散 C={r['C']:.2e}/更新")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
