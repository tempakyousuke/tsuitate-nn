"""F2b: ゲーム用 PC（RTX 3070・WSL2）で、collect の GPU 側の部品を測る（作った環境に GPU が無いので未実行）。

    python rnad/proto/speed2/box_bench.py [--weights ../tsuitate-bot/models/rl_deduce_v7.bin] [--reps 20]

1. 1手番ぶん（全 768 局が生きている手番）の H2D: 観測 float32 / fp16、密なマスク、CSR の添字。pageable と pinned
2. update 側の `batch["obs"][sl].to(dev, torch.float32)`（fp16 の CPU テンソル → float32 の GPU）と
   `.to(dev).float()`（fp16 のまま送って GPU で float32 に）の比較（どちらで型変換されるかで送る量が倍違う）
3. 生きている局数 n ごとの「送る + 順伝播 + サンプリング + .cpu()」の時間（元の sample_actions と、fp16・CSR で送る版）
"""
import argparse
import time

import numpy as np
import torch

from proto_common import DEFAULT_WEIGHTS
import rnad  # noqa: E402
from export_weights import read  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
dev = torch.device(args.device)
sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
rnad.set_precision("tf32")
net = read(args.weights).to(dev).eval()
P, A, B = net.config["planes"], rnad.NUM_ACTIONS, 768
rng = np.random.default_rng(0)
obs = (rng.random((B, P, 9, 9)) < 0.1).astype(np.float32)
mask = np.zeros((B, A), bool)
for i in range(B):
    mask[i, rng.choice(A, 79, replace=False)] = True


def timeit(name, f, reps=args.reps, nbytes=None):
    f()
    sync()
    ts = []
    for _ in range(reps):
        sync()
        s = time.perf_counter()
        f()
        sync()
        ts.append(time.perf_counter() - s)
    m = float(np.median(ts))
    bw = f"  {nbytes / m / 1e9:6.2f} GB/s" if nbytes else ""
    print(f"{name:55s} {m * 1e3:8.3f} ms{bw}", flush=True)
    return m


print("== 1. H2D（768 局ぶん）")
o32 = torch.from_numpy(obs)
o16 = o32.half()
m_t = torch.from_numpy(mask)
cols = torch.from_numpy(np.nonzero(mask)[1].astype(np.int16))
cnts = torch.from_numpy(mask.sum(1).astype(np.int32))
for pinned in ((False, True) if dev.type == "cuda" else (False,)):
    tag = "pinned" if pinned else "pageable"
    a32, a16, am, ac, an = ((x.pin_memory() if pinned else x) for x in (o32, o16, m_t, cols, cnts))
    timeit(f"obs float32 {a32.numel() * 4 / 1e6:.1f}MB {tag}", lambda: a32.to(dev, non_blocking=True), nbytes=a32.numel() * 4)
    timeit(f"obs fp16 {a16.numel() * 2 / 1e6:.1f}MB {tag}", lambda: a16.to(dev, non_blocking=True), nbytes=a16.numel() * 2)
    timeit(f"mask bool {am.numel() / 1e6:.1f}MB {tag}", lambda: am.to(dev, non_blocking=True), nbytes=am.numel())
    timeit(f"CSR cols+counts {(ac.numel() * 2 + an.numel() * 4) / 1e6:.2f}MB {tag}",
           lambda: (ac.to(dev, non_blocking=True), an.to(dev, non_blocking=True)))

print("== 2. update 側の観測の送り方（1024 行の fp16 チャンク）")
chunk = o16.repeat(2, 1, 1, 1)[:1024].contiguous()
timeit("chunk.to(dev, torch.float32)  pageable", lambda: chunk.to(dev, torch.float32))
timeit("chunk.to(dev).float()          pageable", lambda: chunk.to(dev).float())
if dev.type == "cuda":
    chunk_p = chunk.pin_memory()
    timeit("chunk.to(dev, torch.float32)  pinned", lambda: chunk_p.to(dev, torch.float32))
    timeit("chunk.to(dev).float()          pinned", lambda: chunk_p.to(dev).float())

print("== 3. 1手番の GPU 側（送る + 順伝播 + サンプリング + .cpu()）を生きている局数 n ごとに")
g = torch.Generator(device=dev)
for n in (768, 512, 384, 256, 128, 64, 16, 4):
    o_n, m_n = obs[:n], mask[:n]
    timeit(f"n={n:4d} 元の sample_actions（float32 + 密なマスクを送る）",
           lambda: rnad.sample_actions(net, o_n, m_n, dev, g))
    oh = torch.from_numpy(o_n).half()
    c_n = torch.from_numpy(np.nonzero(m_n)[1].astype(np.int16))
    k_n = torch.from_numpy(m_n.sum(1).astype(np.int32))

    def fast():
        x = oh.to(dev, non_blocking=True).float()
        logits, _ = rnad.forward(net, x)
        c = c_n.to(dev, non_blocking=True).long()
        k = k_n.to(dev, non_blocking=True).long()
        rows = torch.repeat_interleave(torch.arange(n, device=dev), k, output_size=len(c_n))  # 渡さないと同期が入る
        legal = torch.zeros((n, A), dtype=torch.bool, device=dev)
        legal[rows, c] = True
        masked = logits.float().masked_fill(~legal, float("-inf"))
        u = torch.rand(masked.shape, device=dev, generator=g).clamp_(1e-20, 1.0)
        action = (masked - torch.log(-torch.log(u))).argmax(-1)
        prob = masked.softmax(-1).gather(1, action[:, None]).squeeze(1)
        return action.cpu(), prob.cpu()
    timeit(f"n={n:4d} fp16 + CSR で送ってデバイスでマスクを作る", fast)
    x_dev = torch.from_numpy(o_n).to(dev).half().float()
    timeit(f"n={n:4d} 順伝播だけ", lambda: rnad.forward(net, x_dev))
