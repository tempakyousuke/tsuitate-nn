"""F2: rnad.collect（元のまま）と fast_collect / prof_collect の batch がビット単位で同じかを確かめ、壁時計と部品の時間を
並べる（docs/rl-learning-speed.md「さらに速くする」の S1）。

    # CPU（どこでも）
    python rnad/proto/speed2/verify_collect.py --batch 16 --threads 4
    # ゲーム用 PC（CUDA での一致と壁時計。同じ env・同じ seed なので完全なペア）
    python rnad/proto/speed2/verify_collect.py --device cuda --batch 768 --modes orig,fast_np,fast_dev,orig --out verify768.json
    # observe_alive.diff を当てて tsuitate_rl を作り直した後（S1(b)）
    python rnad/proto/speed2/verify_collect.py --device cuda --batch 768 --modes orig,fast_rs,fast_rs_np

腕（`--modes`。最初の腕が基準）:
- `orig`     … rnad.collect（compact_legal=True）
- `prof`     … prof_collect.collect_prof（部品ごとに計時する rnad.collect の写し）
- `fast_np`  … fast_collect（密なマスクを numpy から送る）。docs の推奨の既定
- `fast_dev` … fast_collect（CSR を送ってデバイスで密なマスクを作る）
- `fast_rs` / `fast_rs_np` … fast_collect で Rust の observe_alive_f16_csr を使う（無ければ飛ばす）
どの腕も同じ env を reset して、同じ seed の乱数列で指す（局は手の列だけで決まる）。全部の腕が基準とビット単位で
同じなら終了コード 0、違えば 1。CUDA でも各腕の順伝播は同じ形の入力に同じカーネルが選ばれるので一致する見込み
（推測。一致しなければ、その腕を本線へ入れる前に原因を調べる）
"""

import argparse
import importlib
import json
import sys
import time

import numpy as np
import torch

from proto_common import DEFAULT_WEIGHTS, env_kwargs
import rnad  # noqa: E402
from export_weights import read  # noqa: E402
from fast_collect import collect_fast, same  # noqa: E402
from prof_collect import collect_prof  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7, help="手のサンプリングの乱数")
    ap.add_argument("--env-seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--modes", default="orig,fast_dev,fast_np")
    ap.add_argument("--out", default="", help="結果の JSON（省略可）")
    ap.add_argument("--save-actions", default="", help="基準の腕の行動列を npz に（Rust だけを再生して計時するとき）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default="tf32")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--env-module", default="tsuitate_rl",
                    help="VecEnv を読むモジュール（別名でビルドした試作の env を比べるとき）")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    rnad.set_precision(args.precision)
    envmod = importlib.import_module(args.env_module)
    net = read(args.weights).to(dev).eval()
    env = envmod.VecEnv(args.batch, seed=args.env_seed, auto_reset=False, **env_kwargs(net))
    has_rs = hasattr(env, "observe_alive_f16_csr")
    out = {"batch": args.batch, "seed": args.seed, "device": args.device, "runs": {}}
    ref = None
    ok = True
    for k, mode in enumerate(args.modes.split(",")):
        if mode.startswith("fast_rs") and not has_rs:
            print(f"{mode}: {args.env_module}.VecEnv に observe_alive_f16_csr が無いので飛ばす（observe_alive.diff）")
            continue
        g = torch.Generator(device=dev)
        g.manual_seed(args.seed)
        prof = {}
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if mode == "orig":
            b = rnad.collect(env, net, dev, generator=g, compact_legal=True)
        elif mode == "prof":
            b = collect_prof(env, net, dev, generator=g, compact_legal=True, prof=prof)
        elif mode in ("fast_dev", "fast_np", "fast_rs", "fast_rs_np"):
            b = collect_fast(env, net, dev, generator=g, mask_on_device=mode in ("fast_dev", "fast_rs"), prof=prof,
                             rust_alive=mode.startswith("fast_rs"))
        else:
            raise SystemExit(f"腕 {mode} は orig / prof / fast_np / fast_dev / fast_rs / fast_rs_np のどれか")
        if dev.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        # 全局が終わった状態での observe（終局済みの局だけの観測の費用）
        tt = []
        for _ in range(3):
            s = time.perf_counter()
            env.observe()
            tt.append(time.perf_counter() - s)
        rec = {"wall": wall, "parts": prof, "samples": int(b["action"].numel()), "T": int(b["valid"].shape[0]),
               "observe_all_dead": tt}
        if ref is None:
            ref = b
            if args.save_actions:
                np.savez(args.save_actions, t=b["t"].numpy(), b=b["b"].numpy(), action=b["action"].numpy(),
                         B=args.batch, T=int(b["valid"].shape[0]))
        else:
            rec["bitwise_vs_first"] = same(ref, b)
            rec["all_equal"] = all(rec["bitwise_vs_first"].values())
            ok &= rec["all_equal"]
        out["runs"][f"{k}:{mode}"] = rec
        diff = "" if rec.get("all_equal", True) else " 違うキー: " + ",".join(
            key for key, v in rec["bitwise_vs_first"].items() if not v)
        print(f"{mode:10s} wall {wall:7.2f}s  T {rec['T']}  決定点 {rec['samples']}  終局済みの observe "
              f"{np.mean(tt) * 1e3:.1f}ms  基準とビット一致 {rec.get('all_equal', '（基準）')}{diff}", flush=True)
        if prof:
            tot = sum(prof.values())
            print("    " + " ".join(f"{p} {v:.3f}s({100 * v / tot:.0f}%)" for p, v in prof.items()))
    out["all_equal"] = ok
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f)
    print("全部の腕が基準とビット単位で同じ" if ok else "基準と違う腕がある")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
