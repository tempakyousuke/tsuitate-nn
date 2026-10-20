"""1更新の壁時計を、**数式を変えない高速化**ごとに測る（tsuitate-bot の docs/rl-learning-speed.md）。

    # 本線の checkpoint で（GPU のゲーム用 PC）
    .venv/bin/python rnad/bench_update.py --ckpt out_rnad/selfplay_v2planes/checkpoint.pt --batch 768
    # 凍結版の重みから（学習状態が無いので正則化方策・ターゲットは同じネットで始める）
    .venv/bin/python rnad/bench_update.py --weights ../tsuitate-bot/models/rl_deduce_v4.bin --batch 64

測るもの（ウォームアップ1回の後、--reps 回の平均）:
- `collect`: 自己対局1回（batch 局）
- `update`: 従来の更新 / `packed`: 合法手に詰めた更新（`RNaDLearner(packed=True)`）。同じ batch・同じ学習状態から
  1回ずつ回し、更新量（パラメータの変化）の一致（コサイン・相対誤差）も出す
- `sequential` / `pipelined`: 「集める → 更新」を --loop 回まわした1更新あたりの壁時計。pipelined は
  `PipelinedCollector`（1更新遅れの方策で集めて更新と重ねる）。更新は packed で回す
"""

import argparse
import copy
import json
import time

import torch

import tsuitate_rl
from export_weights import read as read_weights
from model import NUM_PLANES, PolicyValueNet
from rnad import PipelinedCollector, RNaDConfig, RNaDLearner, collect
from train_rnad import pick_device


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_learner(args, device, packed):
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device)
        c = ck["config"]
        net = PolicyValueNet(c["channels"], c["blocks"], c["planes"])
        cfg = RNaDConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
        learner = RNaDLearner(net, cfg, device, packed=packed)
        learner.load_state_dict(copy.deepcopy(ck))
        return learner
    net = read_weights(args.weights) if args.weights else PolicyValueNet(args.channels, args.blocks, args.planes)
    cfg = RNaDConfig(beta=16.0, lr=1e-4, schedule_sizes=(2000,), schedule_repeats=(1,))
    learner = RNaDLearner(net, cfg, device, packed=packed)
    learner.step_count = 1000  # α = 1（段階の途中）。正則化方策は同じネット
    return learner


def env_kw(learner):
    planes = learner.net.config["planes"]
    return {"planes": planes} if planes != NUM_PLANES else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="train_rnad.py の checkpoint.pt（学習状態一式）")
    ap.add_argument("--weights", default="", help="TSRLPV01 の重み（--ckpt が無いとき）")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--planes", type=int, default=91)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--loop", type=int, default=6, help="sequential / pipelined の反復回数（最初の1回は数えない）")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    print(f"device {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} "
          f"torch {torch.__version__} threads {torch.get_num_threads()}", flush=True)

    base = make_learner(args, device, packed=False)
    kw = env_kw(base)
    env = tsuitate_rl.VecEnv(args.batch, seed=1, auto_reset=False, **kw)
    rec = {"batch": args.batch, "net": base.net.config}
    cs, us, ps, coss, errs = [], [], [], [], []
    for r in range(args.reps + 1):
        torch.manual_seed(r)
        sync(device)
        t0 = time.time()
        batch = collect(env, base.net, device)
        sync(device)
        t1 = time.time()
        env.pop_finished()
        a = copy.deepcopy(base)
        b = copy.deepcopy(base)
        b.packed = True
        p0 = torch.nn.utils.parameters_to_vector(a.net.parameters()).detach().clone()
        sync(device)
        t2 = time.time()
        a.update(batch)
        sync(device)
        t3 = time.time()
        b.update(batch)
        sync(device)
        t4 = time.time()
        da = torch.nn.utils.parameters_to_vector(a.net.parameters()).detach() - p0
        db = torch.nn.utils.parameters_to_vector(b.net.parameters()).detach() - p0
        if r == 0:
            continue  # ウォームアップ
        cs.append(t1 - t0)
        us.append(t3 - t2)
        ps.append(t4 - t3)
        coss.append(float(torch.nn.functional.cosine_similarity(da, db, dim=0)))
        errs.append(float((da - db).norm() / da.norm()))
        print(f"rep {r}: collect {t1 - t0:.2f}s update {t3 - t2:.2f}s packed {t4 - t3:.2f}s "
              f"更新量のコサイン {coss[-1]:.6f} 相対誤差 {errs[-1]:.2e} 標本 {len(batch['action'])}", flush=True)
    rec.update(collect_sec=sum(cs) / len(cs), update_sec=sum(us) / len(us), packed_sec=sum(ps) / len(ps),
               delta_cos_min=min(coss), delta_rel_err_max=max(errs))

    # 集める → 更新 を回したときの1更新の壁時計（更新は packed）
    for mode in ("sequential", "pipelined"):
        learner = make_learner(args, device, packed=True)
        times = []
        if mode == "pipelined":
            pipe = PipelinedCollector(lambda: tsuitate_rl.VecEnv(args.batch, seed=2, auto_reset=False, **kw), device)
            pipe.submit(learner.net)
        else:
            env2 = tsuitate_rl.VecEnv(args.batch, seed=2, auto_reset=False, **kw)
        for i in range(args.loop):
            sync(device)
            t0 = time.time()
            if mode == "pipelined":
                batch, _, _ = pipe.result()
                pipe.submit(learner.net)
            else:
                batch = collect(env2, learner.net, device)
                env2.pop_finished()
            learner.update(batch)
            sync(device)
            if i:
                times.append(time.time() - t0)
        if mode == "pipelined":
            pipe.result()
            pipe.close()
        rec[f"{mode}_sec"] = sum(times) / len(times)
        print(f"{mode}: 1更新 {rec[f'{mode}_sec']:.2f}s", flush=True)
    print(json.dumps(rec))
    print(f"\n従来: collect {rec['collect_sec']:.2f} + update {rec['update_sec']:.2f} = "
          f"{rec['collect_sec'] + rec['update_sec']:.2f}s / packed の更新 {rec['packed_sec']:.2f}s "
          f"（{rec['update_sec'] / rec['packed_sec']:.2f}倍）/ 連続 {rec['sequential_sec']:.2f}s / "
          f"重ねる {rec['pipelined_sec']:.2f}s（従来比 {(rec['collect_sec'] + rec['update_sec']) / rec['pipelined_sec']:.2f}倍）")


if __name__ == "__main__":
    main()
