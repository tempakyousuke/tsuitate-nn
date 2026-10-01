"""計算精度（rnad.set_precision）ごとの速さと数値のずれを、学習中の checkpoint で測る。

    .venv/bin/python rnad/bench_precision.py --ckpt out_rnad/rnad_gpu_b256/checkpoint.pt

精度ごとに:
- 速さ: 自己対局（collect）と更新（update）の秒数（ウォームアップ1回の後、--reps 回の平均）
- 自己無撞着: 更新ログの π_a/μ_a のずれ（行動時と学習時で同じネットなので本来 1）
- fp32 との一致: fp32 で集めた同じバッチに対する ロジット（合法手）・log π・価値の差と、
  1回更新した後のパラメータ変化量の fp32 とのコサイン類似度
"""

import argparse
import copy
import json
import time

import torch

import tsuitate_rl
from model import PolicyValueNet
from rnad import (RNaDConfig, RNaDLearner, collect, forward, legal_log_policy,
                  set_precision)
from train_rnad import pick_device


def make_learner(ck, device):
    c = ck["config"]
    net = PolicyValueNet(c["channels"], c["blocks"], c["planes"])
    cfg = RNaDConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
    learner = RNaDLearner(net, cfg, device)
    learner.load_state_dict(copy.deepcopy(ck))
    return learner


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def outputs(net, batch, device, n):
    obs = batch["obs"][:n].to(device, torch.float32)
    legal = batch["legal"][:n].to(device)
    logits, v = forward(net, obs)
    return logits, legal_log_policy(logits, legal), v, legal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--modes", default="fp32,default,tf32,bf16")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--compare-n", type=int, default=2048, help="出力の一致を見る標本数")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location=device)
    print(f"device {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} "
          f"torch {torch.__version__} step {ck['step']}", flush=True)

    # 一致を見る共通のバッチ（fp32 で集める）と、fp32 での基準
    set_precision("fp32")
    torch.manual_seed(0)
    env = tsuitate_rl.VecEnv(args.batch, seed=0, auto_reset=False)
    base = make_learner(ck, device)
    fixed = collect(env, base.net, device)
    env.pop_finished()
    ref_logits, ref_logp, ref_v, ref_legal = outputs(base.net, fixed, device, args.compare_n)
    p0 = torch.nn.utils.parameters_to_vector(base.net.parameters()).detach().clone()
    base.update(fixed)
    ref_delta = torch.nn.utils.parameters_to_vector(base.net.parameters()).detach() - p0
    del base

    results = []
    for mode in args.modes.split(","):
        set_precision(mode)
        rec = {"mode": mode}
        # fp32 との一致
        learner = make_learner(ck, device)
        logits, logp, v, legal = outputs(learner.net, fixed, device, args.compare_n)
        rec["logit_max_abs"] = float((logits - ref_logits)[legal].abs().max())
        rec["logp_mean_abs"] = float((logp - ref_logp)[legal].abs().mean())
        rec["logp_max_abs"] = float((logp - ref_logp)[legal].abs().max())
        rec["v_max_abs"] = float((v - ref_v).abs().max())
        p0 = torch.nn.utils.parameters_to_vector(learner.net.parameters()).detach().clone()
        learner.update(fixed)
        delta = torch.nn.utils.parameters_to_vector(learner.net.parameters()).detach() - p0
        rec["delta_cos"] = float(torch.nn.functional.cosine_similarity(delta, ref_delta, dim=0))
        rec["delta_rel_err"] = float((delta - ref_delta).norm() / ref_delta.norm())
        # 速さと自己無撞着（この精度で集めて、この精度で更新する = 学習と同じ）
        learner = make_learner(ck, device)
        torch.manual_seed(1)
        env = tsuitate_rl.VecEnv(args.batch, seed=1, auto_reset=False)
        cs, us, dev_mean, dev_max = [], [], [], []
        for r in range(args.reps + 1):
            sync(device)
            t0 = time.time()
            batch = collect(env, learner.net, device)
            sync(device)
            t1 = time.time()
            env.pop_finished()
            info = learner.update(batch)
            sync(device)
            t2 = time.time()
            if r == 0:
                continue  # ウォームアップ
            cs.append(t1 - t0)
            us.append(t2 - t1)
            dev_mean.append(info["ratio_dev_mean"])
            dev_max.append(info["ratio_dev_max"])
        rec.update(collect_sec=sum(cs) / len(cs), update_sec=sum(us) / len(us),
                   ratio_dev_mean=sum(dev_mean) / len(dev_mean), ratio_dev_max=max(dev_max),
                   samples=info["samples"])
        results.append(rec)
        print(json.dumps(rec), flush=True)

    print()
    print(f"{'mode':8} {'collect':>8} {'update':>8} {'計':>7} | {'|π/μ-1|平均':>12} {'最大':>8} | "
          f"{'Δlogit最大':>10} {'Δlogπ平均':>10} {'Δv最大':>8} | {'更新cos':>8} {'相対誤差':>8}")
    for r in results:
        print(f"{r['mode']:8} {r['collect_sec']:8.2f} {r['update_sec']:8.2f} "
              f"{r['collect_sec'] + r['update_sec']:7.2f} | {r['ratio_dev_mean']:12.2e} "
              f"{r['ratio_dev_max']:8.2e} | {r['logit_max_abs']:10.2e} {r['logp_mean_abs']:10.2e} "
              f"{r['v_max_abs']:8.2e} | {r['delta_cos']:8.5f} {r['delta_rel_err']:8.4f}")


if __name__ == "__main__":
    main()
