"""1更新の壁時計を、**数式を変えない高速化**ごとに測る（tsuitate-bot の docs/rl-learning-speed.md）。

    # 本線の checkpoint で（GPU のゲーム用 PC）
    .venv/bin/python rnad/bench_update.py --ckpt out_rnad/selfplay_v2planes/checkpoint.pt --batch 768
    # 凍結版の重みから（学習状態が無いので正則化方策・ターゲットは同じネットで始める）
    .venv/bin/python rnad/bench_update.py --weights ../tsuitate-bot/models/rl_deduce_v4.bin --batch 64
    # 「さらに速くする」の S1〜S3 の腕（収集・更新とも同じ batch・同じ乱数で並べるので完全なペア）
    .venv/bin/python rnad/bench_update.py --ckpt out_rnad/deduce_s10000/checkpoint.pt --batch 768 --compact-legal \\
        --collect-arms orig,py,rust --update-arms packed,device,fused,fused+device \\
        --fast-collect rust --device-batch --fused

測るもの（ウォームアップ1回の後、--reps 回の平均）:
- `collect`: 自己対局1回（batch 局）
- `update`: 従来の更新 / `packed`: 合法手に詰めた更新（`RNaDLearner(packed=True)`）。同じ batch・同じ学習状態から
  1回ずつ回し、**勾配**の一致（コサイン・相対誤差。数式が同じなら 1e-5 程度）と更新量の一致も出す
  （`--weights` は Adam の状態が新しく、1歩目の更新量は勾配の符号に近いので、0 付近の成分の符号の揺れで
  更新量のコサインは 0.9994 程度まで下がる。`--ckpt` の本物の状態なら 1 に近い）
- `--collect-arms`: 収集の腕（orig = 従来の `collect(compact_legal=True)`、py / rust = `collect(fast=...)`）。同じ env の
  seed・同じ乱数列で集めるので、腕どうしの batch は**ビット単位で同じ**はず（違えば `collect_mismatch` に出る）
- `--update-arms`: 更新の腕（packed = 基準、device = `device_batch`、fused、fused+device）。同じ batch・同じ学習状態から
  1回ずつ回し、更新後の全パラメータが基準とビット単位で同じか（CPU では同じ。CUDA は畳み込みの逆伝播のカーネルが
  非決定的なら基準を2回回しても揺れるので、`packed` を2回並べると揺れの床が見える）と、更新量の相対誤差を出す
- `sequential` / `pipelined`: 「集める → 更新」を --loop 回まわした1更新あたりの壁時計。pipelined は
  `PipelinedCollector`（1更新遅れの方策で集めて更新と重ねる）。更新は packed で回し、`--fast-collect` /
  `--device-batch` / `--fused` を付けるとこのループにも使う
"""

import argparse
import copy
import json
import time

import torch

import tsuitate_rl
from export_weights import read as read_weights
from model import NUM_PLANES, PolicyValueNet
from rnad import PRECISIONS, PipelinedCollector, RNaDConfig, RNaDLearner, collect, set_precision
from train_rnad import pick_device

UPDATE_ARMS = {"packed": {}, "device": {"device_batch": True}, "fused": {"fused": True},
               "fused+device": {"fused": True, "device_batch": True}}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_learner(args, device, packed, **flags):
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device)
        c = ck["config"]
        net = PolicyValueNet(c["channels"], c["blocks"], c["planes"])
        cfg = RNaDConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
        learner = RNaDLearner(net, cfg, device, packed=packed, **flags)
        learner.load_state_dict(copy.deepcopy(ck))
        return learner
    net = read_weights(args.weights) if args.weights else PolicyValueNet(args.channels, args.blocks, args.planes)
    cfg = RNaDConfig(beta=16.0, lr=1e-4, schedule_sizes=(2000,), schedule_repeats=(1,))
    learner = RNaDLearner(net, cfg, device, packed=packed, **flags)
    learner.step_count = 1000  # α = 1（段階の途中）。正則化方策は同じネット
    return learner


def env_kw(learner):
    planes = learner.net.config["planes"]
    return {"planes": planes} if planes != NUM_PLANES else {}


def batch_diff(a, b):
    """batch どうしで違うキー（ビット単位）"""
    bad = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if torch.is_tensor(x) and torch.is_tensor(y):
            if not (x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)):
                bad.append(k)
        elif hasattr(x, "shape") and hasattr(y, "shape"):
            if not (x.shape == y.shape and (x == y).all()):
                bad.append(k)
        elif x is None or y is None:
            bad.append(k)
    return bad


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
    ap.add_argument("--compact-legal", action="store_true", help="batch の合法手を添字で持つ（本線と同じ形）")
    ap.add_argument("--collect-arms", default="", help="収集の腕（カンマ区切り。orig / py / rust）")
    ap.add_argument("--update-arms", default="",
                    help="更新の腕（カンマ区切り。packed / device / fused / fused+device。同じ名前を2回並べてもよい）")
    ap.add_argument("--fast-collect", choices=["py", "rust"], default="", help="sequential / pipelined の収集に使う")
    ap.add_argument("--device-batch", action="store_true", help="sequential / pipelined の更新に使う")
    ap.add_argument("--fused", action="store_true", help="sequential / pipelined の更新に使う")
    ap.add_argument("--skip-loop", action="store_true", help="sequential / pipelined を測らない")
    ap.add_argument("--precision", choices=PRECISIONS, default="default",
                    help="ネットの計算精度（train_rnad.py と同じ。本線は tf32）")
    args = ap.parse_args()
    set_precision(args.precision)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    collect_arms = [a for a in args.collect_arms.split(",") if a]
    update_arms = [a for a in args.update_arms.split(",") if a]
    for a in collect_arms:
        if a not in ("orig", "py", "rust"):
            raise SystemExit(f"--collect-arms の {a}（orig / py / rust）")
    for a in update_arms:
        if a not in UPDATE_ARMS:
            raise SystemExit(f"--update-arms の {a}（{' / '.join(UPDATE_ARMS)}）")
    if any("device" in a for a in update_arms) and not args.compact_legal:
        raise SystemExit("device の腕は --compact-legal の batch で測る")
    if (args.fast_collect or args.device_batch) and not args.compact_legal:
        raise SystemExit("--fast-collect / --device-batch は --compact-legal と一緒に使う")
    print(f"device {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} "
          f"torch {torch.__version__} threads {torch.get_num_threads()}", flush=True)

    base = make_learner(args, device, packed=False)
    kw = env_kw(base)
    env = tsuitate_rl.VecEnv(args.batch, seed=1, auto_reset=False, **kw)
    rec = {"batch": args.batch, "net": base.net.config, "compact_legal": args.compact_legal}
    cs, us, ps, coss, errs, gcos, gerr = [], [], [], [], [], [], []
    arm_sec = {a: [] for a in collect_arms}
    mismatch = []
    upd_sec = {f"{i}:{a}": [] for i, a in enumerate(update_arms)}
    upd_err = {k: [] for k in upd_sec}
    upd_bitwise = {k: True for k in upd_sec}
    for r in range(args.reps + 1):
        torch.manual_seed(r)
        sync(device)
        t0 = time.time()
        batch = collect(env, base.net, device, compact_legal=args.compact_legal)
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
        # 勾配そのものの一致（Adam の状態が新しいと1歩目は勾配の符号に近く、0 付近の成分の符号の揺れで
        # 更新量のコサインが 0.9994 程度まで下がる。数式の一致は勾配で見る）
        ga = torch.cat([q.grad.flatten() for q in a.net.parameters()])
        gb = torch.cat([q.grad.flatten() for q in b.net.parameters()])
        gcos.append(float(torch.nn.functional.cosine_similarity(ga, gb, dim=0)))
        gerr.append(float((ga - gb).norm() / ga.norm()))

        # 収集の腕: 同じ env の seed・同じ乱数列で集め、最初の腕の batch とビット単位で比べる
        arm_t, first = {}, None
        for arm in collect_arms:
            env_c = tsuitate_rl.VecEnv(args.batch, seed=100 + r, auto_reset=False, **kw)
            gen = torch.Generator(device=device)
            gen.manual_seed(r)
            sync(device)
            s0 = time.time()
            bc = collect(env_c, base.net, device, compact_legal=True, generator=gen,
                         fast="" if arm == "orig" else arm)
            sync(device)
            arm_t[arm] = time.time() - s0
            if first is None:
                first = bc
            else:
                bad = batch_diff(first, bc)
                if bad:
                    mismatch.append({"rep": r, "arm": arm, "keys": bad})

        # 更新の腕: 同じ batch・同じ学習状態から1回ずつ。基準（最初の腕）と更新後のパラメータを比べる
        upd_t, ref_vec = {}, None
        for key in upd_sec:
            arm = key.split(":", 1)[1]
            L = copy.deepcopy(base)
            L.packed = True
            for k, v in UPDATE_ARMS[arm].items():
                setattr(L, k, v)
            q0 = torch.nn.utils.parameters_to_vector(L.net.parameters()).detach().clone()
            sync(device)
            s0 = time.time()
            L.update(batch)
            sync(device)
            upd_t[key] = time.time() - s0
            vec = torch.nn.utils.parameters_to_vector(L.net.parameters()).detach()
            if ref_vec is None:
                ref_vec, ref_d = vec, vec - q0
            elif r > 0:
                upd_bitwise[key] &= bool(torch.equal(vec, ref_vec))
                upd_err[key].append(float(((vec - q0) - ref_d).norm() / ref_d.norm()))
            del L

        if r == 0:
            gcos.clear()
            gerr.clear()
            continue  # ウォームアップ
        cs.append(t1 - t0)
        us.append(t3 - t2)
        ps.append(t4 - t3)
        coss.append(float(torch.nn.functional.cosine_similarity(da, db, dim=0)))
        errs.append(float((da - db).norm() / da.norm()))
        for arm, s in arm_t.items():
            arm_sec[arm].append(s)
        for key, s in upd_t.items():
            upd_sec[key].append(s)
        extra = "".join(f" / 収集 {arm} {s:.2f}s" for arm, s in arm_t.items())
        extra += "".join(f" / 更新 {key.split(':', 1)[1]} {s:.2f}s" for key, s in upd_t.items())
        print(f"rep {r}: collect {t1 - t0:.2f}s update {t3 - t2:.2f}s packed {t4 - t3:.2f}s "
              f"勾配のコサイン {gcos[-1]:.8f} 相対誤差 {gerr[-1]:.2e} / 更新量のコサイン {coss[-1]:.6f} "
              f"標本 {len(batch['action'])}{extra}", flush=True)
    rec.update(collect_sec=sum(cs) / len(cs), update_sec=sum(us) / len(us), packed_sec=sum(ps) / len(ps),
               grad_cos_min=min(gcos), grad_rel_err_max=max(gerr),
               delta_cos_min=min(coss), delta_rel_err_max=max(errs))
    if collect_arms:
        rec["collect_arms_sec"] = {arm: sum(v) / len(v) for arm, v in arm_sec.items()}
        rec["collect_mismatch"] = mismatch
        print("収集の腕（秒）: " + " / ".join(f"{a} {s:.2f}" for a, s in rec["collect_arms_sec"].items())
              + ("（全部ビット単位で同じ batch）" if not mismatch else f"（違う batch: {mismatch}）"), flush=True)
    if update_arms:
        rec["update_arms_sec"] = {k: sum(v) / len(v) for k, v in upd_sec.items()}
        rec["update_arms_bitwise"] = {k: v for k, v in list(upd_bitwise.items())[1:]}
        rec["update_arms_delta_rel_err_max"] = {k: max(v) for k, v in upd_err.items() if v}
        print("更新の腕（秒）: " + " / ".join(f"{k.split(':', 1)[1]} {s:.2f}" for k, s in rec["update_arms_sec"].items())
              + " / 基準とビット単位で同じ: " + ", ".join(f"{k.split(':', 1)[1]}={v}" for k, v in
                                                     rec["update_arms_bitwise"].items()), flush=True)

    # 集める → 更新 を回したときの1更新の壁時計（更新は packed）
    loop_flags = {"device_batch": args.device_batch, "fused": args.fused}
    collect_kw = {"compact_legal": args.compact_legal, "fast": args.fast_collect}
    for mode in () if args.skip_loop else ("sequential", "pipelined"):
        learner = make_learner(args, device, packed=True, **loop_flags)
        times = []
        if mode == "pipelined":
            pipe = PipelinedCollector(lambda: tsuitate_rl.VecEnv(args.batch, seed=2, auto_reset=False, **kw), device,
                                      **collect_kw)
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
                batch = collect(env2, learner.net, device, **collect_kw)
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
    rec["loop_flags"] = {**loop_flags, "fast_collect": args.fast_collect}
    print(json.dumps(rec))
    msg = (f"\n従来: collect {rec['collect_sec']:.2f} + update {rec['update_sec']:.2f} = "
           f"{rec['collect_sec'] + rec['update_sec']:.2f}s / packed の更新 {rec['packed_sec']:.2f}s "
           f"（{rec['update_sec'] / rec['packed_sec']:.2f}倍）")
    if not args.skip_loop:
        msg += (f" / 連続 {rec['sequential_sec']:.2f}s / 重ねる {rec['pipelined_sec']:.2f}s"
                f"（従来比 {(rec['collect_sec'] + rec['update_sec']) / rec['pipelined_sec']:.2f}倍）")
    print(msg)


if __name__ == "__main__":
    main()
