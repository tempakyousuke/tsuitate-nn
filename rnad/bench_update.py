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
  非決定的なら基準を2回回しても揺れるので、`packed` を2回並べると揺れの床が見える）と、更新量の相対誤差を出す。
  更新が返すログの値（損失・見張り・π/μ のずれ・v-trace の目標と s。順伝播と v-trace だけで決まる）が基準と完全に同じかも
  出す（`update_arms_info_same`。パラメータが逆伝播の揺れで違っても、ここが同じなら順伝播と v-trace は同じ）。
  `--ckpt` の区切りの checkpoint は α=0 なので、α=1 の区間は `--step` で区切り + Δm/2 以上を渡して別に測る
- `sequential` / `pipelined`: 「集める → 更新」を --loop 回まわした1更新あたりの壁時計。pipelined は
  `PipelinedCollector`（1更新遅れの方策で集めて更新と重ねる）。更新は packed で回し、`--fast-collect` /
  `--device-batch` / `--fused` を付けるとこのループにも使う。`--loop-mmd <割合>` なら更新を MMD 学習器にする
  （pipelined は decoupled の近い方策の順伝播も入る）

MMD 学習器（`--algo mmd`。tsuitate-bot の docs/rl-learning-speed.md 10.5）の A/B の前に決めること（tsuitate-bot の
docs/rl-mmd-design.md 6.0c）:

    .venv/bin/python rnad/bench_update.py --ckpt out_rnad/ab_base_108000.pt --batch 768 --compact-legal \\
        --collect-arms rust,rust+extras --mmd-arms 0.5,1 --fast-collect rust --device-batch --precision tf32 \\
        --calib net,target --loop-mmd 0.5 --mmd-stale

- `--collect-arms` に `+extras` の腕（`collect(actor_extras=True)`）: 既存のキーが extras なしとビット単位で同じか
- `--mmd-arms`: 行を残す割合ごと（1 = 絞り込みなし）の MMD の1更新の秒、クリップ前の勾配ノルムの分位（歩ごと）、
  `torch.cuda.max_memory_allocated`。R-NaD の checkpoint なら net ← 現在のネット、ターゲットと磁石 ← ターゲットで始める
  （`--init-which both` と同じ）。`--mmd-stale` で pipeline の decoupled の順伝播も入れて測る
- `--calib net,target`: checkpoint の現在のネットとターゲットの価値の較正（|v| の帯・終局までの手番数の帯・V の帯の残差）。
  `--mmd-value-warmup` を使うかを決める
"""

import argparse
import copy
import json
import time

import numpy as np
import torch

import tsuitate_rl
from export_weights import read as read_weights
from model import NUM_PLANES, PolicyValueNet
from mmd import MMDLearner, _value_monitors, player_lambda_returns
from mmd import make_config as make_mmd_config
from rnad import PRECISIONS, PipelinedCollector, RNaDConfig, RNaDLearner, collect, forward, set_precision
from train_rnad import pick_device

UPDATE_ARMS = {"packed": {}, "device": {"device_batch": True}, "fused": {"fused": True},
               "fused+device": {"fused": True, "device_batch": True}}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


_CK = {}


def load_ck(args, device):
    if args.ckpt not in _CK:
        _CK[args.ckpt] = torch.load(args.ckpt, map_location=device)
    return _CK[args.ckpt]


def make_learner(args, device, packed, **flags):
    if args.ckpt:
        ck = copy.deepcopy(load_ck(args, device))
        c = ck["config"]
        net = PolicyValueNet(c["channels"], c["blocks"], c["planes"])
        if ck.get("algo", "rnad") == "mmd":
            # MMD の checkpoint: R-NaD の腕は現在のネットから（正則化方策・ターゲットも同じネット。α = 1）
            net.load_state_dict(ck["net"])
            cfg = RNaDConfig(beta=16.0, lr=1e-4, schedule_sizes=(2000,), schedule_repeats=(1,))
            learner = RNaDLearner(net, cfg, device, packed=packed, **flags)
            learner.step_count = 1000 if args.step < 0 else args.step
            return learner
        cfg = RNaDConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
        learner = RNaDLearner(net, cfg, device, packed=packed, **flags)
        learner.load_state_dict(ck)
        if args.step >= 0:
            learner.step_count = args.step  # α を選ぶ（計時と一致の確認だけ。学習状態の中身は checkpoint のまま）
        return learner
    net = read_weights(args.weights) if args.weights else PolicyValueNet(args.channels, args.blocks, args.planes)
    cfg = RNaDConfig(beta=16.0, lr=1e-4, schedule_sizes=(2000,), schedule_repeats=(1,))
    learner = RNaDLearner(net, cfg, device, packed=packed, **flags)
    learner.step_count = 1000  # α = 1（段階の途中）。正則化方策は同じネット
    return learner


def make_mmd(args, device, rate, stale=False, device_batch=False):
    """MMD 学習器（tsuitate preset、行を残す割合 rate。1 なら絞り込みなし）。新しい Adam で始める（A/B と同じ）"""
    cfg = make_mmd_config("tsuitate", filter="none" if rate >= 1 else f"pps:{rate},floor=0.05",
                          max_grad_norm=args.mmd_max_grad_norm, chunk=args.mmd_chunk, steps=args.mmd_steps)
    if args.ckpt:
        ck = load_ck(args, device)
        c = ck["config"]
        L = MMDLearner(PolicyValueNet(c["channels"], c["blocks"], c["planes"]), cfg, device, device_batch=device_batch,
                       stale_actor=stale)
        L.start_from(ck["net"], ck["target"], int(ck.get("step", 0)))
        if ck.get("algo") == "mmd":
            L.magnet_net.load_state_dict(ck["magnet_net"])
        return L
    net = read_weights(args.weights) if args.weights else PolicyValueNet(args.channels, args.blocks, args.planes)
    return MMDLearner(net, cfg, device, device_batch=device_batch, stale_actor=stale)


@torch.no_grad()
def calib_report(args, device, batch):
    """checkpoint の現在のネット・ターゲットの価値の較正（`mmd._value_monitors` と同じ帯。λ=0.8 の TD 目標）"""
    ck = load_ck(args, device) if args.ckpt else None
    out = {}
    for which in [w for w in args.calib.split(",") if w]:
        if ck is None:
            raise SystemExit("--calib は --ckpt と一緒に使う")
        c = ck["config"]
        net = PolicyValueNet(c["channels"], c["blocks"], c["planes"]).to(device).eval()
        net.load_state_dict(ck[which])
        v = torch.cat([forward(net, batch["obs"][i:i + 2048].to(device, torch.float32))[1].cpu()
                       for i in range(0, len(batch["action"]), 2048)])
        g = player_lambda_returns(batch, v, 0.8)
        everything = torch.ones(len(v), dtype=torch.bool)
        mon = _value_monitors(batch, v, g, g - v, everything, everything)
        out[which] = {k: mon[k] for k in ("calib_band", "calib_slope_by_remain", "resid_band")}
        print(f"較正（{which}）: |v| の帯 " + " / ".join(
            f"[{b['lo']:.1f},{b['hi']:.1f}) n={b['n']} z·sign(v) {b['z_sign_v'] if b['z_sign_v'] is None else round(b['z_sign_v'], 3)}"
            f" |v| {b['abs_v'] if b['abs_v'] is None else round(b['abs_v'], 3)}" for b in mon["calib_band"])
            + " ／ 終局までの手番数の帯の傾き " + " / ".join(
            f"{b['lo']}〜{b['hi']} {b['slope'] if b['slope'] is None else round(b['slope'], 2)}"
            for b in mon["calib_slope_by_remain"]), flush=True)
    return out


def env_kw(learner):
    planes = learner.net.config["planes"]
    return {"planes": planes} if planes != NUM_PLANES else {}


_INT_VIEW = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32,
             torch.float64: torch.int64}


def same_bits(x, y):
    """テンソルどうしがビット単位で同じか（torch.equal は値で比べ、+0 と −0 を同じ・NaN を違うとみなすので、整数として見る）"""
    if x.dtype != y.dtype or x.shape != y.shape:
        return False
    x, y = x.detach().contiguous(), y.detach().contiguous()
    if x.dtype in _INT_VIEW:
        x, y = x.view(_INT_VIEW[x.dtype]), y.view(_INT_VIEW[y.dtype])
    return bool(torch.equal(x, y))


def batch_diff(a, b):
    """batch どうしで違うキー（ビット単位）"""
    bad = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            bad.append(k)
        elif torch.is_tensor(x) and torch.is_tensor(y):
            if not same_bits(x, y):
                bad.append(k)
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if not (x.dtype == y.dtype and x.shape == y.shape
                    and np.array_equal(np.ascontiguousarray(x).view(np.uint8), np.ascontiguousarray(y).view(np.uint8))):
                bad.append(k)
        elif x != y:
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
    ap.add_argument("--step", type=int, default=-1,
                    help="--ckpt の step を上書きする（段階の中の位置 = α を選ぶ。区切りの checkpoint は α=0 なので、"
                         "α=1 の区間を測るには区切り + Δm/2 以上を渡す）")
    ap.add_argument("--precision", choices=PRECISIONS, default="default",
                    help="ネットの計算精度（train_rnad.py と同じ。本線は tf32）")
    ap.add_argument("--mmd-arms", default="", help="MMD の更新の腕（行を残す割合のカンマ区切り。1 = 絞り込みなし）")
    ap.add_argument("--mmd-stale", action="store_true",
                    help="MMD の腕を pipeline と同じ持ち方（decoupled の近い方策の順伝播あり）で測る")
    ap.add_argument("--mmd-max-grad-norm", type=float, default=0.267, help="MMD の腕の勾配ノルムのクリップ")
    ap.add_argument("--mmd-chunk", type=int, default=2048, help="MMD の腕の逆伝播のチャンク")
    ap.add_argument("--mmd-steps", type=int, default=4, help="MMD の腕の1更新の歩数（tsuitate preset の既定と同じ 4）")
    ap.add_argument("--loop-mmd", type=float, default=0.0,
                    help="sequential / pipelined の更新を MMD（この割合）にする（0 = R-NaD）")
    ap.add_argument("--calib", default="", help="価値の較正を測るネット（net / target をカンマ区切り。--ckpt が要る）")
    args = ap.parse_args()
    set_precision(args.precision)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    collect_arms = [a for a in args.collect_arms.split(",") if a]
    update_arms = [a for a in args.update_arms.split(",") if a]
    for a in collect_arms:
        if a.removesuffix("+extras") not in ("orig", "py", "rust"):
            raise SystemExit(f"--collect-arms の {a}（orig / py / rust、どれも +extras を付けてよい）")
    mmd_rates = [float(x) for x in args.mmd_arms.split(",") if x]
    if (mmd_rates or args.loop_mmd or args.calib) and not args.compact_legal:
        raise SystemExit("--mmd-arms / --loop-mmd / --calib は --compact-legal の batch で測る")
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
    # 更新が返すログの値（損失・見張りの和・π/μ のずれ・v-trace の目標と s）は全部、順伝播と CPU の v-trace から
    # 決まる（逆伝播と Adam を通らない）。CUDA で逆伝播のカーネルが非決定的でパラメータが揺れても、ここが基準と
    # 完全に同じなら、順伝播と v-trace は腕どうしで同じ
    upd_info_same = {k: True for k in upd_sec}
    mmd_sec = {r: [] for r in mmd_rates}
    mmd_mem = {r: [] for r in mmd_rates}
    mmd_gn = {r: [] for r in mmd_rates}
    mmd_rows = {r: [] for r in mmd_rates}
    want_extras = bool(mmd_rates or args.calib)
    for r in range(args.reps + 1):
        torch.manual_seed(r)
        sync(device)
        t0 = time.time()
        batch = collect(env, base.net, device, compact_legal=args.compact_legal, actor_extras=want_extras)
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
            kind = arm.removesuffix("+extras")
            bc = collect(env_c, base.net, device, compact_legal=True, generator=gen,
                         fast="" if kind == "orig" else kind, actor_extras=arm.endswith("+extras"))
            sync(device)
            arm_t[arm] = time.time() - s0
            if first is None:
                first = bc
            else:
                # extras の腕どうしは extras も、extras の有無が違う腕は共通のキーだけ比べる
                keys = set(first) & set(bc)
                bad = batch_diff({k: first[k] for k in keys}, {k: bc[k] for k in keys})
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
            info = L.update(batch)
            sync(device)
            upd_t[key] = time.time() - s0
            vec = torch.nn.utils.parameters_to_vector(L.net.parameters()).detach()
            if ref_vec is None:
                ref_vec, ref_d, ref_info = vec, vec - q0, info
            elif r > 0:
                upd_bitwise[key] &= same_bits(vec, ref_vec)
                upd_info_same[key] &= info == ref_info
                upd_err[key].append(float(((vec - q0) - ref_d).norm() / ref_d.norm()))
            del L

        # MMD の腕: 同じ batch・同じ出発点から1更新ずつ
        mmd_t = {}
        for rate in mmd_rates:
            L = make_mmd(args, device, rate, stale=args.mmd_stale, device_batch=args.device_batch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            s0 = time.time()
            info = L.update(batch)
            sync(device)
            mmd_t[rate] = time.time() - s0
            if r > 0:
                mmd_sec[rate].append(mmd_t[rate])
                mmd_gn[rate] += L.last_grad_norms
                mmd_rows[rate].append(info["rows_kept"])
                if device.type == "cuda":
                    mmd_mem[rate].append(torch.cuda.max_memory_allocated(device) / 2**30)
            del L
        if r == 0 and args.calib:
            rec["calib"] = calib_report(args, device, batch)

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
        extra += "".join(f" / MMD {rate:g} {s:.2f}s" for rate, s in mmd_t.items())
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
        rec["update_arms_info_same"] = {k: v for k, v in list(upd_info_same.items())[1:]}
        rec["update_arms_delta_rel_err_max"] = {k: max(v) for k, v in upd_err.items() if v}
        print("更新の腕（秒）: " + " / ".join(f"{k.split(':', 1)[1]} {s:.2f}" for k, s in rec["update_arms_sec"].items())
              + " / 基準とビット単位で同じ: " + ", ".join(f"{k.split(':', 1)[1]}={v}" for k, v in
                                                     rec["update_arms_bitwise"].items())
              + " / ログの値が基準と同じ: " + ", ".join(f"{k.split(':', 1)[1]}={v}" for k, v in
                                                  rec["update_arms_info_same"].items()), flush=True)

    if mmd_rates:
        rec["mmd_arms"] = {}
        for rate in mmd_rates:
            gn = torch.tensor(mmd_gn[rate], dtype=torch.float64)
            q = {f"p{int(x * 100)}": float(gn.quantile(x)) for x in (0.1, 0.5, 0.9)}
            rec["mmd_arms"][f"{rate:g}"] = {
                "sec": sum(mmd_sec[rate]) / len(mmd_sec[rate]), "rows_kept": sum(mmd_rows[rate]) / len(mmd_rows[rate]),
                "grad_norm": {**q, "max": float(gn.max())},
                "clip_rate_at": {f"{c:g}": float((gn > c).double().mean()) for c in (0.267, 0.5, 1.0, 2.0)},
                **({"max_mem_gb": max(mmd_mem[rate])} if mmd_mem[rate] else {})}
            a = rec["mmd_arms"][f"{rate:g}"]
            print(f"MMD {rate:g}: 1更新 {a['sec']:.2f}s 残した行 {a['rows_kept']:.0f} "
                  f"クリップ前の勾配ノルム p10 {q['p10']:.3f} p50 {q['p50']:.3f} p90 {q['p90']:.3f} 最大 {a['grad_norm']['max']:.3f}"
                  + (f" メモリ {a['max_mem_gb']:.2f}GB" if "max_mem_gb" in a else ""), flush=True)

    # 集める → 更新 を回したときの1更新の壁時計（更新は packed、--loop-mmd なら MMD）
    loop_flags = {"device_batch": args.device_batch, "fused": args.fused}
    collect_kw = {"compact_legal": args.compact_legal, "fast": args.fast_collect}
    if args.loop_mmd:
        collect_kw["actor_extras"] = True
    for mode in () if args.skip_loop else ("sequential", "pipelined"):
        if args.loop_mmd:
            learner = make_mmd(args, device, args.loop_mmd, stale=mode == "pipelined", device_batch=args.device_batch)
        else:
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
    rec["loop_flags"] = {**loop_flags, "fast_collect": args.fast_collect, "loop_mmd": args.loop_mmd}
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
