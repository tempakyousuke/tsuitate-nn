"""batch をデバイスに置いた更新（`RNaDLearner(device_batch=True)`）と、1回目の現在のネットの順伝播を2回目へまとめた
更新（`fused=True`）が、従来の更新と**ビット単位で同じ**結果になることの検査（CPU）。tsuitate_rl が要る。

    .venv/bin/python rnad/tests/test_fast_update.py

- 実際の自己対局の batch（チャンクの境目が時刻の途中に来る小さいチャンク）を2回続けて更新し、4つのネット（現在・
  ターゲット・正則化2つ）の全パラメータ、勾配、Adam の状態、ログの値がビット単位で一致する
- α = 0 / 補間中 / 1 と正則化方策の差し替えの更新、補助ヘッド、終盤だけの λ（c_ply）、1/μ の上限、報酬の縮め、
  密なマスクの batch（fused だけ。device_batch は添字の batch に限る）
- 評価モードの局（相手は Rust の戦略。学習側が後手の局を含む）を混ぜた batch でも同じ
- fused が使う v-trace のチャンク版（`_v_trace_chunk`）を、合成した batch（途中の報酬・反則で同じ側が続く手番・
  長さ1の局・片方の手番が無い局・いろいろなチャンクの大きさと設定）で `v_trace_scalars` と突き合わせる
  （`v_trace_scalars` だけを直して fused が黙って違う目標で学ぶのを止める）
- 相手のネットに指させた局（手番に隙間がある）を fused は止める。device_batch は packed なしを止める
"""

import copy
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import MultiEnv, RNaDConfig, RNaDLearner, collect  # noqa: E402

DEV = torch.device("cpu")


def bits(t):
    return t.detach().contiguous().view(-1).view(torch.int32) if t.dtype == torch.float32 else t.detach().view(-1)


def same_tensor(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))


def diff_learners(a, b):
    """違うところの一覧（空ならビット単位で同じ）"""
    bad = []
    for name in ("net", "target", "prev", "prev_"):
        for (k, x), (_, y) in zip(getattr(a, name).state_dict().items(), getattr(b, name).state_dict().items()):
            if not same_tensor(x, y):
                bad.append(f"{name}.{k}")
    for i, (p, q) in enumerate(zip(a.net.parameters(), b.net.parameters())):
        if not same_tensor(p.grad, q.grad):
            bad.append(f"grad[{i}]")
    if a.aux is not None:
        for i, (p, q) in enumerate(zip(a.aux.parameters(), b.aux.parameters())):
            if not same_tensor(p, q) or not same_tensor(p.grad, q.grad):
                bad.append(f"aux[{i}]")
    for opt in ("opt", "aux_opt"):
        oa, ob = getattr(a, opt), getattr(b, opt)
        if oa is None:
            continue
        for sa, sb in zip(oa.state.values(), ob.state.values()):
            for k in sa:
                if torch.is_tensor(sa[k]) and not same_tensor(sa[k], sb[k]):
                    bad.append(f"{opt}.{k}")
    if a.step_count != b.step_count:
        bad.append("step_count")
    return bad


def make_base(step, cfg, packed):
    torch.manual_seed(0)
    net = PolicyValueNet(16, 2, 91).eval()
    base = RNaDLearner(net, cfg, DEV, packed=packed)
    # 正則化方策・ターゲットを現在のネットと違う値にする（同じだと lpr が 0 になり η の項を検査できない）
    with torch.no_grad():
        for m, scale in ((base.prev, 0.3), (base.prev_, -0.2), (base.target, 0.1)):
            for p in m.parameters():
                p.add_(scale * torch.randn_like(p) * p.abs().mean())
    base.step_count = step
    if cfg.aux_truth > 0:
        base._ensure_aux()
    return base


def batches(net, compact, truth, n=2, league=False):
    out = []
    for i in range(n):
        torch.manual_seed(10 + i)
        if league:
            # 評価モードの局（相手は Rust の戦略で、相手の手は step の中で進む）を自己対局の局と混ぜる
            env = MultiEnv([tsuitate_rl.VecEnv(4, seed=20 + i, auto_reset=False, planes=91),
                            tsuitate_rl.VecEnv(8, opponents=["heuristic", "rl_v15"], seed=30 + i,
                                               auto_reset=False, planes=91)])
        else:
            env = tsuitate_rl.VecEnv(10, seed=20 + i, auto_reset=False, planes=91)
        out.append(collect(env, net, DEV, truth=truth, compact_legal=compact))
    return out


def check(name, step, cfg, compact=True, packed=True, reward_scale=1.0, league=False):
    base = make_base(step, cfg, packed)
    bs = batches(base.net, compact, cfg.aux_truth > 0, league=league)
    arms = {"fused": dict(fused=True)}
    if compact and packed:
        arms["device_batch"] = dict(device_batch=True)
        arms["fused+device_batch"] = dict(fused=True, device_batch=True)
    ref = copy.deepcopy(base)
    logs_ref = [ref.update(b, reward_scale=reward_scale) for b in bs]
    for arm, kw in arms.items():
        x = copy.deepcopy(base)
        for k, v in kw.items():
            setattr(x, k, v)
        logs = [x.update(b, reward_scale=reward_scale) for b in bs]
        bad = diff_learners(ref, x)
        for la, lb in zip(logs_ref, logs):
            if la != lb:
                bad.append(f"log {[k for k in la if la[k] != lb.get(k)]}")
        assert not bad, f"{name} {arm}: 違うところ {bad[:8]}"
    n = sum(len(b["action"]) for b in bs)
    print(f"{name}: {'/'.join(arms)} が従来とビット単位で一致（2更新、決定点 {n}、α {logs_ref[0]['alpha']:.2f}"
          f"{'、差し替えあり' if any(lg['reg_update'] for lg in logs_ref) else ''}）")


def check_vtrace_chunk_synthetic():
    g = torch.Generator().manual_seed(7)
    cfgs = [RNaDConfig(), RNaDConfig(c_ply=(4, 0.8), inv_mu_cap=10.0), RNaDConfig(eta=0.5, c_vtrace=0.9),
            RNaDConfig(c_ply=(0, 0.7))]
    cases = 0
    for trial in range(30):
        B = int(torch.randint(1, 12, (1,), generator=g))
        lengths = torch.randint(1, 15, (B,), generator=g)
        T = int(lengths.max())
        valid = torch.arange(T)[:, None] < lengths[None, :]
        # 手番の色は乱数（反則で同じ側が続く・片方の手番が無い局も出る）。報酬は途中の手番にも置く
        player = torch.randint(0, 2, (T, B), generator=g) * valid
        reward = (torch.randn(T, B, 2, generator=g) * (torch.rand(T, B, 1, generator=g) < 0.3)) * valid[..., None]
        t_idx, b_idx = valid.nonzero(as_tuple=True)  # 時刻の昇順（collect と同じ並び）
        n = len(t_idx)
        batch = {"t": t_idx, "b": b_idx, "valid": valid, "player": player, "reward": reward,
                 "mu_a": 10.0 ** (-6 * torch.rand(n, generator=g)), "action": torch.zeros(n, dtype=torch.long)}
        pi_a = 10.0 ** (-6 * torch.rand(n, generator=g))
        ent = torch.randn(n, generator=g) * 0.1
        v = torch.rand(n, generator=g) * 2 - 1
        for cfg in cfgs:
            vt_ref, s_ref = RNaDLearner._v_trace(None, batch, cfg, pi_a, ent, v)
            c_ply = tuple(cfg.c_ply) != (0, 1.0)
            for chunk in sorted({1, 2, 3, 7, B, n}):
                carry = {k: [torch.zeros(B), torch.zeros(B), torch.zeros(B), torch.zeros(B), torch.ones(B)] for k in (0, 1)}
                vt, s = torch.zeros(n), torch.zeros(n)
                for i in reversed(range(0, n, chunk)):
                    sl = slice(i, min(i + chunk, n))
                    vt[sl], s[sl] = RNaDLearner._v_trace_chunk(cfg, carry, t_idx[sl], b_idx[sl], player, reward,
                                                               pi_a[sl], batch["mu_a"][sl], ent[sl], v[sl], c_ply)
                assert same_tensor(vt, vt_ref) and same_tensor(s, s_ref), f"v-trace のチャンク版が違う（試行 {trial}、チャンク {chunk}、{cfg}）"
                cases += 1
    print(f"_v_trace_chunk: 合成した batch の {cases} 通り（設定 × チャンクの大きさ）で v_trace_scalars とビット単位で一致")


def check_rejects():
    torch.manual_seed(0)
    net = PolicyValueNet(8, 1, 91).eval()
    try:
        RNaDLearner(net, RNaDConfig(), DEV, packed=False, device_batch=True)
    except ValueError:
        pass
    else:
        raise AssertionError("packed なしの device_batch を通した")
    # 相手のネットに指させた局（学習側の手番の間に相手の手番の隙間がある）
    env = tsuitate_rl.VecEnv(4, seed=1, auto_reset=False, planes=91)
    batch = collect(env, net, DEV, opponents=[None, None, ("opp", 0), ("opp", 1)], opp_nets={"opp": net},
                    compact_legal=True)
    learner = RNaDLearner(net, RNaDConfig(beta=16.0, chunk=64), DEV, packed=True, fused=True)
    try:
        learner.update(batch)
    except ValueError:
        pass
    else:
        raise AssertionError("手番に隙間のある batch を fused で通した")
    print("device_batch は packed なしを、fused は手番に隙間のある batch を止める")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_vtrace_chunk_synthetic()
    check_rejects()
    common = dict(beta=16.0, lr=1e-3, schedule_sizes=(10,), schedule_repeats=(1,), chunk=97)
    check("α=0（段階の頭）", 10, RNaDConfig(**common))
    check("補間中", 13, RNaDConfig(**common))
    check("α=1・差し替え", 18, RNaDConfig(**common))
    check("補助ヘッド", 13, RNaDConfig(**common, aux_truth=1.0))
    check("c_ply・1/μ の上限・報酬の縮め", 13, RNaDConfig(**common, c_ply=(30, 0.8), inv_mu_cap=50.0),
          reward_scale=2.0)
    check("Adam の β1・c_vtrace", 16, RNaDConfig(**common, adam_b1=0.9, c_vtrace=0.9))
    check("密なマスク・packed", 13, RNaDConfig(**common), compact=False)
    check("密なマスク・packed なし", 13, RNaDConfig(**common), compact=False, packed=False)
    check("添字・packed なし", 13, RNaDConfig(**common), packed=False)
    check("チャンク1つ", 13, RNaDConfig(**{**common, "chunk": 100_000}))
    check("PG 型", 13, RNaDConfig(**common, policy_loss="pg"))
    check("評価モードの局を混ぜた batch", 13, RNaDConfig(**common), league=True)
