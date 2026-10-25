"""F1: rnad.collect（自己対局だけ・λ=0・compact_legal）と**ビット単位で同じ batch** を返す、収集の CPU 経路を軽くした試作
（docs/rl-learning-speed.md「さらに速くする」の S1(a)。学習には使っていない）。一致と時間は verify_collect.py で見る。

変えたところ（どれも数式・乱数の消費順を変えない）:
  1. 生きている局の行を最初に1回だけ抜く（obs[idx] / mask[idx] を2回ずつ作らない。全局が生きている手番は抜かない）
  2. 観測は CPU で1回だけ fp16 にして、それを batch にも積み、デバイスへも fp16 のまま送る（送ってから .float()）。
     元のコードは float32 を送ってから .half().float() するので、送る量が半分になり値は同じ（fp16 への丸めは同じ RNE）
  3. 合法手の添字（CSR）を1回だけ作り、batch の legal_cols / legal_counts と、デバイス上の密なマスクの両方に使う
     （mask_on_device=True なら密なマスクを送らずデバイスで scatter して作る。repeat_interleave に output_size を
     渡さないと CUDA では同期が入る）
  4. 入玉の記帳（λ=0 では entered / advance を付けるだけ）を局ごとの Python ループからベクトル演算へ
     （学習ログの入玉率・king_advance_hist に使うので省かずに同じ値を作る）
  5. rust_alive=True なら、Rust 側で生きている局だけを fp16＋CSR で返す `VecEnv.observe_alive_f16_csr`
     （S1(b)、observe_alive.diff を当ててビルドした tsuitate_rl が要る）を使う。終局した局の観測（全体の約6割）を作らない
"""

import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
import rnad  # noqa: E402

KING_PLANE = rnad.KING_PLANE
NUM_ACTIONS = rnad.NUM_ACTIONS

FAST_PARTS = ["alive", "observe", "select", "ny_vec", "half", "csr", "send_fwd", "mask_dev", "sample", "step", "rec",
              "store", "final"]


@torch.no_grad()
def collect_fast(env, net, device, max_steps=260, generator=None, mask_on_device=True, prof=None, rust_alive=False):
    T0 = time.perf_counter
    acc = {k: 0.0 for k in FAST_PARTS}
    env.reset()
    B = env.num_envs
    obs_l, cols_l, cnt_l, act_l, mu_l, t_l, b_l = [], [], [], [], [], [], []
    valid = torch.zeros(max_steps, B, dtype=torch.bool)
    player = torch.zeros(max_steps, B, dtype=torch.long)
    reward = torch.zeros(max_steps, B, 2)
    entered = np.zeros((B, 2), dtype=bool)
    advance = np.zeros((B, 2), dtype=np.int64)
    outcome = np.zeros((B, 2), dtype=np.float32)
    for t in range(max_steps):
        s = T0()
        alive = env.alive()
        a0 = T0()
        acc["alive"] += a0 - s
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        n = len(idx)
        if rust_alive:
            o16, cols, counts, pl_alive = env.observe_alive_f16_csr()
            a1 = T0()
            acc["observe"] += a1 - a0
            o = o16.view(np.float16).reshape(n, -1, 9, 9)
            p_sel = pl_alive.astype(np.int64)
            m = None
            if not mask_on_device:
                m = np.zeros((n, NUM_ACTIONS), dtype=bool)
                m[np.repeat(np.arange(n), counts), cols] = True
        else:
            obs, mask, pl = env.observe()
            a1 = T0()
            acc["observe"] += a1 - a0
            if n == B:
                o, m, p_sel = obs, mask, pl.astype(np.int64)
            else:
                o, m, p_sel = obs[idx], mask[idx], pl[idx].astype(np.int64)
        a2 = T0()
        acc["select"] += a2 - a1
        # 入玉の記帳（λ=0）: 1手番に1局1側なので fancy な代入で重複は起きない
        king_rows = o[:, KING_PLANE].any(axis=2)
        has_king = king_rows.any(axis=1)
        adv_now = 8 - np.where(has_king, king_rows.argmax(axis=1), 8)
        entered[idx, p_sel] |= adv_now >= 6
        advance[idx, p_sel] = np.maximum(advance[idx, p_sel], adv_now)
        a3 = T0()
        acc["ny_vec"] += a3 - a2
        oh = torch.from_numpy(o) if rust_alive else torch.from_numpy(o).half()
        a4 = T0()
        acc["half"] += a4 - a3
        if rust_alive:
            cols16 = torch.from_numpy(cols)
            counts32 = torch.from_numpy(counts)
        else:
            # np.nonzero(m)[1] と m.sum(1) と同じ値（2次元の np.nonzero は 1次元の flatnonzero の約8倍遅い）
            rows_f, cols = np.divmod(np.flatnonzero(m), NUM_ACTIONS)
            cols16 = torch.from_numpy(cols.astype(np.int16))
            counts32 = torch.from_numpy(np.bincount(rows_f, minlength=n).astype(np.int32))
        a5 = T0()
        acc["csr"] += a5 - a4
        x = oh.to(device, non_blocking=True).float()
        logits, _ = rnad.forward(net, x)
        a6 = T0()
        acc["send_fwd"] += a6 - a5
        if mask_on_device:
            c_dev = cols16.to(device, non_blocking=True).long()
            k_dev = counts32.to(device, non_blocking=True).long()
            rows = torch.repeat_interleave(torch.arange(n, device=device), k_dev, output_size=len(cols16))
            legal = torch.zeros((n, NUM_ACTIONS), dtype=torch.bool, device=device)
            legal[rows, c_dev] = True
        else:
            legal = torch.from_numpy(m).to(device, non_blocking=True)
        a7 = T0()
        acc["mask_dev"] += a7 - a6
        masked = logits.float().masked_fill(~legal, float("-inf"))
        u = torch.rand(masked.shape, device=masked.device, generator=generator).clamp_(1e-20, 1.0)
        action = (masked - torch.log(-torch.log(u))).argmax(-1)
        prob = masked.softmax(-1).gather(1, action[:, None]).squeeze(1)
        a, mu = action.cpu(), prob.cpu()
        actions = np.full(B, -1, dtype=np.int64)
        actions[idx] = a.numpy()
        a8 = T0()
        acc["sample"] += a8 - a7
        rew, done = env.step(actions)
        a9 = T0()
        acc["step"] += a9 - a8
        valid[t, idx] = True
        player[t, idx] = torch.from_numpy(p_sel)
        reward[t] += torch.from_numpy(rew)
        outcome[done] = rew[done]
        a10 = T0()
        acc["rec"] += a10 - a9
        obs_l.append(oh)
        cols_l.append(cols16)
        cnt_l.append(counts32)
        act_l.append(a)
        mu_l.append(mu)
        t_l.append(torch.full((n,), t, dtype=torch.long))
        b_l.append(torch.from_numpy(idx))
        acc["store"] += T0() - a10
    else:
        raise RuntimeError(f"{max_steps} 手で終わらない局がある")
    s = T0()
    T = t + 1
    out = {
        "legal_cols": torch.cat(cols_l), "legal_counts": torch.cat(cnt_l),
        "obs": torch.cat(obs_l), "action": torch.cat(act_l),
        "mu_a": torch.cat(mu_l), "t": torch.cat(t_l), "b": torch.cat(b_l),
        "valid": valid[:T], "player": player[:T], "reward": reward[:T],
        "entered": entered, "advance": advance, "outcome": outcome,
    }
    acc["final"] += T0() - s
    if prof is not None:
        prof.update(acc)
    return out


def same(a, b):
    """batch どうしがビット単位で同じか（キーごとの結果）"""
    res = {}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            res[k] = False
            continue
        if isinstance(x, np.ndarray):
            res[k] = bool(x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y))
        else:
            iv = {torch.float16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}
            xv = x.view(iv[x.dtype]) if x.dtype in iv else x
            yv = y.view(iv[y.dtype]) if y.dtype in iv else y
            res[k] = bool(x.dtype == y.dtype and x.shape == y.shape and torch.equal(xv, yv))
    return res
