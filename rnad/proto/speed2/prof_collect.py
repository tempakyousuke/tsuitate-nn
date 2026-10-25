"""rnad.collect の自己対局（本線の設定: 自己対局だけ・λ=0・compact_legal=True）を部品ごとに計時するコピー
（F2 の verify_collect.py の `prof` の腕。学習には使っていない）。

    python rnad/proto/speed2/prof_collect.py --batch 768 --threads 4 --out prof768.json

数式は rnad.collect と同じ（同じ seed で rnad.collect と出力がビット単位で一致することを verify_collect.py の
`--modes orig,prof` で確かめる）。CPU で回すので fwd は GPU の代表ではない。
部品:
  alive      env.alive()
  observe    env.observe()（Rust。全 B 局の観測 float32 とマスク bool を返す）
  king       obs[idx, KING_PLANE] の抜き出しと前進段数の計算（numpy）
  pyloop     局ごとの Python ループ（入玉の記帳。λ=0 でも回る）
  gather     sample_actions に渡す obs[idx] / mask[idx] の fancy index
  h2d_conv   torch.from_numpy(obs).to(device).half().float()（CPU では .to は何もしない）
  fwd        ネットの順伝播（CPU。GPU の代表ではない）
  legal      torch.from_numpy(mask).to(device)
  sample     masked_fill・Gumbel・argmax・softmax・gather・.cpu()
  step       env.step()（Rust）
  rec        valid / player / reward の記帳
  store      obs_l へ obs[idx] の half 化、compact legal（mask[idx]・nonzero・sum）、残りの append
  final      最後の torch.cat
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from proto_common import DEFAULT_WEIGHTS, env_kwargs
import rnad  # noqa: E402
from export_weights import read  # noqa: E402
import tsuitate_rl  # noqa: E402

KING_PLANE = rnad.KING_PLANE

PARTS = ["alive", "observe", "king", "pyloop", "gather", "h2d_conv", "fwd", "legal", "sample", "step", "rec",
         "store", "final"]


@torch.no_grad()
def collect_prof(env, net, device, max_steps=260, generator=None, compact_legal=True, prof=None, steps_log=None):
    """rnad.collect（opponents=None・truth=False・nyugyoku_lambda=0・mode=enter）の計時つきコピー"""
    T0 = time.perf_counter
    acc = {k: 0.0 for k in PARTS}
    env.reset()
    B = env.num_envs
    obs_l, legal_l, act_l, mu_l, t_l, b_l = [], [], [], [], [], []
    valid = torch.zeros(max_steps, B, dtype=torch.bool)
    player = torch.zeros(max_steps, B, dtype=torch.long)
    reward = torch.zeros(max_steps, B, 2)
    entered = np.zeros((B, 2), dtype=bool)
    advance = np.zeros((B, 2), dtype=np.int64)
    last_act = np.full((B, 2), -1, dtype=np.int64)
    outcome = np.zeros((B, 2), dtype=np.float32)
    nyugyoku_lambda = 0.0
    for t in range(max_steps):
        s = T0()
        alive = env.alive()
        a0 = T0(); acc["alive"] += a0 - s
        if not alive.any():
            break
        obs, mask, pl = env.observe()
        a1 = T0(); acc["observe"] += a1 - a0
        idx = np.flatnonzero(alive)
        king_rows = obs[idx, KING_PLANE].any(axis=2)
        has_king = king_rows.any(axis=1)
        row = np.where(has_king, king_rows.argmax(axis=1), 8)
        adv_now = 8 - row
        a2 = T0(); acc["king"] += a2 - a1
        for i, b in enumerate(idx):
            p = int(pl[b])
            bonus = 0.0
            if adv_now[i] >= 6 and not entered[b, p]:
                entered[b, p] = True
                bonus += nyugyoku_lambda
            if adv_now[i] > advance[b, p]:
                advance[b, p] = adv_now[i]
            if bonus > 0 and last_act[b, p] >= 0:
                reward[last_act[b, p], b, p] += bonus
                reward[last_act[b, p], b, 1 - p] -= bonus
            last_act[b, p] = t
        a3 = T0(); acc["pyloop"] += a3 - a2
        actions = np.full(B, -1, dtype=np.int64)
        # sample_actions(net, obs[idx], mask[idx], device, generator) を展開して計時
        o_sel = obs[idx]
        m_sel = mask[idx]
        a4 = T0(); acc["gather"] += a4 - a3
        x = torch.from_numpy(o_sel).to(device, non_blocking=True).half().float()
        a5 = T0(); acc["h2d_conv"] += a5 - a4
        logits, _ = rnad.forward(net, x)
        a6 = T0(); acc["fwd"] += a6 - a5
        legal = torch.from_numpy(m_sel).to(device, non_blocking=True)
        a7 = T0(); acc["legal"] += a7 - a6
        masked = logits.float().masked_fill(~legal, float("-inf"))
        u = torch.rand(masked.shape, device=masked.device, generator=generator).clamp_(1e-20, 1.0)
        action = (masked - torch.log(-torch.log(u))).argmax(-1)
        prob = masked.softmax(-1).gather(1, action[:, None]).squeeze(1)
        a, mu = action.cpu(), prob.cpu()
        actions[idx] = a.numpy()
        a8 = T0(); acc["sample"] += a8 - a7
        rew, done = env.step(actions)
        a9 = T0(); acc["step"] += a9 - a8
        valid[t, idx] = True
        player[t, idx] = torch.from_numpy(pl[idx].astype(np.int64))
        reward[t] += torch.from_numpy(rew)
        outcome[done] = rew[done]
        a10 = T0(); acc["rec"] += a10 - a9
        obs_l.append(torch.from_numpy(obs[idx]).half())
        if compact_legal:
            rows_m = mask[idx]
            legal_l.append((torch.from_numpy(np.nonzero(rows_m)[1].astype(np.int16)),
                            torch.from_numpy(rows_m.sum(1).astype(np.int32))))
        else:
            legal_l.append(torch.from_numpy(mask[idx]))
        act_l.append(a)
        mu_l.append(mu)
        t_l.append(torch.full((len(idx),), t, dtype=torch.long))
        b_l.append(torch.from_numpy(idx))
        a11 = T0(); acc["store"] += a11 - a10
        if steps_log is not None:
            steps_log.append({"t": t, "alive": int(len(idx)), "observe": a1 - a0, "step": a9 - a8,
                              "fwd": a6 - a5, "pyloop": a3 - a2, "gather": a4 - a3, "store": a11 - a10,
                              "h2d_conv": a5 - a4, "sample": a8 - a7, "king": a2 - a1,
                              "legal_sum": int(m_sel.sum())})
    else:
        raise RuntimeError(f"{max_steps} 手で終わらない局がある")
    s = T0()
    T = t + 1
    out = {
        **({"legal_cols": torch.cat([c for c, _ in legal_l]), "legal_counts": torch.cat([n for _, n in legal_l])}
           if compact_legal else {"legal": torch.cat(legal_l)}),
        "obs": torch.cat(obs_l), "action": torch.cat(act_l),
        "mu_a": torch.cat(mu_l), "t": torch.cat(t_l), "b": torch.cat(b_l),
        "valid": valid[:T], "player": player[:T], "reward": reward[:T],
        "entered": entered, "advance": advance, "outcome": outcome,
    }
    acc["final"] += T0() - s
    if prof is not None:
        prof.update(acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    net = read(args.weights).to(device).eval()
    env = tsuitate_rl.VecEnv(args.batch, seed=args.env_seed, auto_reset=False, **env_kwargs(net))
    res = []
    for r in range(args.reps):
        g = torch.Generator(device=device)
        g.manual_seed(args.seed + r)
        prof, steps = {}, []
        t0 = time.perf_counter()
        batch = collect_prof(env, net, device, generator=g, compact_legal=True, prof=prof, steps_log=steps)
        wall = time.perf_counter() - t0
        n = int(batch["action"].numel())
        res.append({"rep": r, "batch": args.batch, "wall": wall, "T": int(batch["valid"].shape[0]),
                    "samples": n, "parts": prof, "steps": steps,
                    "rayon": os.environ.get("RAYON_NUM_THREADS"), "torch_threads": args.threads})
        print(f"B={args.batch} rep={r} wall={wall:.2f}s T={res[-1]['T']} samples={n}", flush=True)
        tot = sum(prof.values())
        for k in PARTS:
            print(f"  {k:9s} {prof[k]:8.3f}s {100 * prof[k] / tot:5.1f}%")
    with open(args.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
