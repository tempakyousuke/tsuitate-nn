"""R-NaD（DeepNash の中核）の PyTorch 実装。

参照: OpenSpiel の `open_spiel/python/algorithms/rnad/rnad.py`（JAX。2025-05 に OpenSpiel から
削除されたので、削除直前の d1dcdf5 を基準にする）。損失と勾配が参照実装と一致することを
`tests/test_rnad.py` が乱数入力のフィクスチャ（参照実装で計算したもの）で検査する。

参照実装との違いは**計算の持ち方だけ**:
- 参照は各ステップで [A] 次元の Q ベクトル（learning_output）を作るが、ここでは行動数が
  11,259 あるので作らない。アドバンテージは
      adv_a = −η·(lpr_a − Σ_b π_b·lpr_b) + s·(1[a=選んだ手] − π_選んだ手)
  に分解できる（s は v-trace のステップごとのスカラー）ので、s だけを持つ
- ネットの順伝播は有効なステップだけを平らにして、チャンクに分けて回す
  （[T, B] の格子は v-trace のスキャンにだけ使う）

記号: lpr = log π − (α·log π_prev + (1−α)·log π_prev_)（正則化方策との対数比）、
π = 現在のネット、v-trace に入れる価値は**ターゲットネット**の v（参照と同じ）。
"""

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- 方策の基本演算


def legal_log_policy(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """マスク内の log softmax。マスク外は 0（参照の legal_log_policy と同じ規約）"""
    masked = logits.masked_fill(~legal, float("-inf"))
    return torch.where(legal, masked.log_softmax(-1), torch.zeros_like(logits))


def legal_policy(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """マスク内の softmax。マスク外は 0"""
    return logits.masked_fill(~legal, float("-inf")).softmax(-1)


# ---------------------------------------------------------------- 正則化の繰り返し


class EntropySchedule:
    """正則化方策を差し替える学習ステップの列（参照の EntropySchedule と同じ）。

    EntropySchedule(sizes=[3, 5, 10], repeats=[2, 4, 1]) → [0, 3, 6, 11, 16, 21, 26, 36]、
    以後は最後の大きさ（10）で繰り返す。
    """

    def __init__(self, sizes, repeats):
        if len(sizes) != len(repeats) or not sizes or repeats[-1] != 1 or min(repeats) <= 0:
            raise ValueError(f"entropy schedule: sizes={sizes} repeats={repeats}")
        schedule = [0]
        for size, repeat in zip(sizes, repeats):
            schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])
        self.schedule = np.array(schedule)

    def __call__(self, step: int):
        """(alpha, この step の後に正則化方策を差し替えるか)"""
        s = self.schedule
        if step >= s[-1]:
            size = s[-1] - s[-2]
            start = s[-1] + (step - s[-1]) // size * size
        else:
            start = s[s <= step].max()
            size = s[s > step].min() - start
        update = step > 0 and step == start + size - 1
        alpha = min(2.0 * (step - start) / size, 1.0)
        return alpha, update


# ---------------------------------------------------------------- v-trace


def v_trace_scalars(v, valid, player, pi_a, mu_a, ent, reward, k, eta, c=1.0, rho=math.inf):
    """プレイヤー k から見た v-trace（参照の v_trace をスカラーに落とした版）。

    引数はどれも [T, B]（reward は k の報酬、ent = Σ_a π_a·lpr_a（手番側の方策で））。
    戻り値:
        vt: k の手番のステップでの価値の目標（それ以外は 0）
        s:  k の手番のステップでの Q のスカラー部分（learning_output = v + η 項 + onehot·s）
    """
    T, B = v.shape
    cs = torch.where(valid, pi_a / mu_a, torch.ones_like(v))
    inv_mu = torch.where(valid, 1.0 / mu_a, torch.ones_like(v))
    player_others = torch.where(player == k, 1.0, -1.0) * valid
    eta_reg_entropy = -eta * ent * player_others

    zeros = torch.zeros(B, dtype=v.dtype)
    ones = torch.ones(B, dtype=v.dtype)
    c_reward, c_reward_unc, c_next_value, c_next_vt, c_is = zeros, zeros, zeros, zeros, ones
    vt = torch.zeros_like(v)
    s = torch.zeros_like(v)
    for t in range(T - 1, -1, -1):
        ok = valid[t]
        ours = ok & (player[t] == k)
        opp = ok & (player[t] != k)
        r_unc = reward[t] + c_reward_unc + eta_reg_entropy[t]
        disc = reward[t] + c_reward
        w = cs[t] * c_is
        our_vt = (v[t] + torch.clamp(w, max=rho) * (r_unc + c_next_value - v[t])
                  + torch.clamp(w, max=c) * (c_next_vt - c_next_value))
        our_s = inv_mu[t] * (disc + c_is * c_next_vt - v[t])
        vt[t] = torch.where(ours, our_vt, zeros)
        s[t] = torch.where(ours, our_s, zeros)
        # 次（1つ前の時刻）へ渡す carry
        n_reward = torch.where(ours, zeros, torch.where(opp, eta_reg_entropy[t] + cs[t] * disc, zeros))
        n_reward_unc = torch.where(ours, zeros, torch.where(opp, r_unc, zeros))
        n_next_value = torch.where(ours, v[t], torch.where(opp, c_next_value, zeros))
        n_next_vt = torch.where(ours, our_vt, torch.where(opp, c_next_vt, zeros))
        n_is = torch.where(ours, ones, torch.where(opp, cs[t] * c_is, ones))
        c_reward, c_reward_unc, c_next_value, c_next_vt, c_is = (
            n_reward, n_reward_unc, n_next_value, n_next_vt, n_is)
    return vt, s


# ---------------------------------------------------------------- 損失


def loss_terms(logits, v, legal, action, player, lpr, vt, s, counts, eta, beta, clip=10_000.0):
    """有効なステップの集まり（平らにした [N]）について、損失の和を返す。

    counts[k] はバッチ全体でのプレイヤー k の手番数（参照は損失をこれで割る）。チャンクに
    分けて呼んでも、全体をまとめて呼んだのと同じ値の和になる。
    lpr は勾配を止めた正則化の対数比 [N, A]、vt / s は v-trace の結果 [N]。
    """
    norm = torch.where(player == 0, 1.0 / max(counts[0], 1), 1.0 / max(counts[1], 1))
    loss_v = (norm * (v - vt) ** 2).sum()

    with torch.no_grad():
        pi = legal_policy(logits, legal)
        pi_taken = pi.gather(1, action[:, None]).squeeze(1)
        mean_lpr = (pi * lpr).sum(-1, keepdim=True)
        onehot = F.one_hot(action, logits.shape[-1]).to(logits.dtype)
        adv = -eta * (lpr - mean_lpr) + s[:, None] * (onehot - pi_taken[:, None])
        adv = adv.clamp(-clip, clip)
    legal_f = legal.to(logits.dtype)
    mean_logit = (logits * legal_f).sum(-1, keepdim=True) / legal_f.sum(-1, keepdim=True)
    centered = logits - mean_logit
    with torch.no_grad():
        can_decrease = centered > -beta
        can_increase = centered < beta
        force = can_decrease * adv.clamp(max=0.0) + can_increase * adv.clamp(min=0.0)
    nerd = (legal_f * centered * force).sum(-1)
    loss_nerd = -(norm * nerd).sum()
    return loss_v, loss_nerd


# ---------------------------------------------------------------- 学習器


@dataclass
class RNaDConfig:
    eta: float = 0.2
    beta: float = 2.0
    clip: float = 10_000.0
    c_vtrace: float = 1.0
    lr: float = 5e-5
    target_avg: float = 0.001
    schedule_sizes: tuple = (200,)
    schedule_repeats: tuple = (1,)
    chunk: int = 1024


class RNaDLearner:
    """現在のネット・ターゲット・正則化方策2つ（prev / prev_）を持ち、1バッチで1回更新する。

    BatchNorm は**常に eval モード**で使う（初期化に使う模倣ネットの統計に固定し、BN は
    アフィン変換として学習する）。行動・学習・評価で統計が変わらないようにするため。
    """

    def __init__(self, net, cfg: RNaDConfig, device):
        self.cfg = cfg
        self.device = device
        self.net = net.to(device).eval()
        self.target = copy.deepcopy(self.net).eval()
        self.prev = copy.deepcopy(self.net).eval()
        self.prev_ = copy.deepcopy(self.net).eval()
        for m in (self.target, self.prev, self.prev_):
            m.requires_grad_(False)
        # 参照: adam(b1=0, b2=0.999, eps=1e-7) → scale(-lr) → clip(clip_gradient)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, betas=(0.0, 0.999), eps=1e-7)
        self.schedule = EntropySchedule(cfg.schedule_sizes, cfg.schedule_repeats)
        self.step_count = 0

    def _chunks(self, n):
        for i in range(0, n, self.cfg.chunk):
            yield slice(i, min(i + self.cfg.chunk, n))

    def _lpr_mix(self, obs, legal, alpha):
        """正則化方策の対数確率（2つの前の版の α 混合）"""
        lp_prev = legal_log_policy(self.prev(obs)[0], legal)
        lp_prev_ = legal_log_policy(self.prev_(obs)[0], legal)
        return alpha * lp_prev + (1 - alpha) * lp_prev_

    def update(self, batch):
        """batch（`collect` の戻り値）で1回更新する。ログ用の dict を返す"""
        cfg, dev = self.cfg, self.device
        alpha, update_reg = self.schedule(self.step_count)
        obs_all = batch["obs"]
        legal_all = batch["legal"]
        action_all = batch["action"]
        n = len(action_all)
        t_idx, b_idx = batch["t"], batch["b"]
        T, B = batch["valid"].shape

        # 1回目: 勾配なしで v-trace の入力（π_a・Σπ·lpr・ターゲットの v）を作る
        pi_a = torch.zeros(n)
        ent = torch.zeros(n)
        v_tgt = torch.zeros(n)
        with torch.no_grad():
            for sl in self._chunks(n):
                obs = obs_all[sl].to(dev, torch.float32)
                legal = legal_all[sl].to(dev)
                act = action_all[sl].to(dev)
                logits, _ = self.net(obs)
                log_pi = legal_log_policy(logits, legal)
                pi = log_pi.exp() * legal
                lpr = log_pi - self._lpr_mix(obs, legal, alpha)
                pi_a[sl] = pi.gather(1, act[:, None]).squeeze(1).cpu()
                ent[sl] = (pi * lpr).sum(-1).cpu()
                v_tgt[sl] = self.target(obs)[1].cpu()

        def grid(x):
            g = torch.zeros(T, B)
            g[t_idx, b_idx] = x
            return g

        valid = batch["valid"]
        player = batch["player"]
        mu_g = torch.where(valid, grid(batch["mu_a"]), torch.ones(T, B))
        vt = torch.zeros(n)
        s = torch.zeros(n)
        for k in (0, 1):
            vt_k, s_k = v_trace_scalars(grid(v_tgt), valid, player, grid(pi_a), mu_g, grid(ent),
                                        batch["reward"][:, :, k], k, cfg.eta, c=cfg.c_vtrace)
            mine = player[t_idx, b_idx] == k
            vt = torch.where(mine, vt_k[t_idx, b_idx], vt)
            s = torch.where(mine, s_k[t_idx, b_idx], s)
        player_flat = player[t_idx, b_idx].long()
        counts = [int((player_flat == 0).sum()), int((player_flat == 1).sum())]

        # 2回目: 勾配つきで損失を積む（チャンクごとに backward して勾配を足し合わせる）
        self.opt.zero_grad(set_to_none=True)
        tot_v = tot_nerd = 0.0
        for sl in self._chunks(n):
            obs = obs_all[sl].to(dev, torch.float32)
            legal = legal_all[sl].to(dev)
            act = action_all[sl].to(dev)
            logits, v = self.net(obs)
            with torch.no_grad():
                lpr = legal_log_policy(logits, legal) - self._lpr_mix(obs, legal, alpha)
            lv, ln = loss_terms(logits, v, legal, act, player_flat[sl].to(dev), lpr,
                                vt[sl].to(dev), s[sl].to(dev), counts, cfg.eta, cfg.beta, cfg.clip)
            (lv + ln).backward()
            tot_v += lv.item()
            tot_nerd += ln.item()
        self.opt.step()
        with torch.no_grad():
            # 参照の optax.clip は更新量に掛かるが、lr=5e-5 では事実上効かないので省く
            for pt, p in zip(self.target.parameters(), self.net.parameters()):
                pt.add_(cfg.target_avg * (p - pt))
            for bt, b in zip(self.target.buffers(), self.net.buffers()):
                bt.copy_(b)
            if update_reg:
                self.prev_.load_state_dict(self.prev.state_dict())
                self.prev.load_state_dict(self.target.state_dict())
        self.step_count += 1
        return {"loss_v": tot_v, "loss_nerd": tot_nerd, "alpha": alpha,
                "reg_update": bool(update_reg), "samples": n,
                "v_target_mean": float(vt.mean()), "s_abs_mean": float(s.abs().mean())}

    def state_dict(self):
        return {"net": self.net.state_dict(), "target": self.target.state_dict(),
                "prev": self.prev.state_dict(), "prev_": self.prev_.state_dict(),
                "opt": self.opt.state_dict(), "step": self.step_count, "cfg": self.cfg.__dict__}

    def load_state_dict(self, d):
        self.net.load_state_dict(d["net"])
        self.target.load_state_dict(d["target"])
        self.prev.load_state_dict(d["prev"])
        self.prev_.load_state_dict(d["prev_"])
        self.opt.load_state_dict(d["opt"])
        self.step_count = d["step"]


# ---------------------------------------------------------------- 自己対局


@torch.no_grad()
def sample_actions(net, obs, mask, device):
    """マスク内の softmax からサンプリングし (行動, その確率) を返す。

    サンプリングはデバイス上で行い、戻すのは選んだ手と確率だけ（[局数, 11259] の確率を
    CPU へ運んでサンプリングすると、GPU では自己対局の律速になる。2026-09-28 実測で L4 の
    1更新 20 秒のうち 15 秒が自己対局）
    """
    logits, _ = net(torch.from_numpy(obs).to(device, non_blocking=True))
    legal = torch.from_numpy(mask).to(device, non_blocking=True)
    masked = logits.float().masked_fill(~legal, float("-inf"))
    # Gumbel-max（softmax からのサンプリングと同じ分布）。torch.multinomial は MPS で
    # 確率 0 の手（マスク外）を選ぶことがあったので使わない（2026-09-28 実測）
    u = torch.rand_like(masked).clamp_(1e-20, 1.0)
    action = (masked - torch.log(-torch.log(u))).argmax(-1)
    prob = masked.softmax(-1).gather(1, action[:, None]).squeeze(1)
    return action.cpu(), prob.cpu()


def collect(env, net, device, max_steps=260):
    """`env`（auto_reset=False の VecEnv）の全局を1局ずつ最後まで自己対局し、学習用に並べる。

    戻り値の obs / legal / action / mu_a / t / b は有効なステップだけを平らにしたもの、
    valid / player / reward は [T, B] の格子（v-trace 用）。
    """
    env.reset()
    B = env.num_envs
    obs_l, legal_l, act_l, mu_l, t_l, b_l = [], [], [], [], [], []
    valid = torch.zeros(max_steps, B, dtype=torch.bool)
    player = torch.zeros(max_steps, B, dtype=torch.long)
    reward = torch.zeros(max_steps, B, 2)
    for t in range(max_steps):
        alive = env.alive()
        if not alive.any():
            break
        obs, mask, pl = env.observe()
        idx = np.flatnonzero(alive)
        a, mu = sample_actions(net, obs[idx], mask[idx], device)
        actions = np.full(B, -1, dtype=np.int64)
        actions[idx] = a.numpy()
        rew, _done = env.step(actions)
        valid[t, idx] = True
        player[t, idx] = torch.from_numpy(pl[idx].astype(np.int64))
        reward[t] = torch.from_numpy(rew)
        obs_l.append(torch.from_numpy(obs[idx]).half())
        legal_l.append(torch.from_numpy(mask[idx]))
        act_l.append(a)
        mu_l.append(mu)
        t_l.append(torch.full((len(idx),), t, dtype=torch.long))
        b_l.append(torch.from_numpy(idx))
    else:
        raise RuntimeError(f"{max_steps} 手で終わらない局がある")
    T = t + 1
    return {
        "obs": torch.cat(obs_l), "legal": torch.cat(legal_l), "action": torch.cat(act_l),
        "mu_a": torch.cat(mu_l), "t": torch.cat(t_l), "b": torch.cat(b_l),
        "valid": valid[:T], "player": player[:T], "reward": reward[:T],
    }


@torch.no_grad()
def play_match(env, net_a, net_b, device):
    """net_a と net_b を env（auto_reset=False）の全局で1局ずつ対戦させる。
    偶数番の局は net_a が先手。net_a から見た (勝, 負, 分) を返す"""
    env.reset()
    B = env.num_envs
    a_color = np.array([i % 2 for i in range(B)])  # 0 = 先手
    wins = losses = draws = 0
    while True:
        alive = env.alive()
        if not alive.any():
            break
        obs, mask, pl = env.observe()
        actions = np.full(B, -1, dtype=np.int64)
        for net, sel in ((net_a, pl == a_color), (net_b, pl != a_color)):
            idx = np.flatnonzero(alive & sel)
            if len(idx):
                actions[idx] = sample_actions(net, obs[idx], mask[idx], device)[0].numpy()
        rew, done = env.step(actions)
        for i in np.flatnonzero(done):
            r = rew[i, a_color[i]]
            wins += r > 0
            losses += r < 0
            draws += r == 0
    return int(wins), int(losses), int(draws)
