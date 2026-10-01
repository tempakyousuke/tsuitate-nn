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


# ---------------------------------------------------------------- 計算精度

PRECISIONS = ("default", "fp32", "tf32", "bf16")
_AMP_DTYPE = None


def set_precision(mode: str) -> None:
    """ネットの順伝播・逆伝播の精度（プロセス全体）。CUDA（Ampere 以降）でだけ意味がある。

    default = PyTorch の既定（**畳み込みは TF32・全結合は FP32**。畳み込みだけのこのネットは
              既定のままでほぼ TF32 で回っている）
    fp32    = TF32 を全部切る（比較用の厳密な基準）
    tf32    = 全結合も TF32 にする（価値の頭の 2層だけなので差は小さい）
    bf16    = TF32 に加えて順伝播を bf16 の autocast で回す。出力（ロジット・価値）は float32 へ
              戻すので、log softmax・v-trace・損失は float32 のまま

    L4・30,000更新の checkpoint・256局/更新での実測（rnad/bench_precision.py、2026-10-01）:
        fp32    自己対局 3.29s + 更新 5.20s = 8.49s
        tf32    2.94s + 3.49s = 6.43s。|π/μ−1| 最大 2.6e-6、更新量の fp32 とのコサイン 0.9996
        bf16    3.34s + 2.37s = 5.71s。ただし |π/μ−1| 最大 0.089、log π の差 平均 0.059、
                更新量のコサイン 0.992 と数値のずれが桁違いに大きい
    default は学習ログで比べた（同じ VM・同じ設定、200 / 130 更新の平均）: default（29,801〜30,000）
    更新 3.714s ± 0.12、tf32（30,006〜）更新 3.717s = **差なし**（既定で畳み込みが TF32 のため）。
    速くするなら bf16 だが、上の数値のずれがあるので錨への勝率でペア比較してから
    （ベンチの初版は fp32 の直後に default を測り、default が TF32 を戻さなかったので
    default = fp32 と誤って見えた。今は default で既定値へ戻す）
    """
    global _AMP_DTYPE
    if mode not in PRECISIONS:
        raise ValueError(f"precision {mode}（{PRECISIONS} のどれか）")
    # default は PyTorch の既定値へ戻す（同じプロセスで順に切り替えるベンチで前の設定が残らないように）
    torch.backends.cuda.matmul.allow_tf32 = mode in ("tf32", "bf16")
    torch.backends.cudnn.allow_tf32 = mode != "fp32"
    _AMP_DTYPE = torch.bfloat16 if mode == "bf16" else None


def forward(net, x: torch.Tensor):
    """`net(x)` を `set_precision` の精度で回し、(ロジット, 価値) を float32 で返す"""
    if _AMP_DTYPE is None:
        return net(x)
    if x.device.type != "cuda":
        raise RuntimeError("bf16 は CUDA でだけ使う")
    with torch.autocast("cuda", dtype=_AMP_DTYPE):
        logits, v = net(x)
    return logits.float(), v.float()


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
        lp_prev = legal_log_policy(forward(self.prev, obs)[0], legal)
        lp_prev_ = legal_log_policy(forward(self.prev_, obs)[0], legal)
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
                logits, _ = forward(self.net, obs)
                log_pi = legal_log_policy(logits, legal)
                pi = log_pi.exp() * legal
                lpr = log_pi - self._lpr_mix(obs, legal, alpha)
                pi_a[sl] = pi.gather(1, act[:, None]).squeeze(1).cpu()
                ent[sl] = (pi * lpr).sum(-1).cpu()
                v_tgt[sl] = forward(self.target, obs)[1].cpu()

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
            logits, v = forward(self.net, obs)
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
        # 集めた直後に1回更新するので π_a/μ_a は本来 1。ずれは行動時と学習時の数値誤差
        # （バッチの形で変わるカーネル・精度）で、v-trace の重要度比へそのまま入る
        ratio = pi_a / batch["mu_a"]
        return {"loss_v": tot_v, "loss_nerd": tot_nerd, "alpha": alpha,
                "reg_update": bool(update_reg), "samples": n,
                "ratio_dev_mean": float((ratio - 1).abs().mean()),
                "ratio_dev_max": float((ratio - 1).abs().max()),
                "v_target_mean": float(vt.mean()), "s_abs_mean": float(s.abs().mean())}

    def state_dict(self):
        """再開に要る全部: 4つのネット・optimizer・step・設定・ネットの形・乱数状態"""
        return {"net": self.net.state_dict(), "target": self.target.state_dict(),
                "prev": self.prev.state_dict(), "prev_": self.prev_.state_dict(),
                "opt": self.opt.state_dict(), "step": self.step_count,
                "cfg": dict(self.cfg.__dict__), "config": dict(self.net.config),
                "rng": rng_state(self.device)}

    def load_state_dict(self, d):
        self.net.load_state_dict(d["net"])
        self.target.load_state_dict(d["target"])
        self.prev.load_state_dict(d["prev"])
        self.prev_.load_state_dict(d["prev_"])
        self.opt.load_state_dict(d["opt"])
        self.step_count = d["step"]
        if "rng" in d:
            set_rng_state(d["rng"], self.device)

    def apply_cfg(self, cfg: RNaDConfig):
        """設定を差し替える（再開時に明示的に上書きするとき。optimizer の学習率も合わせる）"""
        self.cfg = cfg
        for g in self.opt.param_groups:
            g["lr"] = cfg.lr
        self.schedule = EntropySchedule(cfg.schedule_sizes, cfg.schedule_repeats)


def cfg_diff(saved: dict, cfg: RNaDConfig) -> dict:
    """checkpoint の設定と今の設定の差（tuple / list の違いは無視する）"""
    now = cfg.__dict__
    norm = lambda v: list(v) if isinstance(v, (tuple, list)) else v  # noqa: E731
    return {k: (saved.get(k), now[k]) for k in now if norm(saved.get(k)) != norm(now[k])}


def rng_state(device) -> dict:
    st = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        st["mps"] = torch.mps.get_rng_state()
    return st


def set_rng_state(st: dict, device) -> None:
    # 乱数状態は CPU の ByteTensor で渡す（torch.load(map_location=device) で運ばれていても戻す）
    torch.set_rng_state(st["cpu"].cpu())
    if device.type == "cuda" and "cuda" in st:
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])
    elif device.type == "mps" and "mps" in st:
        torch.mps.set_rng_state(st["mps"].cpu())


# ---------------------------------------------------------------- 自己対局


@torch.no_grad()
def sample_actions(net, obs, mask, device, generator=None):
    """マスク内の softmax からサンプリングし (行動, その確率) を返す。

    サンプリングはデバイス上で行い、戻すのは選んだ手と確率だけ（[局数, 11259] の確率を
    CPU へ運んでサンプリングすると、GPU では自己対局の律速になる。2026-09-28 実測で L4 の
    1更新 20 秒のうち 15 秒が自己対局）。

    観測は **fp16 に丸めてから**使う（学習用に fp16 で保存するので、行動時と学習時で同じ入力に
    する。丸めない版は未更新でも π_a/μ_a が 0.998〜1.004 にずれた。2026-09-28 codex 指摘）。
    `generator` を渡すとその乱数列でサンプリングする（評価を学習の乱数列から切り離す）
    """
    x = torch.from_numpy(obs).to(device, non_blocking=True).half().float()
    logits, _ = forward(net, x)
    legal = torch.from_numpy(mask).to(device, non_blocking=True)
    masked = logits.float().masked_fill(~legal, float("-inf"))
    # Gumbel-max（softmax からのサンプリングと同じ分布）。torch.multinomial は MPS で
    # 確率 0 の手（マスク外）を選ぶことがあったので使わない（2026-09-28 実測）
    u = torch.rand(masked.shape, device=masked.device, generator=generator).clamp_(1e-20, 1.0)
    action = (masked - torch.log(-torch.log(u))).argmax(-1)
    prob = masked.softmax(-1).gather(1, action[:, None]).squeeze(1)
    return action.cpu(), prob.cpu()


# 観測の自玉のプレーン（tsuitate-bot の rl::encode の ROLES 順で玉は 7 番目）。盤は手番側を
# 先手向きに正規化しているので、行 0〜2（正規化後の段 1〜3）が常に敵陣
KING_PLANE = 7
# 段階的な入玉ボーナスの節目（玉の初期位置からの前進段数）。6 = 敵陣3段目への進入
GRADED_MILESTONES = (3, 4, 5, 6)


def collect(env, net, device, max_steps=260, nyugyoku_lambda=0.0, nyugyoku_mode="enter"):
    """`env`（auto_reset=False の VecEnv）の全局を1局ずつ最後まで自己対局し、学習用に並べる。

    戻り値の obs / legal / action / mu_a / t / b は有効なステップだけを平らにしたもの、
    valid / player / reward は [T, B] の格子（v-trace 用）。entered / outcome は [B, 2]
    （各局・各側が入玉したか、終局の勝敗 ±1/0）。

    `nyugyoku_lambda` > 0 なら**入玉ボーナス**（docs/rl-deepnash-design.md の「報酬を変えた
    スタイル特化モデル」）: 玉が敵陣3段目以内に**初めて入った**側に +λ、相手に −λ（ゼロ和）。
    入ったかどうかはその側が次に観測したときの自玉の位置（自駒なので観測にある情報）で判定し、
    報酬はその側の直前の着手（= 玉を入れた手）に付ける。

    `nyugyoku_mode="graded"` は**段階的なボーナス**: 玉が初期位置から 3・4・5・6 段前進した地点へ
    **初めて**着くたびに λ/4（敵陣へ入るまでに計 λ）。入玉そのものは稀すぎてボーナスが学習の信号に
    ならなかった（arm C、2026-09-28）ので、毎局のように起きる「最初の数段の前進」から信号を届ける。
    ポテンシャル型ではない（最適方策を変える = 入玉を好むモデルにする）のは意図どおり。
    戻り値の `advance` は各局・各側の玉の最大前進段数 [B, 2]
    """
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
    for t in range(max_steps):
        alive = env.alive()
        if not alive.any():
            break
        obs, mask, pl = env.observe()
        idx = np.flatnonzero(alive)
        # 自玉の段（正規化後の行 0〜8。玉が無い = 取られた局面は -1）→ 前進段数 = 8 − 行
        king_rows = obs[idx, KING_PLANE].any(axis=2)
        has_king = king_rows.any(axis=1)
        row = np.where(has_king, king_rows.argmax(axis=1), 8)
        adv_now = 8 - row
        for i, b in enumerate(idx):
            p = int(pl[b])
            bonus = 0.0
            if adv_now[i] >= 6 and not entered[b, p]:
                entered[b, p] = True
                if nyugyoku_mode == "enter":
                    bonus += nyugyoku_lambda
            if adv_now[i] > advance[b, p]:
                if nyugyoku_mode == "graded":
                    newly = [m for m in GRADED_MILESTONES if advance[b, p] < m <= adv_now[i]]
                    bonus += nyugyoku_lambda / len(GRADED_MILESTONES) * len(newly)
                advance[b, p] = adv_now[i]
            if bonus > 0 and last_act[b, p] >= 0:
                reward[last_act[b, p], b, p] += bonus
                reward[last_act[b, p], b, 1 - p] -= bonus
            last_act[b, p] = t
        a, mu = sample_actions(net, obs[idx], mask[idx], device)
        actions = np.full(B, -1, dtype=np.int64)
        actions[idx] = a.numpy()
        rew, done = env.step(actions)
        valid[t, idx] = True
        player[t, idx] = torch.from_numpy(pl[idx].astype(np.int64))
        # ボーナスは過去の時刻へ足すので、この時刻の報酬は上書きでなく加算
        reward[t] += torch.from_numpy(rew)
        outcome[done] = rew[done]
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
        "entered": entered, "advance": advance, "outcome": outcome,
    }


@torch.no_grad()
def play_match(env, net_a, net_b, device, generator=None):
    """net_a と net_b を env（auto_reset=False）の全局で1局ずつ対戦させる。
    偶数番の局は net_a が先手（局数は偶数 = 先後が同数）。net_a から見た (勝, 負, 分) を返す。
    `generator` を渡すと学習の乱数列を消費しない"""
    env.reset()
    B = env.num_envs
    if B % 2:
        raise ValueError(f"評価の局数 {B} は偶数にする（先後を同数にするため）")
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
                actions[idx] = sample_actions(net, obs[idx], mask[idx], device, generator)[0].numpy()
        rew, done = env.step(actions)
        for i in np.flatnonzero(done):
            r = rew[i, a_color[i]]
            wins += r > 0
            losses += r < 0
            draws += r == 0
    return int(wins), int(losses), int(draws)
