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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

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


def forward_aux(net, aux, x: torch.Tensor):
    """`forward` と同じだが、幹の出力を補助ヘッドにも通す: (ロジット, 価値, 補助のロジット [B, 81, 15])"""
    if _AMP_DTYPE is None:
        f = net.trunk(x)
        logits, v = net.heads(f)
        return logits, v, aux(f)
    if x.device.type != "cuda":
        raise RuntimeError("bf16 は CUDA でだけ使う")
    with torch.autocast("cuda", dtype=_AMP_DTYPE):
        f = net.trunk(x)
        logits, v = net.heads(f)
        a = aux(f)
    return logits.float(), v.float(), a.float()


def aux_truth_loss(aux_logits: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """決定点ごとの補助損失 [B]: 自駒のマス（ラベル −1）を除いた交差エントロピーのマス平均"""
    ce = F.cross_entropy(aux_logits.reshape(-1, aux_logits.shape[-1]), truth.reshape(-1),
                         ignore_index=-1, reduction="none").view(truth.shape)
    return ce.sum(1) / (truth >= 0).sum(1).clamp(min=1)


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

NUM_ACTIONS = 139 * 81  # tsuitate-bot の rl::action（種類 × マス）


def legal_log_policy(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """マスク内の log softmax。マスク外は 0（参照の legal_log_policy と同じ規約）"""
    masked = logits.masked_fill(~legal, float("-inf"))
    return torch.where(legal, masked.log_softmax(-1), torch.zeros_like(logits))


def legal_policy(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """マスク内の softmax。マスク外は 0"""
    return logits.masked_fill(~legal, float("-inf")).softmax(-1)


def pack_csr(cols: torch.Tensor, counts: torch.Tensor, action: torch.Tensor):
    """`pack_legal` と同じ (idx, 詰めたマスク, 選んだ手の位置) を、合法手の添字の列（行ごとに昇順でつなげたもの）と
    行ごとの個数から作る（`collect(compact_legal=True)` の batch 用。密なマスクを経由しない）"""
    idx, pmask, apos, ok = _pack_csr(cols, counts, action, int(counts.long().max()))
    if not bool(ok):
        raise ValueError("マスク外の手が batch にある")
    return idx, pmask, apos


def _pack_csr(cols, counts, action, k, out_size=None):
    """`pack_csr` の中身。行の数 k と添字の総数 out_size を呼び出し側が渡せば CUDA で同期しない
    （`out_size` なしの repeat_interleave は同期する）。マスク外の手の検査は ok（0次元の bool）で返す"""
    n = len(counts)
    counts = counts.long()
    rows = torch.repeat_interleave(torch.arange(n, device=cols.device), counts, output_size=out_size)
    pos = torch.arange(len(rows), device=cols.device) - (torch.cumsum(counts, 0) - counts)[rows]
    idx = torch.zeros(n, k, dtype=torch.long, device=cols.device)
    idx[rows, pos] = cols.long()
    pmask = torch.zeros(n, k, dtype=torch.bool, device=cols.device)
    pmask[rows, pos] = True
    apos = ((idx == action[:, None]) & pmask).to(torch.uint8).argmax(1)
    return idx, pmask, apos, (idx.gather(1, apos[:, None]).squeeze(1) == action).all()


def pack_legal(legal: torch.Tensor, action: torch.Tensor):
    """合法手だけを左に詰めた添字 (idx [n, K], 詰めたマスク [n, K], 選んだ手の詰めた位置 [n])。K は行ごとの合法手数の最大。

    方策の計算（マスク内の softmax・NeuRD の損失）はマスク内の手しか使わないので、11,259 手の配列の代わりに
    `logits.gather(1, idx)` の [n, K] で同じ値・同じ勾配になる（合法手は rl_deduce_v4 の自己対局で平均 約80・最大 約330 手）。
    `RNaDLearner(packed=True)` が使う。並びは行ごとに行動番号の昇順（nonzero が行優先で返すため）
    """
    n = legal.shape[0]
    counts = legal.sum(1)
    k = int(counts.max())
    rows, cols = legal.nonzero(as_tuple=True)
    pos = torch.arange(len(rows), device=legal.device) - (torch.cumsum(counts, 0) - counts)[rows]
    idx = torch.zeros(n, k, dtype=torch.long, device=legal.device)
    idx[rows, pos] = cols
    pmask = torch.zeros(n, k, dtype=torch.bool, device=legal.device)
    pmask[rows, pos] = True
    apos = ((idx == action[:, None]) & pmask).to(torch.uint8).argmax(1)
    if not bool((idx.gather(1, apos[:, None]).squeeze(1) == action).all()):
        raise ValueError("マスク外の手が batch にある")
    return idx, pmask, apos


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

    def position(self, step: int):
        """(この step を含む段階の始まり, 段階の長さ)"""
        s = self.schedule
        if step >= s[-1]:
            size = s[-1] - s[-2]
            return int(s[-1] + (step - s[-1]) // size * size), int(size)
        start = s[s <= step].max()
        return int(start), int(s[s > step].min() - start)

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


def v_trace_scalars(v, valid, player, pi_a, mu_a, ent, reward, k, eta, c=1.0, rho=math.inf, inv_mu_cap=0.0):
    """プレイヤー k から見た v-trace（参照の v_trace をスカラーに落とした版）。

    引数はどれも [T, B]（reward は k の報酬、ent = Σ_a π_a·lpr_a（手番側の方策で））。
    戻り値:
        vt: k の手番のステップでの価値の目標（それ以外は 0）
        s:  k の手番のステップでの Q のスカラー部分（learning_output = v + η 項 + onehot·s）
    `c` は数か [T, B] のテンソル（時刻ごとの c。`RNaDConfig.c_ply` で序盤は 1・終盤だけ λ にするため）。
    **式を変えたら `RNaDLearner._v_trace_chunk`（`--fused` が使う同じ再帰のチャンク版）も同じように直す**
    （`tests/test_fast_update.py` の合成データの検査が食い違いを止める）。
    `inv_mu_cap` > 0 なら s に掛かる 1/μ に上限を付ける（参照には無い。まれにしか選ばれない手の
    s が勾配の分散のほとんどを占めるため。docs/rl-learning-speed.md）。価値の目標 vt には効かない
    """
    T, B = v.shape
    cs = torch.where(valid, pi_a / mu_a, torch.ones_like(v))
    inv_mu = torch.where(valid, 1.0 / mu_a, torch.ones_like(v))
    if inv_mu_cap > 0:
        inv_mu = inv_mu.clamp(max=inv_mu_cap)
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
        w_c = torch.minimum(w, c[t]) if torch.is_tensor(c) else torch.clamp(w, max=c)
        our_vt = (v[t] + torch.clamp(w, max=rho) * (r_unc + c_next_value - v[t])
                  + w_c * (c_next_vt - c_next_value))
        our_s = inv_mu[t] * (disc + c_is * c_next_vt - v[t])
        vt[t] = torch.where(ours, our_vt, zeros)
        s[t] = torch.where(ours, our_s, zeros)
        # 次（1つ前の時刻）へ渡す carry。**無効なステップ（どちらの手番でもない）では carry をそのまま通す**
        # （2026-11-10。それまでは 0 に戻していた）。局の後ろの埋め草では carry は初期値のままなので結果は同じ。
        # 局の途中に隙間がある batch（相手の手番を記録しない `collect(opponents=...)` = `--league-device gpu`）では、
        # 0 に戻すと終局の報酬が各局の最後の学習側の手番にしか届かなかった（docs/rl-learning-speed.md 9.8）。
        # `_v_trace_chunk`（有効な手番だけを進める）はもともと隙間を通す
        n_reward = torch.where(ours, zeros, torch.where(opp, eta_reg_entropy[t] + cs[t] * disc, c_reward))
        n_reward_unc = torch.where(ours, zeros, torch.where(opp, r_unc, c_reward_unc))
        n_next_value = torch.where(ours, v[t], c_next_value)
        n_next_vt = torch.where(ours, our_vt, c_next_vt)
        n_is = torch.where(ours, ones, torch.where(opp, cs[t] * c_is, c_is))
        c_reward, c_reward_unc, c_next_value, c_next_vt, c_is = (
            n_reward, n_reward_unc, n_next_value, n_next_vt, n_is)
    return vt, s


# ---------------------------------------------------------------- 損失


def loss_terms(logits, v, legal, action, player, lpr, vt, s, counts, eta, beta, clip=10_000.0, policy_loss="neurd"):
    """有効なステップの集まり（平らにした [N]）について、損失の和を返す。

    counts[k] はバッチ全体でのプレイヤー k の手番数（参照は損失をこれで割る）。チャンクに
    分けて呼んでも、全体をまとめて呼んだのと同じ値の和になる。
    lpr は勾配を止めた正則化の対数比 [N, A]、vt / s は v-trace の結果 [N]。

    `policy_loss="pg"` は NeuRD の押し出し（アドバンテージそのもの）に各手の π を掛ける（参照には無い。
    docs/rl-learning-speed.md）。η の項は −η·KL(π‖π_reg) のロジットでの勾配、s の項は選んだ手の
    (1[b=a] − π_b)·(結果 − v)（集めた方策で更新するので π = μ）になり、まれな手の 1/μ が消える。
    NeuRD の「確率の低い手でも押し出しが消えない」性質は失う
    """
    norm = torch.where(player == 0, 1.0 / max(counts[0], 1), 1.0 / max(counts[1], 1))
    loss_v = (norm * (v - vt) ** 2).sum()

    with torch.no_grad():
        pi = legal_policy(logits, legal)
        pi_taken = pi.gather(1, action[:, None]).squeeze(1)
        mean_lpr = (pi * lpr).sum(-1, keepdim=True)
        onehot = F.one_hot(action, logits.shape[-1]).to(logits.dtype)
        adv = -eta * (lpr - mean_lpr) + s[:, None] * (onehot - pi_taken[:, None])
        if policy_loss == "pg":
            adv = adv * pi
        elif policy_loss != "neurd":
            raise ValueError(f"policy_loss {policy_loss}（neurd / pg）")
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
    # 真実の盤面を予測する補助ヘッドの損失の重み（0 = 使わない）。損失は1決定点あたりの
    # 「自駒以外のマスの交差エントロピーの平均」の決定点平均（loss_v・nerd と同じく標本の平均の単位）
    aux_truth: float = 0.0
    # Adam の1次モーメントの係数（参照は 0 = モーメンタムなし）。学習の速さの実験用（docs/rl-learning-speed.md）
    adam_b1: float = 0.0
    # v-trace の s に掛かる 1/μ の上限（0 = 上限なし = 参照と同じ）。学習の速さの実験用
    inv_mu_cap: float = 0.0
    # 方策の損失（neurd = 参照 / pg = 押し出しに π を掛ける）。学習の速さの実験用
    policy_loss: str = "neurd"
    # 終盤だけの λ: (時刻, λ)。collect の時刻（両者の試行の通し番号。反則を含む）がこれ以上の手番だけ c = λ、
    # それより前は c_vtrace。(0, 1.0) = 使わない。学習の速さの実験用
    c_ply: tuple = (0, 1.0)


class RNaDLearner:
    """現在のネット・ターゲット・正則化方策2つ（prev / prev_）を持ち、1バッチで1回更新する。

    BatchNorm は**常に eval モード**で使う（初期化に使う模倣ネットの統計に固定し、BN は
    アフィン変換として学習する）。行動・学習・評価で統計が変わらないようにするため。
    """

    def __init__(self, net, cfg: RNaDConfig, device, packed: bool = False, device_batch: bool = False,
                 fused: bool = False):
        self.cfg = cfg
        self.device = device
        # packed = 方策の計算を合法手に詰めて行い、正則化方策の対数確率を1回目から2回目へ持ち越す
        # （`pack_legal`）。**数式は同じ**（浮動小数の足し順が違うだけ）で、2回目の prev / prev_ の
        # 順伝播と [n, 11259] の要素演算がなくなる。設定（cfg）には入れない = checkpoint の互換に影響しない
        self.packed = packed
        # device_batch = batch を更新の頭で1回だけデバイスへ送り、チャンクごとの転送と同期をなくす（`_passes_resident`）。
        # fused = 1回目の現在のネットの順伝播を2回目へまとめる（`_passes_fused`）。どちらも docs/rl-learning-speed.md
        # 「さらに速くする」の S2 / S3 で、結果は従来とビット単位で同じ（CPU）。packed と同じく設定には入れない
        if device_batch and not packed:
            raise ValueError("device_batch は packed と一緒に使う")
        self.device_batch = device_batch
        self.fused = fused
        self.net = net.to(device).eval()
        self.target = copy.deepcopy(self.net).eval()
        self.prev = copy.deepcopy(self.net).eval()
        self.prev_ = copy.deepcopy(self.net).eval()
        for m in (self.target, self.prev, self.prev_):
            m.requires_grad_(False)
        # 参照: adam(b1=0, b2=0.999, eps=1e-7) → scale(-lr) → clip(clip_gradient)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, betas=(cfg.adam_b1, 0.999), eps=1e-7)
        self.schedule = EntropySchedule(cfg.schedule_sizes, cfg.schedule_repeats)
        self.step_count = 0
        # 補助ヘッドは使うときだけ作る（使わない run の checkpoint・乱数の消費は従来と同じ）
        self.aux = None
        self.aux_opt = None

    def _ensure_aux(self):
        if self.aux is None:
            from model import TruthHead
            self.aux = TruthHead(self.net.config["channels"]).to(self.device)
            self.aux_opt = torch.optim.Adam(self.aux.parameters(), lr=self.cfg.lr, betas=(0.0, 0.999), eps=1e-7)

    def _chunks(self, n):
        for i in range(0, n, self.cfg.chunk):
            yield slice(i, min(i + self.cfg.chunk, n))

    def _lpr_mix(self, obs, legal, alpha, sel=None):
        """正則化方策の対数確率（2つの前の版の α 混合）。`sel` はロジットを詰める関数（packed のとき）"""
        sel = sel or (lambda x: x)
        # 重みが 0 の側は順伝播しない（段階の後半は α = 1 で prev_ が要らない）。対数確率は合法手で有限なので
        # 1.0·x + 0.0·y = x がビット単位で成り立ち、結果は混ぜた場合と同じ
        if alpha >= 1.0:
            return legal_log_policy(sel(forward(self.prev, obs)[0]), legal)
        if alpha <= 0.0:
            return legal_log_policy(sel(forward(self.prev_, obs)[0]), legal)
        lp_prev = legal_log_policy(sel(forward(self.prev, obs)[0]), legal)
        lp_prev_ = legal_log_policy(sel(forward(self.prev_, obs)[0]), legal)
        return alpha * lp_prev + (1 - alpha) * lp_prev_

    def _chunk_view(self, batch, sl):
        """チャンクの (観測, マスク, 行動, ロジットを選ぶ関数)。packed なら合法手に詰めたもの。
        batch のマスクは密（`legal`）でも添字（`legal_cols` / `legal_counts`、`collect(compact_legal=True)`）でもよい"""
        dev = self.device
        obs = batch["obs"][sl].to(dev, torch.float32)
        act = batch["action"][sl].to(dev)
        if "legal" in batch:
            legal = batch["legal"][sl].to(dev)
            if not self.packed:
                return obs, legal, act, (lambda x: x)
            idx, pmask, apos = pack_legal(legal, act)
        else:
            ptr = batch.get("_legal_ptr")
            if ptr is None:
                ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(batch["legal_counts"].long(), 0)])
                batch["_legal_ptr"] = ptr
            counts = batch["legal_counts"][sl].to(dev)
            cols = batch["legal_cols"][int(ptr[sl.start]):int(ptr[sl.stop])].to(dev)
            idx, pmask, apos = pack_csr(cols, counts, act)
            if not self.packed:
                legal = torch.zeros(len(act), NUM_ACTIONS, dtype=torch.bool, device=dev)
                legal.scatter_(1, idx, pmask)
                return obs, legal, act, (lambda x: x)
        return obs, pmask, apos, (lambda x: x.gather(1, idx))

    def update(self, batch, reward_scale=1.0):
        """batch（`collect` の戻り値）で1回更新する。ログ用の dict を返す。

        `reward_scale` > 1 なら**報酬と η を同じ倍率で縮める**（issue #4）。スタイル特化のボーナス込みの
        終局報酬は ±1 を超え、価値の頭（tanh、−1〜1）が勝ち側で上限に張り付く。報酬だけを縮めると
        正則化（η · log(π/π_reg)）が相対的に強く効くので、η も同じ倍率で割って釣り合いを保つ
        （= 報酬・正則化・価値をまとめて単位を変えただけで、学習する方策の目的は変わらない）。
        呼び出し側は「1 + ボーナスの最大値」（λ の予定全体での入玉の λ の最大 + 固めの λ の最大）を、
        **予定全体で固定して**渡す（区切りごとに変えると価値の目標の単位が変わり、ゆっくり追従する
        ターゲットネットの v が古い単位のまま残る）。保存する設定（`self.cfg`）の η は縮める前の値のまま。

        既存の checkpoint（s=1 で学んだもの）から s>1 で再開すると、Adam の2次モーメント（b2=0.999）が
        古い勾配の大きさのままなので、最初の約1,000更新は実効の学習率が最大で 1/s 程度まで下がる

        `device_batch` / `fused`（`__init__`）は計算の持ち方だけを変える。勾配・更新後のネット・ログの値は従来と
        ビット単位で同じ（CPU。`tests/test_fast_update.py`）"""
        cfg = self.cfg
        if reward_scale != 1.0:
            if reward_scale < 1.0:
                raise ValueError(f"reward_scale {reward_scale} は 1 以上")
            cfg = replace(cfg, eta=cfg.eta / reward_scale)
            batch = {**batch, "reward": batch["reward"] / reward_scale}
        alpha, update_reg = self.schedule(self.step_count)
        n = len(batch["action"])
        if self.fused:
            pi_a, vt, s, sums = self._passes_fused(batch, cfg, alpha)
        elif self.device_batch:
            pi_a, vt, s, sums = self._passes_resident(batch, cfg, alpha)
        else:
            pi_a, vt, s, sums = self._passes(batch, cfg, alpha)
        use_aux = cfg.aux_truth > 0
        self.opt.step()
        if use_aux:
            self.aux_opt.step()
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
        out = {"loss_v": sums["loss_v"], "loss_nerd": sums["loss_nerd"], "alpha": alpha,
               "pi_top_mean": sums["top"] / n, "pi_top_gt_0999": sums["top_hi"] / n, "entropy_mean": sums["ent"] / n,
                "reg_update": bool(update_reg), "samples": n,
                "ratio_dev_mean": float((ratio - 1).abs().mean()),
                "ratio_dev_max": float((ratio - 1).abs().max()),
                "v_target_mean": float(vt.mean()), "s_abs_mean": float(s.abs().mean())}
        if use_aux:
            out["loss_aux"] = sums["loss_aux"]  # 重みを掛ける前の値（決定点平均の交差エントロピー）
        return out

    def _passes(self, batch, cfg, alpha):
        """従来の更新の中身: 1回目（勾配なし）→ v-trace → 2回目（勾配つき）で勾配を積む。
        (π_a, vt, s, ログ用の和) を返す（π_a・vt・s は CPU の [N]）"""
        dev = self.device
        n = len(batch["action"])

        # 1回目: 勾配なしで v-trace の入力（π_a・Σπ·lpr・ターゲットの v）を作る
        pi_a = torch.zeros(n)
        # 見張り（方策が決定的に固まっていないか。PG・1/μ の上限で NeuRD の「確率の低い手でも押し出しが消えない」
        # 性質が弱まると、ここが先に動く）: 最善手の確率の平均・0.999 を超える決定点の数・エントロピーの和
        top_sum = top_hi = ent_sum = 0.0
        ent = torch.zeros(n)
        v_tgt = torch.zeros(n)
        mix_cache = []  # packed のとき、チャンクごとの正則化方策の対数確率（詰めたもの。2回目で使い回す）
        with torch.no_grad():
            for sl in self._chunks(n):
                obs, legal, act, sel = self._chunk_view(batch, sl)
                logits = sel(forward(self.net, obs)[0])
                log_pi = legal_log_policy(logits, legal)
                pi = log_pi.exp() * legal
                mix = self._lpr_mix(obs, legal, alpha, sel)
                if self.packed:
                    mix_cache.append(mix)
                lpr = log_pi - mix
                pi_a[sl] = pi.gather(1, act[:, None]).squeeze(1).cpu()
                ent[sl] = (pi * lpr).sum(-1).cpu()
                top = pi.max(1).values
                top_sum += float(top.sum())
                top_hi += float((top > 0.999).sum())
                ent_sum += float(-(pi * log_pi).sum())
                v_tgt[sl] = forward(self.target, obs)[1].cpu()

        vt, s = self._v_trace(batch, cfg, pi_a, ent, v_tgt)
        player_flat = batch["player"][batch["t"], batch["b"]].long()
        counts = [int((player_flat == 0).sum()), int((player_flat == 1).sum())]

        # 2回目: 勾配つきで損失を積む（チャンクごとに backward して勾配を足し合わせる）
        use_aux = cfg.aux_truth > 0
        if use_aux:
            if "truth" not in batch:
                raise ValueError("aux_truth > 0 には collect(..., truth=True) の batch が要る")
            self._ensure_aux()
            self.aux_opt.zero_grad(set_to_none=True)
        self.opt.zero_grad(set_to_none=True)
        tot_v = tot_nerd = tot_aux = 0.0
        for ci, sl in enumerate(self._chunks(n)):
            obs, legal, act, sel = self._chunk_view(batch, sl)
            if use_aux:
                logits, v, aux_logits = forward_aux(self.net, self.aux, obs)
            else:
                logits, v = forward(self.net, obs)
            logits = sel(logits)
            with torch.no_grad():
                mix = mix_cache[ci] if self.packed else self._lpr_mix(obs, legal, alpha)
                lpr = legal_log_policy(logits, legal) - mix
            lv, ln = loss_terms(logits, v, legal, act, player_flat[sl].to(dev), lpr,
                                vt[sl].to(dev), s[sl].to(dev), counts, cfg.eta, cfg.beta, cfg.clip,
                                cfg.policy_loss)
            loss = lv + ln
            if use_aux:
                la = aux_truth_loss(aux_logits, batch["truth"][sl].to(dev, torch.long)).sum() / n
                loss = loss + cfg.aux_truth * la
                tot_aux += la.item()
            loss.backward()
            tot_v += lv.item()
            tot_nerd += ln.item()
        return pi_a, vt, s, {"loss_v": tot_v, "loss_nerd": tot_nerd, "loss_aux": tot_aux,
                             "top": top_sum, "top_hi": top_hi, "ent": ent_sum}

    def _v_trace(self, batch, cfg, pi_a, ent, v_tgt):
        """全局の v-trace（[T, B] の格子で時間方向に後ろから）。π_a・ent（Σπ·lpr）・v_tgt は平らな [N]（CPU）"""
        n = len(batch["action"])
        t_idx, b_idx = batch["t"], batch["b"]
        T, B = batch["valid"].shape

        def grid(x):
            g = torch.zeros(T, B)
            g[t_idx, b_idx] = x
            return g

        valid = batch["valid"]
        player = batch["player"]
        mu_g = torch.where(valid, grid(batch["mu_a"]), torch.ones(T, B))
        vt = torch.zeros(n)
        s = torch.zeros(n)
        c_arg = cfg.c_vtrace
        if tuple(cfg.c_ply) != (0, 1.0):
            ply0, lam = cfg.c_ply
            c_arg = torch.where(torch.arange(T)[:, None] >= int(ply0), float(lam), float(cfg.c_vtrace)).expand(T, B)
        for k in (0, 1):
            vt_k, s_k = v_trace_scalars(grid(v_tgt), valid, player, grid(pi_a), mu_g, grid(ent),
                                        batch["reward"][:, :, k], k, cfg.eta, c=c_arg,
                                        inv_mu_cap=cfg.inv_mu_cap)
            mine = player[t_idx, b_idx] == k
            vt = torch.where(mine, vt_k[t_idx, b_idx], vt)
            s = torch.where(mine, s_k[t_idx, b_idx], s)
        return vt, s

    def _passes_resident(self, batch, cfg, alpha):
        """`_passes` と同じ計算を、batch をデバイスに置いたまま回す（`device_batch`。S2）。
        観測・行動・合法手は最初に1回だけ送り、チャンクの詰め（pack）は1回目に作って2回目で使い回す。
        π_a・Σπ·lpr・ターゲットの v・見張りの和・損失はデバイスに貯めて、1回目と2回目の終わりに1回ずつだけ CPU へ戻す
        （従来はチャンクごとに .cpu() / float() / .item() で同期していた）。足す順は従来と同じなので和もビット単位で同じ"""
        dev = self.device
        n = len(batch["action"])
        res = _ResidentBatch(self, batch)
        chunks = list(self._chunks(n))
        pi_a_d = torch.zeros(n, device=dev)
        ent_d = torch.zeros(n, device=dev)
        v_tgt_d = torch.zeros(n, device=dev)
        stats, mix_cache = [], []
        with torch.no_grad():
            for sl in chunks:
                obs, legal, act, sel = res.chunk(sl)
                logits = sel(forward(self.net, obs)[0])
                log_pi = legal_log_policy(logits, legal)
                pi = log_pi.exp() * legal
                mix = self._lpr_mix(obs, legal, alpha, sel)
                mix_cache.append(mix)
                lpr = log_pi - mix
                pi_a_d[sl] = pi.gather(1, act[:, None]).squeeze(1)
                ent_d[sl] = (pi * lpr).sum(-1)
                top = pi.max(1).values
                # 0.999 を超える数（チャンク ≤ 2^24 行）は float32 でも正確
                stats.append(torch.stack([top.sum(), (top > 0.999).sum().to(top.dtype), -(pi * log_pi).sum()]))
                v_tgt_d[sl] = forward(self.target, obs)[1]
            res.check()
            pi_a, ent, v_tgt = pi_a_d.cpu(), ent_d.cpu(), v_tgt_d.cpu()
            top_sum = top_hi = ent_sum = 0.0
            for a, b, c in torch.stack(stats).cpu().tolist():
                top_sum += a
                top_hi += b
                ent_sum += c

        vt, s = self._v_trace(batch, cfg, pi_a, ent, v_tgt)
        player_flat = batch["player"][batch["t"], batch["b"]].long()
        counts = [int((player_flat == 0).sum()), int((player_flat == 1).sum())]

        use_aux = cfg.aux_truth > 0
        if use_aux:
            if res.truth is None:
                raise ValueError("aux_truth > 0 には collect(..., truth=True) の batch が要る")
            self._ensure_aux()
            self.aux_opt.zero_grad(set_to_none=True)
        self.opt.zero_grad(set_to_none=True)
        vt_d, s_d, pl_d = vt.to(dev), s.to(dev), player_flat.to(dev)
        losses = []
        for ci, sl in enumerate(chunks):
            obs, legal, act, sel = res.chunk(sl)
            if use_aux:
                logits, v, aux_logits = forward_aux(self.net, self.aux, obs)
            else:
                logits, v = forward(self.net, obs)
            logits = sel(logits)
            with torch.no_grad():
                lpr = legal_log_policy(logits, legal) - mix_cache[ci]
            lv, ln = loss_terms(logits, v, legal, act, pl_d[sl], lpr, vt_d[sl], s_d[sl], counts,
                                cfg.eta, cfg.beta, cfg.clip, cfg.policy_loss)
            loss = lv + ln
            parts = [lv.detach(), ln.detach()]
            if use_aux:
                la = aux_truth_loss(aux_logits, res.truth[sl]).sum() / n
                loss = loss + cfg.aux_truth * la
                parts.append(la.detach())
            loss.backward()
            losses.append(torch.stack(parts))
        return pi_a, vt, s, {**_sum_losses(losses, use_aux), "top": top_sum, "top_hi": top_hi, "ent": ent_sum}

    def _passes_fused(self, batch, cfg, alpha):
        """`_passes` と同じ勾配を、1回目の現在のネットの順伝播を2回目へまとめて作る（`fused`。S3）。

        batch の決定点は時刻の昇順（collect が手番ごとに足す）で、各局の手番は時刻 0 から途切れずに続く（自己対局と
        Rust の相手の局。相手のネットに指させた局〔`opponents`〕は相手の手番が隙間になるので使えない）。
        v-trace の後ろ向きの再帰は「時刻 t の出力が t 以降の入力だけで決まる」ので、チャンクを**後ろから**回し、
        局ごとの持ち越しを [B] で持てば、チャンクごとに
            現在のネットの順伝播（勾配つき）→ 正則化・ターゲットの順伝播 → そのチャンクの時刻の v-trace → 損失 → backward
        を1回で済ませられる（決定点あたり順伝播1回分が消える）。v-trace の1ステップは `v_trace_scalars` と同じ式を
        同じ順に計算するので vt・s はビット単位で同じ。チャンクの勾配は取っておき、最後に**従来と同じ順（チャンクの
        昇順）に足す**ので、全パラメータの勾配もビット単位で同じになる（チャンク数 × パラメータ分のメモリを使う。
        64ch×6・6.5万決定点で約130MB）。ログの和も同じ順に足す"""
        dev = self.device
        n = len(batch["action"])
        t_idx, b_idx = batch["t"], batch["b"]
        T, B = batch["valid"].shape
        valid, player = batch["valid"], batch["player"]
        if n > 1 and not bool((t_idx[1:] >= t_idx[:-1]).all()):
            raise ValueError("fused: 決定点が時刻の昇順でない")
        if not bool((valid == (torch.arange(T)[:, None] < valid.sum(0)[None, :])).all()):
            raise ValueError("fused: 手番が時刻 0 から途切れずに続かない局がある（相手のネットに指させた局は使えない）")
        res = _ResidentBatch(self, batch, cache=False) if self.device_batch else None
        view = res.chunk if res is not None else (lambda sl: self._chunk_view(batch, sl))
        chunks = list(self._chunks(n))
        player_flat = player[t_idx, b_idx].long()
        counts = [int((player_flat == 0).sum()), int((player_flat == 1).sum())]
        pl_d = player_flat.to(dev)
        use_aux = cfg.aux_truth > 0
        if use_aux:
            if "truth" not in batch:
                raise ValueError("aux_truth > 0 には collect(..., truth=True) の batch が要る")
            self._ensure_aux()
            self.aux_opt.zero_grad(set_to_none=True)
        self.opt.zero_grad(set_to_none=True)
        params = list(self.net.parameters()) + (list(self.aux.parameters()) if use_aux else [])

        mu = batch["mu_a"]
        reward = batch["reward"]
        c_ply = tuple(cfg.c_ply) != (0, 1.0)
        # 局ごとの持ち越し（v_trace_scalars の carry）をプレイヤー k ごとに: 報酬・報酬（割り引かない）・次の価値・
        # 次の v-trace の目標・重要度の積
        carry = {k: [torch.zeros(B), torch.zeros(B), torch.zeros(B), torch.zeros(B), torch.ones(B)] for k in (0, 1)}
        pi_a, vt, s = torch.zeros(n), torch.zeros(n), torch.zeros(n)
        grads, losses, stats = [None] * len(chunks), [None] * len(chunks), [None] * len(chunks)
        for ci in reversed(range(len(chunks))):
            sl = chunks[ci]
            obs, legal, act, sel = view(sl)
            if use_aux:
                logits, v, aux_logits = forward_aux(self.net, self.aux, obs)
            else:
                logits, v = forward(self.net, obs)
            logits = sel(logits)
            with torch.no_grad():
                log_pi = legal_log_policy(logits, legal)
                pi = log_pi.exp() * legal
                lpr = log_pi - self._lpr_mix(obs, legal, alpha, sel)
                top = pi.max(1).values
                stats[ci] = torch.stack([top.sum(), (top > 0.999).sum().to(top.dtype), -(pi * log_pi).sum()])
                pa_c, ent_c, vv_c = torch.stack([pi.gather(1, act[:, None]).squeeze(1), (pi * lpr).sum(-1),
                                                 forward(self.target, obs)[1]]).cpu()
            pi_a[sl] = pa_c
            vt_c, s_c = self._v_trace_chunk(cfg, carry, t_idx[sl], b_idx[sl], player, reward, pa_c, mu[sl],
                                            ent_c, vv_c, c_ply)
            vt[sl], s[sl] = vt_c, s_c
            vt_d, s_d = torch.stack([vt_c, s_c]).to(dev)
            lv, ln = loss_terms(logits, v, legal, act, pl_d[sl], lpr, vt_d, s_d, counts,
                                cfg.eta, cfg.beta, cfg.clip, cfg.policy_loss)
            loss = lv + ln
            parts = [lv.detach(), ln.detach()]
            if use_aux:
                truth = res.truth[sl] if res is not None else batch["truth"][sl].to(dev, torch.long)
                la = aux_truth_loss(aux_logits, truth).sum() / n
                loss = loss + cfg.aux_truth * la
                parts.append(la.detach())
            loss.backward()
            losses[ci] = torch.stack(parts)
            grads[ci] = [p.grad for p in params]
            for p in params:
                p.grad = None
        if res is not None:
            res.check()
        # 従来の2回目と同じ順（チャンクの昇順）に勾配を足す（AccumulateGrad の grad += new_grad と同じ演算）
        for j, p in enumerate(params):
            g = None
            for ci in range(len(chunks)):
                x = grads[ci][j]
                if x is None:
                    continue
                if g is None:
                    g = x
                else:
                    g += x
            p.grad = g
        top_sum = top_hi = ent_sum = 0.0
        for a, b, c in torch.stack(stats).cpu().tolist():
            top_sum += a
            top_hi += b
            ent_sum += c
        return pi_a, vt, s, {**_sum_losses(losses, use_aux), "top": top_sum, "top_hi": top_hi, "ent": ent_sum}

    @staticmethod
    def _v_trace_chunk(cfg, carry, ct, cb, player, reward, pi_a, mu, ent, v, c_ply):
        """`_passes_fused` の v-trace: チャンク（時刻 ct・局 cb の決定点）の時刻を後ろから1つずつ進め、`carry` を更新する。
        式と計算の順は `v_trace_scalars` の1ステップと同じ（有効な手番だけなので ok = 真、opp = ours の否定）"""
        cs_all = pi_a / mu
        inv_mu_all = 1.0 / mu
        if cfg.inv_mu_cap > 0:
            inv_mu_all = inv_mu_all.clamp(max=cfg.inv_mu_cap)
        vt = torch.zeros(len(ct))
        s = torch.zeros(len(ct))
        for t in range(int(ct[-1]), int(ct[0]) - 1, -1):
            rows = (ct == t).nonzero().squeeze(1)
            bs = cb[rows]
            pl = player[t, bs]
            cs = cs_all[rows]
            vv = v[rows]
            if c_ply:
                ply0, lam = cfg.c_ply
                c_t = torch.full((len(rows),), float(lam) if t >= int(ply0) else float(cfg.c_vtrace))
            for k in (0, 1):
                c_reward, c_reward_unc, c_next_value, c_next_vt, c_is = (x[bs] for x in carry[k])
                ours = pl == k
                eta_reg_entropy = -cfg.eta * ent[rows] * torch.where(ours, 1.0, -1.0)
                r = reward[t, bs, k]
                r_unc = r + c_reward_unc + eta_reg_entropy
                disc = r + c_reward
                w = cs * c_is
                w_c = torch.minimum(w, c_t) if c_ply else torch.clamp(w, max=cfg.c_vtrace)
                our_vt = (vv + torch.clamp(w, max=math.inf) * (r_unc + c_next_value - vv)
                          + w_c * (c_next_vt - c_next_value))
                our_s = inv_mu_all[rows] * (disc + c_is * c_next_vt - vv)
                vt[rows] = torch.where(ours, our_vt, vt[rows])
                s[rows] = torch.where(ours, our_s, s[rows])
                zeros = torch.zeros(len(rows))
                new = (torch.where(ours, zeros, eta_reg_entropy + cs * disc), torch.where(ours, zeros, r_unc),
                       torch.where(ours, vv, c_next_value), torch.where(ours, our_vt, c_next_vt),
                       torch.where(ours, torch.ones(len(rows)), cs * c_is))
                for x, y in zip(carry[k], new):
                    x[bs] = y
        return vt, s

    def state_dict(self):
        """再開に要る全部: 4つのネット・optimizer・step・設定・ネットの形・乱数状態"""
        return {"net": self.net.state_dict(), "target": self.target.state_dict(),
                "prev": self.prev.state_dict(), "prev_": self.prev_.state_dict(),
                "opt": self.opt.state_dict(), "step": self.step_count,
                "cfg": dict(self.cfg.__dict__), "config": dict(self.net.config),
                "rng": rng_state(self.device),
                **({"aux": self.aux.state_dict(), "aux_opt": self.aux_opt.state_dict()}
                   if self.aux is not None else {})}

    def load_state_dict(self, d):
        self.net.load_state_dict(d["net"])
        self.target.load_state_dict(d["target"])
        self.prev.load_state_dict(d["prev"])
        self.prev_.load_state_dict(d["prev_"])
        self.opt.load_state_dict(d["opt"])
        self.step_count = d["step"]
        if "aux" in d:
            self._ensure_aux()
            self.aux.load_state_dict(d["aux"])
            self.aux_opt.load_state_dict(d["aux_opt"])
        if "rng" in d:
            set_rng_state(d["rng"], self.device)

    def reset_optimizer_state(self):
        """Adam の状態（1次・2次モーメントと step）を捨てる。**途中から更新の式を変える A/B の両腕で使う**:
        checkpoint の2次モーメントは今の式の勾配（本線ではまれな手の 1/μ のスパイク）の大きさを覚えていて、
        分散を減らす変更をすると実効の学習率が何千更新も小さいまま残る（b2 = 0.999。docs/rl-learning-speed.md）"""
        self.opt.state.clear()
        if self.aux_opt is not None:
            self.aux_opt.state.clear()

    def apply_cfg(self, cfg: RNaDConfig):
        """設定を差し替える（再開時に明示的に上書きするとき。optimizer の学習率も合わせる）"""
        self.cfg = cfg
        for g in self.opt.param_groups:
            g["lr"] = cfg.lr
            g["betas"] = (cfg.adam_b1, g["betas"][1])
        if self.aux_opt is not None:
            for g in self.aux_opt.param_groups:
                g["lr"] = cfg.lr
        self.schedule = EntropySchedule(cfg.schedule_sizes, cfg.schedule_repeats)


def _sum_losses(losses, use_aux):
    """チャンクごとの損失 [lv, ln(, la)]（デバイス上）を1回で CPU へ戻し、従来（チャンクごとの .item() を順に足す）と
    同じ順に足す。float32 の値を double で足すのも従来と同じなので、和はビット単位で同じ"""
    tot = [0.0, 0.0, 0.0]
    for row in torch.stack(losses).cpu().tolist():
        for i, x in enumerate(row):
            tot[i] += x
    out = {"loss_v": tot[0], "loss_nerd": tot[1]}
    if use_aux:
        out["loss_aux"] = tot[2]
    return out


class _ResidentBatch:
    """batch をデバイスへ1回だけ送って持つ（`RNaDLearner(device_batch=True)`）。`collect(compact_legal=True)` の batch。

    観測は fp16 のまま送り、チャンクごとにデバイス上で float32 にする（従来の `_chunk_view` の CPU からの変換と同じ値）。
    合法手の詰め（pack）は行の数 K を CPU の個数から決めて同期せずに作り、`cache` なら1回目と2回目で使い回す。
    マスク外の手の検査は `check()` でまとめて1回"""

    def __init__(self, learner, batch, cache=True):
        if "legal_cols" not in batch:
            raise ValueError("device_batch は collect(compact_legal=True) の batch に限る")
        dev = learner.device
        self.obs = batch["obs"].to(dev)
        self.act = batch["action"].to(dev)
        self.cols = batch["legal_cols"].to(dev)
        self.counts_cpu = batch["legal_counts"]
        self.counts = self.counts_cpu.to(dev)
        self.ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(self.counts_cpu.long(), 0)])
        self.truth = batch["truth"].to(dev, torch.long) if "truth" in batch else None
        self.cache = cache
        self.packs = {}
        self.oks = []

    def chunk(self, sl):
        """`RNaDLearner._chunk_view`（packed）と同じ (観測, 詰めたマスク, 選んだ手の位置, ロジットを選ぶ関数)"""
        obs = self.obs[sl].to(torch.float32)
        pk = self.packs.get(sl.start)
        if pk is None:
            a0, a1 = int(self.ptr[sl.start]), int(self.ptr[sl.stop])
            idx, pmask, apos, ok = _pack_csr(self.cols[a0:a1], self.counts[sl], self.act[sl],
                                             int(self.counts_cpu[sl].max()), a1 - a0)
            self.oks.append(ok)
            pk = (idx, pmask, apos)
            if self.cache:
                self.packs[sl.start] = pk
        idx, pmask, apos = pk
        return obs, pmask, apos, (lambda x: x.gather(1, idx))

    def check(self):
        if self.oks and not bool(torch.stack(self.oks).all()):
            raise ValueError("マスク外の手が batch にある")


def cfg_diff(saved: dict, cfg: RNaDConfig) -> dict:
    """checkpoint の設定と今の設定の差（tuple / list の違いは無視する）。
    checkpoint に無い項目（後から足した設定）は既定値だったものとして比べる"""
    now = cfg.__dict__
    default = type(cfg)().__dict__
    norm = lambda v: list(v) if isinstance(v, (tuple, list)) else v  # noqa: E731
    return {k: (saved.get(k, default[k]), now[k]) for k in now
            if norm(saved.get(k, default[k])) != norm(now[k])}


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
    return _gumbel_sample(logits, legal, generator)


@torch.no_grad()
def _sample_actions_extras(net, obs, mask, device, generator=None):
    """`sample_actions` と同じ手と確率（同じ乱数の消費）に加えて、行動したネットの価値と合法手全体の対数確率
    （行ごとに行動番号の昇順でつなげた CSR の並び）を返す（`collect(actor_extras=True)`。MMD 学習器が使う）"""
    x = torch.from_numpy(obs).to(device, non_blocking=True).half().float()
    logits, v = forward(net, x)
    legal = torch.from_numpy(mask).to(device, non_blocking=True)
    a, mu = _gumbel_sample(logits, legal, generator)
    lsm = logits.float().masked_fill(~legal, float("-inf")).log_softmax(-1)
    return a, mu, v.float().cpu(), lsm[legal].cpu()


def _gumbel_sample(logits, legal, generator):
    """`sample_actions` の後半（マスク内の softmax から引く）。`collect(fast=...)` も同じ関数を通す"""
    masked = logits.float().masked_fill(~legal, float("-inf"))
    # Gumbel-max（softmax からのサンプリングと同じ分布）。torch.multinomial は MPS で
    # 確率 0 の手（マスク外）を選ぶことがあったので使わない（2026-09-28 実測）
    u = torch.rand(masked.shape, device=masked.device, generator=generator).clamp_(1e-20, 1.0)
    action = (masked - torch.log(-torch.log(u))).argmax(-1)
    prob = masked.softmax(-1).gather(1, action[:, None]).squeeze(1)
    return action.cpu(), prob.cpu()


def pad_planes(net, planes: int):
    """入力を `planes` 枚に広げた、**同じ関数**のネット（足した stem の列は 0。観測の先頭が同じなので、広い観測を渡しても
    元のネットと同じ出力になる）。v3 / v4 の「使わない組を 0 にしてから読む」マスクは外す（その列の重みはもともと 0 のまま
    なので結果は同じ）。相手のネットをまとめて推論する `OpponentPool` が、入力の版の違う相手を1つに積むために使う"""
    from model import EXTENDED_PLANES, EXTRA_PLANES, inference_net
    cfg = net.config
    if cfg["planes"] > planes:
        raise ValueError(f"入力 {cfg['planes']}枚のネットを {planes}枚へは狭められない")
    if cfg["planes"] == planes and getattr(net, "in_off", None) is None:
        return net
    # 広げた側は、その枚数に入る組を全部読む（マスクなし）。v4（102枚）で組を空にすると、v3 / v4 の相手が読む組まで
    # 0 にしてしまう
    fits = [n for n, (s, k) in EXTRA_PLANES.items() if s + k <= planes] if planes in EXTENDED_PLANES else ()
    # 重みは下で全部書き込むので、初期化でグローバルの乱数を引かない（学習の再開で戻した乱数状態をずらさない）
    wide = inference_net({"channels": cfg["channels"], "blocks": cfg["blocks"], "planes": planes, "extras": fits})
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    w = sd["stem.weight"]
    stem = torch.zeros((w.shape[0], planes, *w.shape[2:]), dtype=w.dtype, device=w.device)
    stem[:, :w.shape[1]] = w
    sd["stem.weight"] = stem
    wide.load_state_dict(sd)
    return wide.to(w.device)


def module_shape(net) -> tuple:
    """推論の形（ネットの型・パラメータとバッファの名前・形・dtype）。同じなら `stack_module_state` で積んで
    先頭のネットの構造で推論しても、各ネット単体と同じ計算になる"""
    return (type(net),
            tuple((k, tuple(v.shape), v.dtype) for k, v in net.named_parameters()),
            tuple((k, tuple(v.shape), v.dtype) for k, v in net.named_buffers()))


class OpponentPool:
    """リーグの相手のネットを**まとめて1回の推論**で指させる（`collect` の `opp_nets` に渡す）。

    相手ごとに推論すると、手番ごとに相手の数だけ小さな推論が直列に走り、GPU の起動の待ちで
    Rust の評価モードとほとんど変わらなかった（相手12体・128局で 10〜11秒 対 12〜14秒）。
    同じ形のネットの重みを積み重ね（`torch.func.stack_module_state`）、`vmap` で全相手を一度に
    推論する。局の数が相手ごとに違うので、相手ごとの入力を最大の局数まで 0 で埋めて並べる
    （埋めた行は捨てる）。サンプリングは `sample_actions` と同じ（fp16 に丸めた入力・Gumbel-max）

    積めるのは**推論の形（層の名前と形）が同じネットどうし**だけなので、形ごとに組に分けて組ごとに積む
    （チャネル数・ブロック数が全員同じなら組は1つで、従来と同じ1回の推論）。`stack_module_state` は先頭のネットの
    層の名前だけを並べ、推論も先頭のネットの構造で走るので、形の違う相手を1つに積むと、先頭より深い相手の余分な
    ブロックが黙って落ちていた（浅い相手が後ろなら KeyError、チャネル数が違えば形の不一致で落ちる）。
    `params` / `buffers` は組ごとの積み重ねのリスト。`forward(params, buffers, x)` は全相手の並び（`names`）の
    入力 [K, m, ...] を受け、組ごとに推論して同じ並びの [K, m, A] を返す
    """

    def __init__(self, nets: dict):
        from torch.func import functional_call, stack_module_state
        self.names = list(nets)
        self.index = {n: i for i, n in enumerate(self.names)}
        # 入力の版（v1 = 86枚 / v2 = 91枚 / v3 = 101枚 / v4 = 102枚）が違う相手は、いちばん広い版にそろえてから積む
        # （`pad_planes`。足した列は 0 なので同じ関数）。それまでは版が混ざると積めずに落ちていた
        planes = max(n.config["planes"] for n in nets.values())
        self.planes = planes
        mods = [pad_planes(nets[n], planes).eval() for n in self.names]
        by_shape = {}
        for k, m in enumerate(mods):
            by_shape.setdefault(module_shape(m), []).append(k)
        self.groups = []  # [(全相手の並びでの添字, vmap した推論)]
        self.params, self.buffers = [], []
        for ks in by_shape.values():
            p, b = stack_module_state([mods[k] for k in ks])
            base = copy.deepcopy(mods[ks[0]]).to("meta")

            def one(p, b, x, base=base):
                return functional_call(base, (p, b), (x,))[0]
            self.groups.append((ks, torch.vmap(one)))
            self.params.append(p)
            self.buffers.append(b)

    def forward(self, params, buffers, x):
        """x = 全相手の並び（`names`）の入力 [K, m, planes, 9, 9] → ロジット [K, m, A]（形の組ごとに推論して戻す）"""
        if len(self.groups) == 1:
            return self.groups[0][1](params[0], buffers[0], x)  # 組が1つ（全員同じ形）なら並びもそのまま
        out = None
        for (ks, fn), p, b in zip(self.groups, params, buffers):
            idx = torch.as_tensor(ks, dtype=torch.long, device=x.device)
            y = fn(p, b, x.index_select(0, idx))
            if out is None:
                out = y.new_empty((x.shape[0], *y.shape[1:]))
            out[idx] = y
        return out

    def __getitem__(self, name):
        if name not in self.index:
            raise KeyError(name)
        return name

    @torch.no_grad()
    def sample(self, groups: dict, obs, mask, device):
        """groups = {相手の名前: 局の添字の配列}。{名前: 行動の配列} を返す"""
        K = len(self.names)
        m = max(len(bs) for bs in groups.values())
        planes = obs.shape[1:]
        x = torch.zeros((K, m, *planes), device=device)
        legal = torch.zeros((K, m, mask.shape[1]), dtype=torch.bool, device=device)
        for name, bs in groups.items():
            k = self.index[name]
            x[k, :len(bs)] = torch.from_numpy(obs[bs]).to(device, non_blocking=True).half().float()
            legal[k, :len(bs)] = torch.from_numpy(mask[bs]).to(device, non_blocking=True)
        logits = self.forward(self.params, self.buffers, x).float()  # [K, m, A]
        legal[:, :, 0] |= ~legal.any(-1)  # 埋めた行に合法手が無いと Gumbel-max が NaN になるので仮に1つ立てる
        masked = logits.masked_fill(~legal, float("-inf"))
        u = torch.rand(masked.shape, device=device).clamp_(1e-20, 1.0)
        act = (masked - torch.log(-torch.log(u))).argmax(-1).cpu()
        return {name: act[self.index[name], :len(bs)].numpy() for name, bs in groups.items()}


# 観測の自玉のプレーン（tsuitate-bot の rl::encode の ROLES 順で玉は 7 番目）。盤は手番側を
# 先手向きに正規化しているので、行 0〜2（正規化後の段 1〜3）が常に敵陣
KING_PLANE = 7
# 段階的な入玉ボーナスの節目（玉の初期位置からの前進段数）。6 = 敵陣3段目への進入
GRADED_MILESTONES = (3, 4, 5, 6)


def split_noise(spec: str):
    """相手の指定 `<名前>` か `<名前>@<ε>`（ハンデ = 相手の手番ごとに確率 ε で合法手の乱数手。tsuitate-bot の
    docs/rl-fixed-opponent.md）→ (名前, ε)。tsuitate_rl と同じ読み方: 末尾の `@` の後ろが 0〜1 の数でなければ全体が名前"""
    name, sep, eps = spec.rpartition("@")
    if sep:
        try:
            e = float(eps)
        except ValueError:
            return spec, 0.0
        if 0.0 <= e <= 1.0:
            return name, e
    return spec, 0.0


class MultiEnv:
    """複数の VecEnv（auto_reset=False）を1つの VecEnv に見せる（局の並びは渡した順につなげる）。

    リーグ学習で、自己対局の env と相手ごとの評価モードの env を `collect` に一度に渡すためのもの。
    1回の `observe` で全 env の局面を集めるので、ネットの推論は全局まとめて1回になる
    （env ごとに `collect` を回すと、相手の数だけ手番のループが直列になる）。
    評価モードの env は学習側の手番だけを返し、相手の手は `step` の中で指し進める。報酬は
    [先手, 後手] の両方が入るが、相手側の手番は記録に現れないので学習には学習側の列だけが効く。

    env は順番に呼ぶ（VecEnv は unsendable で、Python のスレッドからは呼べない）。相手ごとに env を
    分けると凍結版の相手の推論が env の数だけ直列になる（相手10体・128局で収集 2秒 → 18秒、
    2026-10-03 実測）ので、リーグの局は `VecEnv(opponents=[...])` の1つの env にまとめること
    """

    def __init__(self, envs):
        self.envs = list(envs)
        self.sizes = [e.num_envs for e in self.envs]
        self.offsets = np.cumsum([0] + self.sizes)
        self.num_envs = int(self.offsets[-1])

    def _split(self, x):
        return [x[self.offsets[i]:self.offsets[i + 1]] for i in range(len(self.envs))]

    def reset(self):
        for e in self.envs:
            e.reset()

    def alive(self):
        return np.concatenate([e.alive() for e in self.envs])

    def observe(self):
        parts = [e.observe() for e in self.envs]
        return tuple(np.concatenate([p[j] for p in parts]) for j in range(3))

    def truth(self):
        return np.concatenate([e.truth() for e in self.envs])

    def signals(self):
        return np.concatenate([e.signals() for e in self.envs])

    def forced_moves(self):
        return np.concatenate([e.forced_moves() for e in self.envs])

    def random_legal_actions(self, u):
        return np.concatenate([e.random_legal_actions(np.ascontiguousarray(x)) if (x >= 0).any()
                               else np.full(e.num_envs, -1, dtype=np.int64)
                               for e, x in zip(self.envs, self._split(u))])

    def observe_alive_f16_csr(self):
        # 生きている局だけを env の順につなげる = 全体の局の添字の昇順（VecEnv と同じ並び）
        parts = [e.observe_alive_f16_csr() for e in self.envs]
        return tuple(np.concatenate([p[j] for p in parts]) for j in range(4))

    def step(self, actions):
        rews, dones = [], []
        for e, a in zip(self.envs, self._split(actions)):
            if (a >= 0).any():
                r, d = e.step(a)
            else:  # この env の局は全部終わっている
                r, d = np.zeros((e.num_envs, 2), np.float32), np.zeros(e.num_envs, bool)
            rews.append(r)
            dones.append(d)
        return np.concatenate(rews), np.concatenate(dones)


def collect(env, net, device, max_steps=260, nyugyoku_lambda=0.0, nyugyoku_mode="enter",
            nyugyoku_zero_sum=True, opponents=None, opp_nets=None, nyugyoku_win_only=False, truth=False,
            generator=None, compact_legal=False, fast="", actor_extras=False, shaping=None, shape_mult=1.0,
            watch_signals=False, noise_rng=None):
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

    `nyugyoku_zero_sum=False` は**非ゼロ和**: 相手から −λ を引かない（前進した側に +λ だけ）。
    ゼロ和だと自己対局の両側が「前進する」と「前進させない」を同じ重みで学び、λ=1.0 を 800更新
    続けても入玉率が 2.2% で横ばいだった（arm D、2026-10-03）。非ゼロ和なら前進を止める理由は
    勝敗だけになる。R-NaD の収束保証（ゼロ和2人ゲームが前提）は外れる

    `nyugyoku_win_only=True` は**勝った局でだけ**入玉ボーナスを付ける: 節目に着いた時点では付けずに覚えておき、
    終局でその側が勝っていれば、節目に着いた手番へ同じ額を足す（負け・引き分けの局では付かない）。
    前進するたびに付ける形は「負けても玉を前に出せば得」になり、λ=4→2→1 と下げても入玉率 13〜15% のまま
    強さが戻らなかった（2026-10-06）。勝ちを条件にすると、入玉の指し方と強さが同じ向きにそろう。
    勝ち負けは終局の報酬の符号で見るので、固め・速攻のボーナス（負けた側にも正の値が乗りうる）とは併用しない

    `opponents`（長さ B、局ごとに None か (相手の名前, 学習側の色 0/1)）を渡すと、その局は
    **相手を固定した局**になる（リーグの相手を GPU で指させる。`env` は自己対局モードのまま）:
    相手の手番は `opp_nets[名前]` が同じくサンプリングで指し、**記録しない**（valid にならない）。
    相手の手番で入った報酬（相手の手で終局した局の勝敗・ボーナス）は学習側の直前の手番へ足す
    （Rust の評価モード `VecEnv(opponents=...)` で相手の手が学習側の step の中で進むのと同じ帰属）。
    入玉ボーナスも学習側だけを数える。None の局は従来どおりの自己対局で、`opponents` を渡さなければ
    結果も乱数の消費も従来と同じ。相手の名前が `<名前>@<ε>` なら**ハンデつき**: ネットは `opp_nets[<名前>]` を使い、
    相手の手番ごとに確率 ε で、その手を真実の盤面の合法手から一様に選んだ手に差し替える（`VecEnv.random_legal_actions`。
    Rust の評価モードの `<名前>@<ε>` と同じ弱め方）。その乱数は `noise_rng`（numpy の Generator。ハンデつきの局が
    あれば必須）から引き、学習側の手の乱数は消費しない

    `truth=True` なら記録する各手番の**真実の盤面のラベル**（`VecEnv.truth()`、[N, 81] int8）を
    `truth` に入れて返す（補助ヘッド用。乱数は消費しない）

    `compact_legal=True` なら合法手のマスクを密な [N, 11259] の bool でなく、**合法手の添字**（行ごとに昇順でつなげた
    `legal_cols` int16 と行ごとの個数 `legal_counts`）で持つ（768局で約 0.79GB → 約 11MB。`--pipeline` で batch が
    2つ同時にあるときのメモリ対策。`RNaDLearner.update` はどちらの形でも同じ結果になる）

    `generator` を渡すと学習側の手のサンプリングはその乱数列を使う（`PipelinedCollector` が別スレッドでも
    再現するように。渡さなければ従来どおり torch の大域の乱数）

    `actor_extras=True` なら、学習側の各手番について**行動したネットの価値** `v_actor` [N] と**合法手全体の対数確率**
    `mu_logp`（`legal_cols` と同じ CSR の並び）も返す（MMD 学習器〔`mmd.py`〕が使う。`compact_legal=True` に限り、
    相手のネットに指させる局〔opponents〕とは使えない）。乱数は消費しないので、ほかのキーは付けないときと同じ

    `shaping`（`shaping.Shaping`）を渡すと、**局の信号**（`VecEnv.signals()`: 駒得・反則・王手・手数など）の差分に重みを
    掛けた報酬を足す（`shape_mult` は重みの倍率。tsuitate-bot の docs/rl-fixed-opponent.md）。足す先は env の報酬と同じ
    （自己対局は [先手, 後手] の両方をその時刻へ、相手を固定した局は学習側の列だけを学習側の直前の手番へ）。
    `outcome` は env の勝敗のまま。`watch_signals=True`（`shaping` があれば常に）なら、局ごとの信号の監視値の元
    （`shaping.ShapeRun`）を `shape_run` に入れて返す。どちらも乱数は消費しないので、ほかのキーは付けないときと同じ。
    `fast` とは一緒に使えない

    `fast="py"` / `"rust"` は**同じ batch をビット単位で同じに作る、CPU の仕事を減らした経路**（`_collect_fast`。
    docs/rl-learning-speed.md「さらに速くする」の S1）。自己対局（opponents なし）・入玉ボーナスなし・truth なし・
    `compact_legal=True` に限る（報酬の整形と信号の監視は使える）。"rust" は `VecEnv.observe_alive_f16_csr`（rl-env を作り直した tsuitate_rl）を使う
    """
    if actor_extras and not compact_legal:
        raise ValueError("collect(actor_extras=True) は compact_legal=True に限る")
    watch_signals = watch_signals or shaping is not None
    if fast:
        if fast not in ("py", "rust"):
            raise ValueError(f"fast {fast!r}（'py' か 'rust'）")
        if opponents is not None or nyugyoku_lambda != 0 or truth or not compact_legal:
            raise ValueError("collect(fast=...) は自己対局（opponents なし）・入玉ボーナスなし・truth なし・"
                             "compact_legal=True に限る")
        if fast == "rust" and not all(hasattr(e, "observe_alive_f16_csr") for e in getattr(env, "envs", [env])):
            raise ValueError("collect(fast='rust') には VecEnv.observe_alive_f16_csr が要る（tsuitate-bot の rl-env を"
                             " maturin develop で作り直す）")
        shape_run = None
        if watch_signals:
            env.reset()  # 整形の基準（収集の頭の信号）は reset の後に読む
            shape_run = _start_shape_run(env, shaping, shape_mult)
        return _collect_fast(env, net, device, max_steps, generator, rust=fast == "rust", extras=actor_extras,
                             shape_run=shape_run)
    env.reset()
    B = env.num_envs
    obs_l, legal_l, act_l, mu_l, t_l, b_l, truth_l = [], [], [], [], [], [], []
    v_l, lp_l = [], []  # actor_extras
    valid = torch.zeros(max_steps, B, dtype=torch.bool)
    player = torch.zeros(max_steps, B, dtype=torch.long)
    reward = torch.zeros(max_steps, B, 2)
    entered = np.zeros((B, 2), dtype=bool)
    advance = np.zeros((B, 2), dtype=np.int64)
    last_act = np.full((B, 2), -1, dtype=np.int64)
    outcome = np.zeros((B, 2), dtype=np.float32)
    pending = []  # nyugyoku_win_only のとき、終局まで保留する (時刻, 局, 側, ボーナス)
    shape_run = _start_shape_run(env, shaping, shape_mult) if watch_signals else None
    for t in range(max_steps):
        alive = env.alive()
        if not alive.any():
            break
        obs, mask, pl = env.observe()
        tr = env.truth() if truth else None
        idx = np.flatnonzero(alive)
        all_idx = idx
        opp_by = {}
        if opponents is not None:
            # 相手を固定した局で相手の手番のもの → 相手のネットごとに束ねる（記録しない）
            mine = []
            for b in all_idx:
                o = opponents[b]
                if o is not None and int(pl[b]) != o[1]:
                    opp_by.setdefault(o[0], []).append(b)
                else:
                    mine.append(b)
            idx = np.array(mine, dtype=np.int64)
        # 自玉の段（正規化後の行 0〜8。玉が無い = 取られた局面は -1）→ 前進段数 = 8 − 行
        # （学習側の手番だけ。相手の玉の前進はボーナスに数えない）
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
                if nyugyoku_win_only:
                    pending.append((last_act[b, p], b, p, bonus))
                else:
                    reward[last_act[b, p], b, p] += bonus
                    if nyugyoku_zero_sum:
                        reward[last_act[b, p], b, 1 - p] -= bonus
            last_act[b, p] = t
        actions = np.full(B, -1, dtype=np.int64)
        if len(idx):
            if actor_extras:
                a, mu, v_a, lp_a = _sample_actions_extras(net, obs[idx], mask[idx], device, generator)
                v_l.append(v_a)
                lp_l.append(lp_a)
            else:
                a, mu = sample_actions(net, obs[idx], mask[idx], device, generator)
            actions[idx] = a.numpy()
        if opp_by:
            # ハンデつきの相手（<名前>@<ε>）は同じネット（<名前>）にまとめて推論し、後から雑音の手に差し替える
            merged, noisy = {}, []
            for spec, bs in opp_by.items():
                name, eps = split_noise(spec)
                merged.setdefault(name, []).extend(bs)
                if eps > 0:
                    noisy.append((bs, eps))
            groups = {name: np.array(bs, dtype=np.int64) for name, bs in merged.items()}
            if isinstance(opp_nets, OpponentPool):
                for name, a_opp in opp_nets.sample(groups, obs, mask, device).items():
                    actions[groups[name]] = a_opp
            else:
                for name, bs in groups.items():
                    actions[bs] = sample_actions(opp_nets[name], obs[bs], mask[bs], device)[0].numpy()
            if noisy:
                if noise_rng is None:
                    raise ValueError("ハンデつきの相手（<名前>@<ε>）には noise_rng が要る")
                u = np.full(B, -1.0)
                for bs, eps in noisy:
                    bs = np.array(bs, dtype=np.int64)
                    hit = noise_rng.random(len(bs)) < eps
                    u[bs[hit]] = noise_rng.random(int(hit.sum()))
                if (u >= 0).any():
                    a_noise = env.random_legal_actions(u)
                    sel = a_noise >= 0
                    actions[sel] = a_noise[sel]
        rew, done = env.step(actions)
        outcome[done] = rew[done]
        if shape_run is not None:
            # 勝敗（outcome）は env の報酬のまま。学習の報酬にだけ整形を足す
            rew = rew + _shape_step(shape_run, env, done, rew)
        valid[t, idx] = True
        player[t, idx] = torch.from_numpy(pl[idx].astype(np.int64))
        # ボーナスは過去の時刻へ足すので、この時刻の報酬は上書きでなく加算
        if opponents is None:
            reward[t] += torch.from_numpy(rew)
        else:
            for b in all_idx:
                o = opponents[b]
                if o is None:
                    reward[t, b] += torch.from_numpy(rew[b])
                elif last_act[b, o[1]] >= 0 and rew[b, o[1]] != 0:
                    # 学習側の列だけを、学習側の直前の手番（この時刻の手番なら t）へ
                    reward[last_act[b, o[1]], b, o[1]] += float(rew[b, o[1]])
        if not len(idx):
            continue
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
        if truth:
            truth_l.append(torch.from_numpy(tr[idx]))
    else:
        raise RuntimeError(f"{max_steps} 手で終わらない局がある")
    for t0, b, p, bonus in pending:
        if outcome[b, p] > 0:  # その側が勝った局だけ
            reward[t0, b, p] += bonus
            if nyugyoku_zero_sum:
                reward[t0, b, 1 - p] -= bonus
    T = t + 1
    extra = {"truth": torch.cat(truth_l)} if truth else {}
    if actor_extras:
        extra.update(v_actor=torch.cat(v_l), mu_logp=torch.cat(lp_l))
    if shape_run is not None:
        extra["shape_run"] = shape_run
    return {**extra,
        **({"legal_cols": torch.cat([c for c, _ in legal_l]), "legal_counts": torch.cat([n for _, n in legal_l])}
           if compact_legal else {"legal": torch.cat(legal_l)}),
        "obs": torch.cat(obs_l), "action": torch.cat(act_l),
        "mu_a": torch.cat(mu_l), "t": torch.cat(t_l), "b": torch.cat(b_l),
        "valid": valid[:T], "player": player[:T], "reward": reward[:T],
        "entered": entered, "advance": advance, "outcome": outcome,
    }


def _start_shape_run(env, shaping, shape_mult):
    """収集の頭（env.reset の後）で整形・監視の状態を作る（`shaping=None` なら監視だけ）。
    env が強制した手の数（`forced_moves`）を出せるなら渡す = 強制の手で動いた Φ を報酬にしない（`ShapeRun`）"""
    from shaping import ShapeRun, env_signal_names
    envs = getattr(env, "envs", [env])
    names = shaping.names if shaping is not None else env_signal_names()
    if names is None or not all(hasattr(e, "signals") for e in envs):
        raise ValueError("局の信号（VecEnv.signals）が無い。tsuitate-bot の rl-env を maturin develop で作り直す")
    forced0 = env.forced_moves() if all(hasattr(e, "forced_moves") for e in envs) else None
    return ShapeRun(names, env.signals(), shaping, shape_mult, forced0=forced0)


def _shape_step(shape_run, env, done, rew):
    """env.step の後に呼ぶ: この step の整形の報酬 [B, 2]（監視値も進める）"""
    forced = env.forced_moves() if shape_run.forced_n is not None else None
    return shape_run.step(env.signals(), done, rew, forced=forced)


@torch.no_grad()
def _collect_fast(env, net, device, max_steps, generator, rust, extras=False, shape_run=None):
    """`collect(..., compact_legal=True)`（自己対局・入玉ボーナスなし・truth なし）と**ビット単位で同じ batch** を返す。

    数式と乱数の消費の順は同じで、CPU の仕事だけを減らす（2026-10-24 の計測で、768局の収集の CPU 時間の半分が
    終局した局まで観測する `observe`、15% が行の2回目の抜き出し・fp16 化・2次元の np.nonzero だった）:
    - 生きている局の行を1回だけ抜く（全局が生きている手番は抜かない）。`rust` なら Rust 側で生きている局だけを観測し、
      fp16 のビットと合法手の添字で返す（`observe_alive_f16_csr`。終局した局の観測を作らない）
    - 観測は CPU で1回だけ fp16 にし、batch に積むのと同じものをデバイスへ送る（送る量が半分。値は従来の
      「float32 を送ってデバイスで .half().float()」と同じ。fp16 への丸めはどちらも最近接偶数）
    - 合法手の添字（CSR）を1回だけ作り、batch とデバイス上の密なマスク（scatter で作る。密なマスクを送らない）の両方に使う
    - 入玉・前進の記帳（λ=0 では entered / advance を付けるだけ）を局ごとの Python ループからベクトル演算にする
    `extras` なら `collect(actor_extras=True)` と同じ `v_actor` / `mu_logp` も返す（乱数は消費しない）。
    `shape_run`（`_start_shape_run`。呼ぶ側が env.reset の後に作るので、ここでは reset しない）があれば報酬へ整形を足し、
    `shape_run` に入れて返す（`collect` と同じ。勝敗 outcome は env の報酬のまま）
    """
    if shape_run is None:
        env.reset()
    B = env.num_envs
    obs_l, cols_l, cnt_l, act_l, mu_l, t_l, b_l = [], [], [], [], [], [], []
    v_l, lp_l = [], []
    valid = torch.zeros(max_steps, B, dtype=torch.bool)
    player = torch.zeros(max_steps, B, dtype=torch.long)
    reward = torch.zeros(max_steps, B, 2)
    entered = np.zeros((B, 2), dtype=bool)
    advance = np.zeros((B, 2), dtype=np.int64)
    outcome = np.zeros((B, 2), dtype=np.float32)
    for t in range(max_steps):
        alive = env.alive()
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        n = len(idx)
        if rust:
            bits, cols, counts, pl = env.observe_alive_f16_csr()
            # 玉のプレーンは 0/1 なので、下の記帳は fp16 の観測で見ても float32 と同じ
            o = bits.view(np.float16).reshape(n, -1, 9, 9)
            oh = torch.from_numpy(o)
            cols16 = torch.from_numpy(cols)
            counts32 = torch.from_numpy(counts)
        else:
            obs, mask, pl = env.observe()
            if n < B:
                obs, mask, pl = obs[idx], mask[idx], pl[idx]
            o = obs
            oh = torch.from_numpy(obs).half()
            # np.nonzero(mask)[1] と mask.sum(1) と同じ値（2次元の np.nonzero は1次元の flatnonzero よりずっと遅い）
            rows_f, cols = np.divmod(np.flatnonzero(mask), NUM_ACTIONS)
            cols16 = torch.from_numpy(cols.astype(np.int16))
            counts32 = torch.from_numpy(np.bincount(rows_f, minlength=n).astype(np.int32))
        p_sel = pl.astype(np.int64)
        # 自玉の段（正規化後の行 0〜8。玉が無い = 取られた局面は 8 = 前進 0）。1手番に1局1側なので fancy な代入は重ならない
        king_rows = o[:, KING_PLANE].any(axis=2)
        adv_now = 8 - np.where(king_rows.any(axis=1), king_rows.argmax(axis=1), 8)
        entered[idx, p_sel] |= adv_now >= 6
        advance[idx, p_sel] = np.maximum(advance[idx, p_sel], adv_now)

        logits, v_a = forward(net, oh.to(device, non_blocking=True).float())
        rows = torch.repeat_interleave(torch.arange(n, device=device), counts32.to(device, non_blocking=True).long(),
                                       output_size=len(cols16))  # output_size が無いと CUDA で同期が入る
        cols_d = cols16.to(device, non_blocking=True).long()
        legal = torch.zeros((n, NUM_ACTIONS), dtype=torch.bool, device=device)
        legal[rows, cols_d] = True
        a, mu = _gumbel_sample(logits, legal, generator)
        if extras:
            lsm = logits.float().masked_fill(~legal, float("-inf")).log_softmax(-1)
            v_l.append(v_a.float().cpu())
            lp_l.append(lsm[rows, cols_d].cpu())
        actions = np.full(B, -1, dtype=np.int64)
        actions[idx] = a.numpy()
        rew, done = env.step(actions)
        outcome[done] = rew[done]
        if shape_run is not None:
            rew = rew + _shape_step(shape_run, env, done, rew)
        valid[t, idx] = True
        player[t, idx] = torch.from_numpy(p_sel)
        reward[t] += torch.from_numpy(rew)
        obs_l.append(oh)
        cols_l.append(cols16)
        cnt_l.append(counts32)
        act_l.append(a)
        mu_l.append(mu)
        t_l.append(torch.full((n,), t, dtype=torch.long))
        b_l.append(torch.from_numpy(idx))
    else:
        raise RuntimeError(f"{max_steps} 手で終わらない局がある")
    T = t + 1
    return {
        **({"v_actor": torch.cat(v_l), "mu_logp": torch.cat(lp_l)} if extras else {}),
        **({"shape_run": shape_run} if shape_run is not None else {}),
        "legal_cols": torch.cat(cols_l), "legal_counts": torch.cat(cnt_l),
        "obs": torch.cat(obs_l), "action": torch.cat(act_l),
        "mu_a": torch.cat(mu_l), "t": torch.cat(t_l), "b": torch.cat(b_l),
        "valid": valid[:T], "player": player[:T], "reward": reward[:T],
        "entered": entered, "advance": advance, "outcome": outcome,
    }


class PipelinedCollector:
    """自己対局を**別スレッドで集めて更新と重ねる**（`train_rnad.py --pipeline`。docs/rl-learning-speed.md）。

    `submit(net)` の時点のネットの写しで次の batch を集め始め、`result()` で受け取る。学習ループは
    「batch を受け取る → 今のネットで次の batch を頼む → 受け取った batch で更新」と回すので、
    **batch は1更新前の方策で集めたもの**になる（π/μ ≠ 1。v-trace の重要度比がそのまま補正する）。
    自己対局（Rust の env と行動の推論）と更新（ネットの順伝播・逆伝播）が重なるぶん、1更新の壁時計が
    「自己対局 + 更新」から「大きいほう」へ縮む（GPU では自己対局の大半が CPU 側の env なので重なる）。

    env は Python のスレッドをまたげない（VecEnv は unsendable）ので、集める側のスレッドの中で作って持つ。
    行動のサンプリングは専用の乱数列（`seed`）を使うので、スレッドの割り込み方によらず run は再現する
    （乱数列が従来と違うので、`--pipeline` なしの run とは別の局になる）。

    CUDA では集める側を**優先度の高い別ストリーム**で回す。既定のストリームを共有すると、集める側が手ごとに
    `.cpu()` で待つたびに更新側の積まれたカーネルの後ろに並び、GPU の仕事が重ならない
    """

    def __init__(self, env_factory, device, seed=0, **collect_kw):
        self.env_factory = env_factory
        self.device = device
        self.collect_kw = collect_kw
        self.local = threading.local()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.stream = torch.cuda.Stream(device=device, priority=-1) if device.type == "cuda" else None

    def _work(self, net, ready, opening_seed, league, shape_mult):
        if not hasattr(self.local, "env"):
            self.local.env = self.env_factory()  # 自己対局の局が無い（リーグだけ）なら None
        self_env = self.local.env
        if opening_seed is not None and self_env is not None:
            self_env.set_opening_seed(opening_seed)  # 序盤の強制の乱数列（collect の reset から効く）
        # リーグの局（相手を固定した評価モードの env）は batch ごとに相手の配分が変わるので毎回作る。
        # 自己対局の env の後ろにつなげる（局の並びは --pipeline なしと同じ [自己対局, リーグ]）
        league_env = league() if league is not None else None
        envs = [e for e in (self_env, league_env) if e is not None]
        env = envs[0] if len(envs) == 1 else MultiEnv(envs)
        t0 = time.time()
        kw = self.collect_kw if shape_mult is None else {**self.collect_kw, "shape_mult": shape_mult}
        if self.stream is None:
            batch = collect(env, net, self.device, generator=self.generator, **kw)
        else:
            self.stream.wait_event(ready)  # 写しを作り終えてから使う
            with torch.cuda.stream(self.stream):
                batch = collect(env, net, self.device, generator=self.generator, **kw)
            self.stream.synchronize()
        finished = self_env.pop_finished() if self_env is not None else []
        league_finished = league_env.pop_finished() if league_env is not None else []
        return batch, finished, time.time() - t0, league_finished

    def submit(self, net, opening_seed=None, league=None, shape_mult=None):
        """`league` はリーグの env を作る関数（集める側のスレッドの中で呼ぶ。VecEnv は unsendable）。None = 自己対局だけ。
        `shape_mult` は報酬の整形の倍率（collect_kw に shaping を渡したとき。batch ごとに変わるのでここで渡す）"""
        if self.future is not None:
            raise RuntimeError("前の batch を受け取る前に次を頼んだ")
        snap = copy.deepcopy(net).eval()
        snap.requires_grad_(False)
        ready = None
        if self.stream is not None:
            ready = torch.cuda.Event()
            ready.record()  # 既定のストリーム上で写しを作り終えた印
        self.future = self.pool.submit(self._work, snap, ready, opening_seed, league, shape_mult)

    def result(self):
        """(batch, 自己対局の終局した局, 集めるのにかかった秒)。リーグの終局した局は `league_finished` に置く"""
        *out, self.league_finished = self.future.result()
        self.future = None
        return tuple(out)

    def close(self):
        self.pool.shutdown(wait=True)


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
