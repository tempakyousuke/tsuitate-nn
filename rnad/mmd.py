"""MMD 学習器（Ataraxos〔Stratego、AtaraxosAI/stratego〕型: PPO の比クリップ ＋ 近い方策への逆KL ＋ 磁石への逆KL × 温度）。

`train_rnad.py --algo mmd` が使う（既定は R-NaD のまま）。設計・根拠・計測は tsuitate-bot の docs/rl-learning-speed.md
「10. MMD 学習器」。R-NaD（`rnad.RNaDLearner`）との違い:

- **1/μ がどの項にも無い**。R-NaD の勾配の雑音の 99.6〜99.9% は v-trace の s = (G − v)/μ のスパイク
- アドバンテージは手番側ごとの GAE（λ_A）、価値の目標は TD(λ)。どちらも**収集時の価値**（`collect(actor_extras=True)`
  の `v_actor`）で作り、学習時に計算し直さない（Ataraxos と同じ）。反則で同じ側が続く手番も、既存の `v_trace_scalars` を
  η=0・π=μ=1・c=λ で呼べばそのまま扱える（λ は試行ごと。`lambda_unit="turn"` なら反則の試行では λ をかけない）
- 学習に使う行を |A| に比例した確率で間引き（pps）、重み s̄/p で戻す（期待値は全行と同じ = 固定点を動かさない）。
  Ataraxos どおりの「|A| の 75%点以上だけ」（quantile）は、クーンポーカーで固定点を均衡から外す（NashConv は期待値の
  更新で 0.67〜0.71、標本ありで 0.60〜0.62。絞り込みなしは α-QRE の 0.038）ので既定にしない。`preset="ataraxos"` の腕として残す
- 1回の収集を k 歩（既定 4）の Adam に分ける（行を無作為に並べて k 等分）。正則化の段階（Δm）は無い。磁石への KL の
  係数（温度）と学習率は、MMD を始めてからの更新回数 t の予定（`power_schedule`、Ataraxos の rl.py と同じ式）で決める
  （既定はどちらも固定。`preset="mixinit"` は温度を、`"ataraxos"` は温度と学習率を t のべき乗で下げる）
- `--pipeline` では batch が1更新前のネットで集めたものなので、クリップと KL の中心を**更新の頭の方策**にし、μ との比は
  外側の重要度（上限 `prox_cap`）にする（decoupled PPO、Hilton ら 2021）
- 書き出すのは現在のネットの EMA（ターゲット、target_avg）。学習には使わない。BN は R-NaD と同じく eval モードのまま

損失（ミニバッチ m 行、w = 方策の行の重み、w_v = 価値の行の重み、中心 c = 近い方策〔pipeline なしなら μ〕）:

    r     = π_θ(a) / c(a)、is = min(c(a) / μ(a), prox_cap)（pipeline なしなら 1）
    pol   = −Σ w·is·min(A·r, A·clip(r, 1 − ε, 1 + ε)) / m_pol
    kl    = Σ w·KL(π_θ ‖ c) / m_pol、mag = Σ w·KL(π_θ ‖ ρ) / m_pol、lv = Σ w_v·(v − G_td)² / m_val
    loss  = pol_scale·pol + α_t·mag + vf·lv + kl_coef·kl

磁石 ρ は既定で**ターゲット（現在のネットの EMA）= 動く磁石**、温度は 0.05 固定（MMD 論文 H.5 の形。R-NaD の段階ごとの正則化方策の
差し替えを滑らかにしたもの）。設計の初版の「始めたときの方策と一様の混合を固定の磁石にし、温度を t で下げる」形（`preset="mixinit"`）は、
クーンで R-NaD の約8倍遅く、代理 P2 では自己対局が短い詰みの筋に固まった（docs/rl-learning-speed.md 10.4）。Ataraxos のコードの既定
（合法手に一様）は、ついたて将棋の合法手では質量の 39% を打ちに置き、強い方策を打ちの側へ引き戻す
"""

import copy
import math
import time
from dataclasses import dataclass, fields

import numpy as np
import torch

import rnad as R
from rnad import _pack_csr, forward, legal_log_policy, rng_state, set_rng_state, v_trace_scalars


# ---------------------------------------------------------------- 設定


@dataclass
class MMDConfig:
    # 方策の行の選び方: "pps:<割合>[,floor=<下限>]"（|A| に比例した確率で残し、重み s̄/p で戻す。不偏）/
    # "quantile:<分位>[,min=<下限>]"（Ataraxos: |A| が分位点以上の行だけ。重み 1）/
    # "topk:<割合>"（|A| の大きい順にちょうどその割合。同値は乱数で切る。重み 1）/ "none"（全行）
    filter: str = "pps:0.5,floor=0.05"
    # 価値の行: "auto"（pps は方策と同じ行・重み、quantile は "uniform:0.25"、none は全行）/ "all" /
    # "uniform:<割合>" / "filtered"（方策と同じ行。Ataraxos。確信のある局面の価値が 0 へ縮む）
    value_rows: str = "auto"
    # quantile の分位点の母集団に入れる過去の更新の数（1 = その更新だけ）
    filt_window: int = 1
    pol_scale: float = 1.0
    clip: float = 0.2
    kl: float = 0.1
    # pipeline（stale_actor）でクリップと KL の中心を更新の頭の方策にする。外側の重要度の上限が prox_cap
    decoupled: bool = True
    prox_cap: float = 2.0
    vf: float = 1.0
    # A の λ_A は 0.5、価値の目標の λ は 0.8（2026-10-26。設計の規則〔λ 0.5 と 0.95 の期待勾配がどちらも検出でき、向きが
    # そろう〕を v8 の 768局が満たし、代理 P2 で 0.8 より悪くなかった。docs/rl-learning-speed.md 10.4）
    gae_lambda: float = 0.5
    td_lambda: float = 0.8
    # λ をかける単位: "attempt"（反則の試行も1回と数える）/ "turn"（受理された手ごと。反則の試行では λ をかけない）
    lambda_unit: str = "attempt"
    # 磁石: "target"（今のターゲット = 現在のネットの EMA。動く磁石）/ "mix:target,<ε>"（今のターゲットと一様の混合）/
    # "mix:init,<ε>"（始めたときのターゲットの写しと一様の混合 = 固定の磁石）/ "uniform"（合法手に一様。Ataraxos のコードの既定）。
    # 既定は動く磁石（2026-10-25。固定の磁石 ＋ 温度を下げる形は、クーンで R-NaD の約8倍遅く、代理 P2 で自己対局が短い詰みの筋に
    # 固まった。docs/rl-learning-speed.md 10.4）
    magnet: str = "target"
    # 温度と学習率の予定: (coef, decay, ceil, floor) → clip(coef / (t + 1)^decay, floor, ceil)。既定の温度は 0.05 固定
    temp: tuple = (0.05, 0.0, 0.05, 0.05)
    temp_t_off: float = 0.0
    lr: tuple = (1e-4, 0.0, 1e-4, 1e-4)
    lr_t_off: float = 0.0
    # t = (更新回数 − MMD を始めた更新回数) × t_scale
    t_scale: float = 1.0
    # 1更新の Adam の歩数（行の数を k 等分）・逆伝播のチャンク・残した行を何周するか。k は 4（2026-10-26。A/B の割合では
    # B_noise が設計の規則の「100〜768局 → k 2〜4」の帯に入るか近い〔pps 1 は約260〜500局、pps 0.5 は未計測〕。
    # 代理 P2 で 8 より悪くなかった。docs/rl-learning-speed.md 10.4）
    steps: int = 4
    chunk: int = 2048
    epochs: int = 1
    max_grad_norm: float = 0.267
    adam_b1: float = 0.9
    # 最初の N 更新は価値の頭だけを全行で学ぶ（幹と方策の頭は止める）
    value_warmup: int = 0
    target_avg: float = 0.001
    # 動く磁石（target / mix:target）の速さ。0 = 書き出しのターゲット（target_avg）をそのまま磁石にする。> 0 なら磁石だけを
    # 別の EMA（magnet_net、この係数）で動かし、書き出しの EMA とは切り離す
    magnet_avg: float = 0.0


PRESETS = {
    "tsuitate": {},
    # 設計の初版の既定（固定の混合磁石・温度を t で下げる・起点 2,000・pps 0.25・λ_A 0.8・8歩）。代理 P2 で崩れた形の再現用
    "mixinit": {"filter": "pps:0.25,floor=0.05", "magnet": "mix:init,0.03", "temp": (0.05, 0.3, 0.1, 0.001),
                "temp_t_off": 2000.0, "gae_lambda": 0.8, "steps": 8},
    # Ataraxos のコードの既定（scripts/train/rl_main.py・pyengine/core/rl.py）。pipeline でも μ を中心にする
    "ataraxos": {"filter": "quantile:0.75,min=0.01", "value_rows": "filtered", "magnet": "uniform",
                 "gae_lambda": 0.5, "lr": (0.5, 1.1, 1e-4, 5e-6), "temp": (0.05, 0.3, 0.1, 0.001), "temp_t_off": 0.0,
                 "decoupled": False, "steps": 8},
}


def make_config(preset="tsuitate", **overrides) -> MMDConfig:
    """preset の値を入れ、`overrides`（None は「指定なし」）で上書きした設定"""
    if preset not in PRESETS:
        raise ValueError(f"preset {preset}（{'/'.join(PRESETS)}）")
    names = {f.name for f in fields(MMDConfig)}
    vals = dict(PRESETS[preset])
    for k, v in overrides.items():
        if k not in names:
            raise ValueError(f"MMDConfig に {k} は無い")
        if v is not None:
            vals[k] = v
    cfg = MMDConfig(**vals)
    parse_config(cfg)
    return cfg


def _spec(s):
    """"kind:a,b,k=v" → (kind, [位置引数], {名前つき})"""
    kind, _, rest = s.partition(":")
    pos, kw = [], {}
    for part in rest.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, _, v = part.partition("=")
            kw[k.strip()] = v.strip()
        else:
            pos.append(part)
    return kind.strip(), pos, kw


def parse_config(cfg: MMDConfig) -> dict:
    """文字列の設定を解いて検査する: {"filter": (種類, 引数...), "value_rows": ..., "magnet": ...}"""
    kind, pos, kw = _spec(cfg.filter)
    if kind == "none" and not pos and not kw:
        filt = ("none",)
    elif kind == "pps" and len(pos) == 1 and set(kw) <= {"floor"}:
        rate, floor = float(pos[0]), float(kw.get("floor", 0.05))
        if not 0.0 < floor < rate <= 1.0:
            raise ValueError(f"filter {cfg.filter}: 0 < floor < 割合 ≤ 1 にする")
        filt = ("pps", rate, floor)
    elif kind == "quantile" and len(pos) == 1 and set(kw) <= {"min"}:
        q, mn = float(pos[0]), float(kw.get("min", 0.0))
        if not 0.0 <= q < 1.0 or mn < 0:
            raise ValueError(f"filter {cfg.filter}: 0 ≤ 分位 < 1、min ≥ 0 にする")
        filt = ("quantile", q, mn)
    elif kind == "topk" and len(pos) == 1 and not kw:
        frac = float(pos[0])
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"filter {cfg.filter}: 0 < 割合 ≤ 1 にする")
        filt = ("topk", frac)
    else:
        raise ValueError(f"filter {cfg.filter}（pps:<割合>[,floor=] / quantile:<分位>[,min=] / topk:<割合> / none）")
    kind, pos, kw = _spec(cfg.value_rows)
    if kind in ("auto", "all", "filtered") and not pos and not kw:
        vrows = (kind,)
    elif kind == "uniform" and len(pos) == 1 and not kw:
        r = float(pos[0])
        if not 0.0 < r <= 1.0:
            raise ValueError(f"value_rows {cfg.value_rows}: 0 < 割合 ≤ 1 にする")
        vrows = ("uniform", r)
    else:
        raise ValueError(f"value_rows {cfg.value_rows}（auto / all / filtered / uniform:<割合>）")
    if vrows == ("auto",):
        vrows = {"pps": ("filtered",), "quantile": ("uniform", 0.25), "topk": ("uniform", 0.25),
                 "none": ("all",)}[filt[0]]
    kind, pos, kw = _spec(cfg.magnet)
    if kind in ("uniform", "target") and not pos and not kw:
        mag = (kind,)
    elif kind == "mix" and len(pos) == 2 and pos[0] in ("init", "target") and not kw:
        eps = float(pos[1])
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"magnet {cfg.magnet}: 0 ≤ ε ≤ 1 にする")
        mag = ("mix", pos[0], eps)
    elif kind == "twostage":
        raise ValueError("magnet twostage（駒→手の二段の一様、Ataraxos の論文の形）はまだ無い。"
                         "使うなら α を一様のときの約半分に読み替える（引く強さが約2.1倍。docs/rl-learning-speed.md 10節）")
    else:
        raise ValueError(f"magnet {cfg.magnet}（mix:init,<ε> / mix:target,<ε> / uniform / target）")
    if cfg.lambda_unit not in ("attempt", "turn"):
        raise ValueError(f"lambda_unit {cfg.lambda_unit}（attempt / turn）")
    for name in ("gae_lambda", "td_lambda"):
        if not 0.0 <= getattr(cfg, name) <= 1.0:
            raise ValueError(f"{name} は 0〜1")
    for name in ("temp", "lr"):
        if len(getattr(cfg, name)) != 4:
            raise ValueError(f"{name} は (coef, decay, ceil, floor) の4つ")
    if cfg.steps < 1 or cfg.epochs < 1 or cfg.chunk < 1 or cfg.filt_window < 1:
        raise ValueError("steps・epochs・chunk・filt_window は 1 以上")
    if not 0.0 <= cfg.magnet_avg <= 1.0:
        raise ValueError("magnet_avg は 0〜1")
    if cfg.magnet_avg > 0 and not (mag[0] == "target" or (mag[0] == "mix" and mag[1] == "target")):
        raise ValueError("magnet_avg は動く磁石（target / mix:target,<ε>）と一緒に使う")
    if cfg.prox_cap < 1.0 or cfg.clip <= 0 or cfg.max_grad_norm < 0:
        raise ValueError("prox_cap ≥ 1、clip > 0、max_grad_norm ≥ 0 にする")
    return {"filter": filt, "value_rows": vrows, "magnet": mag}


def power_schedule(coef: float, t: float, decay: float, ceil: float, floor: float) -> float:
    """Ataraxos（pyengine/core/rl.py の power_schedule）と同じ: clip(coef / (t + 1)^decay, floor, ceil)"""
    x = coef / ((t + 1) ** decay)
    x = max(x, floor)
    x = min(x, ceil)
    return x


# ---------------------------------------------------------------- アドバンテージと価値の目標


def foul_rows(valid, player):
    """[T, B] の bool: その試行の次の時刻に同じ局で同じ側が指す = 反則した試行（自己対局の batch に限る）"""
    nxt_valid = torch.zeros_like(valid)
    nxt_valid[:-1] = valid[1:]
    nxt_player = torch.zeros_like(player)
    nxt_player[:-1] = player[1:]
    return valid & nxt_valid & (nxt_player == player)


def player_lambda_returns(batch, v, lam, unit="attempt"):
    """手番側ごとの λ-return G^λ（平らな [N]）。v は各決定点の価値（平らな [N]）。

    プレイヤー k の決定点を時刻の順に i = 1..n とすると（次の自分の決定点は反則なら次の時刻、通常は2つ先）
        r_i = 自分の決定点 i から次の自分の決定点の手前までに k が受けた報酬の和（最後の決定点は局の終わりまで）
        G_i = r_i + [i < n]·((1 − λ_i)·V_{i+1} + λ_i·G_{i+1})
    （λ_i = λ。unit="turn" なら反則の試行だけ λ_i = 1）。`v_trace_scalars` を η=0・π=μ=1・c=λ で呼んだ vt がこれと同じ
    （相手の手番の行は報酬を次の自分の手番へ運ぶだけ）。局の途中で切らないので bootstrap は無い。
    **自己対局の batch に限る**（相手の手番が記録されない局では、有効でない行で carry が 0 に戻り途中が終局扱いになる）
    """
    valid, player, reward = batch["valid"], batch["player"], batch["reward"]
    t_idx, b_idx = batch["t"], batch["b"]
    T, B = valid.shape
    vg = torch.zeros(T, B)
    vg[t_idx, b_idx] = v.float()
    if unit == "attempt":
        c = float(lam)
    elif unit == "turn":
        c = torch.where(foul_rows(valid, player), 1.0, float(lam))
    else:
        raise ValueError(f"unit {unit}（attempt / turn）")
    ones, zeros = torch.ones(T, B), torch.zeros(T, B)
    out = torch.zeros(len(t_idx))
    mine_flat = player[t_idx, b_idx]
    for k in (0, 1):
        vt_k, _ = v_trace_scalars(vg, valid, player, ones, ones, zeros, reward[:, :, k], k, 0.0, c=c)
        out = torch.where(mine_flat == k, vt_k[t_idx, b_idx], out)
    return out


# ---------------------------------------------------------------- 行の選び方


def pps_probs(a, rate, floor):
    """p = clip(a / c, floor, 1) で mean p = rate になる c を二分法で決める（double、決定的）。(p [N] double, c)。
    a が全部 0 なら p = rate（一様に間引く）"""
    a = a.double()
    if rate >= 1.0:
        return torch.ones_like(a), 0.0
    amax = float(a.max()) if len(a) else 0.0
    if amax <= 0.0:
        return torch.full_like(a, rate), 0.0
    # mean p は c について単調に減る。c = amax/floor で全行が下限、c → 0 で a > 0 の行が全部 1
    lo, hi = amax * 1e-12, amax / floor
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float((a / mid).clamp(floor, 1.0).mean()) > rate:
            lo = mid
        else:
            hi = mid
    return (a / hi).clamp(floor, 1.0), hi


def select_rows(adv, filt, vrows, gen, hist=None):
    """(方策の行 keep_p, 重み w_p, 価値の行 keep_v, 重み w_v, ログ)。重みは float32 [N]（残さない行は 0）。
    `filt` / `vrows` は `parse_config` の解いた形。`hist` は quantile の窓（過去の更新の |A| の list）"""
    n = len(adv)
    a = adv.abs()
    info = {}
    if filt[0] == "none":
        keep = torch.ones(n, dtype=torch.bool)
        w = torch.ones(n)
    elif filt[0] == "pps":
        p, c = pps_probs(a, filt[1], filt[2])
        keep = torch.rand(n, generator=gen, dtype=torch.float64) < p
        w = (p.mean() / p).float() * keep
        info.update(pps_c=c, p_mean=float(p.mean()), p_one=float((p >= 1.0).double().mean()),
                    p_floor=float((p <= filt[2]).double().mean()))
    elif filt[0] == "topk":
        k = min(n, math.ceil(filt[1] * n))
        perm = torch.randperm(n, generator=gen)  # 同値は乱数の並びで切る
        top = perm[torch.sort(a[perm], descending=True, stable=True).indices[:k]]
        keep = torch.zeros(n, dtype=torch.bool)
        keep[top] = True
        w = keep.float()
        info["adv_thr"] = float(a[top].min()) if k else 0.0
    else:
        _, q, mn = filt
        pool = torch.cat([*(hist or []), a]) if hist else a
        thr = max(float(pool.float().quantile(q)), mn) if len(pool) else mn
        keep = a >= thr
        w = keep.float()
        info["adv_thr"] = thr
    if vrows[0] == "filtered":
        keep_v, wv = keep, w
    elif vrows[0] == "all":
        keep_v, wv = torch.ones(n, dtype=torch.bool), torch.ones(n)
    else:
        keep_v = torch.rand(n, generator=gen) < vrows[1]
        wv = keep_v.float()
    return keep, w * keep, keep_v, wv * keep_v, info


# ---------------------------------------------------------------- CSR の行の拾い直し


def csr_positions(ptr, counts, rows, total=None):
    """CSR（行 r の区間は [ptr[r], ptr[r] + counts[r])）から、行 `rows` の区間をこの順につなげた位置。
    `total`（= counts[rows] の和）を渡せば CUDA で同期しない"""
    c = counts[rows].long()
    rep = torch.repeat_interleave(torch.arange(len(rows), device=c.device), c, output_size=total)
    offs = torch.arange(len(rep), device=c.device) - (torch.cumsum(c, 0) - c)[rep]
    return ptr[rows][rep] + offs


def magnet_log_probs(kind, lp_magnet, pmask, counts):
    """磁石の対数確率（詰めた [m, K]、マスク外は 0）。lp_magnet は磁石のネットの合法手の対数確率（uniform なら None）"""
    log_n = counts.float().log()[:, None]
    if kind[0] == "uniform":
        return torch.where(pmask, -log_n, torch.zeros_like(pmask, dtype=torch.float32))
    if kind[0] == "target":
        return lp_magnet
    eps = kind[2]
    if eps <= 0.0:
        return lp_magnet
    if eps >= 1.0:
        return torch.where(pmask, -log_n, torch.zeros_like(lp_magnet))
    mix = torch.logaddexp(lp_magnet + math.log1p(-eps), (math.log(eps) - log_n).expand_as(lp_magnet))
    return torch.where(pmask, mix, torch.zeros_like(mix))


# ---------------------------------------------------------------- 学習器


VALUE_HEAD = ("v1.", "v1_bn.", "v_fc1.", "v_fc2.")
V_BANDS = (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0001)
ABS_BANDS = (0.0, 0.3, 0.6, 1.0001)
REMAIN_BANDS = (0, 10, 40, 80, 10_000)


class MMDLearner:
    """現在のネット・ターゲット（書き出し用の EMA）・磁石のネット（`magnet_net`）を持ち、
    1つの batch（`collect(compact_legal=True, actor_extras=True)`）で `steps` 歩の Adam を進める。

    磁石: 既定の動く磁石（magnet target / mix:target、magnet_avg 0）はターゲットそのもの。`magnet_net` は始めたときの
    ターゲットの写しで、mix:init の磁石はこれを動かさずに使い、magnet_avg > 0 ならこれを別の EMA で動かす。

    `stale_actor` = batch を1更新前のネットで集める（`--pipeline`）。設定（cfg）には入れない（R-NaD の packed と同じく
    計算の持ち方の指定）。`device_batch` = 残した行の観測を更新の頭で1回だけデバイスへ送る。
    行の間引きとミニバッチの並べ替えには専用の乱数（`seed`）を使い、checkpoint に入れる（pipeline なしなら再開が再現する）
    """

    algo = "mmd"

    def __init__(self, net, cfg: MMDConfig, device, seed=0, device_batch=False, stale_actor=False):
        self.cfg = cfg
        self.parsed = parse_config(cfg)
        self.device = device
        self.device_batch = device_batch
        self.stale_actor = stale_actor
        self.net = net.to(device).eval()
        self.target = copy.deepcopy(self.net).eval()
        self.magnet_net = copy.deepcopy(self.net).eval()
        for m in (self.target, self.magnet_net):
            m.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr[0], betas=(cfg.adam_b1, 0.999), eps=1e-8)
        self.value_params = [p for k, p in self.net.named_parameters() if k.startswith(VALUE_HEAD)]
        self.step_count = 0
        self.mmd_t0 = 0
        # 学習率に掛ける倍率（train_rnad.py が warmup の値を入れる）
        self.lr_scale = 1.0
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)
        self.filt_hist = []
        self.init_from = None
        self.last_grad_norms = []  # 直前の更新の歩ごとのクリップ前の勾配ノルム（bench_update.py が読む）

    # ---- 予定

    def mmd_time(self, step=None):
        return ((self.step_count if step is None else step) - self.mmd_t0) * self.cfg.t_scale

    def temperature(self, step=None):
        return power_schedule(*_sched(self.cfg.temp, self.mmd_time(step) + self.cfg.temp_t_off))

    def scheduled_lr(self, step=None):
        return power_schedule(*_sched(self.cfg.lr, self.mmd_time(step) + self.cfg.lr_t_off))

    def start_from(self, net_sd, target_sd, step, source=None):
        """R-NaD などの checkpoint から始める: net ← net_sd、ターゲットと磁石 ← target_sd、step の番号を続ける"""
        self.net.load_state_dict(net_sd)
        self.target.load_state_dict(target_sd)
        self.magnet_net.load_state_dict(target_sd)
        self.step_count = step
        self.mmd_t0 = step
        self.init_from = source

    # ---- 1回の更新

    def update(self, batch, reward_scale=1.0):
        """batch で `steps × epochs` 歩進める。ログ用の dict を返す（R-NaD と共通のキー samples / loss_v / loss_nerd /
        alpha / reg_update を含む。stage_log.py がそのまま読む）"""
        if reward_scale != 1.0:
            raise ValueError("MMD は報酬の縮め（ボーナス）を使わない")
        for key in ("v_actor", "mu_logp", "legal_cols"):
            if key not in batch:
                raise ValueError(f"batch に {key} が無い（collect(compact_legal=True, actor_extras=True) の batch が要る）")
        cfg, parsed = self.cfg, self.parsed
        dev = self.device
        t_start = time.time()
        mmd_t = self.mmd_time()
        temp = self.temperature()
        lr = self.scheduled_lr() * self.lr_scale
        for g in self.opt.param_groups:
            g["lr"] = lr
        value_only = self.step_count - self.mmd_t0 < cfg.value_warmup

        n = len(batch["action"])
        v_act = batch["v_actor"].float()
        g_td = player_lambda_returns(batch, v_act, cfg.td_lambda, cfg.lambda_unit)
        g_a = g_td if cfg.gae_lambda == cfg.td_lambda else player_lambda_returns(batch, v_act, cfg.gae_lambda,
                                                                                  cfg.lambda_unit)
        adv = g_a - v_act
        t_gae = time.time()

        if value_only:
            keep_p = torch.zeros(n, dtype=torch.bool)
            wp = torch.zeros(n)
            keep_v = torch.ones(n, dtype=torch.bool)
            wv = torch.ones(n)
            sel_info = {}
        else:
            hist = self.filt_hist if parsed["filter"][0] == "quantile" else None
            keep_p, wp, keep_v, wv, sel_info = select_rows(adv, parsed["filter"], parsed["value_rows"], self.gen, hist)
            if parsed["filter"][0] == "quantile" and cfg.filt_window > 1:
                self.filt_hist = (self.filt_hist + [adv.abs()])[-(cfg.filt_window - 1):]

        st, capped = self.prepare(batch, adv, g_td, v_act, keep_p, wp, keep_v, wv, value_only)
        t_prox = time.time()

        # k 歩: 行を無作為に並べて k 等分し、ミニバッチごとにチャンクで逆伝播して勾配を足してから1歩
        params = self.value_params if value_only else list(self.net.parameters())
        step_stats, gnorms = [], []
        ratio_dev_first = None
        for _ in range(cfg.epochs):
            perm = torch.randperm(len(st.U), generator=self.gen)
            for mb in torch.tensor_split(perm, cfg.steps):
                if len(mb) == 0:
                    continue
                m_pol = max(int((st.wp_cpu[mb] > 0).sum()), 1)
                m_val = max(int((st.wv_cpu[mb] > 0).sum()), 1)
                self.opt.zero_grad(set_to_none=True)
                acc = torch.zeros(9, device=dev)
                for j in range(0, len(mb), cfg.chunk):
                    rows = mb[j:j + cfg.chunk]
                    loss, stats = self._chunk_loss(st, rows, m_pol, m_val, temp, value_only,
                                                   first=(ratio_dev_first is None))
                    loss.backward()
                    acc += stats
                gn = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm if cfg.max_grad_norm > 0 else math.inf)
                gnorms.append(gn.detach().reshape(()))
                self.opt.step()
                step_stats.append(acc)
                if ratio_dev_first is None:
                    ratio_dev_first = (acc[7], acc[8])
        st.check()
        self.last_grad_norms = [float(x) for x in torch.stack(gnorms).cpu()] if gnorms else []
        with torch.no_grad():
            for pt, p in zip(self.target.parameters(), self.net.parameters()):
                pt.add_(cfg.target_avg * (p - pt))
            for bt, b in zip(self.target.buffers(), self.net.buffers()):
                bt.copy_(b)
            if cfg.magnet_avg > 0:
                for pm, p in zip(self.magnet_net.parameters(), self.net.parameters()):
                    pm.add_(cfg.magnet_avg * (p - pm))
                for bm, b in zip(self.magnet_net.buffers(), self.net.buffers()):
                    bm.copy_(b)
        self.step_count += 1
        t_train = time.time()

        # ---- ログ（ここから下は学習に影響しない）
        out = {"samples": n, "alpha": 1.0, "reg_update": False, "algo": "mmd", "mmd_t": mmd_t,
               "temperature": temp, "lr_sched": lr, "value_only": value_only,
               "rows_kept": int(keep_p.sum()), "rows_val": int(keep_v.sum()), "rows_union": len(st.U),
               "keep_rate": float(keep_p.float().mean()), "w_max": float(wp.max()) if n else 0.0,
               "prox_is_capped": capped, **sel_info}
        if step_stats:
            ss = torch.stack(step_stats).cpu().double()
            gn = torch.stack(gnorms).cpu().double()
            # loss_* は各歩の前の値（ミニバッチの損失）の歩の平均。KL は中心（pipeline の decoupled なら更新の頭の方策、
            # それ以外は μ）への KL(π‖中心) で、方策の行の重み付き平均。vf_change は (v − v_actor)² の価値の行の平均
            out.update(loss_nerd=float(ss[:, 0].mean()), loss_kl=float(ss[:, 1].mean()),
                       loss_mag=float(ss[:, 2].mean()), loss_v=float(ss[:, 3].mean()),
                       vf_change=float(ss[:, 4].mean()),
                       kl_prox_mean=float(ss[:, 1].mean()), kl_prox_last=float(ss[-1, 1]),
                       magnet_kl=float(ss[:, 2].mean()),
                       clip_frac=float(ss[:, 5].sum() / max(float(ss[:, 6].sum()), 1.0)),
                       minibatches=len(step_stats),
                       grad_norm_p50=float(gn.quantile(0.5)), grad_norm_p90=float(gn.quantile(0.9)),
                       grad_norm_max=float(gn.max()),
                       clip_rate=float((gn > cfg.max_grad_norm).double().mean()) if cfg.max_grad_norm > 0 else 0.0)
            rd_sum, rd_cnt = (float(x) for x in ratio_dev_first)
            out["ratio_dev_first"] = rd_sum / rd_cnt if rd_cnt else 0.0
        else:
            out.update(loss_nerd=0.0, loss_v=0.0, minibatches=0)
        out.update(_policy_monitors(batch, n))
        out.update(_value_monitors(batch, v_act, g_td, adv, keep_p, keep_v))
        out["kl_mu_unkept"] = self._kl_unkept(batch, keep_p, batch["_legal_ptr"])
        out.update(gae_sec=t_gae - t_start, prox_sec=t_prox - t_gae, train_sec=t_train - t_prox)
        return out

    def prepare(self, batch, adv, g_td, v_act, keep_p, wp, keep_v, wv, value_only=False):
        """更新の頭: 残した行（方策か価値のどちらか = U）を U の並びで持ち直し（合法手の CSR も拾い直す）、
        勾配なしで近い方策（pipeline の decoupled）と磁石の対数確率を方策の行について求める。
        (`_Rows`, 外側の重要度が上限に当たった割合) を返す"""
        cfg, parsed = self.cfg, self.parsed
        dev = self.device
        U = (keep_p | keep_v).nonzero().squeeze(1)
        counts_all = batch["legal_counts"].long()
        ptr_all = batch.get("_legal_ptr")
        if ptr_all is None:
            ptr_all = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts_all, 0)])
            batch["_legal_ptr"] = ptr_all
        counts_u = counts_all[U]
        pos_u = csr_positions(ptr_all, counts_all, U, int(counts_u.sum()))
        st = _Rows(self, batch, U, counts_u, batch["legal_cols"][pos_u], batch["mu_logp"][pos_u].float(),
                   adv[U], g_td[U], wp[U], wv[U], v_act[U])
        # 更新の頭（勾配なし）: 近い方策（pipeline の decoupled）と磁石の対数確率を、方策の行について求める
        pol_loc = (st.wp_cpu > 0).nonzero().squeeze(1)
        capped = 0.0
        need_prox = cfg.decoupled and self.stale_actor and not value_only
        mag_kind = parsed["magnet"]
        # 磁石のネット: mix:init は始めたときの写し（magnet_net、動かさない）、動く磁石は書き出しのターゲットか、
        # magnet_avg > 0 なら別の EMA（magnet_net）
        moving_own = cfg.magnet_avg > 0
        mag_net = None if mag_kind[0] == "uniform" else (
            self.magnet_net if (mag_kind[0] == "mix" and mag_kind[1] == "init") or moving_own else self.target)
        with torch.no_grad():
            for i in range(0, len(pol_loc), cfg.chunk):
                rows = pol_loc[i:i + cfg.chunk]
                obs, pmask, apos, idx, pos = st.view(rows)
                cnt = st.counts[rows.to(dev)]
                if need_prox:
                    lp_now = legal_log_policy(forward(self.net, obs)[0].gather(1, idx), pmask)
                    st.prox[pos] = lp_now[pmask]
                lp_m = (legal_log_policy(forward(mag_net, obs)[0].gather(1, idx), pmask)
                        if mag_net is not None else None)
                st.lrho[pos] = magnet_log_probs(mag_kind, lp_m, pmask, cnt)[pmask]
        if need_prox:
            # 外側の重要度 is = min(π_prox(a) / μ(a), 上限)。行ごとの選んだ手の位置で拾う
            ia = st.act_pos[pol_loc.to(dev)]
            log_is = st.prox[ia] - st.mulp[ia]
            is_raw = log_is.exp()
            st.is_w[pol_loc.to(dev)] = is_raw.clamp(max=cfg.prox_cap)
            capped = float((is_raw > cfg.prox_cap).float().mean()) if len(pol_loc) else 0.0
        return st, capped

    def _chunk_loss(self, st, rows, m_pol, m_val, temp, value_only, first):
        """1チャンクの損失（ミニバッチの m で割った和）と、ログ用の和 [pol, kl, mag, lv, vf_change, |r − 1| > clip の行の数,
        方策の行の数, 最初の歩の |π_θ/μ − 1| の和, その数]"""
        cfg = self.cfg
        dev = self.device
        obs, pmask, apos, idx, pos = st.view(rows)
        r = rows.to(dev)
        wv, g, v_old = st.wv[r], st.g[r], st.v_old[r]
        stats = torch.zeros(9, device=dev)
        if value_only:
            v = _value_forward(self.net, obs)
            lv = (wv * (v - g) ** 2).sum() / m_val
            stats[3] = lv.detach()
            stats[4] = ((wv * (v.detach() - v_old) ** 2).sum() / m_val)
            return cfg.vf * lv, stats
        logits, v = forward(self.net, obs)
        lp = legal_log_policy(logits.gather(1, idx), pmask)
        p = lp.exp() * pmask
        zero = torch.zeros_like(lp)
        prox = zero.clone()
        prox[pmask] = st.prox[pos]
        lrho = zero.clone()
        lrho[pmask] = st.lrho[pos]
        w, a, isw = st.wp[r], st.adv[r], st.is_w[r]
        lp_a = lp.gather(1, apos[:, None]).squeeze(1)
        prox_a = prox.gather(1, apos[:, None]).squeeze(1)
        ratio = (lp_a - prox_a).exp()
        surr = torch.minimum(a * ratio, a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip))
        pol = -(w * isw * surr).sum() / m_pol
        kl = (w * (p * (lp - prox)).sum(1)).sum() / m_pol
        mag = (w * (p * (lp - lrho)).sum(1)).sum() / m_pol
        lv = (wv * (v - g) ** 2).sum() / m_val
        loss = cfg.pol_scale * pol + temp * mag + cfg.vf * lv + cfg.kl * kl
        with torch.no_grad():
            on = w > 0
            stats[0], stats[1], stats[2], stats[3] = pol.detach(), kl.detach(), mag.detach(), lv.detach()
            stats[4] = (wv * (v.detach() - v_old) ** 2).sum() / m_val
            stats[5] = ((ratio.detach() - 1).abs() > cfg.clip)[on].float().sum()
            stats[6] = on.float().sum()
            if first:
                mu_a = torch.zeros_like(lp)
                mu_a[pmask] = st.mulp[pos]
                mu_a = mu_a.gather(1, apos[:, None]).squeeze(1)
                stats[7] = ((lp_a.detach() - mu_a).exp() - 1).abs()[on].sum()
                stats[8] = on.float().sum()
        return loss, stats

    @torch.no_grad()
    def _kl_unkept(self, batch, keep_p, ptr_all):
        """残さなかった方策の行の無作為な 1/16 で、更新後の KL(π_new ‖ μ)（共有の幹を通して信頼領域なしで動いた分の見張り）"""
        un = (~keep_p).nonzero().squeeze(1)
        if len(un) == 0:
            return 0.0
        k = max(1, len(un) // 16)
        rows = un[torch.randperm(len(un), generator=self.gen)[:k]].sort().values
        dev = self.device
        counts = batch["legal_counts"].long()
        tot = 0.0
        for i in range(0, len(rows), self.cfg.chunk):
            rr = rows[i:i + self.cfg.chunk]
            cnt = counts[rr]
            pos = csr_positions(ptr_all, counts, rr, int(cnt.sum()))
            cols = batch["legal_cols"][pos].to(dev)
            act = batch["action"][rr].to(dev)
            idx, pmask, _, _ = _pack_csr(cols, cnt.to(dev), act, int(cnt.max()), len(pos))
            obs = batch["obs"][rr].to(dev, torch.float32)
            lp = legal_log_policy(forward(self.net, obs)[0].gather(1, idx), pmask)
            mu = torch.zeros_like(lp)
            mu[pmask] = batch["mu_logp"][pos].to(dev).float()
            tot += float(((lp.exp() * pmask) * (lp - mu)).sum())
        return tot / len(rows)

    # ---- 保存と再開

    def state_dict(self):
        return {"algo": "mmd", "net": self.net.state_dict(), "target": self.target.state_dict(),
                "magnet_net": self.magnet_net.state_dict(), "opt": self.opt.state_dict(),
                "step": self.step_count, "mmd_t0": self.mmd_t0,
                "cfg": dict(self.cfg.__dict__), "config": dict(self.net.config),
                "rng": rng_state(self.device), "gen": self.gen.get_state(),
                **({"filt_hist": list(self.filt_hist)} if self.filt_hist else {}),
                **({"init_from": self.init_from} if self.init_from else {})}

    def load_state_dict(self, d):
        if d.get("algo", "rnad") != "mmd":
            raise ValueError(f"MMD でない checkpoint（algo {d.get('algo', 'rnad')}）")
        self.net.load_state_dict(d["net"])
        self.target.load_state_dict(d["target"])
        self.magnet_net.load_state_dict(d["magnet_net"])
        self.opt.load_state_dict(d["opt"])
        self.step_count = d["step"]
        self.mmd_t0 = d["mmd_t0"]
        if "rng" in d:
            set_rng_state(d["rng"], self.device)
        if "gen" in d:
            self.gen.set_state(d["gen"].cpu())
        self.filt_hist = [x.cpu() for x in d.get("filt_hist", [])]
        self.init_from = d.get("init_from")

    def reset_optimizer_state(self):
        self.opt.state.clear()

    def reseed(self, seed):
        self.gen.manual_seed(seed)

    def apply_cfg(self, cfg: MMDConfig, old=None):
        """設定を差し替える（再開時に明示的に上書きするとき）。学習率は毎更新で予定から入れ直す。
        `old` = 差し替える前の設定（checkpoint の cfg の dict。再開では学習器を新しい設定で作ってから読むので要る。
        None なら今の self.cfg）。

        magnet_net は magnet_avg 0 のあいだ始めたときの写しのまま動かないので、別の EMA（magnet_avg > 0）へ切り替えるときは
        現在のターゲットから始め直す（そのままだと始めの方策へ引き戻す）。mix:init への途中からの切り替えは、magnet_net が
        始めたときの写しのままとは限らないので止める。分位の窓は新しい窓の長さに切り詰める"""
        prev = dict(self.cfg.__dict__) if old is None else {**MMDConfig().__dict__, **old}
        parsed = parse_config(cfg)

        def is_init(magnet):
            kind, pos, _ = _spec(magnet)
            return kind == "mix" and pos[:1] == ["init"]

        if is_init(cfg.magnet) and not is_init(prev["magnet"]) and self.step_count > self.mmd_t0:
            raise ValueError("MMD の途中から mix:init の磁石には切り替えられない（始めたときの写しが残っているとは限らない）")
        if cfg.magnet_avg > 0 and not prev["magnet_avg"] > 0:
            self.magnet_net.load_state_dict(self.target.state_dict())
        if parsed["filter"][0] == "quantile" and cfg.filt_window > 1:
            self.filt_hist = self.filt_hist[-(cfg.filt_window - 1):]
        else:
            self.filt_hist = []
        self.parsed = parsed
        self.cfg = cfg
        for g in self.opt.param_groups:
            g["betas"] = (cfg.adam_b1, g["betas"][1])


def _sched(spec, t):
    coef, decay, ceil, floor = spec
    return coef, t, decay, ceil, floor


def _value_forward(net, x):
    """価値の頭だけに勾配を流す順伝播（幹は止める）"""
    if R._AMP_DTYPE is None:
        with torch.no_grad():
            f = net.trunk(x)
        return net.heads(f)[1]
    with torch.autocast("cuda", dtype=R._AMP_DTYPE):
        with torch.no_grad():
            f = net.trunk(x)
        v = net.heads(f)[1]
    return v.float()


class _Rows:
    """1回の更新で使う行（U）の持ち物。小さい配列はデバイスに置き、観測は device_batch なら U の分だけ1回で送る"""

    def __init__(self, learner, batch, U, counts_u, cols_u, mulp_u, adv, g, wp, wv, v_old):
        dev = learner.device
        self.dev = dev
        self.batch = batch
        self.U = U
        self.counts_cpu = counts_u
        self.ptr_cpu = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts_u, 0)])
        self.counts = counts_u.to(dev)
        self.ptr = self.ptr_cpu.to(dev)
        self.cols = cols_u.to(dev)
        self.mulp = mulp_u.to(dev)
        self.prox = self.mulp.clone()  # pipeline なし（または decoupled なし）では中心 = μ
        self.lrho = torch.zeros_like(self.mulp)
        self.act = batch["action"][U].to(dev)
        self.adv, self.g, self.v_old = adv.to(dev), g.to(dev), v_old.to(dev)
        self.wp, self.wv = wp.to(dev), wv.to(dev)
        self.wp_cpu, self.wv_cpu = wp, wv
        self.is_w = torch.ones(len(U), device=dev)
        self.obs = batch["obs"][U].to(dev) if learner.device_batch else None
        # 行ごとの選んだ手の、U の CSR の中の位置（外側の重要度を拾う）
        cols_cpu = cols_u.long()
        act_cpu = batch["action"][U]
        rowid = torch.repeat_interleave(torch.arange(len(U)), counts_u)
        hit = (cols_cpu == act_cpu[rowid]).nonzero().squeeze(1)
        if len(hit) != len(U):
            raise ValueError("マスク外の手が batch にある")
        self.act_pos = hit.to(dev)
        self.oks = []

    def view(self, rows):
        """U の中の行 `rows`（CPU の添字）の (観測 float32, 詰めたマスク, 選んだ手の位置, 詰めの添字, U の CSR の位置)"""
        dev = self.dev
        r = rows.to(dev)
        total = int(self.counts_cpu[rows].sum())
        pos = csr_positions(self.ptr, self.counts, r, total)
        idx, pmask, apos, ok = _pack_csr(self.cols[pos], self.counts[r], self.act[r],
                                         int(self.counts_cpu[rows].max()), total)
        self.oks.append(ok)
        if self.obs is not None:
            obs = self.obs[r].to(torch.float32)
        else:
            obs = self.batch["obs"][self.U[rows]].to(dev, torch.float32)
        return obs, pmask, apos, idx, pos

    def check(self):
        if self.oks and not bool(torch.stack(self.oks).all()):
            raise ValueError("マスク外の手が batch にある")


# ---------------------------------------------------------------- 見張り（ログだけ）


def _policy_monitors(batch, n):
    """収集した方策（μ）の形: エントロピー・最善手の確率の平均・0.999 を超える決定点の割合（R-NaD のログと同じ定義）"""
    counts = batch["legal_counts"].long()
    lp = batch["mu_logp"].float()
    p = lp.exp()
    rowid = torch.repeat_interleave(torch.arange(n), counts)
    ent = torch.zeros(n).index_add_(0, rowid, -(p * lp))
    top = torch.zeros(n).scatter_reduce_(0, rowid, p, "amax", include_self=False)
    return {"entropy_mean": float(ent.mean()), "pi_top_mean": float(top.mean()),
            "pi_top_gt_0999": float((top > 0.999).float().mean())}


def _bands(x, edges):
    return [(edges[i], edges[i + 1], (x >= edges[i]) & (x < edges[i + 1])) for i in range(len(edges) - 1)]


def _value_monitors(batch, v, g_td, adv, keep_p, keep_v):
    """価値の較正の見張り: |v| の帯ごとの mean(z·sign v)・mean|v|、終局までの手番数の帯ごとの z を v へ回帰した傾き、
    V の帯ごとの mean(G_td − v)（全行と価値の行）。z はその手番側の最終結果（±1/0）"""
    t_idx, b_idx = batch["t"], batch["b"]
    pl = batch["player"][t_idx, b_idx]
    z = torch.as_tensor(np.asarray(batch["outcome"]), dtype=torch.float32)[b_idx, pl]
    end = batch["valid"].sum(0)[b_idx] - 1
    remain = end - t_idx
    out = {"v_target_mean": float(g_td.mean()) if len(g_td) else 0.0,
           "adv_abs_mean_all": float(adv.abs().mean()) if len(adv) else 0.0,
           "adv_abs_mean_kept": float(adv[keep_p].abs().mean()) if bool(keep_p.any()) else 0.0}
    far = remain >= 80
    out["kept_share_far"] = float(keep_p[far].float().mean()) if bool(far.any()) else None
    cal = []
    for lo, hi, m in _bands(v.abs(), ABS_BANDS):
        k = int(m.sum())
        cal.append({"lo": lo, "hi": min(hi, 1.0), "n": k,
                    "z_sign_v": float((z[m] * torch.sign(v[m])).mean()) if k else None,
                    "abs_v": float(v[m].abs().mean()) if k else None})
    out["calib_band"] = cal
    slopes = []
    for lo, hi, m in _bands(remain.float(), REMAIN_BANDS):
        k = int(m.sum())
        vm, zm = v[m].double(), z[m].double()
        var = float(((vm - vm.mean()) ** 2).mean()) if k > 1 else 0.0
        slope = float(((vm - vm.mean()) * (zm - zm.mean())).mean() / var) if var > 1e-12 else None
        slopes.append({"lo": lo, "hi": hi, "n": k, "slope": slope})
    out["calib_slope_by_remain"] = slopes
    res = []
    resid = g_td - v
    for lo, hi, m in _bands(v, V_BANDS):
        mv = m & keep_v
        res.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()),
                    "all": float(resid[m].mean()) if bool(m.any()) else None,
                    "val_rows": float(resid[mv].mean()) if bool(mv.any()) else None})
    out["resid_band"] = res
    return out
