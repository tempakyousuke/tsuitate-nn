"""報酬の整形: 勝敗（±1）のほかに、局の信号（駒得・反則・王手・手数など）を報酬へ足す。

tsuitate-bot の docs/rl-fixed-opponent.md。既存のモデルを相手にゼロから学習すると最初は1局も勝てず、勝敗だけの報酬では
どの手が良かったかの信号が届かない。そこで「勝ち以外の、学習の手がかりになる量」を報酬へ足して比べる。

env（tsuitate_rl.VecEnv.signals()）は各側の信号 [B, 2, K] を出すだけで、どの信号にどんな重みを付けるかはここで決める:

    Φ_p(s) = Σ_k ( own_k · sig[p, k] + opp_k · sig[1 − p, k] )      （p から見た「形勢の目安」）
    Φc_p(s) = clip(Φ_p(s) − Φ_p(初期局面), −cap, cap) × 倍率
    r_p(s → s') = Φc_p(s') − Φc_p(s)

- `mode="plain"`: 差分をそのまま足す。1局の合計は Φc_p(終局) で、**目的そのものが変わる**（駒得して負けた局は、
  駒得ぶんだけ負けが軽くなる）。勝敗の信号が届くまでの足場として使い、`decay` で 0 へ下げる
- `mode="potential"`: 終局の Φc を 0 とみなす（終局の遷移で累計を差し戻す）。1局の合計は必ず 0 なので
  **最適方策は変わらない**（Ng ら 1999 のポテンシャル型の整形）。価値の頭に「Φ が形勢」という事前知識を与えるのと同じで、
  効くのは価値が育つまでの信用の割り当て

`lose_len=(λ, H)`: **負けた側**の終局の報酬に λ × min(手数, H) / H を足す（負けるまでの長さ。勝った側・引き分けには付けない）。
λ < 1 なら「負け < 引き分け < 勝ち」の順は保たれる。目的は変わる（plain と同じ扱いで、倍率で下げる）。

どの形でも 1局の報酬は ±(1 + cap + λ) に収まるので、学習器へは `reward_scale` = 1 + cap + λ を**学習の全体で固定して**渡す
（価値の頭は tanh。倍率で整形を下げ切った後も、単位を途中で変えない）。

重みの書式（カンマ区切りの `鍵=重み`）:
    <信号>        自分の信号（例 `fouls=-0.03` = 自分の反則1回につき −0.03）
    opp_<信号>    相手の信号（例 `opp_fouls=0.01`）
    d_<信号>      自分 − 相手（例 `d_capture_value=0.02` = 駒を取ると +、取られると −。零和を保つ）
"""

import json
import math
from dataclasses import dataclass

import numpy as np

MODES = ("plain", "potential")
# 監視値を切り出す手数（両者の受理手の合計がこの値に初めて届いた時点の信号。終局の値は詰み・反則負けそのものを
# 映すので、「途中のどの量が勝ちを先取りするか」はこちらで見る）
SNAP_PLIES = (20, 40, 60)


def env_signal_names():
    """tsuitate_rl が知っている信号の名前（古いビルドは None）"""
    import tsuitate_rl
    names = getattr(tsuitate_rl, "SIGNAL_NAMES", None)
    return list(names) if names is not None else None


def parse_weights(spec: str, names):
    """`鍵=重み,...` → (own [K], opp [K])。知らない鍵・重複は ValueError"""
    own = np.zeros(len(names), dtype=np.float64)
    opp = np.zeros(len(names), dtype=np.float64)
    seen = set()
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"'{item}' は 鍵=重み の形にする")
        if key in seen:
            raise ValueError(f"鍵 {key} が2回ある")
        seen.add(key)
        try:
            w = float(val)
        except ValueError:
            raise ValueError(f"'{item}' の重みが数でない") from None
        if not math.isfinite(w):
            raise ValueError(f"'{item}' の重みが有限でない")
        kind, name = "own", key
        if key not in names:
            for prefix, k in (("opp_", "opp"), ("d_", "diff")):
                if key.startswith(prefix) and key[len(prefix):] in names:
                    kind, name = k, key[len(prefix):]
                    break
            else:
                raise ValueError(f"信号 {key} を知らない（{' / '.join(names)}。相手の信号は opp_<名前>、差は d_<名前>）")
        i = names.index(name)
        if kind in ("own", "diff"):
            own[i] += w
        if kind == "opp":
            opp[i] += w
        if kind == "diff":
            opp[i] -= w
    return own, opp


def parse_decay(spec: str):
    """`A:B` → (A, B)。更新 A までは倍率 1、B 以後は 0、その間は線形。空なら None（ずっと 1）"""
    if not spec:
        return None
    a, sep, b = spec.partition(":")
    if not sep:
        raise ValueError("--shape-decay は '始め:終わり'（更新回数）")
    a, b = int(a), int(b)
    if not 0 <= a < b:
        raise ValueError("--shape-decay は 0 ≤ 始め < 終わり")
    return a, b


def parse_lose_len(spec: str):
    """`λ[:H]` → (λ, H)。空なら (0, 200)"""
    if not spec:
        return 0.0, 200
    lam, _, h = spec.partition(":")
    lam, h = float(lam), int(h or 200)
    if not 0.0 <= lam < 1.0:
        raise ValueError("--lose-len の λ は 0 以上 1 未満（1 以上だと長い負けが引き分け以上になる）")
    if h < 1:
        raise ValueError("--lose-len の H は 1 以上")
    return lam, h


@dataclass
class Shaping:
    """報酬の整形の設定（1つの学習で固定。倍率だけが更新回数で変わる）"""
    names: list
    own: np.ndarray
    opp: np.ndarray
    mode: str = "plain"
    cap: float = 1.0
    lose_len: tuple = (0.0, 200)
    decay: tuple = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode {self.mode}（{' / '.join(MODES)}）")
        if not self.cap > 0:
            raise ValueError("cap は正")
        if "moves" not in self.names:
            raise ValueError("信号に moves が無い")

    @classmethod
    def from_spec(cls, spec="", mode="plain", cap=1.0, lose_len="", decay="", names=None):
        names = env_signal_names() if names is None else list(names)
        if names is None:
            raise ValueError("tsuitate_rl が局の信号（VecEnv.signals）を知らない。tsuitate-bot の rl-env を maturin develop で作り直す")
        own, opp = parse_weights(spec, names)
        return cls(names, own, opp, mode, cap, parse_lose_len(lose_len), parse_decay(decay))

    @property
    def has_weights(self):
        return bool(np.any(self.own != 0) or np.any(self.opp != 0))

    @property
    def active(self):
        return self.has_weights or self.lose_len[0] > 0

    @property
    def zero_sum(self):
        """両者の整形の和が常に 0 か（自己対局の R-NaD の前提。重みが反対称で、負けの長さの項が無いとき）"""
        return bool(np.allclose(self.own, -self.opp)) and self.lose_len[0] == 0

    @property
    def reward_scale(self):
        """1局の報酬の絶対値の上限（学習器へ渡す倍率。学習の全体で固定）"""
        return 1.0 + (self.cap if self.has_weights else 0.0) + self.lose_len[0]

    def multiplier(self, step: int) -> float:
        if self.decay is None:
            return 1.0
        a, b = self.decay
        return float(min(1.0, max(0.0, (b - step) / (b - a))))

    def potential(self, sig):
        """Φ [B, 2]（切る前・倍率を掛ける前）。sig は [B, 2, K]"""
        sig = sig.astype(np.float64)
        return sig @ self.own + sig[:, ::-1] @ self.opp

    def config(self):
        return {"own": {n: float(w) for n, w in zip(self.names, self.own) if w},
                "opp": {n: float(w) for n, w in zip(self.names, self.opp) if w},
                "mode": self.mode, "cap": self.cap, "lose_len": list(self.lose_len),
                "decay": list(self.decay) if self.decay else None, "reward_scale": self.reward_scale}


class ShapeRun:
    """1回の収集（`rnad.collect`）のあいだの整形の状態。作る（収集の頭の信号を渡す）→ env.step のたびに `step`。

    監視値も一緒に集める: 終局の信号と、両者の受理手の合計が `SNAP_PLIES` に届いた時点の信号（局ごと）。
    `shaping=None`（整形なし）なら監視だけ（`step` は 0 を返す）。

    **序盤の強制**（env の `--open-p`）: 強制した手は方策が選んだ手ではないので、その手で動いた Φ は報酬にしない。
    `forced0` / `step(forced=...)` に各側の強制した手の数（`VecEnv.forced_moves()` [B, 2]）を渡すと、**強制の手が
    指された env.step の Φ の変化を基準（Φ の初期値）へ繰り入れる**（その step の整形は 0。同じ step に方策の手が
    混ざっていても、その step は丸ごと数えない = 相手がまだ強制の途中の間は整形が付かない）。収集の頭の reset で
    指した強制の手は `sig0` に入っているので、もともと数えない。強制の手が反則で止まったとき（手数は増えない）は
    見分けないが、反則では盤面が動かない（動くのは反則の数の信号だけ）"""

    def __init__(self, names, sig0, shaping=None, mult=1.0, forced0=None):
        self.shaping = shaping
        self.mult = float(mult)
        self.sig = sig0.copy()
        B = len(sig0)
        self.moves_k = list(names).index("moves")
        self.phi0 = shaping.potential(sig0) if shaping is not None else None
        self.phic = np.zeros((B, 2))
        self.raw = np.zeros((B, 2))  # Φ − Φ(基準)（切る前・倍率を掛ける前）
        self.forced_n = None if forced0 is None else np.asarray(forced0).sum(axis=1)
        self.total = np.zeros((B, 2))  # 局ごとの整形の合計（監視用）
        self.capped = np.zeros((B, 2), dtype=bool)  # Φ が上限に当たったことがあるか（監視用）
        self.ended = np.zeros(B, dtype=bool)
        self.snaps = np.full((len(SNAP_PLIES),) + sig0.shape, np.nan, dtype=np.float32)
        self.snap_done = np.zeros((len(SNAP_PLIES), B), dtype=bool)

    def step(self, sig, done, rew, forced=None):
        """env.step の後の信号 `sig` [B, 2, K]・この step で終局した局 `done` [B]・env の報酬 `rew` [B, 2] → 足す報酬 [B, 2]。
        前の step までに終局した局（auto_reset=False で止まっている局）には何も足さない。
        `forced` [B, 2] は各側の強制した手の数（作るときに `forced0` を渡したとき。増えた局はこの step の Φ の変化を数えない）"""
        live = ~self.ended
        if (forced is None) != (self.forced_n is None):
            raise ValueError("強制した手の数は、作るとき（forced0）と step（forced）の両方に渡す")
        moved = None
        if forced is not None:
            forced_n = np.asarray(forced).sum(axis=1)
            moved = live & (forced_n > self.forced_n)
            self.forced_n = np.where(live, forced_n, self.forced_n)
        add = np.zeros((len(sig), 2))
        plies = sig[:, 0, self.moves_k] + sig[:, 1, self.moves_k]
        for j, thr in enumerate(SNAP_PLIES):
            hit = live & (plies >= thr) & ~self.snap_done[j]
            self.snaps[j, hit] = sig[hit]
            self.snap_done[j] |= hit
        sh = self.shaping
        if sh is not None and sh.active:
            if sh.has_weights:
                raw = sh.potential(sig) - self.phi0
                if moved is not None and moved.any():
                    # 強制の手が指された step: Φ の変化を基準へ繰り入れる（Φ − 基準は前の step のまま）
                    self.phi0[moved] += raw[moved] - self.raw[moved]
                    raw[moved] = self.raw[moved]
                self.raw = np.where(live[:, None], raw, self.raw)
                self.capped |= live[:, None] & (np.abs(raw) >= sh.cap)
                phic = np.clip(raw, -sh.cap, sh.cap) * self.mult
                if sh.mode == "potential":
                    phic[done] = 0.0  # 終局の Φ は 0（累計を差し戻す）
                phic = np.where(live[:, None], phic, self.phic)
                add += phic - self.phic
                self.phic = phic
            lam, horizon = sh.lose_len
            if lam > 0 and done.any():
                bonus = lam * self.mult * np.minimum(plies, horizon) / horizon
                add += np.where((done & live)[:, None] & (rew < 0), bonus[:, None], 0.0)
            self.total += add
        self.sig = np.where(live[:, None, None], sig, self.sig)
        self.ended |= done
        return add.astype(np.float32)


# ---------------------------------------------------------------- 監視値


def _moments(x):
    """[n, D] → {"n", "mean", "sd"}（NaN の行は除く）"""
    x = x[~np.isnan(x).any(axis=1)] if len(x) else x
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean": [round(float(v), 4) for v in x.mean(0)],
            "sd": [round(float(v), 4) for v in x.std(0)]}


def summarize(run: ShapeRun, outcome, learner=None):
    """局の信号の監視値。`outcome` [B, 2]（各側の勝敗 ±1 / 0）、`learner` [B]（学習側の色 0/1。自己対局は −1 か None）。

    学習側から見た並び [自分の K 個, 相手の K 個, 自分 − 相手の K 個] で、勝ち・負け・引き分けの局に分けて平均と標準偏差を出す。
    自己対局の局は両方の側を1標本ずつ数える（勝った側の視点と負けた側の視点）。
    `final` = 終局時、`at` = 両者の受理手の合計が SNAP_PLIES に届いた時点（届かなかった局は除く）"""
    B = len(outcome)
    learner = np.full(B, -1) if learner is None else np.asarray(learner)
    rows_b, rows_p = [], []
    for b in range(B):
        for p in ((int(learner[b]),) if learner[b] >= 0 else (0, 1)):
            rows_b.append(b)
            rows_p.append(p)
    rows_b, rows_p = np.array(rows_b, dtype=np.int64), np.array(rows_p, dtype=np.int64)
    res = outcome[rows_b, rows_p]

    def view(sig):  # [B, 2, K] → [rows, 3K]（自分, 相手, 自分 − 相手）
        own, opp = sig[rows_b, rows_p], sig[rows_b, 1 - rows_p]
        return np.concatenate([own, opp, own - opp], axis=1)

    out = {}
    final = view(run.sig)
    snaps = [view(s) for s in run.snaps]
    for name, sel in (("win", res > 0), ("loss", res < 0), ("draw", res == 0)):
        out[name] = {"final": _moments(final[sel]),
                     "at": {str(thr): _moments(s[sel]) for thr, s in zip(SNAP_PLIES, snaps)}}
    if run.shaping is not None and run.shaping.active:
        tot = run.total[rows_b, rows_p]
        # 1局の整形の合計（plain なら終局の Φc ＋負けの長さ、potential なら負けの長さだけ）と、Φ が上限に当たった割合
        out["shaped_total"] = {"mean": float(tot.mean()), "abs_mean": float(np.abs(tot).mean()),
                               "capped": float(run.capped[rows_b, rows_p].mean())}
    return out


def dumps_config(shaping) -> str:
    return json.dumps(shaping.config(), ensure_ascii=False, sort_keys=True)
