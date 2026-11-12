"""序盤の強制の定跡ブック（`train_rnad.py --open-book`、`opening_eval.py`）。

自己対局で序盤を散らすため、局ごと・先後それぞれ独立に確率 p で、その側の最初の数手を
定跡のラインから指させる（`tsuitate_rl.VecEnv` の `openings` / `opening_prob`。env の中で指すので
学習器から見ると環境の手になる）。設計と計測は tsuitate-bot の docs/rl-opening-diversity.md。

ブックは tsuitate-bot の `joseki.json` の形式（`{"lines": [{"name", "moves", "category"}]}`、手は先手視点の
USI。後手は env が点対称にミラーする）。**`joseki.json` は凍結版の共有 pin なので書き換えない**。
別のラインを使うときは同じ形式の別ファイルを作る。各ラインは env が自駒だけの盤で検査し、
初期配置の相手の駒に触れる手（最後の駒取りなど）の手前で切る。
"""

import collections
import dataclasses
import hashlib
import json
import math
import pathlib

import numpy as np

import tsuitate_rl


@dataclasses.dataclass
class Book:
    lines: list  # 先手視点の USI の並び
    weights: list  # ラインごとの重み（env の opening_weights）
    categories: list
    names: list
    sha256: str

    def subset(self, category):
        """1つの系統だけのブック（重みは一様）"""
        idx = [i for i, c in enumerate(self.categories) if c == category]
        if not idx:
            raise ValueError(f"系統 {category} のラインが無い（{sorted(set(self.categories))}）")
        return Book([self.lines[i] for i in idx], [1.0] * len(idx), [category] * len(idx),
                    [self.names[i] for i in idx], self.sha256)


def load_book(path, weight="category"):
    """`weight`: category = 系統ごとに同じ重み（系統の中は一様。ラインの数の多い系統に寄らない）/
    line = ラインごとに一様。どちらもラインに "weight" があればそれを掛ける"""
    raw = pathlib.Path(path).read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} を JSON として読めない: {e}") from e
    lines = data.get("lines") if isinstance(data, dict) else None
    if not isinstance(lines, list) or not lines:
        raise ValueError(f"{path} にラインが無い（{{\"lines\": [{{\"moves\": [...]}}, ...]}} の形）")
    for i, line in enumerate(lines):
        if not isinstance(line, dict) or not isinstance(line.get("moves"), list) or not line["moves"] \
                or not all(isinstance(m, str) for m in line["moves"]):
            raise ValueError(f"{path} のライン {i}: moves が USI の文字列の並びでない")
        w = line.get("weight", 1.0)
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
            raise ValueError(f"{path} のライン {i}: weight は 0 以上の有限の数（{w!r}）")
    moves = [list(line["moves"]) for line in lines]
    cats = [str(line.get("category", "?")) for line in lines]
    names = [str(line.get("name", "?")) for line in lines]
    base = [float(line.get("weight", 1.0)) for line in lines]
    if weight == "category":
        count = collections.Counter(cats)
        weights = [w / count[c] for w, c in zip(base, cats)]
    elif weight == "line":
        weights = base
    else:
        raise ValueError(f"weight は category / line（{weight}）")
    return Book(moves, weights, cats, names, hashlib.sha256(raw).hexdigest())


def env_kwargs(book, prob, length=None, sides="both"):
    """`tsuitate_rl.VecEnv` へ渡す引数（prob = 0 なら何も渡さない = 古い tsuitate_rl でも動く）"""
    if not prob:
        return {}
    kw = {"openings": book.lines, "opening_weights": book.weights, "opening_prob": float(prob), "opening_sides": sides}
    if length is not None:
        kw["opening_len"] = tuple(length)
    return kw


def parse_len(spec):
    """'LO:HI' / 'N' → (lo, hi)。空なら None（ラインを切った後の最後まで）"""
    if not spec:
        return None
    lo, _, hi = spec.partition(":")
    lo, hi = int(lo), int(hi or lo)
    if not 1 <= lo <= hi:
        raise ValueError(f"手数の範囲は 1 ≤ LO ≤ HI（{spec}）")
    return lo, hi


def check_env_support():
    """入っている tsuitate_rl が序盤の強制（と終局の記録の `fouls_early`）を知っているか。知らなければ理由を返す
    （古いビルドは学習の最初の更新の後に KeyError で落ちるので、起動時に1局だけ指して確かめる）"""
    if not hasattr(tsuitate_rl.VecEnv, "set_opening_seed"):
        return "VecEnv に set_opening_seed が無い"
    env = tsuitate_rl.VecEnv(1, seed=0, auto_reset=False, openings=[["7g7f"]], opening_prob=1.0)
    for _ in range(600):
        if not env.alive().any():
            break
        _, mask, _ = env.observe()
        env.step(np.array([int(np.flatnonzero(mask[0])[0])], dtype=np.int64))
    fin = env.pop_finished()
    missing = [k for k in ("forced_moves", "forced_line", "forced_stop", "contact_ply", "fouls_precontact",
                           "fouls_early") if fin and k not in fin[0]]
    return f"終局の記録に {missing} が無い" if missing or not fin else None


def summarize(finished):
    """終局した局の記録（`VecEnv.pop_finished`）から、序盤の強制の監視値を出す"""
    sides = [(f["forced_line"][c], f["forced_moves"][c], f["forced_stop"][c]) for f in finished for c in (0, 1)]
    forced = [s for s in sides if s[0] is not None]
    contact = [f["contact_ply"] for f in finished if f.get("contact_ply") is not None]
    out = {
        "forced_frac": len(forced) / max(len(sides), 1),
        "forced_moves": (sum(s[1] for s in forced) / len(forced)) if forced else 0.0,
        "forced_stop": dict(collections.Counter(s[2] for s in forced)),
        "both_forced_frac": (sum(1 for f in finished if all(x is not None for x in f["forced_line"]))
                             / max(len(finished), 1)),
        "fouls_precontact_per_side": (sum(sum(f["fouls_precontact"]) for f in finished) / max(2 * len(finished), 1)),
        # 各側の受理手が 12 に届くまでの反則（速攻どうしは7〜8手目でぶつかるので、接触の前の反則はほぼ 0 になる）
        "fouls_early_per_side": (sum(sum(f["fouls_early"]) for f in finished) / max(2 * len(finished), 1)),
    }
    if contact:
        out["contact_ply"] = sum(contact) / len(contact)
    return out


class LineAdapter:
    """リーグの局の定跡のラインの重みを、学習側の直近の勝率で付け替える（`train_rnad.py --league-open-adapt γ`）。

    定跡に従う相手を搾取する学習では、勝てるようになったライン（同じ手口が通るライン）の局が batch の大半を占め、
    まだ勝てないラインを崩す手がかりが薄まる。そこで 重み ∝ 元の重み × max(1 − 勝率, floor)^γ とし、勝てていない
    ラインの局を増やす（γ = 0 は付け替えなし。勝てているラインも floor ぶんは残る = 忘れないための復習）。
    勝率はラインごとに直近 `window` 更新の [勝, 負, 分]（引き分けは 0.5 勝）で、`min_games` 局に満たないラインは
    勝率 0 として扱う（まだ分からないラインは減らさない）。状態はログの `league_opening.by_line` から作り直せる"""

    def __init__(self, base_weights, gamma, window=50, floor=0.05, min_games=8):
        if gamma < 0 or window < 1:
            raise ValueError(f"γ は 0 以上・window は 1 以上（{gamma}, {window}）")
        self.base = np.asarray(base_weights, dtype=np.float64)
        self.gamma, self.window, self.floor, self.min_games = float(gamma), int(window), float(floor), int(min_games)
        self.recent = collections.deque(maxlen=self.window)

    def push(self, by_line):
        """1更新ぶんの {ラインの添字（文字列）: [勝, 負, 分]} を足す"""
        self.recent.append({int(k): list(v) for k, v in (by_line or {}).items()})

    def win_rates(self):
        w = np.zeros(len(self.base))
        n = np.zeros(len(self.base))
        for rec in self.recent:
            for k, (win, loss, draw) in rec.items():
                w[k] += win + 0.5 * draw
                n[k] += win + loss + draw
        return np.where(n >= self.min_games, w / np.maximum(n, 1), 0.0), n

    def weights(self):
        wr, _ = self.win_rates()
        out = self.base * np.maximum(1.0 - wr, self.floor) ** self.gamma
        return (out / out.sum() * self.base.sum()).tolist()

    def restore(self, log_path, step):
        """再開: ログの step 以下の**更新の行**から直近 window 更新を読み直す（同じ step の更新の行が2回あれば後の行）。
        錨の評価の行（`eval`）は同じ step の更新の行の後に書かれ、`league_opening` を持たないので読まない
        （読むと、その step の勝敗を空で上書きしてしまう）"""
        rows = {}
        path = pathlib.Path(log_path)
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    x = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(x, dict) or "eval" in x:
                    continue
                if isinstance(x.get("step"), int) and x["step"] <= step:
                    rows[x["step"]] = (x.get("league_opening") or {}).get("by_line")
        self.recent.clear()
        for s in sorted(rows)[-self.window:]:
            self.push(rows[s])
