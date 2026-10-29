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
import pathlib


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
    data = json.loads(raw)
    lines = data.get("lines") or []
    if not lines:
        raise ValueError(f"{path} にラインが無い")
    moves = [list(map(str, line["moves"])) for line in lines]
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
    }
    if contact:
        out["contact_ply"] = sum(contact) / len(contact)
    return out
