"""防御特化（玉の周りの固め）モデルの監視値の集計（tsuitate-bot の docs/rl-deepnash-design.md の
「防御特化（玉の周りの固め）モデル」）。

報酬そのものは Rust 側の `VecEnv(guard_bonus=λ, guard_radius=r)` が終局の報酬に足す
（λ × 対局を通した自玉の周りの占有率の平均、両者それぞれ = 非ゼロ和）。ここは `pop_finished()` の
集計を「学習側」と「相手側」に分けて平均するだけ。

- 自己対局の局は両側とも学習側（同じネット）なので、両側を学習側として数える
- 相手を固定した局（`learner` が 0/1）は、学習側の列と相手側の列を分ける。この戦法の狙いは
  相手の反則を誘うことなので、**相手側の反則数**を一緒に見る
"""

FIELDS = ("near_drops", "near_drop_fouls", "near_captures")


def _sides(f):
    """(学習側の列, 相手側の列 or None) の組を返す。自己対局は両側が学習側"""
    if f.get("learner") is None:
        return [(0, None), (1, None)]
    me = f["learner"]
    return [(me, 1 - me)]


def summarize(finished, radius=1):
    """`pop_finished()` の局のリストから、学習側（と相手を固定した局の相手側）の平均を返す。
    局が無ければ空の dict（空の平均は NaN になり、厳密な JSON でなくなるため）"""
    if not finished:
        return {}
    key = f"guard_r{radius}"
    mine = {k: [] for k in (key, "fouls", *FIELDS)}
    opp = {k: [] for k in (key, "fouls")}
    for f in finished:
        for me, other in _sides(f):
            mine[key].append(f[key][me])
            mine["fouls"].append(f["fouls"][me])
            for k in FIELDS:
                mine[k].append(f[k][me])
            if other is not None:
                opp[key].append(f[key][other])
                opp["fouls"].append(f["fouls"][other])
    out = {k: sum(v) / len(v) for k, v in mine.items()}
    if opp[key]:
        out[f"opp_{key}"] = sum(opp[key]) / len(opp[key])
        out["opp_fouls"] = sum(opp["fouls"]) / len(opp["fouls"])
    return {k: round(v, 4) for k, v in out.items()}
