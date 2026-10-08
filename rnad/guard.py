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


def apply_win_bonus(batch, parts, lam, radius=1):
    """**勝った局でだけ**固めのボーナスを付ける（`--guard-cond win`）。

    env には guard_bonus=0 を渡して（ボーナスなしで）対局させ、終局後に `pop_finished()` の記録から
    勝った側の占有率の平均 g を取り、その側の**最後の手番**へ λ × g を足す（Rust の guard_bonus が終局の
    報酬に足すのと同じ位置。勝者は記録の winner で見るので、他のボーナスと併用しても判定はぶれない）。
    負け・引き分けの局と、相手固定の局で相手が勝った局には付けない。
    parts = [(batch の局の並びでの先頭の位置, その env の pop_finished())...]。足した局数を返す
    """
    key = f"guard_r{radius}"
    valid, player, reward = batch["valid"], batch["player"], batch["reward"]
    n = 0
    for offset, finished in parts:
        for j, f in enumerate(sorted(finished, key=lambda f: f["game_no"])):
            w = f["winner"]
            if w is None or (f.get("learner") is not None and w != f["learner"]):
                continue
            b = offset + j
            steps = (valid[:, b] & (player[:, b] == w)).nonzero().squeeze(1)
            if len(steps) == 0:
                continue
            reward[int(steps[-1]), b, w] += lam * float(f[key][w])
            n += 1
    return n
