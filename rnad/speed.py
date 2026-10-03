"""速攻特化（決着の速さのボーナス）モデルの監視値の集計（tsuitate-bot の docs/rl-deepnash-design.md の
速攻特化の案、tsuitate-bot PR #65）。

報酬そのものは Rust 側の `VecEnv(speed_bonus=λ, speed_horizon=H, speed_sum=...)` が終局の報酬に足す
（勝った側に λ × max(0, (H − 手数) / H)。own = 勝った側だけ・非ゼロ和 / zero = 負けた側から同じ額を引く）。
ここは `pop_finished()` から学習側の勝ち方を集計するだけ。

- 自己対局の局は両側とも学習側（同じネット）なので、決着した局はどれも「学習側の勝ち」として数える
- 相手を固定した局（`learner` が 0/1）は学習側の勝ちと負けを分ける
"""


def speed_of(plies, horizon):
    """ボーナスの係数 max(0, (H − 手数) / H) ∈ [0, 1]"""
    return max(0.0, (horizon - plies) / horizon)


def summarize(finished, horizon):
    """`pop_finished()` の局のリストから、学習側の勝ち局の手数・速さ・詰みの割合などを返す。
    局が無ければ空の dict（空の平均は NaN になり、厳密な JSON でなくなるため）"""
    if not finished:
        return {}
    wins, losses, draws = [], [], 0
    for f in finished:
        if f["winner"] is None:
            draws += 1
        elif f.get("learner") is None or f["winner"] == f["learner"]:
            wins.append(f)
        else:
            losses.append(f)
    out = {"games": len(finished), "wins": len(wins), "losses": len(losses), "draws": draws}
    if wins:
        out["win_plies"] = sum(f["plies"] for f in wins) / len(wins)
        out["win_speed"] = sum(speed_of(f["plies"], horizon) for f in wins) / len(wins)
        out["win_mate"] = sum(f["reason"] == "checkmate" for f in wins) / len(wins)
    if losses:
        out["loss_plies"] = sum(f["plies"] for f in losses) / len(losses)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}
