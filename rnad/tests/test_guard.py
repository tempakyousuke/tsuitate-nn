"""玉の周りの固めのボーナス（tsuitate_rl.VecEnv の guard_bonus）が collect の報酬に正しく載るかと、
監視値の集計（rnad/guard.py）を検査する。tsuitate_rl が要る。

    .venv/bin/python rnad/tests/test_guard.py

- 自己対局: 同じ乱数で λ=0 と λ>0 を回すと対局は同じで、局ごとの報酬の差 = λ × 両側の占有率の平均
- 相手固定（heuristic。`opponent=` と、リーグが使う `opponents=[...]` の両方）: 学習側の列の報酬 = 勝敗 + λ × 学習側の占有率の平均
- summarize: 自己対局は両側を学習側として数え、相手固定の局は相手側を分ける
"""

import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from guard import apply_win_bonus, summarize  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import MultiEnv, collect  # noqa: E402

LAM = 0.5
DEV = torch.device("cpu")


def net():
    torch.manual_seed(0)
    return PolicyValueNet(8, 1).eval()


def run(env, wrap=False):
    torch.manual_seed(1)
    batch = collect(MultiEnv([env]) if wrap else env, net(), DEV)
    # collect は最初に reset() するので局番号は 0 からではない。並びの位置 = 局番号の順
    fin = sorted(env.pop_finished(), key=lambda f: f["game_no"])
    assert len(fin) == batch["outcome"].shape[0]
    return batch, fin


def check_selfplay():
    n = 8
    base, _ = run(tsuitate_rl.VecEnv(n, seed=0, auto_reset=False))
    bonus, fin = run(tsuitate_rl.VecEnv(n, seed=0, auto_reset=False, guard_bonus=LAM))
    assert torch.equal(base["action"], bonus["action"]), "ボーナスは対局の進行を変えない"
    diff = (bonus["reward"] - base["reward"]).sum(dim=0)  # [B, 2]
    want = torch.tensor([fin[b]["guard_r1"] for b in range(n)]) * LAM
    assert torch.allclose(diff, want, atol=1e-5), (diff, want)
    print(f"OK: 自己対局 {n}局で報酬の差 = λ × 占有率（平均 {want.mean() / LAM:.3f}）")


def check_versus(**opp):
    n = 6
    env = tsuitate_rl.VecEnv(n, seed=2, auto_reset=False, guard_bonus=LAM, guard_radius=2, **opp)
    batch, fin = run(env, wrap=True)
    # 学習側の色は終局記録から取る（collect の reset で局番号が変わるので、事前の learner_colors は使わない）
    total = batch["reward"].sum(dim=0)
    for b in range(n):
        f = fin[b]
        me = f["learner"]
        got = total[b]
        result = 0.0 if f["winner"] is None else (1.0 if f["winner"] == me else -1.0)
        want = result + LAM * f["guard_r2"][me]
        assert abs(float(got[me]) - want) < 1e-5, (b, float(got[me]), want)
    print(f"OK: heuristic 相手 {n}局（{next(iter(opp))}=）で学習側の報酬 = 勝敗 + λ × 学習側の占有率（半径2）")


def check_win_only():
    """勝ち限定（apply_win_bonus）: 同じ対局で、勝った側の報酬の合計は Rust の guard_bonus（両者に付ける）と同じ、
    負けた側・引き分けは勝敗だけ。自己対局と相手固定（opponents=）の両方"""
    for kw, label in (({}, "自己対局"), ({"opponents": ["heuristic"] * 8}, "相手固定")):
        n = 8 if not kw else 8
        ref, fin_ref = run(tsuitate_rl.VecEnv(n, seed=4, auto_reset=False, guard_bonus=LAM, **kw), wrap=bool(kw))
        win, fin = run(tsuitate_rl.VecEnv(n, seed=4, auto_reset=False, guard_bonus=0.0, **kw), wrap=bool(kw))
        assert torch.equal(ref["action"], win["action"]), "ボーナスの付け方で対局が変わった"
        added = apply_win_bonus(win, [(0, fin)], LAM, radius=1)
        r_ref, r_win = ref["reward"].sum(dim=0), win["reward"].sum(dim=0)
        winners = 0
        for b, f in enumerate(fin):
            w = f["winner"]
            for side in (0, 1):
                outcome = 0.0 if w is None else (1.0 if w == side else -1.0)
                learner_side = f.get("learner") is None or f["learner"] == side
                if w == side and learner_side:
                    assert abs(float(r_win[b, side]) - float(r_ref[b, side])) < 1e-5, (label, b, side)
                    winners += 1
                elif learner_side:
                    assert abs(float(r_win[b, side]) - outcome) < 1e-5, (label, b, side, float(r_win[b, side]))
        assert winners == added and winners > 0, (winners, added)
        print(f"OK: 勝ち限定（{label}）: 勝った {winners} 側だけ Rust の guard_bonus と同じ、負け・引き分けは勝敗だけ")


def check_summarize():
    assert summarize([]) == {}
    common = {"near_drops": [1, 3], "near_drop_fouls": [0, 2], "near_captures": [1, 1]}
    selfplay = {"learner": None, "guard_r1": [0.2, 0.6], "fouls": [2, 4], **common}
    s = summarize([selfplay])
    assert s["guard_r1"] == 0.4 and s["fouls"] == 3 and s["near_drops"] == 2, s
    assert "opp_fouls" not in s, "自己対局だけなら相手側は無い"
    versus = {"learner": 1, "guard_r1": [0.2, 0.6], "fouls": [5, 1], **common}
    v = summarize([versus])
    assert v["guard_r1"] == 0.6 and v["fouls"] == 1 and v["near_drop_fouls"] == 2, v
    assert v["opp_guard_r1"] == 0.2 and v["opp_fouls"] == 5, v
    print("OK: summarize の学習側・相手側の振り分け")


if __name__ == "__main__":
    check_summarize()
    check_selfplay()
    check_versus(opponent="heuristic")
    # リーグ学習（train_rnad.py）が使う、相手を局ごとに並べる形
    check_versus(opponents=["heuristic"] * 6)
    check_win_only()
