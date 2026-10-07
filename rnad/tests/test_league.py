"""リーグ学習（rnad/league.py と rnad.MultiEnv）の検査。tsuitate_rl が要る。

    .venv/bin/python rnad/tests/test_league.py

- League: 配分は2局単位で合計が合う・負けている相手ほど重い・pop_finished の取り込み
- MultiEnv: env 1つを包んだだけなら collect の結果が素の env と完全に同じ
- 自己対局 + 相手固定（heuristic）の混在: 相手固定の局は学習側の手番だけが記録され、
  終局の報酬が勝敗と一致する
"""

import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from league import League  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import MultiEnv, collect  # noqa: E402


def check_league():
    lg = League(["a", "b", "c"], window=8, power=2.0)
    assert np.allclose(lg.weights(), 1 / 3)  # 未対局は全員 0.5
    got = lg.record([{"opponent": "a", "winner": 0, "learner": 0}] * 8  # a には全勝
                    + [{"opponent": "b", "winner": 1, "learner": 0}] * 8  # b には全敗
                    + [{"opponent": "c", "winner": None, "learner": 0}] * 8  # c とは全分け
                    + [{"opponent": "外", "winner": 0, "learner": 0}])  # リーグの外は数えない
    assert got == {"a": [8, 0, 0], "b": [0, 8, 0], "c": [0, 0, 8]}, got
    w = dict(zip(lg.names, lg.weights()))
    assert w["b"] > w["c"] > w["a"], w
    rng = np.random.default_rng(0)
    for n in (0, 2, 64, 128):
        alloc = lg.allocate(n, rng)
        assert sum(alloc.values()) == n and all(k % 2 == 0 for k in alloc.values()), alloc
    lg2 = League(["a", "b", "c"], window=8)
    lg2.load_state_dict(lg.state_dict())
    assert lg2.summary() == lg.summary()
    print("OK: League の重み・配分・保存")

    # even: 五分の相手が一番重く、全勝・全敗の相手は軽い
    ev = League(["a", "b", "c"], window=8, power=1.0, weighting="even")
    ev.record([{"opponent": "a", "winner": 0, "learner": 0}] * 8
              + [{"opponent": "b", "winner": 1, "learner": 0}] * 8
              + [{"opponent": "c", "winner": 0, "learner": 0}] * 4 + [{"opponent": "c", "winner": 1, "learner": 0}] * 4)
    w = dict(zip(ev.names, ev.weights()))
    assert w["c"] > w["a"] and w["c"] > w["b"], w
    assert abs(w["a"] - w["b"]) < 1e-9, w  # 勝率 p と 1−p は同じ重み
    try:
        League(["a"], weighting="x")
    except ValueError:
        pass
    else:
        raise AssertionError("未知の weighting が通った")
    print("OK: even は五分の相手が一番重い")


def net():
    torch.manual_seed(0)
    return PolicyValueNet(8, 1).eval()


def check_single_env_identity():
    dev = torch.device("cpu")
    torch.manual_seed(1)
    a = collect(tsuitate_rl.VecEnv(8, seed=3, auto_reset=False), net(), dev)
    torch.manual_seed(1)
    b = collect(MultiEnv([tsuitate_rl.VecEnv(8, seed=3, auto_reset=False)]), net(), dev)
    for k in a:
        x, y = a[k], b[k]
        same = torch.equal(x, y) if isinstance(x, torch.Tensor) else np.array_equal(x, y)
        assert same, k
    print("OK: MultiEnv で包んでも collect の結果は同じ")


def check_mixed():
    dev = torch.device("cpu")
    selfplay = tsuitate_rl.VecEnv(4, seed=5, auto_reset=False)
    vs = tsuitate_rl.VecEnv(6, opponents=["heuristic"] * 6, seed=7, auto_reset=False)
    batch = collect(MultiEnv([selfplay, vs]), net(), dev)
    # collect は最初に reset するので、色は終わった後（auto_reset=False なので終局した局のまま）に取る
    colors = vs.learner_colors().astype(np.int64)
    finished = {f["game_no"]: f for f in vs.pop_finished()}
    valid, player, reward = batch["valid"], batch["player"], batch["reward"]
    for j in range(6):
        b = 4 + j
        steps = valid[:, b].nonzero().squeeze(1)
        assert len(steps) > 0
        assert (player[steps, b] == int(colors[j])).all(), "相手固定の局に相手の手番が記録された"
        last = int(steps[-1])
        f = finished[sorted(finished)[j]]
        want = 0.0 if f["winner"] is None else (1.0 if f["winner"] == f["learner"] else -1.0)
        assert f["learner"] == colors[j]
        got = float(reward[last, b, colors[j]])
        assert got == want, (j, got, want)
        assert float(reward[:, b, colors[j]].sum()) == want  # ボーナスなしなら終局の1回だけ
    # 自己対局の局は両方の手番が記録される
    assert set(player[valid[:, 0], 0].tolist()) == {0, 1}
    print(f"OK: 自己対局4局 + heuristic 相手6局（学習側 {colors.tolist()}）の混在")


if __name__ == "__main__":
    check_league()
    check_single_env_identity()
    check_mixed()
