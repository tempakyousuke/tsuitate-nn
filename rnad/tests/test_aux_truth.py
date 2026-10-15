"""真実の盤面を予測する補助ヘッド（RNaDConfig.aux_truth）の検査。tsuitate_rl（VecEnv.truth がある版）が要る。

    .venv/bin/python rnad/tests/test_aux_truth.py

- collect(truth=True) は記録した手番ごとに [N, 81] のラベルを返し、行動・報酬は truth=False と同じ
- aux_truth=0 の update は補助ヘッドを作らず、truth の有無で結果が変わらない（従来と同じ）
- aux_truth>0 の update は同じ batch で繰り返すと補助損失が下がり、方策・価値のネットの形は変わらない
- state_dict は補助ヘッドと optimizer を持ち回り、読み直すと同じ更新になる
- checkpoint に無い設定（aux_truth）は既定値として比べる（古い checkpoint の再開で差にならない）
"""

import copy
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import RNaDConfig, RNaDLearner, cfg_diff, collect  # noqa: E402

DEV = torch.device("cpu")


def net():
    torch.manual_seed(0)
    return PolicyValueNet(8, 1).eval()


def batch(truth):
    torch.manual_seed(1)
    env = tsuitate_rl.VecEnv(16, seed=7, auto_reset=False)
    return collect(env, net(), DEV, truth=truth)


def params(learner):
    return [p.detach().clone() for p in learner.net.parameters()]


def check_collect():
    a, b = batch(False), batch(True)
    assert "truth" not in a
    t = b["truth"]
    assert t.shape == (len(b["action"]), 81) and t.dtype == torch.int8, t.shape
    assert torch.equal(a["action"], b["action"]) and torch.equal(a["reward"], b["reward"])
    # 初手（各局の最初の記録）は相手の駒20枚・自駒20枚
    first = b["t"] == 0
    assert ((t[first] > 0).sum(1) == 20).all() and ((t[first] < 0).sum(1) == 20).all()
    assert int(t.max()) <= 14 and int(t.min()) >= -1


def check_off_is_unchanged():
    cfg = RNaDConfig(eta=0.2, beta=16.0, lr=2e-4, schedule_sizes=(200,), schedule_repeats=(1,))
    base = RNaDLearner(net(), cfg, DEV)
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    ia = a.update(batch(False))
    ib = b.update(batch(True))
    assert a.aux is None and b.aux is None and "loss_aux" not in ib
    for x, y in zip(params(a), params(b)):
        assert torch.equal(x, y)
    assert ia["loss_v"] == ib["loss_v"]
    assert "aux" not in a.state_dict()


def check_on_learns_and_roundtrips():
    bt = batch(True)
    cfg = RNaDConfig(eta=0.2, beta=16.0, lr=1e-3, schedule_sizes=(200,), schedule_repeats=(1,), aux_truth=1.0)
    lr = RNaDLearner(net(), cfg, DEV)
    shapes = [p.shape for p in lr.net.parameters()]
    losses = [lr.update(bt)["loss_aux"] for _ in range(30)]
    assert losses[-1] < losses[0] * 0.8, losses
    assert [p.shape for p in lr.net.parameters()] == shapes
    sd = copy.deepcopy(lr.state_dict())
    assert "aux" in sd and "aux_opt" in sd
    other = RNaDLearner(net(), cfg, DEV)
    other.load_state_dict(sd)
    lr.load_state_dict(copy.deepcopy(sd))  # 乱数の状態もそろえる
    x, y = lr.update(bt), other.update(bt)
    assert x["loss_aux"] == y["loss_aux"], (x["loss_aux"], y["loss_aux"])
    for p, q in zip(params(lr), params(other)):
        assert torch.equal(p, q)
    # aux_truth=0 の batch（truth なし）を渡すと止まる
    try:
        lr.update(batch(False))
    except ValueError:
        pass
    else:
        raise AssertionError("truth の無い batch で aux_truth>0 の update が通った")


def check_cfg_diff_default():
    old = dict(RNaDConfig().__dict__)
    del old["aux_truth"]
    assert cfg_diff(old, RNaDConfig()) == {}
    assert cfg_diff(old, RNaDConfig(aux_truth=0.1)) == {"aux_truth": (0.0, 0.1)}


if __name__ == "__main__":
    check_collect()
    check_off_is_unchanged()
    check_on_learns_and_roundtrips()
    check_cfg_diff_default()
    print("ok")
