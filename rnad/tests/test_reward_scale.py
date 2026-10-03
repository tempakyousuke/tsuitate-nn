"""ボーナス込みの報酬の縮め方（issue #4、RNaDLearner.update の reward_scale）の検査。tsuitate_rl が要る。

    .venv/bin/python rnad/tests/test_reward_scale.py

- 各局・各側の報酬の合計を「1 + 入玉の λ + 固めの λ」で割ると −1〜1 に収まる
  （入玉: ゼロ和・非ゼロ和、固め、両方。価値の頭は tanh なので、収まらないと勝ち側が上限に張り付く）
- reward_scale を固定する元になる λ の予定の最大
- update(batch, reward_scale=s) は「報酬を s で割り、η を s で割った設定」での update と完全に同じ
"""

import copy
import pathlib
import sys
from dataclasses import replace

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import RNaDConfig, RNaDLearner, collect  # noqa: E402

DEV = torch.device("cpu")


def net():
    torch.manual_seed(0)
    return PolicyValueNet(8, 1).eval()


def returns(nyu=0.0, zero_sum=True, guard=0.0):
    kw = {"guard_bonus": guard} if guard else {}
    env = tsuitate_rl.VecEnv(32, seed=4, auto_reset=False, **kw)
    model = net()
    torch.manual_seed(3)
    batch = collect(env, model, DEV, nyugyoku_lambda=nyu, nyugyoku_mode="graded",
                    nyugyoku_zero_sum=zero_sum)
    return batch, batch["reward"].sum(dim=0)  # [B, 2] 各局・各側の合計


def check_bounds():
    for nyu, zero_sum, guard in ((4.0, True, 0.0), (4.0, False, 0.0), (0.0, True, 0.5), (4.0, False, 0.5)):
        _, ret = returns(nyu, zero_sum, guard)
        # 縮める必要のある報酬（合計が 1 を超える局・側）を実際に含んでいること（無ければ検査が空振りする）
        assert ret.abs().max() > 1.0, (nyu, zero_sum, guard, float(ret.abs().max()))
        scale = 1.0 + nyu + guard
        scaled = ret / scale
        assert scaled.abs().max() <= 1.0 + 1e-6, (nyu, zero_sum, guard, float(ret.abs().max()))
        print(f"OK: 入玉 λ={nyu}（{'ゼロ和' if zero_sum else '非ゼロ和'}）・固め λ={guard}: "
              f"報酬の最大 {float(ret.abs().max()):.2f} → 縮めて {float(scaled.abs().max()):.3f}")


def params(learner):
    return [p.detach().clone() for p in learner.net.parameters()]


def check_update_equivalence():
    batch, _ = returns(4.0, False, 0.0)
    cfg = RNaDConfig(eta=0.2, beta=16.0, lr=2e-4, schedule_sizes=(200,), schedule_repeats=(1,))
    base = RNaDLearner(net(), cfg, DEV)
    a = copy.deepcopy(base)
    b = copy.deepcopy(base)
    b.cfg = replace(cfg, eta=cfg.eta / 5.0)
    a.update(batch, reward_scale=5.0)
    b.update({**batch, "reward": batch["reward"] / 5.0})
    for x, y in zip(params(a), params(b)):
        assert torch.equal(x, y)
    assert a.cfg.eta == cfg.eta, "保存する設定の η が書き換わった"
    c = copy.deepcopy(base)
    d = copy.deepcopy(base)
    c.update(batch, reward_scale=1.0)
    d.update(batch)
    for x, y in zip(params(c), params(d)):
        assert torch.equal(x, y)
    try:
        copy.deepcopy(base).update(batch, reward_scale=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("reward_scale < 1 が通った")
    print("OK: update(reward_scale=s) は報酬と η を s で割った update と同じ")


def check_schedule_max():
    """reward_scale は λ の予定全体の最大で固定する（区切りごとに変えない）。その最大値の取り出し"""
    from train_rnad import parse_lambda_schedule
    at, mx = parse_lambda_schedule("1.0:800,0.5:400,0.25:400,0", 200)
    assert mx == 1.0 and at(0) == 1.0 and at(900) == 0.5 and at(5000) == 0.0
    at, mx = parse_lambda_schedule("0.5:400,2.0", 200)
    assert mx == 2.0 and at(0) == 0.5 and at(400) == 2.0
    at, mx = parse_lambda_schedule("", 200)
    assert mx == 0.0 and at(10) == 0.0
    print("OK: λ の予定の最大（reward_scale を予定全体で固定する元）")


if __name__ == "__main__":
    check_schedule_max()
    check_bounds()
    check_update_equivalence()
