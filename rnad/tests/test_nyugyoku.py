"""入玉ボーナス（rnad.collect）の報酬の配り方を検査する。tsuitate_rl が要る。

    .venv/bin/python rnad/tests/test_nyugyoku.py

- ゼロ和（既定）: 両側の報酬の和は 0（勝敗も ±1 の対なので、局ごとの和が 0）
- 非ゼロ和（own）: 両側の報酬の和 = 前進した側が得たボーナスの総額
  （各側の玉の最大前進段数から、届いた節目の数 × λ/4 を数え直したものと一致）
"""

import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import GRADED_MILESTONES, collect  # noqa: E402

LAM = 1.0


def run(zero_sum, win_only=False, n=16):
    torch.manual_seed(0)
    net = PolicyValueNet(8, 1).eval()
    env = tsuitate_rl.VecEnv(n, seed=0, auto_reset=False)
    return collect(env, net, torch.device("cpu"), nyugyoku_lambda=LAM, nyugyoku_mode="graded",
                   nyugyoku_zero_sum=zero_sum, nyugyoku_win_only=win_only)


def check_win_only():
    """nyugyoku_win_only: 勝った側だけが、届いた節目の数 × λ/4 を受け取る（負け・引き分けは 0）。
    対局は同じ乱数なら any と同じで、ゼロ和なら負けた側から勝った側のボーナスを引く"""
    for zero_sum in (False, True):
        base = run(zero_sum, n=48)
        win = run(zero_sum, win_only=True, n=48)
        assert np.array_equal(base["advance"], win["advance"]), "ボーナスの付け方で対局が変わった"
        reached = np.array([[sum(m <= a for m in GRADED_MILESTONES) for a in row] for row in win["advance"]])
        bonus = reached * LAM / len(GRADED_MILESTONES)  # [B, 2] 各側が届いた節目のボーナス
        outcome = win["outcome"]
        won = outcome > 0
        want = outcome + np.where(won, bonus, 0.0)
        if zero_sum:
            want = want - np.where(won[:, ::-1], bonus[:, ::-1], 0.0)
        got = win["reward"].sum(dim=0).numpy()
        assert np.allclose(got, want, atol=1e-5), (got, want)
        lost_with_bonus = int(((~won) & (bonus > 0)).sum())
        won_with_bonus = int((won & (bonus > 0)).sum())
        assert lost_with_bonus > 0 and won_with_bonus > 0, "勝ち・負けの両方で節目に届いた側が無く、検査になっていない"
        print(f"OK: 勝った局だけ（{'ゼロ和' if zero_sum else '非ゼロ和'}）: 勝って節目 {won_with_bonus} 側に付き、"
              f"負けて節目 {lost_with_bonus} 側には付かない")


def main():
    zero = run(True)
    per_game = zero["reward"].sum(dim=(0, 2))
    assert torch.allclose(per_game, torch.zeros_like(per_game), atol=1e-5), per_game

    own = run(False)
    reached = np.array([[sum(m <= a for m in GRADED_MILESTONES) for a in row] for row in own["advance"]])
    # ボーナスはその側の直前の着手に付けるので、最初の観測（着手前）の前進は数えない。
    # 初期配置の玉は前進0段で、節目（3段〜）には着手なしで届かない
    want = torch.from_numpy(reached.sum(axis=1) * LAM / len(GRADED_MILESTONES)).float()
    got = own["reward"].sum(dim=(0, 2))
    assert torch.allclose(got, want, atol=1e-5), (got, want)
    assert reached.sum() > 0, "節目に届いた玉が無く、検査になっていない"
    print(f"OK: ゼロ和は局ごとの和 0、非ゼロ和は和 = ボーナス総額（節目 {reached.sum()} 回）")


if __name__ == "__main__":
    main()
    check_win_only()
