"""speedlab の鍵つきの乱数と Elo の当てはめの検査（tsuitate_rl は要らない）。

    .venv/bin/python rnad/tests/test_speedlab.py

- keyed_uniform は (salt, 局, 手番, 行動) だけで決まり、行の並びや一緒に作る行に依らない。一様に近い
- keyed_actions は合法手だけで乱数を作っても、全行動で作ったときと同じ手を選ぶ（マスク外は選ばない）
- fit_fixed はパネルのレートから離れた値でも発散しない
"""

import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import speedlab as S  # noqa: E402

DEV = torch.device("cpu")


def check_uniform():
    u = S.keyed_uniform(5, [0, 1, 2], [0, 0, 3], 300, DEV)
    v = S.keyed_uniform(5, [2], [3], 300, DEV)
    assert torch.equal(u[2], v[0])
    ids = torch.tensor([[7, 100, 299]])
    w = S.keyed_uniform(5, [2], [3], ids, DEV)
    assert torch.equal(w[0], u[2, [7, 100, 299]])
    x = S.keyed_uniform(1, np.arange(64), np.zeros(64, dtype=np.int64), 2000, DEV).flatten()
    assert 0 < float(x.min()) and float(x.max()) < 1
    assert abs(float(x.mean()) - 0.5) < 0.01 and abs(float(x.var()) - 1 / 12) < 0.002
    print("keyed_uniform: 鍵だけで決まり、一様に近い")


class FixedNet(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, x):
        return self.logits[: len(x)], torch.zeros(len(x))


def check_keyed_actions():
    torch.manual_seed(0)
    n, a = 40, 11259
    logits = torch.randn(n, a) * 3
    mask = np.random.default_rng(0).random((n, a)) < 0.01
    mask[:, 5] = True
    obs = np.zeros((n, 1, 9, 9), dtype=np.float32)
    games, decs = np.arange(n), np.arange(n) * 3
    got = S.keyed_actions(FixedNet(logits), obs, mask, games, decs, 11, DEV)
    # 全行動で乱数を作った場合
    u = S.keyed_uniform(11, games, decs, a, DEV)
    want = (logits.masked_fill(~torch.from_numpy(mask), float("-inf")) - torch.log(-torch.log(u)).float()).argmax(1)
    assert np.array_equal(got, want.numpy()), (got, want)
    assert mask[np.arange(n), got].all()
    print("keyed_actions: 合法手だけで乱数を作っても全行動の場合と同じ手")


def check_fit():
    ratings = {"a": 1431.0, "b": 1490.0, "c": 1539.0}
    r, se = S.fit_fixed([("a", 50, 14, 0), ("b", 40, 24, 0), ("c", 30, 34, 0)], ratings)
    assert 1400 < r < 1700 and 0 < se < 100, (r, se)
    r2, _ = S.fit_fixed([("a", 64, 0, 0)], ratings)
    assert r2 > 1431 and np.isfinite(r2)
    print(f"fit_fixed: {r:.0f}±{se:.0f}、全勝でも有限（{r2:.0f}）")


if __name__ == "__main__":
    check_uniform()
    check_keyed_actions()
    check_fit()
