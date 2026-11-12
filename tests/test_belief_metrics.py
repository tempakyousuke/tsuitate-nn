"""信念ネットの指標（train_belief.py の binary_metrics）の検査。

    .venv/bin/python tests/test_belief_metrics.py

- AUC は同率の予測に平均順位を与える: 全部同率なら 0.5、行の並びを入れ替えても変わらない
- 対数損失のための clip の前の値で順位を付ける（端に張り付いた予測どうしを同率にしない）
- 同率を含む予測で sklearn.metrics.roc_auc_score と一致する
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_belief import average_ranks, binary_metrics  # noqa: E402


def check_ties():
    p = np.array([0.5, 0.5])
    assert binary_metrics(p, np.array([1.0, 0.0]))["auc"] == 0.5
    assert binary_metrics(p, np.array([0.0, 1.0]))["auc"] == 0.5
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.3).astype(float)
    assert binary_metrics(np.full(200, 0.2), y)["auc"] == 0.5
    # 粗く丸めた予測（同率が多い）: 行の並びを入れ替えても AUC は同じ
    p = np.round(np.clip(y * 0.2 + rng.random(200) * 0.8, 0, 1), 1)
    a = binary_metrics(p, y)["auc"]
    for _ in range(5):
        perm = rng.permutation(200)
        assert binary_metrics(p[perm], y[perm])["auc"] == a
    print("AUC: 同率の予測は平均順位（全部同率なら 0.5、行の並びに依らない）")


def check_clip():
    # どちらも clip の下限（1e-4）より小さいが、順位は付く（clip の後で順位を付けると同率になり、行の並びで 0 になる）
    m = binary_metrics(np.array([1e-5, 1e-6]), np.array([1.0, 0.0]))
    assert m["auc"] == 1.0, m
    assert np.isfinite(m["logloss"])  # 対数損失は clip した値で
    print("AUC: clip の前の値で順位を付ける")


def check_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(5, 300))
        y = (rng.random(n) < rng.uniform(0.1, 0.9)).astype(float)
        if y.min() == y.max():
            continue
        p = np.round(rng.random(n), int(rng.integers(1, 3)))  # 同率を多く含む
        assert abs(binary_metrics(p, y)["auc"] - roc_auc_score(y, p)) < 1e-12
    assert np.array_equal(average_ranks(np.array([3.0, 1.0, 3.0, 2.0, 3.0])), [4.0, 1.0, 4.0, 2.0, 4.0])
    print("AUC: 同率を含む予測で sklearn の roc_auc_score と一致")


if __name__ == "__main__":
    check_ties()
    check_clip()
    check_sklearn()
    print("ok")
