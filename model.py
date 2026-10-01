"""局面valueネット（フェーズ1）のモデル定義。

入力: tsuitate-bot の src/value_features.rs::VALUE_FEATURES 次元の特徴量
（正規化済み）。出力: その局面の手番側の勝率相当（0..1、sigmoid）。

小さいMLPから始める（14次元入力・数万行のデータセット向け。GPUは不要）。
"""

import torch
from torch import nn

# tsuitate-bot の value_features.rs の現行出力（state 16 + transition 6 = 22）と一致させること。
# 接近特徴量＋紐（2026-07-25〜26 の 35 次元の実験）は不採用で終了し、2026-08-21 の
# データ刷新で本番構成へ戻した（train.py の STATE_FEATURE_NAMES の注を参照）
INPUT_DIM = 22


class ValueNet(nn.Module):
    def __init__(
        self, input_dim: int = INPUT_DIM, hidden: int = 64, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
