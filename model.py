"""局面valueネット（フェーズ1）のモデル定義。

入力: tsuitate-bot の src/value_features.rs::VALUE_FEATURES 次元の特徴量
（正規化済み）。出力: その局面の手番側の勝率相当（0..1、sigmoid）。

小さいMLPから始める（14次元入力・数万行のデータセット向け。GPUは不要）。
"""

import torch
from torch import nn

# tsuitate-bot の value_features.rs::VALUE_FEATURES + APPROACH_STATE_FEATURES
# + TRANSITION_FEATURES + APPROACH_TRANSITION_FEATURES と一致させること
# （16 + 4 + 6 + 5 = 31。接近特徴量 22→29、紐の2列を足して 31、2026-07-26）
INPUT_DIM = 31


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
