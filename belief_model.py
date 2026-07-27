"""信念ネット（NN段階②）のモデル定義。

入力: tsuitate-bot の `src/belief_features.rs::BELIEF_FEATURES` 次元
（マス1つぶんの特徴量、正規化済み）。出力は**占有ロジット**
（そのマスに相手の駒が居る確率のロジット）と、駒種ヘッドのロジット9次元
（0=空 / 1..8=成る前の駒種）。

pointwise（マスごとに独立に評価）なので、盤面の同時制約（駒数・二歩・
行き所・駒間相関）は表現しない。これは意図的で、用途は粒子の**提案分布の
prior**であり、盤面の直接合成には使わない（docs/improvement-plan-2026-07-25.md
項目3の設計方針）。

1決定点あたり81回しか評価しない（粒子に依存しない）ので、value ネットや
opp_move ネットより大きめの隠れ層を許容できる。
"""

import torch
from torch import nn

# tsuitate-bot の belief_features.rs::BELIEF_FEATURES と一致させること
INPUT_DIM = 35
# 駒種ヘッドのクラス数（0=空, 1..8 = 歩香桂銀金角飛玉）
ROLE_CLASSES = 9


class BeliefNet(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.occ_head = nn.Linear(h2, 1)
        self.role_head = nn.Linear(h2, ROLE_CLASSES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.occ_head(h).squeeze(-1), self.role_head(h)
