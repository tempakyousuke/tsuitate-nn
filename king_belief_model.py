"""玉位置ネット（king belief net）のモデル定義。

入力: tsuitate-bot の `belief_features.rs` の35特徴量 + KING_EXTRA（玉専用）
= 36次元（候補マス1つぶん、正規化済み）。出力は**スカラーのスコア**で、
決定点内の候補マス集合（deduce::opp_king_candidates − 自駒マス）上の
softmax がカテゴリカル分布になる。

段階②の pointwise 信念ネットとの違いは正規化の位置: あちらはマスごとの
独立確率（複数駒の同時制約を表現できない）、こちらは**玉1枚**の分布なので
候補集合内 softmax で同時制約の問題がそもそも起きない。
"""

import torch
from torch import nn

# tsuitate-bot の BELIEF_FEATURES + KING_EXTRA_FEATURES と一致させること
INPUT_DIM = 36


class KingBeliefNet(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.score_head = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score_head(self.trunk(x)).squeeze(-1)
