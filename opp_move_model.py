"""相手手モデル（NN段階①、①-a: 線形 vs MLPの比較用）のモデル定義。

入力: tsuitate-bot の src/opp_move_features.rs::OPP_MOVE_FEATURES 次元
（正規化済み）。出力はlogit（Sigmoidではない）— 推論側（Rust）は
`w = exp(logit)` として使うため（既存の opp_move_weight と同じ規約）。

hidden=None なら線形（nn.Linear一発、fit_opp.rsの条件付きMLEと同じ表現力の
NN版）、hidden指定ありなら1隠れ層のMLP。同じ学習ループ・同じ損失関数で
比較し、「非線形化がそもそも効くか」を見る（①-a）。
"""

import torch
from torch import nn

# tsuitate-bot の opp_move_features.rs::OPP_MOVE_FEATURES と一致させること
# （2026-07-23: my_foul_count_last_turn 追加で23→24、
#  2026-07-24: en_prise_flee 追加で24→25。en_prise_flee_home 交互作用の
#  26次元化は bishop×home セルの反実仮想悪化で不採用）
INPUT_DIM = 26


class OppMoveNet(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden: int | None = 16) -> None:
        super().__init__()
        if hidden is None:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
