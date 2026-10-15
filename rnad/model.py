"""方策・価値ネット（DeepNash 路線。tsuitate-bot の docs/rl-deepnash-design.md）。

入力は tsuitate-bot の `rl::encode`（[86, 9, 9]、手番側を先手向きに正規化）、
方策の出力は `rl::action` と同じ並び（[139, 9, 9] を平らにした 11,259、種類×81＋マス）。

Rust 側（`src/rl/policy_net.rs`）が同じ計算を手書きで持つので、層の種類は
3×3/1×1 畳み込み・BatchNorm（書き出し時に畳み込みへ畳み込む）・ReLU・残差・全結合・tanh
だけにとどめる。層を足したら Rust 側と `export_weights.py` も直すこと。
"""

import torch
import torch.nn as nn

NUM_PLANES = 86
NUM_ACTION_KINDS = 139


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 6, planes: int = NUM_PLANES):
        super().__init__()
        self.config = {"planes": planes, "channels": channels, "blocks": blocks}
        self.stem = nn.Conv2d(planes, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(ResBlock(channels) for _ in range(blocks))
        # 方策: 1×1 で 32ch → 1×1 で 139 種
        self.p1 = nn.Conv2d(channels, 32, 1, bias=False)
        self.p1_bn = nn.BatchNorm2d(32)
        self.p2 = nn.Conv2d(32, NUM_ACTION_KINDS, 1, bias=True)
        # 価値: 1×1 で 4ch → 全結合 128 → 1 → tanh
        self.v1 = nn.Conv2d(channels, 4, 1, bias=False)
        self.v1_bn = nn.BatchNorm2d(4)
        self.v_fc1 = nn.Linear(4 * 81, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def trunk(self, x):
        """幹（stem と残差ブロック）の出力 [B, channels, 9, 9]。補助ヘッドはここから読む"""
        x = torch.relu(self.stem_bn(self.stem(x)))
        for b in self.blocks:
            x = b(x)
        return x

    def forward(self, x):
        return self.heads(self.trunk(x))

    def heads(self, x):
        """幹の出力 → (方策のロジット, 価値)"""
        p = torch.relu(self.p1_bn(self.p1(x)))
        logits = self.p2(p).flatten(1)  # [B, 139*81]、種類×81＋マス
        v = torch.relu(self.v1_bn(self.v1(x))).flatten(1)
        v = torch.tanh(self.v_fc2(torch.relu(self.v_fc1(v)))).squeeze(1)
        return logits, v


NUM_TRUTH_CLASSES = 15


class TruthHead(nn.Module):
    """**真実の盤面を予測する補助ヘッド**（tsuitate-bot の docs/rl-deepnash-design.md「中盤の信号」の改善案1）。

    幹の出力から、手番側から見た各マスの相手駒（空き + 駒種14 = 15クラス。ラベルは
    `tsuitate_rl.VecEnv.truth()`）を予測する。学習の補助損失にだけ使い、**ネットの書き出しには入らない**
    （`PolicyValueNet` の外に持つので、`export_weights`・Rust の推論・凍結版は変わらない）。
    1×1 畳み込みはマスを区別しないので、マスごとのバイアス（事前）を足す
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(channels, NUM_TRUTH_CLASSES, 1)
        self.sq_bias = nn.Parameter(torch.zeros(81, NUM_TRUTH_CLASSES))

    def forward(self, x):
        return self.conv(x).flatten(2).transpose(1, 2) + self.sq_bias  # [B, 81, 15]


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """マスク外を -inf 相当にする（softmax・argmax・交差エントロピーの前に通す）"""
    return logits.masked_fill(~mask, -1e9)
