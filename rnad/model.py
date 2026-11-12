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
# エンコーダ v2（tsuitate-bot の `rl::encode::NUM_PLANES_DEDUCED`）: v1 の 86枚の後ろに観測から確定する事実を5枚。
# v2 の観測の前半は v1 と同じなので、v1 のネットは v2 の観測の先頭 86枚だけを読めばよい（trunk が切る）
NUM_PLANES_DEDUCED = 91
# エンコーダ v3（tsuitate-bot の `rl::encode::NUM_PLANES_V3` / `EXTRA_PLANE_GROUPS`、2026-11-08）: v2 の後ろに、観測と
# 自分側の情報だけで決まる事実を2組。**並びは固定**で、どの組を使うかはネットの `config["extras"]` が持つ。
# 使わない組は入力を 0 にしてから stem に通す（stem のその列は 0 のまま動かない = 書き出した重みでも列が 0 なので、
# Rust 側は組の選び方を知らずに v3 の観測を全部書けばよい）。並びが固定なので、組の選び方が違うネットどうし
# （錨・リーグ）も同じ観測を読める。v1・v2 のネットは先頭だけを読む（trunk が切る）
NUM_PLANES_V3 = 101
# エンコーダ v4（tsuitate-bot の `rl::encode::NUM_PLANES_V4`）: v3 の後ろに1枚。組の扱いは v3 と同じで、v3 のネットは
# v4 の観測の先頭を読む（v3 の実験の checkpoint をそのまま使える）
NUM_PLANES_V4 = 102
# 組を持てるネットのプレーン数
EXTENDED_PLANES = (NUM_PLANES_V3, NUM_PLANES_V4)
EXTRA_PLANES = {
    "opp_hand": (91, 7),  # 案3: 相手の持ち駒の上界（駒種別）
    "opp_fouls": (98, 3),  # 案4: 相手の反則の時系列（直前の手番 / 王手を受けていた手番 / 受けていない手番）
    "king_fouls": (101, 1),  # v4: 自玉の移動で反則したマス（鮮度は反則した手数から）
}
NUM_ACTION_KINDS = 139


def planes_for(extras) -> int:
    """組 `extras` を入れられる最小のプレーン数（組が無ければ v2）"""
    need = max((s + n for name, (s, n) in EXTRA_PLANES.items() if name in set(extras or ())), default=0)
    if need == 0:
        return NUM_PLANES_DEDUCED
    return min(p for p in EXTENDED_PLANES if p >= need)


def normalize_extras(planes: int, extras) -> list:
    """v3 / v4 のネットが使う組の名前（並べ替え済み）。v1・v2 のネットは組を持てない。
    組はネットのプレーン数に収まっていること（v4 の組は v3 のネットには入らない）"""
    extras = sorted(set(extras or ()))
    unknown = set(extras) - set(EXTRA_PLANES)
    if unknown:
        raise ValueError(f"知らない入力プレーンの組: {sorted(unknown)}（{sorted(EXTRA_PLANES)} のどれか）")
    if extras and planes not in EXTENDED_PLANES:
        raise ValueError(f"入力プレーンの組 {extras} は v3 / v4（{EXTENDED_PLANES}枚）のネットだけが持てる（planes {planes}）")
    over = [n for n in extras if sum(EXTRA_PLANES[n]) > planes]
    if over:
        raise ValueError(f"入力プレーンの組 {over} は {planes}枚のネットに入らない（{planes_for(over)}枚が要る）")
    return extras


def unused_planes(planes: int, extras) -> list:
    """ネットが使わない入力プレーンの番号（v3 / v4 で選ばなかった組）。v1・v2 は空"""
    if planes not in EXTENDED_PLANES:
        return []
    extras = normalize_extras(planes, extras)
    return [i for name, (s, n) in EXTRA_PLANES.items()
            if name not in extras and s + n <= planes for i in range(s, s + n)]


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
    def __init__(self, channels: int = 64, blocks: int = 6, planes: int = NUM_PLANES, extras=()):
        super().__init__()
        extras = normalize_extras(planes, extras)
        self.config = {"planes": planes, "channels": channels, "blocks": blocks}
        if planes in EXTENDED_PLANES:
            self.config["extras"] = extras
        self.stem = nn.Conv2d(planes, channels, 3, padding=1, bias=False)
        # 使わない入力プレーン（v3 で選ばなかった組）。入力を 0 にするので stem のその列には勾配が来ず、
        # 0 で始めた列は 0 のまま（書き出した重みも 0）。state_dict には入れない（config から作り直せる）
        off = unused_planes(planes, extras)
        if off:
            mask = torch.zeros(planes, dtype=torch.bool)
            mask[off] = True
            self.register_buffer("in_off", mask.view(1, planes, 1, 1), persistent=False)
            with torch.no_grad():
                self.stem.weight[:, off] = 0
        else:
            self.in_off = None
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
        """幹（stem と残差ブロック）の出力 [B, channels, 9, 9]。補助ヘッドはここから読む。
        入力のプレーンがネットより多いとき（v2 の観測を v1 のネットへ）は先頭だけを使う"""
        planes = self.config["planes"]
        if x.shape[1] > planes:
            x = x[:, :planes]
        if self.in_off is not None:
            x = x.masked_fill(self.in_off, 0)
        x = torch.relu(self.stem_bn(self.stem(x)))
        for b in self.blocks:
            x = b(x)
        return x

    @classmethod
    def from_config(cls, config: dict) -> "PolicyValueNet":
        """checkpoint の `config`（や重みファイルのヘッダ）からネットの形を作る（v3 の組の選び方も含む）"""
        return cls(config["channels"], config["blocks"], config["planes"], config.get("extras", ()))

    def forward(self, x):
        return self.heads(self.trunk(x))

    def heads(self, x):
        """幹の出力 → (方策のロジット, 価値)"""
        p = torch.relu(self.p1_bn(self.p1(x)))
        logits = self.p2(p).flatten(1)  # [B, 139*81]、種類×81＋マス
        v = torch.relu(self.v1_bn(self.v1(x))).flatten(1)
        v = torch.tanh(self.v_fc2(torch.relu(self.v_fc1(v)))).squeeze(1)
        return logits, v


def inference_net(config: dict) -> PolicyValueNet:
    """推論にだけ使うネット（重みを後から書き込む、または初期値の乱数で手が変わらないネット）を、
    **グローバルの乱数（CPU）を消費せずに**作る。形は `config`（`PolicyValueNet.from_config` と同じ）。

    `PolicyValueNet` の初期化は CPU のグローバル乱数を引く。checkpoint から乱数状態を戻した後でリーグの相手や
    パネルのネットを作ると、その分だけ続く対局のサンプリングがずれ、同じ seed・設定でも中断なしの学習と
    再開した学習が一致しなくなる。初期化で引いた乱数の状態を元に戻す（作られるネットの初期値は同じ）"""
    with torch.random.fork_rng(devices=[]):
        return PolicyValueNet.from_config(config)


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
