"""checkpoint の入力プレーンを広げる・足す（tsuitate-bot の `rl::encode`）。

    .venv/bin/python rnad/widen_planes.py <元の checkpoint.pt> <出力の checkpoint.pt> [--planes 91]
    .venv/bin/python rnad/widen_planes.py <元> <出力> [--opp-hand-planes] [--opp-foul-planes] [--king-foul-planes]

- **v1 → v2**（`--planes 91`、既定）: v1 の 86枚の後ろに確定事実のプレーン 5枚
- **v2 → v3**（`--planes 101`）: v2 の後ろに相手の持ち駒の上界（7枚）と相手の反則の時系列（3枚）
- **→ v4**（`--planes 102`）: v3 の後ろに玉の移動で反則したマス（1枚）
- 組のフラグを付けると、プレーン数の既定はその組が入る最小の版。どの組を**使う**かはフラグで選ぶ
  （`--opp-hand-planes` = 案3、`--opp-foul-planes` = 案4、`--king-foul-planes` = 玉の反則マス）。
  選ばなかった組はネットが入力を 0 にして読まない（`model.PolicyValueNet`）。v3 / v4 の checkpoint に組を後から足すときも
  同じフラグで足せる（組を外すことはできない）。学習の `train_rnad.py` の同名のフラグはこれを読み込み時に行う

足した列（と新しく使い始める組の列）は stem の重みを **0** にするので、元と同じ関数から学習を続けられる。
R-NaD（net / target / prev / prev_）・MMD（net / target / magnet_net）・模倣学習（state_dict）の checkpoint を読める。

optimizer（Adam）の状態も stem の分を広げる:
- exp_avg は 0
- exp_avg_sq は**同じ出力チャネル・同じカーネル位置の、今使っている列の平均**（0 なら全体の平均）。0 にすると bias 補正後の
  2次モーメントが勾配の 2乗の 0.1% から始まり、足した列だけ最初の数百更新で約30倍の歩幅で動く。v3 で使っていなかった組の
  列は勾配が来ないあいだに 2次モーメントが減衰しているので、組を後から足すときも同じように入れ直す
ネットの形（`config["planes"]` と v3 の `config["extras"]`）以外の設定・step・乱数・補助ヘッドは元のまま。
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import (EXTENDED_PLANES, EXTRA_PLANES, NUM_PLANES_DEDUCED, normalize_extras, planes_for,  # noqa: E402
                   unused_planes)

STEM = "stem.weight"
NETS = ("net", "target", "prev", "prev_", "magnet_net", "state_dict")
# 学習のフラグ名 → 組の名前（train_rnad.py と共有）
EXTRA_FLAGS = {"opp_hand_planes": "opp_hand", "opp_foul_planes": "opp_fouls", "king_foul_planes": "king_fouls"}


def add_extra_flags(ap: argparse.ArgumentParser) -> None:
    """v3 の入力プレーンの組を選ぶフラグ（widen_planes.py と train_rnad.py で共有）"""
    ap.add_argument("--opp-hand-planes", action="store_true",
                    help="v3: 相手の持ち駒の上界（駒種別 7枚。取られた自駒の数で抑える）を入力に使う")
    ap.add_argument("--opp-foul-planes", action="store_true",
                    help="v3: 相手の反則の時系列（直前の手番 / 王手を受けていた手番 / 受けていない手番の 3枚）を入力に使う")
    ap.add_argument("--king-foul-planes", action="store_true",
                    help="v4: 自玉の移動で反則したマス（1枚。鮮度は反則した手数から）を入力に使う")


def extras_from_args(args) -> list:
    """フラグで選んだ組の名前。フラグを持たない引数（テストが手で作る Namespace など）は組なし"""
    return sorted(name for flag, name in EXTRA_FLAGS.items() if getattr(args, flag, False))


def widen_weight(w: torch.Tensor, planes: int) -> torch.Tensor:
    c, old, kh, kw = w.shape
    out = torch.zeros((c, planes, kh, kw), dtype=w.dtype, device=w.device)
    out[:, :old] = w
    return out


def widen_checkpoint(ck: dict, planes: int = NUM_PLANES_DEDUCED, extras=None) -> dict:
    """`ck` の入力を `planes` 枚へ広げ、v3 なら使う組を `extras` にする（None = 今の組のまま）。
    元と同じ関数のまま（足した列・新しく使う組の列は 0）。組を外す・狭める・何も変わらない指定は止める"""
    cfg = ck["config"]
    old = cfg["planes"]
    old_extras = normalize_extras(old, cfg.get("extras", ()))
    new_extras = old_extras if extras is None else normalize_extras(planes, extras)
    if planes < old:
        raise ValueError(f"プレーン数 {old} → {planes} は広げる向きではない")
    if set(old_extras) - set(new_extras):
        raise ValueError(f"入力プレーンの組は外せない（今 {old_extras} → 指定 {new_extras}）")
    if planes == old and new_extras == old_extras:
        raise ValueError(f"プレーン数 {old}・組 {old_extras} のままで、何も変わらない")
    off_before = set(unused_planes(old, old_extras))
    used_before = [i for i in range(old) if i not in off_before]
    # 新しく読み始める列: 広げた列と、新しく使う組のうち既にあった列
    added = [n for n in new_extras if n not in old_extras]
    fresh = list(range(old, planes)) + [i for n in added for i in range(*_span(n)) if i < old]

    ck = dict(ck)
    nets = [n for n in NETS if n in ck]
    if not nets:
        raise ValueError(f"ネットの重みが無い（{sorted(ck)}）")
    for name in nets:
        sd = dict(ck[name])
        w = widen_weight(sd[STEM], planes) if planes > old else sd[STEM].clone()
        w[:, fresh] = 0
        sd[STEM] = w
        ck[name] = sd
    # Adam の状態: param の順は net.parameters() = state_dict の重みの順（BatchNorm の running 統計を除く）。
    # stem.weight は最初の param。MMD を --init から始めたばかりなど、状態が無ければそのまま
    state0 = (ck.get("opt") or {}).get("state", {}).get(0)
    if state0 is not None and "exp_avg_sq" in state0:
        want = tuple(ck[nets[0]][STEM].shape[:1]) + (old,) + tuple(ck[nets[0]][STEM].shape[2:])
        if tuple(state0["exp_avg"].shape) != want:
            raise ValueError(f"optimizer の最初の状態が stem の形ではない: {tuple(state0['exp_avg'].shape)}")
        opt = {**ck["opt"], "state": dict(ck["opt"]["state"])}
        st = dict(state0)
        avg = widen_weight(st["exp_avg"], planes) if planes > old else st["exp_avg"].clone()
        avg[:, fresh] = 0
        sq = widen_weight(st["exp_avg_sq"], planes) if planes > old else st["exp_avg_sq"].clone()
        mean = st["exp_avg_sq"][:, used_before].mean(dim=1, keepdim=True)
        # 勾配が来ていない出力チャネル（死んだ ReLU）は全体の平均で埋める
        fill = torch.where(mean > 0, mean, st["exp_avg_sq"][:, used_before].mean())
        sq[:, fresh] = fill.expand(-1, len(fresh), -1, -1)
        st["exp_avg"], st["exp_avg_sq"] = avg, sq
        opt["state"][0] = st
        ck["opt"] = opt
    config = {k: v for k, v in cfg.items() if k != "extras"}
    config["planes"] = planes
    if planes in EXTENDED_PLANES:
        config["extras"] = new_extras
    ck["config"] = config
    return ck


def _span(name: str) -> tuple:
    start, n = EXTRA_PLANES[name]
    return start, start + n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--planes", type=int, default=None,
                    help=f"広げた後のプレーン数（既定: 組のフラグがあればその組が入る最小の版、無ければ {NUM_PLANES_DEDUCED}）")
    add_extra_flags(ap)
    args = ap.parse_args()
    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    flags = extras_from_args(args)
    # フラグの組は今の組に足す（v3 の checkpoint に後から足すとき、前に選んだ組を書き直さなくてよい）
    extras = sorted(set(ck["config"].get("extras", ())) | set(flags)) if flags else None
    planes = args.planes or (max(ck["config"]["planes"], planes_for(extras)) if flags else NUM_PLANES_DEDUCED)
    out = widen_checkpoint(ck, planes, extras)
    pathlib.Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.dst)
    print(f"{args.src} → {args.dst}: プレーン {ck['config']['planes']} → {planes}"
          f"{'、組 ' + str(out['config']['extras']) if 'extras' in out['config'] else ''}、step {out.get('step')}")


if __name__ == "__main__":
    main()
