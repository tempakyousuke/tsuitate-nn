"""R-NaD の checkpoint の入力プレーンを広げる（エンコーダ v1 → v2。tsuitate-bot の `rl::encode::NUM_PLANES_DEDUCED`）。

    .venv/bin/python rnad/widen_planes.py <元の checkpoint.pt> <出力の checkpoint.pt> [--planes 91]

v2 の観測は v1 の後ろに確定事実のプレーンを足しただけなので、4つのネット（net / target / prev / prev_）の
stem の重みに **0 の列**を足せば、v2 の観測に対して元と同じ関数になる（学習をそのまま続けられる）。

optimizer（Adam、betas=(0, 0.999)）の状態も stem の分を広げる:
- exp_avg は 0（b1=0 なので次の更新で上書きされる）
- exp_avg_sq は**同じ出力チャネル・同じカーネル位置の既存の列の平均**（0 なら全体の平均）。0 にすると bias 補正後の 2次モーメントが
  勾配の 2乗の 0.1% から始まり、足した列だけ最初の数百更新で約30倍の歩幅で動く
ネットの形（`config["planes"]`）以外の設定・step・乱数・補助ヘッドは元のまま。
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import NUM_PLANES_DEDUCED  # noqa: E402

STEM = "stem.weight"
NETS = ("net", "target", "prev", "prev_")


def widen_weight(w: torch.Tensor, planes: int) -> torch.Tensor:
    c, old, kh, kw = w.shape
    out = torch.zeros((c, planes, kh, kw), dtype=w.dtype, device=w.device)
    out[:, :old] = w
    return out


def widen_checkpoint(ck: dict, planes: int = NUM_PLANES_DEDUCED) -> dict:
    old = ck["config"]["planes"]
    if planes <= old:
        raise ValueError(f"プレーン数 {old} → {planes} は広げる向きではない")
    ck = dict(ck)
    for name in NETS:
        sd = dict(ck[name])
        sd[STEM] = widen_weight(sd[STEM], planes)
        ck[name] = sd
    # Adam の状態: param の順は net.parameters() = state_dict の重みの順（BatchNorm の running 統計を除く）。
    # stem.weight は最初の param
    opt = {"state": dict(ck["opt"]["state"]), "param_groups": ck["opt"]["param_groups"]}
    st = dict(opt["state"][0])
    if tuple(st["exp_avg"].shape) != tuple(ck["net"][STEM].shape[:1]) + (old,) + tuple(ck["net"][STEM].shape[2:]):
        raise ValueError(f"optimizer の最初の状態が stem の形ではない: {tuple(st['exp_avg'].shape)}")
    st["exp_avg"] = widen_weight(st["exp_avg"], planes)
    sq = widen_weight(st["exp_avg_sq"], planes)
    mean = st["exp_avg_sq"].mean(dim=1, keepdim=True)
    # 勾配が来ていない出力チャネル（死んだ ReLU）は全体の平均で埋める
    sq[:, old:] = torch.where(mean > 0, mean, st["exp_avg_sq"].mean())
    st["exp_avg_sq"] = sq
    opt["state"][0] = st
    ck["opt"] = opt
    ck["config"] = {**ck["config"], "planes": planes}
    return ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--planes", type=int, default=NUM_PLANES_DEDUCED)
    args = ap.parse_args()
    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    out = widen_checkpoint(ck, args.planes)
    pathlib.Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.dst)
    print(f"{args.src} → {args.dst}: プレーン {ck['config']['planes']} → {args.planes}、step {out['step']}")


if __name__ == "__main__":
    main()
