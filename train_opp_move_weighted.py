"""opp_move NN を、系統ごとに重みを付けて学習し、Rust の定数ブロックを書き出す。

    python3 train_opp_move_weighted.py --data train.csv --out ../tsuitate-bot/src/opp_move_nn.rs --weight estimator_v14=13 --seed 3

入力 CSV は tsuitate-bot の scripts/opp_move/export_by_style.py の出力（export_opp_move_data の列に style を足したもの）。

`--weight 系統=k` の系統の決定点を学習と早期打ち切りの検証の両方で k 回数える（重み k と同じ）。
そのほかは tsuitate-nn の export_opp_move_weights.py と同じ（対局単位で 15% を検証、train_opp_move.train）。
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

import export_opp_move_weights as ex
from opp_move_model import OppMoveNet
from train_opp_move import FEATURE_NAMES, build_groups, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weight", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-groups", type=int, default=64)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    args = ap.parse_args()
    weights = {k: int(v) for k, _, v in (w.partition("=") for w in args.weight)}

    torch.manual_seed(args.seed)
    df = pd.read_csv(args.data)
    styles = df["style"].to_numpy()
    df = df.drop(columns=["style"])
    assert [c for c in df.columns if c not in ("game_id", "decision_id", "chosen")] == FEATURE_NAMES
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_idx, _ = next(splitter.split(df, groups=df["game_id"].to_numpy()))
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True

    x_all = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    chosen = torch.from_numpy(df["chosen"].to_numpy(dtype=np.float32))
    mean = x_all[train_mask].mean(axis=0)
    std = x_all[train_mask].std(axis=0)
    std[std == 0] = 1.0
    x_t = torch.from_numpy((x_all - mean) / std)

    groups = build_groups(df)
    train_groups, val_groups = [], []
    for g in groups:
        k = weights.get(styles[g[0]], 1)
        (train_groups if train_mask[g[0]] else val_groups).extend([g] * k)
    print(f"train {len(train_groups)} / val {len(val_groups)}（重み込みの決定点）  重み {weights}", flush=True)

    model = OppMoveNet(hidden=args.hidden)
    model, best = train(model, x_t, chosen, train_groups, x_t, chosen, val_groups, args.epochs, args.lr,
                        args.weight_decay, args.batch_groups, args.patience)
    print(f"最良val: {best}", flush=True)

    st = model.state_dict()
    w1, b1 = st["net.0.weight"].numpy(), st["net.0.bias"].numpy()
    w2, b2 = st["net.2.weight"].numpy().reshape(-1), float(st["net.2.bias"].numpy()[0])
    tag = ", ".join(f"{k}×{v}" for k, v in weights.items()) or "重みなし"
    gen = "\n".join([
        ex.BEGIN_MARKER,
        f"// 学習: seed={args.seed} hidden={args.hidden} val_nll={best['nll']:.4f} val_top1={best['top1']:.3f} "
        f"({best['n_groups']}決定点、重み込み)。教師は方策ネット13系統＋estimator_v14 を各160局（2026-10-30）、重み {tag}",
        "// 再生成: tsuitate-bot/docs/nn-direction.md の「opp_move NN の学び直し（2026-10-30）」",
        ex.rust_array_1d("OPP_MOVE_NN_MEAN", mean),
        ex.rust_array_1d("OPP_MOVE_NN_STD", std),
        ex.rust_array_2d("OPP_MOVE_NN_W1", w1),
        ex.rust_array_1d("OPP_MOVE_NN_B1", b1),
        ex.rust_array_1d("OPP_MOVE_NN_W2", w2),
        f"pub const OPP_MOVE_NN_B2: f64 = {b2:.8e};",
        ex.END_MARKER,
    ])
    text = args.out.read_text()
    pat = re.compile(re.escape(ex.BEGIN_MARKER) + r".*?" + re.escape(ex.END_MARKER), re.DOTALL)
    assert pat.search(text), "AUTO-GENERATED マーカーが無い"
    args.out.write_text(pat.sub(lambda _: gen, text))
    print(f"書き出し: {args.out}")


if __name__ == "__main__":
    main()
