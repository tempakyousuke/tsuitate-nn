"""局面valueネット（フェーズ1）の学習スクリプト。

使い方:
    python train.py --data data/value_data.csv --out out/

入力CSVは tsuitate-bot の `cargo run --release --bin export_value_data --
records/*.jsonl > data.csv` の出力形式（ヘッダ: game_id,ply,特徴量14個,label）。

出力（--out ディレクトリへ）:
    model.onnx      — 推論用（tsuitate-bot側で将来のONNX統合に使う）
    model.pt        — PyTorch state_dict（held-out lossが最良だったepochの重み。
                       最終epochではない）
    norm_stats.json — 特徴量の正規化統計（**学習split行だけで計算**）と特徴量名。
                      **推論側（Rust）は同じ正規化を適用しないと出力がズレる**

検証はgame_id単位でグループ分割する（同じ対局の別手番が学習側と検証側の
両方に入るとデータリークになるため。codexレビュー指摘、2026-07-20）。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

from model import INPUT_DIM, ValueNet

# tsuitate-bot の value_features.rs::VALUE_FEATURE_NAMES と一致させること
# （名前・順序ともに。列数だけの一致チェックだと並び替え/取り違えを見逃す）
EXPECTED_FEATURE_NAMES = [
    "material_diff",
    "my_hand_value",
    "opp_hand_value",
    "king_pressure_on_me",
    "king_pressure_on_opp",
    "drop_check_danger_me",
    "drop_check_danger_opp",
    "my_in_check",
    "opp_in_check",
    "my_pieces_in_opp_camp",
    "opp_pieces_in_my_camp",
    "my_max_hanging",
    "opp_max_hanging",
    "ply_progress",
]


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    feature_names = [c for c in df.columns if c not in ("game_id", "ply", "label")]
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise ValueError(
            "特徴量の名前・順序が train.py::EXPECTED_FEATURE_NAMES と不一致 "
            "(tsuitate-bot側の value_features.rs が変わったら両方更新すること): "
            f"CSV={feature_names}"
        )
    if len(feature_names) != INPUT_DIM:
        raise ValueError(
            f"特徴量の次元が model.py の INPUT_DIM={INPUT_DIM} と不一致: "
            f"CSVは{len(feature_names)}列"
        )
    if "game_id" not in df.columns:
        raise ValueError(
            "CSVにgame_id列がない（古い形式のexport_value_data出力？ game単位の"
            "分割にgame_idが必須）"
        )
    x = df[feature_names].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    groups = df["game_id"].to_numpy()
    return x, y, groups, feature_names


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
) -> tuple[ValueNet, float]:
    model = ValueNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(y_train)
    xv = torch.from_numpy(x_val)
    yv = torch.from_numpy(y_val)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  train_mse={loss.item():.4f}  val_mse={val_loss:.4f}")

    # 書き出すのは最終epochではなく held-out loss が最良だった時点の重み
    model.load_state_dict(best_state)
    return model, best_val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    x, y, groups, feature_names = load_data(args.data)
    print(f"データ: {x.shape[0]}行 x {x.shape[1]}特徴量 / {len(np.unique(groups))}局")

    # 対局（game_id）単位で分割する。同じ対局の別手番が学習/検証の両方に
    # 入るとデータリークになるため、行単位のtrain_test_splitは使わない
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(x, y, groups))

    # 正規化統計は学習split行だけで計算する（検証行の情報が前処理に
    # 漏れないようにする。codexレビュー指摘）
    mean = x[train_idx].mean(axis=0)
    std = x[train_idx].std(axis=0)
    std[std == 0] = 1.0  # 定数特徴量でゼロ割りしない
    x_norm = (x - mean) / std

    model, best_val = train(
        x_norm[train_idx], y[train_idx], x_norm[val_idx], y[val_idx], args.epochs, args.lr
    )
    print(f"最良 val_mse={best_val:.4f}（held-out {len(val_idx)}行 / {len(np.unique(groups[val_idx]))}局）")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")

    dummy = torch.zeros(1, INPUT_DIM)
    torch.onnx.export(
        model,
        dummy,
        str(args.out / "model.onnx"),
        input_names=["features"],
        output_names=["value"],
        dynamic_axes={"features": {0: "batch"}, "value": {0: "batch"}},
    )

    with (args.out / "norm_stats.json").open("w") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"書き出し完了: {args.out}/model.onnx, model.pt, norm_stats.json")


if __name__ == "__main__":
    main()
