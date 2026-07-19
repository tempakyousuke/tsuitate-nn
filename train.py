"""局面valueネット（フェーズ1）の学習スクリプト。

使い方:
    python train.py --data data/value_data.csv --out out/

入力CSVは tsuitate-bot の `cargo run --release --bin export_value_data --
records/*.jsonl > data.csv` の出力形式（ヘッダに特徴量名12個 + label）。

出力（--out ディレクトリへ）:
    model.onnx      — 推論用（tsuitate-bot側で将来のONNX統合に使う）
    model.pt        — PyTorch state_dict（再学習・デバッグ用）
    norm_stats.json — 特徴量の正規化統計（mean/std）と特徴量名。
                      **推論側（Rust）は同じ正規化を適用しないと出力がズレる**
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from model import INPUT_DIM, ValueNet


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    feature_names = [c for c in df.columns if c != "label"]
    if len(feature_names) != INPUT_DIM:
        raise ValueError(
            f"特徴量の次元が model.py の INPUT_DIM={INPUT_DIM} と不一致: "
            f"CSVは{len(feature_names)}列 ({feature_names})"
        )
    x = df[feature_names].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    return x, y, feature_names


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
        best_val = min(best_val, val_loss)
        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  train_mse={loss.item():.4f}  val_mse={val_loss:.4f}")

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

    x, y, feature_names = load_data(args.data)
    print(f"データ: {x.shape[0]}行 x {x.shape[1]}特徴量")

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0  # 定数特徴量でゼロ割りしない
    x_norm = (x - mean) / std

    x_train, x_val, y_train, y_val = train_test_split(
        x_norm, y, test_size=args.val_fraction, random_state=args.seed
    )

    model, best_val = train(x_train, y_train, x_val, y_val, args.epochs, args.lr)
    print(f"最良 val_mse={best_val:.4f}")

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
