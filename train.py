"""局面valueネット（フェーズ1）の学習スクリプト。

使い方:
    python train.py --data data/value_data.csv --out out/
    python train.py --data data/value_data.csv --pairwise-data data/pairwise.csv --out out/

入力CSVは tsuitate-bot の `cargo run --release --bin export_value_data --
records/*.jsonl > data.csv` の出力形式（ヘッダ: game_id,ply,特徴量22個,label）。

--pairwise-data は `bin/export_pairwise_data` の出力（同一局面・合法手全体の
中でnet_capture_then_recaptureが最大/最小の手のペア）。最終勝敗ラベルだけでは
「駒を危険にさらす手」と「攻めている・優勢な手」が交絡し、transition特徴量の
符号を学習できなかった（docs/nn-value-phase1.md参照、2026-07-20）。同じ局面
（state共通）で手作り特徴量が示す向きを直接教える補助hinge lossとして使う。

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

# tsuitate-bot の value_features.rs::VALUE_FEATURE_NAMES + TRANSITION_FEATURE_NAMES
# と一致させること（名前・順序ともに。列数だけの一致チェックだと並び替え/取り違えを見逃す）
STATE_FEATURE_NAMES = [
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
    "my_max_exchange_loss",
    "opp_max_exchange_loss",
    "ply_progress",
    # 接近特徴量（2026-07-25 追加。value_features.rs::APPROACH_STATE_FEATURE_NAMES）
    "my_prey_proximity",
    "opp_prey_proximity",
    "my_king_siege",
    "opp_king_siege",
]
TRANSITION_FEATURE_NAMES = [
    "moved_piece_value",
    "moved_piece_hanging_value",
    "moved_piece_exchange_loss",
    "captured_value",
    "net_capture_then_recapture",
    "gives_check",
    # 接近特徴量（value_features.rs::APPROACH_TRANSITION_FEATURE_NAMES）
    "king_approach",
    "prey_approach",
    "king_closeness_after",
    "moved_piece_supported",
    "supported_prey_approach",
]
EXPECTED_FEATURE_NAMES = STATE_FEATURE_NAMES + TRANSITION_FEATURE_NAMES


def load_data(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    df = pd.read_csv(path)
    feature_names = [
        c for c in df.columns if c not in ("game_id", "ply", "label", "mat_swing")
    ]
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
    # mat_swing（N手先に実現した駒得。export_value_data::SWING_HORIZON）は
    # 任意列。無ければ勝敗ラベルのみで従来どおり学習する
    swing = (
        df["mat_swing"].to_numpy(dtype=np.float32)
        if "mat_swing" in df.columns
        else None
    )
    return x, y, groups, feature_names, swing


def load_pairwise_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    good_cols = [f"good_{n}" for n in TRANSITION_FEATURE_NAMES]
    bad_cols = [f"bad_{n}" for n in TRANSITION_FEATURE_NAMES]
    missing = [c for c in [*STATE_FEATURE_NAMES, *good_cols, *bad_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"pairwise CSVに想定の列が無い（export_pairwise_dataの出力？）: {missing}")
    state = df[STATE_FEATURE_NAMES].to_numpy(dtype=np.float32)
    good = df[good_cols].to_numpy(dtype=np.float32)
    bad = df[bad_cols].to_numpy(dtype=np.float32)
    groups = df["game_id"].to_numpy()
    return state, good, bad, groups


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
    pairwise: tuple[np.ndarray, np.ndarray] | None,
    pairwise_val: tuple[np.ndarray, np.ndarray] | None,
    pairwise_weight: float,
    pairwise_margin: float,
    pairwise_select_weight: float,
) -> tuple[ValueNet, float]:
    model = ValueNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(y_train)
    xv = torch.from_numpy(x_val)
    yv = torch.from_numpy(y_val)
    n = xt.shape[0]

    pw_good_t = pw_bad_t = None
    if pairwise is not None:
        pw_good_t = torch.from_numpy(pairwise[0])
        pw_bad_t = torch.from_numpy(pairwise[1])
    pw_good_v = pw_bad_v = None
    if pairwise_val is not None:
        pw_good_v = torch.from_numpy(pairwise_val[0])
        pw_bad_v = torch.from_numpy(pairwise_val[1])

    best_val = float("inf")
    best_select_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_since_improve = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        train_loss_sum = 0.0
        pw_loss_sum = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            pred = model(xt[idx])
            loss = loss_fn(pred, yt[idx])

            pw_loss = torch.tensor(0.0)
            if pw_good_t is not None:
                pw_idx = torch.randint(0, pw_good_t.shape[0], (len(idx),))
                score_good = model(pw_good_t[pw_idx])
                score_bad = model(pw_bad_t[pw_idx])
                pw_loss = torch.relu(pairwise_margin - (score_good - score_bad)).mean()

            total = loss + pairwise_weight * pw_loss
            total.backward()
            opt.step()
            train_loss_sum += loss.item() * len(idx)
            pw_loss_sum += pw_loss.item() * len(idx)
            n_batches += 1
        train_loss = train_loss_sum / n
        pw_loss_avg = pw_loss_sum / n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
            pw_acc = float("nan")
            if pw_good_v is not None:
                sg = model(pw_good_v)
                sb = model(pw_bad_v)
                pw_acc = (sg > sb).float().mean().item()
        # モデル選択・early stoppingはval_mse単体ではなくpairwise検証精度も
        # 加味した合成スコアで行う（codexレビュー指摘、2026-07-20: val_mseだけだと
        # pairwiseの目的である候補手ランキング改善が途中のepochを見捨てうる）
        select_score = val_loss
        if pw_good_v is not None:
            select_score = val_loss + pairwise_select_weight * (1.0 - pw_acc)
        if select_score < best_select_score:
            best_select_score = select_score
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch:4d}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}"
                f"  pw_train_hinge={pw_loss_avg:.4f}  pw_val_acc={pw_acc:.3f}"
            )
        # held-out lossがpatienceエポック改善しなければ打ち切る（過学習の暴走を防ぐ）
        if epochs_since_improve >= patience:
            print(f"epoch {epoch:4d}  早期終了（{patience}エポック改善なし）")
            break

    # 書き出すのは最終epochではなく held-out loss が最良だった時点の重み
    model.load_state_dict(best_state)
    return model, best_val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pairwise-data", type=Path, default=None)
    ap.add_argument("--pairwise-weight", type=float, default=0.3)
    ap.add_argument("--pairwise-margin", type=float, default=0.05)
    ap.add_argument("--pairwise-select-weight", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    # 目標値に「N手先に実現した駒得」を混ぜる割合と、そのsigmoidスケール（歩換算）
    ap.add_argument("--swing-weight", type=float, default=0.0)
    ap.add_argument("--swing-scale", type=float, default=8.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x, y, groups, feature_names, swing = load_data(args.data)
    print(f"データ: {x.shape[0]}行 x {x.shape[1]}特徴量 / {len(np.unique(groups))}局")

    # 目標値のブレンド: label（最終勝敗）と「N手先に実現した駒得」の合成。
    # この領域は対局の約66%が反則負けで終わる（アリーナ実測 2026-07-26）ため
    # 勝敗ラベルは反則レースにほぼ支配されていて、局面の良し悪しに対する
    # 教師としては弱い（実測: mat_swing と label の相関は 0.096）。
    # 「多手かけて駒を取りに行く」の価値は実現した駒得のほうが密な信号になる
    if args.swing_weight > 0.0:
        if swing is None:
            raise ValueError(
                "--swing-weight を使うには mat_swing 列が要る"
                "（export_value_data を作り直すこと）"
            )
        swing_target = 1.0 / (1.0 + np.exp(-swing / args.swing_scale))
        y = (1.0 - args.swing_weight) * y + args.swing_weight * swing_target
        print(
            f"目標値をブレンド: label {1 - args.swing_weight:.2f} + "
            f"駒得スイング {args.swing_weight:.2f}（scale={args.swing_scale}）"
        )

    # 対局（game_id）単位で分割する。同じ対局の別手番が学習/検証の両方に
    # 入るとデータリークになるため、行単位のtrain_test_splitは使わない
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(x, y, groups))
    train_games = set(np.unique(groups[train_idx]).tolist())
    val_games = set(np.unique(groups[val_idx]).tolist())

    # 正規化統計は学習split行だけで計算する（検証行の情報が前処理に
    # 漏れないようにする。codexレビュー指摘）
    mean = x[train_idx].mean(axis=0)
    std = x[train_idx].std(axis=0)
    std[std == 0] = 1.0  # 定数特徴量でゼロ割りしない
    x_norm = (x - mean) / std

    state_mean, state_std = mean[: len(STATE_FEATURE_NAMES)], std[: len(STATE_FEATURE_NAMES)]
    trans_mean, trans_std = mean[len(STATE_FEATURE_NAMES) :], std[len(STATE_FEATURE_NAMES) :]

    pairwise_train = pairwise_val = None
    if args.pairwise_data is not None:
        pw_state, pw_good, pw_bad, pw_groups = load_pairwise_data(args.pairwise_data)
        # pairwise CSVのgame_idはexporter内の連番に過ぎず、主データと別コーパス/
        # 別順序で生成されたものだと番号が偶然一致してリーク・誤分割になりうる
        # （codexレビュー指摘、2026-07-20）。同じ対局群から生成された前提を
        # game_id集合の包含関係で最低限検査する
        all_games = set(np.unique(groups).tolist())
        pw_games = set(np.unique(pw_groups).tolist())
        if not pw_games <= all_games:
            raise ValueError(
                "pairwise CSVのgame_idが主データの範囲外を含む: 別コーパス/別順序で"
                "生成された疑いがあるので、同じrecords集合から作り直すこと "
                f"(はみ出し例: {sorted(pw_games - all_games)[:5]})"
            )
        # 主データと同じgame_id分割を使う（held-out局のペアが学習側に漏れないように）
        pw_state_n = (pw_state - state_mean) / state_std
        pw_good_n = (pw_good - trans_mean) / trans_std
        pw_bad_n = (pw_bad - trans_mean) / trans_std
        x_good = np.concatenate([pw_state_n, pw_good_n], axis=1)
        x_bad = np.concatenate([pw_state_n, pw_bad_n], axis=1)
        train_mask = np.isin(pw_groups, list(train_games))
        val_mask = np.isin(pw_groups, list(val_games))
        pairwise_train = (x_good[train_mask], x_bad[train_mask])
        pairwise_val = (x_good[val_mask], x_bad[val_mask])
        print(
            f"pairwise: {train_mask.sum()}ペア学習 / {val_mask.sum()}ペア検証"
            f"（weight={args.pairwise_weight}, margin={args.pairwise_margin}）"
        )

    model, best_val = train(
        x_norm[train_idx],
        y[train_idx],
        x_norm[val_idx],
        y[val_idx],
        args.epochs,
        args.lr,
        args.weight_decay,
        args.batch_size,
        args.patience,
        pairwise_train,
        pairwise_val,
        args.pairwise_weight,
        args.pairwise_margin,
        args.pairwise_select_weight,
    )
    print(f"最良 val_mse={best_val:.4f}（held-out {len(val_idx)}行 / {len(np.unique(groups[val_idx]))}局）")

    model.eval()  # dropoutを無効化した状態で保存・エクスポートする
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
