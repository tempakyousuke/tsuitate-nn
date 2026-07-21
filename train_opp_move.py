"""NN段階①-a: opp_move_weight の線形 vs MLP 比較。

使い方:
    python train_opp_move.py --data data/opp_move_data.csv

入力CSVは tsuitate-bot の
`cargo run --release --bin export_opp_move_data -- records/*.jsonl > data.csv`
の出力形式（ヘッダ: game_id,decision_id,特徴量12個,chosen）。1行=1候補手、
同じ(game_id,decision_id)の複数行が1つの決定点（候補集合）。

損失は候補集合内のsoftmax条件付き交差エントロピー（tsuitate-bot側の
bin/fit_opp.rs::log_likelihood と同じ定式化）。対局(game_id)単位で
train/valを分割し（同一対局の別決定点が漏れないように）、線形モデルと
1隠れ層MLPを同じデータ・同じ損失で学習して比較する。

指標: held-out val NLL（1候補あたりの平均負の対数尤度）・perplexity（exp(NLL)）・
top-1/top-3 accuracy（正解手がスコア上位に入る率）。
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from opp_move_model import INPUT_DIM, OppMoveNet

FEATURE_NAMES = [
    "advance",
    "promote_minor",
    "promote_major",
    "is_drop",
    "threat_known",
    "threat_home",
    "is_king_move",
    "king_flee",
    "deep_unsup_pawn",
    "deep_unsup_piece",
    "hang_minor",
    "hang_major",
    "opp_foul_count_this_turn",
    # 駒種特化ブロック（NN段階①-b、2026-07-22）
    "moved_pawn",
    "moved_lance",
    "moved_knight",
    "moved_silver",
    "moved_gold",
    "moved_bishop",
    "moved_rook",
    "moved_promoted",
    "move_dist",
    "from_home",
]

# ①-bのアブレーション比較用: base = ①-aの13特徴量（駒種ブロックなし）
BASE_FEATURES = FEATURE_NAMES[:13]


def load_data(path: Path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("game_id", "decision_id", "chosen")]
    if feature_cols != FEATURE_NAMES:
        raise ValueError(
            "特徴量の名前・順序が train_opp_move.py::FEATURE_NAMES と不一致 "
            "(tsuitate-bot側の opp_move_features.rs が変わったら両方更新すること): "
            f"CSV={feature_cols}"
        )
    if len(feature_cols) != INPUT_DIM:
        raise ValueError(f"特徴量次元が opp_move_model.INPUT_DIM={INPUT_DIM} と不一致")
    return df


def build_groups(df: pd.DataFrame) -> list[np.ndarray]:
    """(game_id, decision_id)ごとの行indexリスト。エクスポート順で既に
    連続しているのでgroupbyのindicesをそのまま使う"""
    idx = df.groupby(["game_id", "decision_id"], sort=False).indices
    groups = [np.asarray(v) for v in idx.values()]
    # exporter側のバグ・破損CSVを早期検出する（codexレビュー指摘、2026-07-21:
    # 候補が0/複数選択されているグループがあると group_nll が黙って [0][0] を
    # 拾ってしまい事故に気づきにくい）
    chosen = df["chosen"].to_numpy()
    bad = [g for g in groups if chosen[g].sum() != 1]
    if bad:
        raise ValueError(
            f"chosen=1の行がちょうど1つでない決定点が{len(bad)}件ある"
            "（CSVが壊れているかexporterのバグの疑い）"
        )
    return groups


def group_nll(scores: torch.Tensor, chosen: torch.Tensor, group_rows: list[np.ndarray]) -> torch.Tensor:
    """候補集合ごとのsoftmax条件付きNLLの平均（バッチ内のgroup_rowsぶん）"""
    losses = []
    for rows in group_rows:
        s = scores[rows]
        c = chosen[rows]
        logz = torch.logsumexp(s, dim=0)
        chosen_idx = c.nonzero(as_tuple=True)[0][0]
        losses.append(logz - s[chosen_idx])
    return torch.stack(losses).mean()


def evaluate(model, x, chosen, group_rows) -> dict:
    model.eval()
    with torch.no_grad():
        scores = model(x)
    nll = group_nll(scores, chosen, group_rows).item()
    top1 = 0
    top3 = 0
    for rows in group_rows:
        s = scores[rows]
        c = chosen[rows]
        chosen_idx = c.nonzero(as_tuple=True)[0][0].item()
        order = torch.argsort(s, descending=True)
        rank = (order == chosen_idx).nonzero(as_tuple=True)[0][0].item()
        top1 += rank == 0
        top3 += rank < 3
    n = len(group_rows)
    return {
        "nll": nll,
        "perplexity": float(np.exp(nll)),
        "top1": top1 / n,
        "top3": top3 / n,
        "n_groups": n,
    }


def train(
    model: OppMoveNet,
    x_train: torch.Tensor,
    chosen_train: torch.Tensor,
    train_groups: list[np.ndarray],
    x_val: torch.Tensor,
    chosen_val: torch.Tensor,
    val_groups: list[np.ndarray],
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_groups: int,
    patience: int,
) -> tuple[OppMoveNet, dict]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_nll = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: dict = {}
    epochs_since_improve = 0
    rng = np.random.default_rng(0)

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_groups))
        train_loss_sum = 0.0
        for start in range(0, len(order), batch_groups):
            batch_idx = order[start : start + batch_groups]
            batch_rows = [train_groups[i] for i in batch_idx]
            all_rows = np.concatenate(batch_rows)
            opt.zero_grad()
            scores = model(x_train[all_rows])
            # batch_rows は元のグローバル行indexなので、group_nll用にローカル
            # index（allrows内の位置）へ振り直す
            offset = 0
            local_groups = []
            for rows in batch_rows:
                local_groups.append(np.arange(offset, offset + len(rows)))
                offset += len(rows)
            loss = group_nll(scores, chosen_train[all_rows], local_groups)
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * len(batch_rows)
        train_loss = train_loss_sum / len(train_groups)

        val_metrics = evaluate(model, x_val, chosen_val, val_groups)
        if val_metrics["nll"] < best_val_nll:
            best_val_nll = val_metrics["nll"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            print(
                f"  epoch {epoch:4d}  train_nll={train_loss:.4f}  "
                f"val_nll={val_metrics['nll']:.4f}  val_top1={val_metrics['top1']:.3f}"
            )
        if epochs_since_improve >= patience:
            print(f"  epoch {epoch:4d}  早期終了（{patience}エポック改善なし）")
            break

    model.load_state_dict(best_state)
    return model, best_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-groups", type=int, default=64)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--feature-set",
        choices=("base", "piece"),
        default="piece",
        help="base=①-aの13特徴量（アブレーション比較用）、piece=駒種ブロック込み23特徴量",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = load_data(args.data)
    feats = BASE_FEATURES if args.feature_set == "base" else FEATURE_NAMES
    print(f"データ: {len(df)}行 / {df['game_id'].nunique()}局 / 特徴量{len(feats)}次元 ({args.feature_set})")

    groups_col = df["game_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_row_idx, val_row_idx = next(splitter.split(df, groups=groups_col))
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_row_idx] = True

    x_all = df[feats].to_numpy(dtype=np.float32)
    chosen_all = df["chosen"].to_numpy(dtype=np.float32)

    # 正規化統計は学習split行だけで計算
    mean = x_all[train_mask].mean(axis=0)
    std = x_all[train_mask].std(axis=0)
    std[std == 0] = 1.0
    x_norm = (x_all - mean) / std

    x_t = torch.from_numpy(x_norm)
    chosen_t = torch.from_numpy(chosen_all)

    all_groups = build_groups(df)
    train_groups = [g for g in all_groups if train_mask[g[0]]]
    val_groups = [g for g in all_groups if not train_mask[g[0]]]
    print(f"train: {len(train_groups)}決定点 / val: {len(val_groups)}決定点")

    results = {}
    for name, hidden in [("linear", None), ("mlp", args.hidden)]:
        print(f"\n=== {name} (hidden={hidden}) ===")
        model = OppMoveNet(input_dim=len(feats), hidden=hidden)
        model, best_metrics = train(
            model,
            x_t,
            chosen_t,
            train_groups,
            x_t,
            chosen_t,
            val_groups,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.batch_groups,
            args.patience,
        )
        results[name] = best_metrics
        print(f"  最良val: {best_metrics}")

    # 一様分布（常に全候補に均等確率）の基準線
    uniform_nll = float(np.mean([np.log(len(g)) for g in val_groups]))
    print("\n=== 比較（held-out val） ===")
    print(f"{'':10} {'NLL':>8} {'perplexity':>11} {'top1':>7} {'top3':>7}")
    print(f"{'uniform':10} {uniform_nll:8.4f} {np.exp(uniform_nll):11.2f} {'-':>7} {'-':>7}")
    for name in ("linear", "mlp"):
        m = results[name]
        print(
            f"{name:10} {m['nll']:8.4f} {m['perplexity']:11.2f} "
            f"{m['top1'] * 100:6.1f}% {m['top3'] * 100:6.1f}%"
        )


if __name__ == "__main__":
    main()
