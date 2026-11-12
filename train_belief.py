"""NN段階②（信念ネット）の学習。

入力CSVは tsuitate-bot の
`cargo run --release --bin export_belief_data -- records/*.jsonl > data.csv`
（ヘッダ: game_id,decision_id,side,sq,<特徴量35>,occ,role,promoted）。
1行=1マス。同じ (game_id, decision_id) の行が1つの決定点。

損失は占有の二値交差エントロピー + 駒種の交差エントロピー（補助、重み可変）。
**対局(game_id)単位で train/val を分割**する（同一対局の別決定点が漏れない
ようにする。nn-value-phase1 の行単位分割の教訓）。正規化統計も学習split
だけで計算する（同上）。

指標:
- val 対数損失 / Brier / AUC
- **一様事前（prior_occ 特徴量そのもの）との比較**。これを下回れなければ話に
  ならない下限
- `--baseline-col` を指定すると、そのCSV列（belief_probe が出す p_all =
  現行の粒子フィルタの占有信念）を同じ val 行で評価して並べる。
  ゲート1（「ネットが現行粒子より当たるか」）の判定はこの比較で行う

使い方:
    python train_belief.py --data data/belief_data.csv --out out_belief_seed0/ --seed 0
    # ゲート1の評価（belief_probe の出力を val としてそのまま使う）
    python train_belief.py --data data/belief_probe.csv --baseline-col p_all --epochs 0 \
        --load out_belief_seed0/
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

from belief_model import INPUT_DIM, ROLE_CLASSES, BeliefNet

FEATURE_NAMES = [
    "prior_occ",
    "opp_board_n",
    "opp_taken_n",
    "opp_moves_n",
    "in_check",
    "depth_opp",
    "file_center",
    "init_opp",
    "init_pawn",
    "init_lance",
    "init_knight",
    "init_silver",
    "init_gold",
    "init_bishop",
    "init_rook",
    "init_king",
    "opp_seen",
    "opp_seen_rec",
    "empty_seen",
    "empty_seen_rec",
    "untouched",
    "drop_foul",
    "drop_foul_rec",
    "to_foul",
    "path_foul",
    "king_foul",
    "my_attack",
    "my_adjacent",
    "dist_my_king",
    "neighbor_ev",
    "dist_opp_last",
    "king_line",
    "king_step",
    "king_cand",
    "king_cand_frac",
]

assert len(FEATURE_NAMES) == INPUT_DIM


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            "特徴量の列が足りない（tsuitate-bot の belief_features.rs が変わったら "
            f"train_belief.py::FEATURE_NAMES も更新すること）: {missing}"
        )
    # 順序ズレは黙って通すと意味が変わるので明示的に落とす
    present = [c for c in df.columns if c in FEATURE_NAMES]
    if present != FEATURE_NAMES:
        raise ValueError(f"特徴量の順序が不一致: CSV={present}")
    return df


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1 始まりの順位。同率の値には平均順位を与える（scipy.stats.rankdata の method="average" と同じ）"""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    first = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])  # 同率の組の先頭（並べた後の位置）
    counts = np.diff(np.r_[first, len(x)])
    avg = first + (counts + 1) / 2.0  # 組の中の順位 first+1 … first+counts の平均
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.repeat(avg, counts)
    return ranks


def binary_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    # AUC（順位ベース = Mann–Whitney U。scipy 無しで済ませる）。同率の予測には平均順位を与え（与えないと行の並びで
    # AUC が変わる）、対数損失のための clip の前の値で順位を付ける（clip すると端に張り付いた予測どうしが同率になる）
    ranks = average_ranks(p)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) if n_pos and n_neg else float("nan")
    p = np.clip(p, 1e-4, 1 - 1e-4)
    logloss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    brier = float(((p - y) ** 2).mean())
    return {"logloss": logloss, "brier": brier, "auc": auc, "n": int(len(y))}


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> str:
    edges = np.linspace(0, 1, bins + 1)
    lines = ["  予測区間      件数   予測平均  実測占有率"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.sum() == 0:
            continue
        lines.append(f"  [{lo:.1f},{hi:.1f})  {m.sum():7d}   {p[m].mean():.4f}    {y[m].mean():.4f}")
    return "\n".join(lines)


def evaluate(model: BeliefNet, x: torch.Tensor, y: np.ndarray, role: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        occ_logit, role_logit = model(x)
        p = torch.sigmoid(occ_logit).numpy()
        role_pred = role_logit.argmax(dim=1).numpy()
    m = binary_metrics(p, y)
    occupied = y == 1
    m["role_acc"] = float((role_pred[occupied] == role[occupied]).mean()) if occupied.any() else float("nan")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--load", type=Path, default=None, help="学習済みモデルを読んで評価だけする")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--role-weight", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, nargs=2, default=(64, 32))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--baseline-col",
        type=str,
        default=None,
        help="比較対象の確率列（belief_probe の p_all / p_strict）",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = load_data(args.data)
    print(f"データ: {len(df)}行 / {df['game_id'].nunique()}局 / 特徴量{INPUT_DIM}次元")

    x_all = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_all = df["occ"].to_numpy(dtype=np.float32)
    role_all = df["role"].to_numpy(dtype=np.int64)

    groups = df["game_id"].to_numpy()
    if args.epochs > 0:
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
        train_idx, val_idx = next(splitter.split(df, groups=groups))
    else:
        # 評価専用モード: 全行を val として扱う
        train_idx = np.arange(0, 0)
        val_idx = np.arange(len(df))

    if args.load is not None:
        stats = json.loads((args.load / "norm_stats.json").read_text())
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
    else:
        mean = x_all[train_idx].mean(axis=0)
        std = x_all[train_idx].std(axis=0)
        std[std == 0] = 1.0
    x_norm = (x_all - mean) / std

    x_t = torch.from_numpy(x_norm)
    y_t = torch.from_numpy(y_all)
    role_t = torch.from_numpy(role_all)

    model = BeliefNet(hidden=tuple(args.hidden))
    if args.load is not None:
        model.load_state_dict(torch.load(args.load / "model.pt", map_location="cpu"))

    # --- 対照1: 一様事前（prior_occ 特徴量そのもの） ---
    prior = df["prior_occ"].to_numpy(dtype=np.float64)[val_idx]
    y_val = y_all[val_idx].astype(np.float64)
    print("\n=== 対照 ===")
    print("一様事前 :", {k: round(v, 4) for k, v in binary_metrics(prior, y_val).items()})
    if args.baseline_col:
        base = df[args.baseline_col].to_numpy(dtype=np.float64)[val_idx]
        print(f"{args.baseline_col:9s}:", {k: round(v, 4) for k, v in binary_metrics(base, y_val).items()})

    if args.epochs > 0:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        bce = nn.BCEWithLogitsLoss()
        ce = nn.CrossEntropyLoss()
        best = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        since = 0
        rng = np.random.default_rng(args.seed)
        print(f"\ntrain: {len(train_idx)}行 / val: {len(val_idx)}行")
        for epoch in range(args.epochs):
            model.train()
            order = rng.permutation(train_idx)
            total = 0.0
            for start in range(0, len(order), args.batch_size):
                rows = order[start : start + args.batch_size]
                opt.zero_grad()
                occ_logit, role_logit = model(x_t[rows])
                loss = bce(occ_logit, y_t[rows]) + args.role_weight * ce(role_logit, role_t[rows])
                loss.backward()
                opt.step()
                total += loss.item() * len(rows)
            m = evaluate(model, x_t[val_idx], y_val, role_all[val_idx])
            if m["logloss"] < best:
                best = m["logloss"]
                best_state = copy.deepcopy(model.state_dict())
                since = 0
            else:
                since += 1
            if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
                print(
                    f"  epoch {epoch:3d}  train_loss={total / max(1, len(order)):.4f}  "
                    f"val_logloss={m['logloss']:.4f}  auc={m['auc']:.4f}  role_acc={m['role_acc']:.3f}"
                )
            if since >= args.patience:
                print(f"  epoch {epoch:3d}  早期終了（{args.patience}エポック改善なし）")
                break
        model.load_state_dict(best_state)

    m = evaluate(model, x_t[val_idx], y_val, role_all[val_idx])
    print("\n=== 信念ネット（val） ===")
    print({k: round(v, 4) for k, v in m.items()})
    with torch.no_grad():
        p_val = torch.sigmoid(model(x_t[val_idx])[0]).numpy()
    print(calibration_table(p_val, y_val))

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.out / "model.pt")
        (args.out / "norm_stats.json").write_text(
            json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2)
        )
        (args.out / "metrics.json").write_text(json.dumps(m, indent=2))
        print(f"\n保存: {args.out}")


if __name__ == "__main__":
    main()
