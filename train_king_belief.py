"""玉位置ネット（候補集合内 softmax）の学習。

入力CSVは tsuitate-bot の
`cargo run --release --bin export_king_belief_data -- records/*.jsonl > data.csv`
（ヘッダ: game_id,decision_id,side,n_cands,sq,<特徴量36>,is_king）。
1行=1候補マス。同じ (game_id, decision_id) の行が1つの決定点で、
is_king=1 の行がちょうど1つ。

損失は候補集合内の softmax 条件付き NLL（train_opp_move.py と同じ定式化だが、
390万行規模なので pad_sequence でベクトル化してある）。
**対局(game_id)単位で train/val を分割**し、正規化統計も学習splitだけで計算
（nn-value-phase1 の教訓）。

対照は一様分布（1/候補数）。粒子との比較（ゲート1）はここではなく
tsuitate-bot 側の king_probe --net で行う（同一決定点・同一条件で比較する）。

使い方:
    python train_king_belief.py --data data/king_belief_data.csv --out out_king_seed0/ --seed 0
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
from torch.nn.utils.rnn import pad_sequence

from king_belief_model import INPUT_DIM, KingBeliefNet

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
    # KING_EXTRA（belief_features.rs::KING_EXTRA_FEATURE_NAMES）
    "dist_init_king",
]

assert len(FEATURE_NAMES) == INPUT_DIM


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            "特徴量の列が足りない（tsuitate-bot の belief_features.rs が変わったら "
            f"train_king_belief.py::FEATURE_NAMES も更新すること）: {missing}"
        )
    present = [c for c in df.columns if c in FEATURE_NAMES]
    if present != FEATURE_NAMES:
        raise ValueError(f"特徴量の順序が不一致: CSV={present}")
    return df


class Groups:
    """決定点ごとの (開始行, 行数, 正解行のグループ内オフセット)。
    エクスポート順でグループの行は連続している前提（build時に検証する）"""

    def __init__(self, df: pd.DataFrame):
        keys = df[["game_id", "decision_id", "side"]].to_numpy()
        change = np.ones(len(df), dtype=bool)
        change[1:] = (keys[1:] != keys[:-1]).any(axis=1)
        self.starts = np.flatnonzero(change)
        self.sizes = np.diff(np.append(self.starts, len(df)))
        labels = df["is_king"].to_numpy()
        # 正解行のグループ内オフセット（ちょうど1つあることを検証）
        label_rows = np.flatnonzero(labels == 1)
        group_of_label = np.searchsorted(self.starts, label_rows, side="right") - 1
        if len(label_rows) != len(self.starts) or not np.array_equal(
            group_of_label, np.arange(len(self.starts))
        ):
            raise ValueError(
                "is_king=1 の行がちょうど1つでない決定点がある"
                "（CSVが壊れているか exporter のバグの疑い）"
            )
        self.label_offset = label_rows - self.starts

    def __len__(self) -> int:
        return len(self.starts)


def batch_nll(
    scores: torch.Tensor, sizes: np.ndarray, label_pos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """連続配置されたグループ列の (NLL, top1ヒット)。scores はグループ順に連結済み"""
    parts = torch.split(scores, sizes.tolist())
    padded = pad_sequence(list(parts), batch_first=True, padding_value=float("-inf"))
    logz = torch.logsumexp(padded, dim=1)
    starts = torch.from_numpy(np.concatenate(([0], np.cumsum(sizes)[:-1])))
    s_true = scores[starts + label_pos]
    top1 = padded.argmax(dim=1) == label_pos
    return (logz - s_true), top1


def evaluate(model, x, groups: Groups, idx: np.ndarray, batch: int = 4096) -> dict:
    model.eval()
    nll_sum = 0.0
    p_true_sum = 0.0
    top1_sum = 0
    with torch.no_grad():
        for start in range(0, len(idx), batch):
            gs = idx[start : start + batch]
            rows = np.concatenate([np.arange(groups.starts[g], groups.starts[g] + groups.sizes[g]) for g in gs])
            scores = model(x[rows])
            nll, top1 = batch_nll(
                scores, groups.sizes[gs], torch.from_numpy(groups.label_offset[gs])
            )
            nll_sum += nll.sum().item()
            p_true_sum += (-nll).exp().sum().item()
            top1_sum += top1.sum().item()
    n = len(idx)
    return {
        "nll": nll_sum / n,
        "top1": top1_sum / n,
        "p_true": p_true_sum / n,
        "n_groups": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--batch-groups", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--hidden", type=int, nargs=2, default=(64, 32))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = load_data(args.data)
    print(f"データ: {len(df)}行 / {df['game_id'].nunique()}局 / 特徴量{INPUT_DIM}次元", flush=True)

    groups = Groups(df)
    groups_col = df["game_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_row_idx, _ = next(splitter.split(df, groups=groups_col))
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_row_idx] = True

    x_all = df[FEATURE_NAMES].to_numpy(dtype=np.float32)

    mean = x_all[train_mask].mean(axis=0)
    std = x_all[train_mask].std(axis=0)
    std[std == 0] = 1.0
    x_t = torch.from_numpy((x_all - mean) / std)

    group_is_train = train_mask[groups.starts]
    train_g = np.flatnonzero(group_is_train)
    val_g = np.flatnonzero(~group_is_train)
    print(f"train: {len(train_g)}決定点 / val: {len(val_g)}決定点", flush=True)

    # 対照: 一様分布（1/候補数）
    val_sizes = groups.sizes[val_g]
    uniform_nll = float(np.mean(np.log(val_sizes)))
    uniform_top1 = float(np.mean(1.0 / val_sizes))
    print(f"一様事前: nll={uniform_nll:.4f} top1(期待値)={uniform_top1 * 100:.1f}%", flush=True)

    model = KingBeliefNet(hidden=tuple(args.hidden))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: dict = {}
    since = 0
    rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_g)
        loss_sum = 0.0
        for start in range(0, len(order), args.batch_groups):
            gs = order[start : start + args.batch_groups]
            rows = np.concatenate([np.arange(groups.starts[g], groups.starts[g] + groups.sizes[g]) for g in gs])
            opt.zero_grad()
            scores = model(x_t[rows])
            nll, _ = batch_nll(
                scores, groups.sizes[gs], torch.from_numpy(groups.label_offset[gs])
            )
            loss = nll.mean()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(gs)

        m = evaluate(model, x_t, groups, val_g)
        if m["nll"] < best:
            best = m["nll"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = m
            since = 0
        else:
            since += 1
        print(
            f"  epoch {epoch:3d}  train_nll={loss_sum / len(train_g):.4f}  "
            f"val_nll={m['nll']:.4f}  top1={m['top1'] * 100:.1f}%  p_true={m['p_true']:.4f}",
            flush=True,
        )
        if since >= args.patience:
            print(f"  epoch {epoch:3d}  早期終了（{args.patience}エポック改善なし）")
            break

    model.load_state_dict(best_state)
    print("\n=== 玉位置ネット（held-out val） ===")
    print(f"{'':10} {'NLL':>8} {'top1':>7} {'p_true':>8}")
    print(f"{'uniform':10} {uniform_nll:8.4f} {uniform_top1 * 100:6.1f}% {'-':>8}")
    print(
        f"{'net':10} {best_metrics['nll']:8.4f} {best_metrics['top1'] * 100:6.1f}% "
        f"{best_metrics['p_true']:8.4f}"
    )

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.out / "model.pt")
        (args.out / "norm_stats.json").write_text(
            json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2)
        )
        (args.out / "metrics.json").write_text(
            json.dumps({**best_metrics, "uniform_nll": uniform_nll}, indent=2)
        )
        print(f"\n保存: {args.out}")


if __name__ == "__main__":
    main()
