"""粒子尤度モデルのNN化（likelihood.rs FITTED_THETA の後継）: 線形 vs MLP 比較。

使い方:
    python train_particle.py --data data/particle_data.csv

入力CSVは tsuitate-bot の
`cargo run --release --bin export_particle_data -- records/*.jsonl > data.csv`
の出力形式（ヘッダ: game_id,decision_id,offset,chosen,特徴量26個）。
1行=1候補局面、同じ(game_id,decision_id)の複数行が1グループ
（真実+ユニーク粒子群）。chosen=1 の行が真の局面。

損失は fit_particles.rs と同じ条件付きMLE: グループ内 softmax の NLL。
score = offset + model(x)（offset は推論側のベース対数重み = 指紋質量。
学習側にも同じオフセットを入れないと分布がずれる）。

指標: held-out val NLL・実効候補数 exp(NLL)（fit_particles と同じ、
小さいほど真実に質量が寄る）・真実がスコア上位半分に入る率。
基準線: offset のみ（現行のベース重みそのまま）と一様分布。
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from opp_move_model import OppMoveNet

FEATURE_NAMES = [
    # 線形モデルと同じ定義の8特徴量
    "king_moved",
    "king_advance",
    "king_shift",
    "pawn_advance",
    "pieces_home",
    "at_my_death",
    "in_my_camp",
    "past_mid",
    # 駒種別の初期配置残存
    "pawns_home",
    "lances_home",
    "knights_home",
    "silvers_home",
    "golds_home",
    "bishop_home",
    "rook_home",
    # 進出・成り・持ち駒
    "pawn_advance_max",
    "nonpawn_advance_max",
    "promoted_count",
    "opp_hand_count",
    # 自分の駒（既知）との相互作用
    "attacked_by_me",
    "hanging_to_me",
    "defended_frac",
    "king_zone_attackers",
    # 文脈（グループ内で不変。相互作用としてだけ効く）
    "ply",
    "my_dead",
    "you_in_check",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("game_id", "decision_id", "offset", "chosen")]
    if feature_cols != FEATURE_NAMES:
        raise ValueError(
            "特徴量の名前・順序が train_particle.py::FEATURE_NAMES と不一致 "
            "(tsuitate-bot側の likelihood.rs::NN_FEATURE_NAMES が変わったら両方更新すること): "
            f"CSV={feature_cols}"
        )
    return df


def build_groups(df: pd.DataFrame) -> list[np.ndarray]:
    idx = df.groupby(["game_id", "decision_id"], sort=False).indices
    groups = [np.asarray(v) for v in idx.values()]
    chosen = df["chosen"].to_numpy()
    bad = [g for g in groups if chosen[g].sum() != 1]
    if bad:
        raise ValueError(
            f"chosen=1の行がちょうど1つでないグループが{len(bad)}件ある"
            "（CSVが壊れているかexporterのバグの疑い）"
        )
    return groups


# 推論側（particle_nn.rs 呼び出し元）は forward 出力を clamp(-15, 15) してから
# 使うので、学習・評価のスコアにも同じ clamp を入れて分布を揃える
# （codexレビュー指摘、2026-07-23）
CLAMP = 15.0


def group_nll(
    scores: torch.Tensor, chosen: torch.Tensor, group_rows: list[np.ndarray]
) -> torch.Tensor:
    losses = []
    for rows in group_rows:
        s = scores[rows]
        c = chosen[rows]
        logz = torch.logsumexp(s, dim=0)
        chosen_idx = c.nonzero(as_tuple=True)[0][0]
        losses.append(logz - s[chosen_idx])
    return torch.stack(losses).mean()


def evaluate_scores(scores: torch.Tensor, chosen: torch.Tensor, group_rows) -> dict:
    nll = group_nll(scores, chosen, group_rows).item()
    top_half = 0
    for rows in group_rows:
        s = scores[rows]
        c = chosen[rows]
        chosen_idx = c.nonzero(as_tuple=True)[0][0]
        better = (s > s[chosen_idx]).sum().item()
        top_half += better * 2 < len(rows)
    n = len(group_rows)
    return {
        "nll": nll,
        "eff_candidates": float(np.exp(nll)),
        "top_half": top_half / n,
        "n_groups": n,
    }


def evaluate(model, x, offset, chosen, group_rows) -> dict:
    model.eval()
    with torch.no_grad():
        scores = torch.clamp(model(x), -CLAMP, CLAMP) + offset
    return evaluate_scores(scores, chosen, group_rows)


def train(
    model: OppMoveNet,
    x_train: torch.Tensor,
    offset_train: torch.Tensor,
    chosen_train: torch.Tensor,
    train_groups: list[np.ndarray],
    x_val: torch.Tensor,
    offset_val: torch.Tensor,
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
            scores = torch.clamp(model(x_train[all_rows]), -CLAMP, CLAMP) + offset_train[all_rows]
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

        val_metrics = evaluate(model, x_val, offset_val, chosen_val, val_groups)
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
                f"val_nll={val_metrics['nll']:.4f}  val_top_half={val_metrics['top_half']:.3f}"
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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = load_data(args.data)
    print(f"データ: {len(df)}行 / {df['game_id'].nunique()}局 / 特徴量{len(FEATURE_NAMES)}次元")

    groups_col = df["game_id"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_row_idx, _ = next(splitter.split(df, groups=groups_col))
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_row_idx] = True

    x_all = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    offset_all = df["offset"].to_numpy(dtype=np.float32)
    chosen_all = df["chosen"].to_numpy(dtype=np.float32)

    mean = x_all[train_mask].mean(axis=0)
    std = x_all[train_mask].std(axis=0)
    std[std == 0] = 1.0
    x_norm = (x_all - mean) / std

    x_t = torch.from_numpy(x_norm)
    offset_t = torch.from_numpy(offset_all)
    chosen_t = torch.from_numpy(chosen_all)

    all_groups = build_groups(df)
    train_groups = [g for g in all_groups if train_mask[g[0]]]
    val_groups = [g for g in all_groups if not train_mask[g[0]]]
    print(f"train: {len(train_groups)}グループ / val: {len(val_groups)}グループ")

    results = {}
    for name, hidden in [("linear", None), ("mlp", args.hidden)]:
        print(f"\n=== {name} (hidden={hidden}) ===")
        # モデル生成直前にseedをリセットする: export_particle_weights.py が
        # 同じseedで学習したMLPと初期値・学習軌跡を一致させるため
        # （codexレビュー指摘、2026-07-23: 「比較で評価したモデル」と
        # 「出荷するモデル」が別物になる事故の防止）
        torch.manual_seed(args.seed)
        model = OppMoveNet(input_dim=len(FEATURE_NAMES), hidden=hidden)
        model, best_metrics = train(
            model,
            x_t,
            offset_t,
            chosen_t,
            train_groups,
            x_t,
            offset_t,
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

    # 基準線: 一様（offsetも無視）と offsetのみ（現行のベース重み）
    uniform_nll = float(np.mean([np.log(len(g)) for g in val_groups]))
    offset_only = evaluate_scores(offset_t, chosen_t, val_groups)
    print("\n=== 比較（held-out val） ===")
    print(f"{'':12} {'NLL':>8} {'実効候補数':>10} {'top_half':>9}")
    print(f"{'uniform':12} {uniform_nll:8.4f} {np.exp(uniform_nll):10.2f} {'-':>9}")
    print(
        f"{'offset-only':12} {offset_only['nll']:8.4f} "
        f"{offset_only['eff_candidates']:10.2f} {offset_only['top_half'] * 100:8.1f}%"
    )
    for name in ("linear", "mlp"):
        m = results[name]
        print(
            f"{name:12} {m['nll']:8.4f} {m['eff_candidates']:10.2f} "
            f"{m['top_half'] * 100:8.1f}%"
        )


if __name__ == "__main__":
    main()
