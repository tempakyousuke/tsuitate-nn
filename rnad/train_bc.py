"""M2a: 対局記録を真似る教師あり学習（行動クローニング）。

目的は強さより**配管の検証**（tsuitate-bot の docs/rl-deepnash-design.md）:
エンコーダ・行動の符号化・学習・重みの書き出し・Rust 推論が一続きで正しく動くことを、
「うまくいくはずの課題」で確かめる。

    .venv/bin/python rnad/train_bc.py --out out_rnad/bc

教師は `rnad/fetch_records.py` が取った `data/rl_records/**/*.jsonl`。観測とマスクは
tsuitate-bot の `tsuitate_rl.RecordDataset` が作る（arena と同じ審判・同じエンコーダ）。
**棋譜の署名（着手列＋反則試行列）単位で学習/検証に分ける**（同じ局・同じ棋譜の決定点が
両方に入ると検証が甘くなる。別 run でも定跡どおりに進んで完全に同じ棋譜になる局がある）。

学習率は「処理した標本数 / 総標本数」を進み具合にした warmup＋cosine。局の長さが
不揃いでチャンクごとに端数のバッチが出るので、更新回数から決めるスケジュール
（OneCycleLR の total_steps）は実際の更新回数とずれる。
"""

import math

import argparse
import glob
import json
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import tsuitate_rl
from model import PolicyValueNet, masked_logits

ROOT = pathlib.Path(__file__).resolve().parent.parent


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def load_chunk(ds, games):
    (obs, mask, action, side, value, foul, game), outside = ds.encode_games(list(games))
    return {
        "obs": torch.from_numpy(obs),
        "mask": torch.from_numpy(mask),
        "action": torch.from_numpy(action),
        "value": torch.from_numpy(value),
        "foul": torch.from_numpy(foul),
    }, outside


@torch.no_grad()
def evaluate(net, ds, games, device, chunk_games, batch):
    net.eval()
    n = correct = correct_foul = n_foul = 0
    ce_sum = mse_sum = uniform_hit = 0.0
    for g in chunks(games, chunk_games):
        data, _ = load_chunk(ds, g)
        for i in range(0, len(data["action"]), batch):
            obs = data["obs"][i : i + batch].to(device)
            mask = data["mask"][i : i + batch].to(device)
            act = data["action"][i : i + batch].to(device)
            val = data["value"][i : i + batch].to(device)
            foul = data["foul"][i : i + batch].to(device)
            logits, v = net(obs)
            logits = masked_logits(logits, mask)
            ce_sum += F.cross_entropy(logits, act, reduction="sum").item()
            mse_sum += F.mse_loss(v, val, reduction="sum").item()
            hit = logits.argmax(1) == act
            correct += hit.sum().item()
            correct_foul += (hit & foul).sum().item()
            n_foul += foul.sum().item()
            # 対照: マスク内から一様に選んだときの的中率の期待値
            uniform_hit += (1.0 / mask.sum(1).float()).sum().item()
            n += len(act)
    net.train()
    return {
        "samples": n,
        "ce": ce_sum / n,
        "top1": correct / n,
        "top1_on_fouls": correct_foul / max(n_foul, 1),
        "uniform_top1": uniform_hit / n,
        "value_mse": mse_sum / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(ROOT / "data" / "rl_records" / "**" / "*.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "out_rnad" / "bc"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chunk-games", type=int, default=48, help="1回にエンコードする局数")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-weight", type=float, default=0.25)
    ap.add_argument("--max-games", type=int, default=0, help="0 = 全部（動作確認用に絞れる）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    paths = sorted(glob.glob(args.records, recursive=True))
    if args.max_games:
        paths = random.sample(paths, min(args.max_games, len(paths)))
        paths.sort()
    ds = tsuitate_rl.RecordDataset(paths)
    print(f"局 {ds.num_games}（読めなかった局 {len(ds.skipped)}）・試行 {ds.num_attempts}")
    for p, why in ds.skipped[:5]:
        print(f"  skip {p}: {why}")

    # 棋譜の署名でまとめてから分ける
    groups: dict[str, list[int]] = {}
    for i, sig in enumerate(ds.signatures):
        groups.setdefault(sig, []).append(i)
    keys = sorted(groups)
    random.shuffle(keys)
    val_games, train_games = [], []
    target_val = max(1, int(ds.num_games * args.val_frac))
    for k in keys:
        (val_games if len(val_games) < target_val else train_games).extend(groups[k])
    val_games.sort()
    attempts = ds.game_attempts
    total_train_samples = args.epochs * sum(attempts[g] for g in train_games)

    device = pick_device(args.device)
    net = PolicyValueNet(args.channels, args.blocks).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(seen: int) -> float:
        progress = min(seen / total_train_samples, 1.0)
        warmup = 0.05
        if progress < warmup:
            return args.lr * (0.04 + 0.96 * progress / warmup)
        t = (progress - warmup) / (1 - warmup)
        return args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**net.config, "args": vars(args)}, indent=2))
    print(f"device={device} 学習局 {len(train_games)} / 検証局 {len(val_games)}"
          f"（棋譜の署名 {len(keys)} 種）")

    step = 0
    seen = 0
    outside_total = 0
    for epoch in range(args.epochs):
        random.shuffle(train_games)
        t0 = time.time()
        run_ce = run_n = 0.0
        for g in chunks(train_games, args.chunk_games):
            data, outside = load_chunk(ds, g)
            outside_total += outside
            perm = torch.randperm(len(data["action"]))
            for i in range(0, len(perm), args.batch):
                idx = perm[i : i + args.batch]
                obs = data["obs"][idx].to(device)
                mask = data["mask"][idx].to(device)
                act = data["action"][idx].to(device)
                val = data["value"][idx].to(device)
                for group in opt.param_groups:
                    group["lr"] = lr_at(seen)
                logits, v = net(obs)
                ce = F.cross_entropy(masked_logits(logits, mask), act)
                loss = ce + args.value_weight * F.mse_loss(v, val)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
                seen += len(idx)
                run_ce += ce.item() * len(idx)
                run_n += len(idx)
                if step % 200 == 0:
                    print(f"  epoch {epoch} step {step} train_ce {run_ce / run_n:.3f} "
                          f"lr {lr_at(seen):.2e} ({run_n / (time.time() - t0):,.0f} 標本/秒)",
                          flush=True)
            # 次のチャンクを作る前に手放す（1試行 約39KB、48局で 200MB 前後）
            del data, perm
        metrics = evaluate(net, ds, val_games, device, args.chunk_games, args.batch)
        print(f"epoch {epoch}: {json.dumps(metrics)}  マスク外の教師の手 {outside_total}  "
              f"標本 {seen}/{total_train_samples}", flush=True)
        torch.save({"config": net.config, "state_dict": net.state_dict(), "metrics": metrics},
                   out / "checkpoint.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
