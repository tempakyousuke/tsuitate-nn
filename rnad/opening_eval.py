"""序盤の多様さと、序盤を強制したときの強さを測る（tsuitate-bot の docs/rl-opening-diversity.md）。

    # 序盤の手の分布（自己対局、温度 1 = 学習と同じ指し方）。初期局面の方策・最初の k 手の並びのエントロピー
    python rnad/opening_eval.py entropy NAME=weights.bin [NAME=...] --games 256 --k 6

    # 序盤を強制した対局。cand と ref（省略時は cand 自身）を先後入れ替えて指させ、
    # none（強制なし）/ opp（ref の側だけ強制）/ cand（cand の側だけ）/ both の条件ごと・定跡の系統ごとに
    # cand の得点・序盤（各側の受理手 12 まで）の反則・接触の前の反則・最初の接触の手数を出す
    python rnad/opening_eval.py robust --cand a.bin [--ref b.bin] --book ../tsuitate-bot/joseki.json --games 512

どちらも CPU でよい（1局あたり数十手、512局で数十秒〜数分）。強制は tsuitate_rl.VecEnv の中で行うので、
学習の `--open-book` と同じ規則（側ごとに独立・接触で止める・定跡の最後の駒取りの手は指さない）になる。
"""

import argparse
import collections
import json
import math
import pathlib
import sys

import numpy as np
import torch

import openings
import tsuitate_rl
from export_weights import read as read_weights

NSQ, MOVE_KINDS, DROP_BASE = 81, 66, 132
DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
KNIGHT = [(1, -2), (-1, -2)]
HAND = "PLNSGBR"


def decode(action, color):
    """行動番号 → USI（tsuitate-bot の rl::action::decode_usi と同じ。後手は点対称に戻す）"""
    kind, s = divmod(int(action), NSQ)
    f, r = s % 9 + 1, s // 9 + 1
    norm = (lambda a, b: (a, b)) if color == 0 else (lambda a, b: (10 - a, 10 - b))
    sq = lambda a, b: f"{a}{chr(ord('a') + b - 1)}"  # noqa: E731
    if kind >= DROP_BASE:
        return f"{HAND[kind - DROP_BASE]}*{sq(*norm(f, r))}"
    base = kind % MOVE_KINDS
    if base >= 64:
        df, dr = KNIGHT[base - 64]
    else:
        (ux, uy), d = DIRS[base // 8], base % 8 + 1
        df, dr = ux * d, uy * d
    return sq(*norm(f, r)) + sq(*norm(f + df, r + dr)) + ("+" if kind >= MOVE_KINDS else "")


def entropy_bits(counter):
    n = sum(counter.values())
    return -sum(v / n * math.log2(v / n) for v in counter.values()) if n else 0.0


def sample(net, obs, mask, gen):
    logits, _ = net(torch.from_numpy(obs).float())
    lg = logits.float().masked_fill(~torch.from_numpy(mask), float("-inf"))
    p = torch.softmax(lg, -1)
    return torch.multinomial(p, 1, generator=gen).squeeze(1).numpy(), p


@torch.no_grad()
def cmd_entropy(args):
    out = []
    for spec in args.weights:
        name, _, path = spec.partition("=")
        net = read_weights(path or name).eval()
        env = tsuitate_rl.VecEnv(args.games, seed=args.seed, auto_reset=False, planes=net.config["planes"])
        gen = torch.Generator().manual_seed(args.seed)
        seqs = [[[], []] for _ in range(args.games)]
        first = {}
        while True:
            alive = env.alive()
            need = alive & np.array([min(len(s[0]), len(s[1])) < args.k for s in seqs])
            if not need.any():
                break
            obs, mask, pl = env.observe()
            idx = np.flatnonzero(alive)
            drawn, p = sample(net, obs[idx], mask[idx], gen)
            actions = np.full(args.games, -1, dtype=np.int64)
            actions[idx] = drawn
            for j, g in enumerate(idx):
                c = int(pl[g])
                if not seqs[g][c] and c not in first:
                    first[c] = p[j].double().numpy()
            env.step(actions)
            _, _, pl2 = env.observe()
            alive2 = env.alive()
            for g in idx:
                c = int(pl[g])
                if not alive2[g] or pl2[g] != pl[g]:  # 受理（反則なら同じ側が続けて指す）
                    seqs[g][c].append(decode(actions[g], c))
        row = {"name": name}
        for c, side in ((0, "sente"), (1, "gote")):
            p = first[c]
            order = np.argsort(-p)[:5]
            k3 = collections.Counter(" ".join(s[c][:3]) for s in seqs if len(s[c]) >= 3)
            kk = collections.Counter(" ".join(s[c][:args.k]) for s in seqs if len(s[c]) >= args.k)
            row[side] = {
                # 初期局面の方策の上位（log10 で出す。指されない手は 1e-18 まで落ちていることがある）
                "root_top": [(decode(a, c), round(float(np.log10(max(p[a], 1e-300))), 2)) for a in order],
                "root_entropy_bits": round(float(-(p[p > 0] * np.log2(p[p > 0])).sum()), 3),
                "first3_entropy_bits": round(entropy_bits(k3), 3),
                "first3_top": k3.most_common(3),
                f"first{args.k}_entropy_bits": round(entropy_bits(kk), 3),
                f"first{args.k}_mode_share": round(kk.most_common(1)[0][1] / max(sum(kk.values()), 1), 3),
            }
        out.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


COND_SIDES = {"none": (), "cand": ("cand",), "opp": ("opp",), "both": ("cand", "opp")}


@torch.no_grad()
def play_cell(cand, ref, games, seed, book, cond, length):
    """cand を先手・後手で games/2 局ずつ指させる。戻り値は cand から見た局ごとの記録"""
    rows = []
    planes = max(cand.config["planes"], ref.config["planes"])
    for cand_color in (0, 1):
        roles = COND_SIDES[cond]
        colors = sorted({cand_color if r == "cand" else 1 - cand_color for r in roles})
        kw = {}
        if colors:
            sides = "both" if len(colors) == 2 else ("sente" if colors[0] == 0 else "gote")
            kw = openings.env_kwargs(book, 1.0, length, sides=sides)
        n = games // 2
        cell_seed = seed * 2 + cand_color
        env = tsuitate_rl.VecEnv(n, seed=cell_seed, auto_reset=False, planes=planes, opening_seed=cell_seed, **kw)
        gen = torch.Generator().manual_seed(cell_seed)
        while True:
            alive = env.alive()
            if not alive.any():
                break
            obs, mask, pl = env.observe()
            actions = np.full(n, -1, dtype=np.int64)
            for net, mine in ((cand, True), (ref, False)):
                idx = np.flatnonzero(alive & ((pl == cand_color) == mine))
                if len(idx):
                    actions[idx], _ = sample(net, obs[idx], mask[idx], gen)
            env.step(actions)
        for f in env.pop_finished():
            w = f["winner"]
            o = 1 - cand_color
            rows.append({
                "score": 0.5 if w is None else float(w == cand_color),
                "cand_color": cand_color, "reason": f["reason"], "plies": f["plies"],
                "fouls_cand": f["fouls"][cand_color], "fouls_opp": f["fouls"][o],
                "pre_cand": f["fouls_precontact"][cand_color], "pre_opp": f["fouls_precontact"][o],
                "early_cand": f["fouls_early"][cand_color], "early_opp": f["fouls_early"][o],
                "contact_ply": f["contact_ply"],
                "forced_cand": f["forced_moves"][cand_color], "forced_opp": f["forced_moves"][o],
            })
    return rows


def summarize_cell(rows):
    x = np.array([r["score"] for r in rows])
    mean = lambda k: round(float(np.mean([r[k] for r in rows])), 3)  # noqa: E731
    contact = [r["contact_ply"] for r in rows if r["contact_ply"] is not None]
    return {
        "games": len(rows), "score": round(float(x.mean()), 4),
        "se": round(float(x.std(ddof=1) / math.sqrt(len(x))), 4) if len(x) > 1 else None,
        "fouls_cand": mean("fouls_cand"), "fouls_opp": mean("fouls_opp"),
        "precontact_fouls_cand": mean("pre_cand"), "precontact_fouls_opp": mean("pre_opp"),
        # 各側の受理手が 12 に届くまでの反則（相手の序盤が変わったときの「いつもの形を前提にした手」はここに出る）
        "early_fouls_cand": mean("early_cand"), "early_fouls_opp": mean("early_opp"),
        "contact_ply": round(float(np.mean(contact)), 1) if contact else None,
        "forced_cand": mean("forced_cand"), "forced_opp": mean("forced_opp"),
        "reasons": dict(collections.Counter(r["reason"] for r in rows).most_common()),
    }


def cmd_robust(args):
    torch.set_num_threads(args.threads)
    if args.games % 2:
        raise SystemExit("--games は偶数（先後を同数にする）")
    cand = read_weights(args.cand).eval()
    ref = read_weights(args.ref).eval() if args.ref else cand
    book = openings.load_book(args.book, "line")
    length = openings.parse_len(args.open_len)
    cats = sorted(set(book.categories), key=book.categories.index) if args.by_category else [None]
    conds = args.conds.split(",")
    for c in conds:
        if c not in COND_SIDES:
            raise SystemExit(f"--conds は {','.join(COND_SIDES)} から（{c}）")
    out = {"cand": args.cand, "ref": args.ref or args.cand, "book_sha256": book.sha256, "cells": []}
    for cond in conds:
        for k, cat in enumerate([None] if cond == "none" else cats):
            sub = book if cat is None else book.subset(cat)
            rows = play_cell(cand, ref, args.games, args.seed + 1000 * k, sub, cond, length)
            cell = {"cond": cond, "category": cat, **summarize_cell(rows)}
            out["cells"].append(cell)
            print(json.dumps(cell, ensure_ascii=False), flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("entropy", help="序盤の手の分布")
    e.add_argument("weights", nargs="+", help="名前=重みのパス（名前を省くとパスがそのまま名前）")
    e.add_argument("--games", type=int, default=256)
    e.add_argument("--k", type=int, default=6, help="並びを数える手数（各側の受理手）")
    e.add_argument("--seed", type=int, default=20261028)
    e.add_argument("--out", default="")
    r = sub.add_parser("robust", help="序盤を強制した対局")
    r.add_argument("--cand", required=True, help="測るネットの重み（.bin）")
    r.add_argument("--ref", default="", help="相手のネットの重み（省略時は cand 自身）")
    r.add_argument("--book", required=True, help="定跡ブック（joseki.json の形式）")
    r.add_argument("--games", type=int, default=512, help="条件×系統ごとの局数（偶数。先後を半分ずつ）")
    r.add_argument("--conds", default="none,opp,cand,both")
    r.add_argument("--by-category", action=argparse.BooleanOptionalAction, default=True,
                   help="系統ごとに分ける（--no-by-category ならブック全体から一様に引く）")
    r.add_argument("--open-len", default="", help="強制する手数の範囲 'LO:HI'（空 = ラインを切った後の最後まで）")
    r.add_argument("--seed", type=int, default=20261029)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.cmd == "entropy":
        cmd_entropy(args)
    else:
        cmd_robust(args)


if __name__ == "__main__":
    sys.exit(main())
