"""学習ログ（<out>/log.jsonl）だけで見られる事前確認（tsuitate-bot の docs/rl-learning-speed.md「さらに速くする」）。
学習は止めず、GPU も対局も使わない（学習中の log.jsonl をそのまま読んでよい）。

    python rnad/stage_log.py out_rnad/deduce_s10000/log.jsonl
    python rnad/stage_log.py out_rnad/deduce_s10000/log.jsonl --from-step 58000 --bin 1000 \\
        --out-dir out_rnad/deduce_s10000 --exclude-stage 58000 --json stage_log.json

出すもの:
1. **1更新の秒の内訳**: 更新の行を α < 1（段階の前半 = 正則化方策の補間中）と α = 1 に分け、`collect_busy_sec`
   （`--pipeline` のときだけ。別スレッドで集めるのにかかった秒）・`collect_sec`（`--pipeline` では更新側が待った秒、
   なしでは集めた秒）・`update_sec`・1更新の合計（collect_sec + update_sec）の中央値と平均。`--pipeline` の行は
   「集める（busy）と更新の大きいほう」が律速なので、その区間の律速も出す。錨の評価の秒の割合も出す
2. **錨への勝率の段階内の形**: 評価の行（`eval` = target / current）を段階の中の位置（`--bin` 更新ごと）で平均する
   （段階ごとと、同じ長さの段階の合算。± は w/l/d から出した標準誤差）。0 の行は区切りちょうどの評価
3. **段階の平らさ g = (E_end − E_7/10) / (E_end − E_start)**: E は錨への勝率の、段階の頭・7/10・末での値。
   段階数に比例して進む（磁石に着いたら平らになる）なら 0.03〜0.10、更新数に比例する（直線）なら 0.30
   （被覆率の模型、τ=0.001・Δm=10,000）。**g < 0.15 なら Δm を縮める根拠、g > 0.25 なら Δm=10,000 のまま**
   （`--g-lo` / `--g-hi`）。E の求め方（窓）は下の「g の窓」
4. （`--out-dir`）**区切りでの停止**: 止めた時刻 = `policy_<step>.bin` の mtime（`--steps` に着いた run が最後に書く）、
   再開した時刻 = `config_resume_<step>.json` の mtime。差が止まっていた時間。さらに、書き出しどうしの間隔の
   壁時計からログの秒（更新ループ + 錨の評価）を引いた残りを「その他」（checkpoint・書き出し・パネル・スリープなど）とする。
   **mtime を保つ複写（rsync -a / cp -p）でないと意味がない**

段階の区切り: 既定は `reg_update` が真の更新の行（その step で正則化方策を差し替えた = 段階の末）。最初の区切りより前の
段階は、ログがちょうど区切りから始まっているときだけ足す。最初の区切りの前と最後の区切りの後の段階の長さは、補間中の行の
α（= 2·(step − 1 − 始まり)/Δm）から逆算する（Δm=2,000 → 10,000 のように段階の長さが変わる区切りでもずれない）。
`--dm`（と `--schedule-start`）を与えると区切りは --schedule-start + k·Δm（k ≥ 0）にする（train_rnad.py の
--schedule / --schedule-start と同じ。reg_update の行と食い違えば警告する）。落ちて checkpoint から再開した run の
重なった行は後の行を使う（load_log）。

g の窓（既定の幅 W = 0.3·Δm。Δm=10,000 なら 3,000更新 = 50更新ごと・256局の評価で 60回・15,360局、窓1つの
SE ≈ 0.39pt、窓どうしの差の SE ≈ 0.54pt）:
- E_7/10: 位置 0.7·Δm を中心とする幅 W の窓の評価に直線を当て、中心での値
- E_start / E_end: 区切りをまたぐ幅 W の窓（前後の段階の評価も使う）に直線を当て、区切りでの値。ターゲットは区切りで
  連続し、α=0 の磁石は前の段階の末と同じなので、区切りの前後は (a) なら一続きの直線・(b) なら平らで、どちらでも偏らない。
  片側の評価が無い（ログの最初・最後の段階）ときは段階の内側の幅 W の窓の直線を区切りへ延ばす（「片側」。SE は約2倍）
- 1つの段階の g は揺れが大きい（伸び E_end − E_start が 2〜4pt なら g の SE は 0.1〜0.3）ので、**同じ長さの段階を合算した g**
  （分子・分母をそれぞれ足してから割る）で判定する。錨そのものが磁石の段階（本線の 58,000→68,000 は錨 = rl_deduce_v4）は
  独立の証拠として弱いので `--exclude-stage 58000` で合算から外せる
"""

import argparse
import datetime
import json
import pathlib
import re
import sys

import numpy as np

WHICH = ("target", "current")


# ---------------------------------------------------------------- 読み込み


def load_log(path):
    """log.jsonl を (更新の行, 評価の行) に分ける（どちらも step の昇順）。読めない行（学習中の書きかけの最後の行など）は飛ばす。

    落ちて checkpoint.pt から再開した run は、checkpoint の step C より後の行がもう一度（ファイルの後ろに）書かれる
    （log.jsonl は追記）。step が戻ったら、そこから先の前の行を捨てて後の行を使う: 更新の行の step がそれまでの更新の行の
    最大以下なら、前の行のうち step がそれ以上の更新・評価を捨てる。step 0 の評価の行（同じ out で step 0 から回し直した。
    評価だけが先に出るのは step 0 のときだけ）なら前の行を全部捨てる。再開した run は起点で評価しないので、区切り C の
    評価は前の run のものが残る"""
    updates, evals = [], []
    top_u = top = -1  # それまでの更新の行の最大の step / 全部の行の最大の step
    restarts = dropped = 0
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"[stage_log] {path}:{ln} を読めないので飛ばす", file=sys.stderr)
                continue
            if "eval" in r and "eval_vs_anchor" in r:
                is_eval = True
            elif "update_sec" in r:
                is_eval = False
            else:
                continue
            s = r["step"]
            if (s <= top_u) if not is_eval else (s == 0 < top):
                n0 = len(updates) + len(evals)
                updates = [u for u in updates if u["step"] < s]
                evals = [e for e in evals if e["step"] < s]
                restarts, dropped = restarts + 1, dropped + n0 - len(updates) - len(evals)
                top_u = max([u["step"] for u in updates], default=-1)
                top = max([top_u] + [e["step"] for e in evals])
            (evals if is_eval else updates).append(r)
            top = max(top, s)
            if not is_eval:
                top_u = max(top_u, s)
    if restarts:
        print(f"[stage_log] {path}: step が戻った（再開）のが {restarts} 回。前の run の行 {dropped} 行を捨てて後の行を使う",
              file=sys.stderr)
    # 同じ (step, eval) が残っていれば後のものを使う
    ev = {}
    for e in evals:
        ev[(e["step"], e["eval"])] = e
    return sorted(updates, key=lambda r: r["step"]), sorted(ev.values(), key=lambda r: (r["step"], r["eval"]))


# ---------------------------------------------------------------- 1. 1更新の秒


def stat(xs):
    xs = np.asarray([x for x in xs if x is not None], dtype=float)
    if not len(xs):
        return None
    return {"n": int(len(xs)), "median": float(np.median(xs)), "mean": float(xs.mean())}


def timing(updates):
    """(pipeline あり / なし) × (α < 1 / α = 1) ごとの秒の要約"""
    out = {}
    for pipe in (True, False):
        for part, cond in (("ramp", lambda a: a < 1.0), ("alpha1", lambda a: a >= 1.0)):
            rows = [r for r in updates if ("collect_busy_sec" in r) == pipe and cond(float(r.get("alpha", 1.0)))]
            if not rows:
                continue
            total = [r["collect_sec"] + r["update_sec"] for r in rows]
            rec = {"updates": len(rows),
                   "collect": stat([r["collect_sec"] for r in rows]),
                   "update": stat([r["update_sec"] for r in rows]),
                   "total": stat(total),
                   # 1,000決定点あたり（局の長さが違う区間どうしを比べるため）
                   "total_per_1k_samples": stat([t / r["samples"] * 1000 for t, r in zip(total, rows)
                                                 if r.get("samples")])}
            if pipe:
                rec["busy"] = stat([r["collect_busy_sec"] for r in rows])
                rec["bottleneck"] = "collect" if rec["busy"]["median"] > rec["update"]["median"] else "update"
            out[f"{'pipeline' if pipe else 'sequential'}/{part}"] = rec
    return out


def eval_share(updates, evals):
    loop = sum(r["collect_sec"] + r["update_sec"] for r in updates)
    ev = sum(r.get("sec", 0.0) for r in evals)
    return {"loop_hours": loop / 3600, "eval_hours": ev / 3600, "eval_share": ev / (loop + ev) if loop + ev else None}


# ---------------------------------------------------------------- 段階


def ramp_len(updates, B, after):
    """区切り B の後（after）/ 前の段階の長さ Δm を、補間中（0 < α < 1）の更新の行の α から逆算する（0 = 分からない）。
    step s の行の α = 2·(s − 1 − 始まり)/Δm なので、後の段階は Δm = 2(s − 1 − B)/α、前の段階（始まり = B − Δm）は
    Δm = 2(B + 1 − s)/(2 − α)。後は最後の区切りの後、前は最初の区切りの前の行だけに使う（そこには区切りが無い）"""
    xs = []
    for r in updates:
        a, s = float(r.get("alpha", 1.0)), r["step"]
        if 0 < a < 1 and (s > B if after else s <= B):
            xs.append(2 * (s - 1 - B) / a if after else 2 * (B + 1 - s) / (2 - a))
    return int(round(float(np.median(xs)))) if xs else 0


def find_stages(updates, evals, dm=0, schedule_start=0):
    """段階の列 [(始まり, 終わり, 終わりまでログがあるか)]。終わりの step は区切り（次の段階の始まり）"""
    steps = [r["step"] for r in updates] + [r["step"] for r in evals]
    if not steps:
        return []
    lo, hi = min(steps), max(steps)
    reg = sorted({r["step"] for r in updates if r.get("reg_update")})
    tail_len = 0
    if dm:
        bounds = list(range(schedule_start, hi + dm, dm))
        off = [s for s in reg if s > schedule_start and s not in set(bounds)]
        if off:
            print(f"[stage_log] reg_update の行 {off[:5]} が --dm {dm} --schedule-start {schedule_start} の区切りに無い"
                  "（train_rnad.py に渡した値を確かめる）", file=sys.stderr)
    elif reg:
        bounds = list(reg)
        # 最初の区切りの前: ログがちょうど区切りから始まっている（区切り S で再開した run の最初の行は S + 1、または
        # --from-step S で区切りの評価の行から）なら、その段階も足す。長さは α から（無ければ次の段階と同じ）
        first_len = ramp_len(updates, bounds[0], after=False) or (bounds[1] - bounds[0] if len(bounds) >= 2 else 0)
        first_update = min((r["step"] for r in updates), default=hi)
        if first_len and (first_update == bounds[0] - first_len + 1 or lo == bounds[0] - first_len):
            bounds.insert(0, bounds[0] - first_len)
        # 最後の区切りの後の途中の段階。長さは α から（無ければ直前の段階と同じ）
        if hi > bounds[-1]:
            tail_len = ramp_len(updates, bounds[-1], after=True) or (bounds[-1] - bounds[-2] if len(bounds) >= 2 else 0)
    else:
        bounds = []
    stages = []
    for a, b in zip(bounds, bounds[1:]):
        if any(a < s <= b for s in steps):
            stages.append((a, b, hi >= b))
    if tail_len:
        stages.append((bounds[-1], bounds[-1] + tail_len, False))
    return stages


def eval_table(evals, which):
    """評価の行を配列に: step、勝率、局数、勝率の分散（1局あたりの分散 / 局数）"""
    rows = [r for r in evals if r["eval"] == which]
    steps = np.array([r["step"] for r in rows], dtype=np.int64)
    w = np.array([r["w"] for r in rows], dtype=float)
    l = np.array([r["l"] for r in rows], dtype=float)  # noqa: E741
    d = np.array([r["d"] for r in rows], dtype=float)
    n = w + l + d
    m = (w + 0.5 * d) / np.maximum(n, 1)
    s2 = (w * (1 - m) ** 2 + d * (0.5 - m) ** 2 + l * m ** 2) / np.maximum(n, 1)
    if len(s2) and (s2 <= 0).any():
        s2 = np.where(s2 > 0, s2, s2[s2 > 0].mean() if (s2 > 0).any() else 0.25)
    return {"step": steps, "score": m, "n": n, "var": s2 / np.maximum(n, 1),
            "w": w, "l": l, "d": d}


def pooled_cell(t, sel):
    """選んだ評価をまとめた勝率と標準誤差（局を独立とみなす）"""
    W, L, D = t["w"][sel].sum(), t["l"][sel].sum(), t["d"][sel].sum()
    N = W + L + D
    if N == 0:
        return None
    m = (W + 0.5 * D) / N
    s2 = (W * (1 - m) ** 2 + D * (0.5 - m) ** 2 + L * m ** 2) / N
    return {"score": float(m), "se": float(np.sqrt(s2 / N)), "games": int(N), "evals": int(sel.sum())}


def binned(t, stages, size):
    """段階ごとに、段階の中の位置（区切りちょうど = 0、以後 (0, size], (size, 2·size], …）で平均した勝率"""
    out = []
    for a, b, complete in stages:
        L = b - a
        nb = -(-L // size)
        cells = [pooled_cell(t, t["step"] == a)]
        for k in range(nb):
            cells.append(pooled_cell(t, (t["step"] > a + k * size) & (t["step"] <= min(a + (k + 1) * size, b))))
        out.append({"start": a, "end": b, "complete": complete, "positions": [0] + [min((k + 1) * size, L)
                                                                                 for k in range(nb)],
                    "cells": cells})
    return out


def pool_bins(per_stage):
    """同じ長さ・終わりまである段階の、位置ごとの単純平均（段階を1票ずつ）"""
    groups = {}
    for s in per_stage:
        if s["complete"]:
            groups.setdefault(s["end"] - s["start"], []).append(s)
    out = {}
    for L, ss in groups.items():
        cells = []
        for k, pos in enumerate(ss[0]["positions"]):
            xs = [s["cells"][k] for s in ss if s["cells"][k] is not None]
            if not xs:
                cells.append(None)
                continue
            cells.append({"score": float(np.mean([x["score"] for x in xs])),
                          "se": float(np.sqrt(sum(x["se"] ** 2 for x in xs)) / len(xs)), "stages": len(xs)})
        out[L] = {"stages": [s["start"] for s in ss], "positions": ss[0]["positions"], "cells": cells}
    return out


# ---------------------------------------------------------------- 3. g


def line_at(idx, x, n, x0):
    """窓の中の評価（添字 idx・位置 x・局数 n）に局数の重みで直線を当て、x0 での値を Σ c_i·E_i の係数 {添字: c} で返す
    （位置が1種類なら平均）"""
    w = n / n.sum()
    xb = float((w * x).sum())
    sxx = float((w * (x - xb) ** 2).sum())
    c = w * (1 + (x0 - xb) * (x - xb) / sxx) if sxx > 0 else w
    return dict(zip(idx.tolist(), c.tolist()))


def window(t, lo, hi, x0, min_evals=2):
    """step が (lo, hi] の評価に直線を当てた x0 での値の係数。評価が足りなければ None"""
    sel = np.flatnonzero((t["step"] > lo) & (t["step"] <= hi))
    if len(sel) < min_evals or len(np.unique(t["step"][sel])) < min(2, min_evals):
        return None
    return line_at(sel, t["step"][sel].astype(float), t["n"][sel], float(x0))


def boundary_value(t, B, h, inner_lo, inner_hi):
    """区切り B での勝率の係数。区切りをまたぐ (B − h, B + h] に前後とも2点以上あればそれ、無ければ段階の内側の
    (inner_lo, inner_hi] の直線を B へ延ばす"""
    left = int(((t["step"] > B - h) & (t["step"] < B)).sum())
    right = int(((t["step"] > B) & (t["step"] <= B + h)).sum())
    if left >= 2 and right >= 2:
        return window(t, B - h, B + h, B), "straddle"
    c = window(t, inner_lo, inner_hi, B)
    return c, ("one-sided" if c is not None else None)


def lin(c, t):
    return float(sum(v * t["score"][i] for i, v in c.items()))


def var_of(c, t):
    return float(sum(v * v * t["var"][i] for i, v in c.items()))


def cov_of(c1, c2, t):
    return float(sum(v * c2[i] * t["var"][i] for i, v in c1.items() if i in c2))


def sub(c1, c2):
    out = dict(c1)
    for i, v in c2.items():
        out[i] = out.get(i, 0.0) - v
    return out


def add(c1, c2):
    out = dict(c1)
    for i, v in c2.items():
        out[i] = out.get(i, 0.0) + v
    return out


def ratio(num, den, t):
    """g = num / den と、デルタ法の標準誤差"""
    N, D = lin(num, t), lin(den, t)
    vN, vD, cND = var_of(num, t), var_of(den, t), cov_of(num, den, t)
    g = N / D if D else float("nan")
    se = float(np.sqrt(max(vN - 2 * g * cND + g * g * vD, 0.0)) / abs(D)) if D else float("nan")
    return {"g": g, "se": se, "num": N, "num_se": float(np.sqrt(vN)), "den": D, "den_se": float(np.sqrt(vD))}


def plateau(t, stages, g_window=0, frac=0.7, exclude=()):
    """段階ごとと、同じ長さの段階の合算の g"""
    per, groups = [], {}
    for a, b, complete in stages:
        L = b - a
        W = g_window or round(0.3 * L)
        h = W / 2
        rec = {"start": a, "end": b, "complete": complete, "window": W}
        if not complete:
            rec["skip"] = "途中の段階"
            per.append(rec)
            continue
        c_start, how_s = boundary_value(t, a, h, a - 1, a + W)
        c_end, how_e = boundary_value(t, b, h, b - W, b)
        mid = a + frac * L
        c_mid = window(t, mid - h, mid + h, mid)
        if c_start is None or c_end is None or c_mid is None:
            rec["skip"] = "窓に評価が足りない"
            per.append(rec)
            continue
        num, den = sub(c_end, c_mid), sub(c_end, c_start)
        rec.update({"E_start": lin(c_start, t), "E_start_se": float(np.sqrt(var_of(c_start, t))), "how_start": how_s,
                    "E_mid": lin(c_mid, t), "E_mid_se": float(np.sqrt(var_of(c_mid, t))),
                    "E_end": lin(c_end, t), "E_end_se": float(np.sqrt(var_of(c_end, t))), "how_end": how_e,
                    **ratio(num, den, t)})
        per.append(rec)
        if a not in exclude:
            g = groups.setdefault(L, {"num": {}, "den": {}, "stages": []})
            g["num"], g["den"] = add(g["num"], num), add(g["den"], den)
            g["stages"].append(a)
    pooled = {L: {"stages": g["stages"], **ratio(g["num"], g["den"], t)} for L, g in groups.items()}
    return per, pooled


def verdict(r, lo, hi):
    if not np.isfinite(r["g"]) or r["den"] <= 2 * r["den_se"]:
        return "段階の伸びが誤差の内（g は読めない）"
    # 門は点推定で引く（docs の事前の門）。±1SE が両方の門をまたぐときはそう書き添える
    wide = "。ただし ±1SE が両方の門をまたぐ" if r["g"] - r["se"] < lo and r["g"] + r["se"] > hi else ""
    if r["g"] < lo:
        return f"g<{lo:g}: 段階数比例寄り（Δm を縮める根拠）{wide}"
    if r["g"] > hi:
        return f"g>{hi:g}: 更新数比例寄り（Δm はそのまま）{wide}"
    return f"どちらとも言えない{wide}"


# ---------------------------------------------------------------- 4. --out-dir


def step_files(d, prefix, ext):
    out = {}
    for p in d.glob(f"{prefix}_*.{ext}"):
        m = re.fullmatch(rf"{prefix}_(\d+)\.{ext}", p.name)
        if m:
            out[int(m.group(1))] = p.stat().st_mtime
    return out


def scan_out_dir(out_dir, bounds, updates, evals):
    d = pathlib.Path(out_dir)
    pol = step_files(d, "policy", "bin")
    res = step_files(d, "config_resume", "json")
    stops = []
    for R in sorted(res):
        prior = [s for s in pol if s <= R]
        if not prior:
            stops.append({"step": R, "resume_time": res[R], "hours": None, "boundary": R in bounds})
            continue
        s0 = R if R in pol else max(prior)
        stops.append({"step": R, "stop_step": s0, "stop_time": pol[s0], "resume_time": res[R],
                      "hours": (res[R] - pol[s0]) / 3600, "exact": s0 == R, "boundary": R in bounds})
    # 書き出しどうしの間隔ごとの壁時計とログの秒（再開をはさむ間隔は除く）
    steps = sorted(pol)
    u_step = np.array([r["step"] for r in updates], dtype=np.int64)
    u_sec = np.array([r["collect_sec"] + r["update_sec"] for r in updates], dtype=float)
    e_step = np.array([r["step"] for r in evals], dtype=np.int64)
    e_sec = np.array([r.get("sec", 0.0) for r in evals], dtype=float)
    wall = loop = ev = 0.0
    shares = []
    for a, b in zip(steps, steps[1:]):
        if any(a <= R < b for R in res):
            continue
        su = (u_step > a) & (u_step <= b)
        if not su.any():
            continue
        dw = pol[b] - pol[a]
        dl = float(u_sec[su].sum())
        de = float(e_sec[(e_step > a) & (e_step <= b)].sum())
        if dw <= 0:
            continue
        wall, loop, ev = wall + dw, loop + dl, ev + de
        shares.append(1 - (dl + de) / dw)
    over = None
    if wall > 0:
        over = {"wall_hours": wall / 3600, "loop_share": loop / wall, "eval_share": ev / wall,
                "other_share": 1 - (loop + ev) / wall, "other_share_median": float(np.median(shares)),
                "intervals": len(shares)}
    return {"stops": stops, "overhead": over}


# ---------------------------------------------------------------- 表示


def fmt_t(ts):
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def ms(s):
    return "-" if s is None else f"{s['median']:.2f} / {s['mean']:.2f}"


def pt(c):
    return "-" if c is None else f"{100 * c['score']:.1f}±{100 * c['se']:.1f}"


def print_timing(tm, share):
    print("== 1更新の秒（中央値 / 平均）")
    print(f"{'区間':<26}{'更新数':>7}  {'集める(busy)':>14}  {'待ち/集める':>14}  {'更新':>14}  {'合計':>14}  "
          f"{'合計/1000決定点':>14}")
    names = {"ramp": "α<1（補間中）", "alpha1": "α=1"}
    for key, r in tm.items():
        mode, part = key.split("/")
        label = f"{'pipeline' if mode == 'pipeline' else '逐次'} {names[part]}"
        print(f"{label:<26}{r['updates']:>7}  {ms(r.get('busy')):>14}  {ms(r['collect']):>14}  {ms(r['update']):>14}  "
              f"{ms(r['total']):>14}  {ms(r['total_per_1k_samples']):>14}")
    for key, r in tm.items():
        if "bottleneck" in r:
            b, u, w = r["busy"]["median"], r["update"]["median"], r["collect"]["median"]
            who = "集める" if r["bottleneck"] == "collect" else "更新"
            print(f"  律速 {names[key.split('/')[1]]}: {who}（busy {b:.2f}s / 更新 {u:.2f}s、待ち {w:.2f}s）")
    if share["eval_share"] is not None:
        print(f"  更新ループ {share['loop_hours']:.1f}時間、錨の評価 {share['eval_hours']:.2f}時間"
              f"（評価の割合 {100 * share['eval_share']:.1f}%）")


def print_bins(which, per_stage, pooled, max_cols=8):
    for L, p in sorted(pooled.items()):
        ss = [s for s in per_stage if s["complete"] and s["end"] - s["start"] == L]
        show = ss if len(ss) <= max_cols else []
        print(f"== 錨への勝率（{which}、%）の段階内の位置ごとの平均（Δm={L}、{len(ss)}段階"
              + ("" if show else "。段階ごとの列は --json に") + "）")
        head = f"{'位置':>7}" + "".join(f"{s['start']:>8}→" + f"{s['end']:<6}" for s in show) + f"{'合算':>12}"
        print(head)
        for k, pos in enumerate(p["positions"]):
            print(f"{pos:>7}" + "".join(f"{pt(s['cells'][k]):>15}" for s in show) + f"{pt(p['cells'][k]):>12}")
    for s in per_stage:
        if not s["complete"]:
            filled = [(pos, c) for pos, c in zip(s["positions"], s["cells"]) if c is not None]
            print(f"  （途中の段階 {s['start']}→ の {which}: " + " ".join(f"{pos}:{pt(c)}" for pos, c in filled) + "）")


def print_g(which, per, pooled, lo, hi):
    print(f"== 段階の平らさ g（{which}）  g = (E_end − E_7/10) / (E_end − E_start)、E は %")
    print(f"{'段階':<16}{'窓':>6}  {'E_start':>13}  {'E_7/10':>11}  {'E_end':>13}  {'伸び':>10}  {'g':>11}")
    for r in per:
        name = f"{r['start']}→{r['end']}"
        if "skip" in r:
            print(f"{name:<16}{r['window']:>6}  （{r['skip']}）")
            continue
        tag = lambda how: "" if how == "straddle" else "片"  # noqa: E731
        print(f"{name:<16}{r['window']:>6}  {100 * r['E_start']:>7.1f}±{100 * r['E_start_se']:.1f}{tag(r['how_start']):1}"
              f"  {100 * r['E_mid']:>6.1f}±{100 * r['E_mid_se']:.1f}"
              f"  {100 * r['E_end']:>7.1f}±{100 * r['E_end_se']:.1f}{tag(r['how_end']):1}"
              f"  {100 * r['den']:>5.1f}±{100 * r['den_se']:.1f}  {r['g']:>5.2f}±{r['se']:.2f}")
    for L, r in sorted(pooled.items()):
        print(f"  合算（Δm={L}、{len(r['stages'])}段階 {r['stages']}）: g = {r['g']:.2f}±{r['se']:.2f}"
              f"（伸び {100 * r['den']:.1f}±{100 * r['den_se']:.1f}pt）→ {verdict(r, lo, hi)}")
    print("  （片 = 区切りの片側に評価が無いので段階の内側の直線を延ばした値。予測は段階数比例 0.03〜0.10 / 更新数比例 0.30）")


def print_out_dir(o):
    print("== 再開までの停止（--out-dir の mtime）")
    for s in o["stops"]:
        if s["hours"] is None:
            print(f"  再開 {s['step']}: それ以前の policy_*.bin が無い")
            continue
        note = "" if s["exact"] else f"（policy_{s['step']}.bin が無いので policy_{s['stop_step']}.bin から。上限）"
        print(f"  再開 {s['step']:>7} {'区切り' if s['boundary'] else '途中  '}  止めた {fmt_t(s['stop_time'])}"
              f" → 再開 {fmt_t(s['resume_time'])}  {s['hours']:6.2f}時間{note}")
    hs = [s for s in o["stops"] if s["hours"] is not None]
    if hs:
        b = [s["hours"] for s in hs if s["boundary"]]
        nb = [s["hours"] for s in hs if not s["boundary"]]
        print(f"  合計 {sum(s['hours'] for s in hs):.1f}時間（区切り {len(b)}回 {sum(b):.1f}時間、それ以外 {len(nb)}回 "
              f"{sum(nb):.1f}時間）")
    ov = o["overhead"]
    if ov:
        print(f"== ループの外の雑費（書き出しの間隔 {ov['intervals']}個、再開をはさむ間隔は除く）")
        print(f"  壁時計 {ov['wall_hours']:.1f}時間 = 更新ループ {100 * ov['loop_share']:.1f}% + 錨の評価 "
              f"{100 * ov['eval_share']:.1f}% + その他 {100 * ov['other_share']:.1f}%（間隔ごとの中央値 "
              f"{100 * ov['other_share_median']:.1f}%）")


def finite(x):
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", help="train_rnad.py の <out>/log.jsonl")
    ap.add_argument("--from-step", type=int, default=0, help="この step より前の行を使わない")
    ap.add_argument("--to-step", type=int, default=0, help="この step より後の行を使わない（0 = 最後まで）")
    ap.add_argument("--dm", type=int, default=0, help="段階の長さ（0 = reg_update の行から区切りを取る）")
    ap.add_argument("--schedule-start", type=int, default=0,
                    help="--dm の区切りの起点（train_rnad.py と同じ。--dm なしでは使わない）")
    ap.add_argument("--bin", type=int, default=1000, help="段階内の位置の平均の幅（更新回数）")
    ap.add_argument("--g-window", type=int, default=0, help="g の窓の幅（更新回数。0 = 段階の長さの 0.3 倍）")
    ap.add_argument("--g-lo", type=float, default=0.15, help="g がこれ未満なら段階数比例寄り")
    ap.add_argument("--g-hi", type=float, default=0.25, help="g がこれを超えたら更新数比例寄り")
    ap.add_argument("--exclude-stage", type=int, action="append", default=[],
                    help="g の合算から外す段階の始まりの step（錨が磁石そのものの段階など。複数可）")
    ap.add_argument("--out-dir", default="", help="run の出力ディレクトリ（policy_*.bin・config_resume_*.json の mtime を読む）")
    ap.add_argument("--json", default="", help="結果を JSON で書く（- なら標準出力へ、表は出さない）")
    args = ap.parse_args()

    if args.schedule_start and not args.dm:
        print("[stage_log] --schedule-start は --dm と一緒に使う（なしでは reg_update の行から区切りを取る）", file=sys.stderr)
    updates, evals = load_log(args.log)
    keep = lambda s: s >= args.from_step and (not args.to_step or s <= args.to_step)  # noqa: E731
    updates = [r for r in updates if keep(r["step"])]
    evals = [r for r in evals if keep(r["step"])]
    if not updates and not evals:
        raise SystemExit(f"{args.log} に使える行が無い")
    tm = timing(updates)
    share = eval_share(updates, evals)
    stages = find_stages(updates, evals, args.dm, args.schedule_start)
    res = {"log": str(args.log), "timing": tm, "eval_share": share,
           "stages": [{"start": a, "end": b, "complete": c} for a, b, c in stages], "bins": {}, "g": {}}
    for which in WHICH:
        t = eval_table(evals, which)
        if not len(t["step"]) or not stages:
            continue
        per_bins = binned(t, stages, args.bin)
        per_g, pooled_g = plateau(t, stages, args.g_window, exclude=set(args.exclude_stage))
        res["bins"][which] = {"per_stage": per_bins, "pooled": pool_bins(per_bins)}
        res["g"][which] = {"per_stage": per_g, "pooled": pooled_g}
    if args.out_dir:
        bounds = {b for _, b, _ in stages} | {a for a, _, _ in stages}
        res["out_dir"] = scan_out_dir(args.out_dir, bounds, updates, evals)

    if args.json:
        # g が読めない（伸びが 0）ときの NaN は null にする（厳密な JSON にする）
        text = json.dumps(finite(res), indent=1, ensure_ascii=False, default=str, allow_nan=False)
        if args.json == "-":
            print(text)
            return
        pathlib.Path(args.json).write_text(text)
    print_timing(tm, share)
    if stages:
        print("段階: " + "、".join(f"{a}→{b}{'' if c else '（途中）'}" for a, b, c in stages))
    else:
        print("段階: reg_update の行が無い（または段階の長さが分からない）ので区切りを決められない（--dm と --schedule-start を与える）")
    for which in res["bins"]:
        print_bins(which, res["bins"][which]["per_stage"], res["bins"][which]["pooled"])
    for which in res["g"]:
        print_g(which, res["g"][which]["per_stage"], res["g"][which]["pooled"], args.g_lo, args.g_hi)
    if "out_dir" in res:
        print_out_dir(res["out_dir"])


if __name__ == "__main__":
    main()
