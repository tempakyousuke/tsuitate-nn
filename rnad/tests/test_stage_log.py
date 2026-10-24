"""stage_log.py（学習ログからの事前確認）の検査。合成した log.jsonl と run のディレクトリで確かめる（tsuitate_rl も
torch も要らない）。

    .venv/bin/python rnad/tests/test_stage_log.py

- 1更新の秒を α < 1 / α = 1 に分け、--pipeline の行は busy と更新の大きいほうを律速とする
- 段階の区切りを reg_update の行から取り、ログが区切りから始まっていればその段階も足す（--dm でも同じ区切り）
- 段階の平らさ g: 直線（更新数比例）の勝率なら 0.30、被覆率の模型（段階数比例）の勝率なら 0.15 未満になる。
  256局の評価の雑音では、窓どうしの差の SE が約0.54pt（docs の見積もり）になる
- 区切りでの停止の時間と、書き出しの間隔のうちログに出ない秒の割合を mtime から出す
- 落ちて checkpoint から再開した run（step が戻って同じ step の行がもう一度書かれる）は後の行を使う
- 段階の長さが変わる区切り（Δm=2,000 → 10,000）でも、最初と最後の段階の長さを α から逆算してずれない
- 段階の伸びが 0（g が読めない）でも落ちず、--json は厳密な JSON（NaN を書かない）
"""

import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import stage_log as S  # noqa: E402

STAGES = [(58000, 68000), (68000, 78000), (78000, 88000)]


def coverage(L, R, Tr=300, tau=0.001):
    """被覆率の模型（補間 R の線形の動きを、時定数 Tr の現在のネットと τ のターゲットの1次遅れ2段に通す）。f(0)=0、f(L)≈1"""
    o = t = 0.0
    out = np.zeros(L + 1)
    for n in range(L):
        p = min(1.0, (n + 1) / R)
        o += (p - o) / Tr
        t += tau * (o - t)
        out[n + 1] = t
    return out


def score_at(curve, stages):
    """step → 錨への勝率。linear = 更新数に比例して一様に伸びる / plateau = 段階ごとに被覆率の模型で伸びて平らになる"""
    first = stages[0][0]
    if curve == "flat":
        return lambda s: 0.50
    if curve == "linear":
        return lambda s: 0.50 + 0.04 * (s - first) / 10_000
    f = coverage(10_000, 5_000)
    f = f / f[-1]
    gains = [0.10, 0.04, 0.04]

    def at(s):
        e = 0.50
        for (a, b), g in zip(stages, gains):
            if s >= b:
                e += g
            elif s > a:
                e += g * f[s - a]
        return e
    return at


def make_log(path, curve, games=256, seed=0, stages=STAGES, eval_every=50, start_eval=True, pipeline=True, extra=0):
    """train_rnad.py と同じ形の log.jsonl（更新の行は step ごと、評価は eval_every ごとに target と current）"""
    rng = np.random.default_rng(seed)
    E = score_at(curve, stages)
    first, last = stages[0][0], stages[-1][1] + extra
    L = stages[0][1] - stages[0][0]
    rows = []

    def add_eval(s_eval):
        for which in ("target", "current"):
            p = E(s_eval) - (0.01 if which == "current" else 0.0)
            w = int(round(games * p)) if games >= 10**6 else int(rng.binomial(games, p))
            rows.append({"step": s_eval, "eval": which, "eval_vs_anchor": w / games, "w": w,
                         "l": games - w, "d": 0, "ci95": 0.0, "sec": 30.0})
    if start_eval:
        add_eval(first)  # 区切りでの評価（前の run が書いた行）
    for step in range(first + 1, last + 1):
        a = first + (step - 1 - first) // L * L
        alpha = min(2.0 * (step - 1 - a) / L, 1.0)
        reg = (step - 1) == a + L - 1
        busy, upd = (8.0, 7.3) if alpha < 1 else (6.0, 6.5)
        rec = {"step": step, "alpha": alpha, "reg_update": reg, "samples": 60_000, "loss_v": 1.0,
               "update_sec": upd + rng.normal(0, 0.05)}
        if pipeline:
            rec["collect_busy_sec"] = busy + rng.normal(0, 0.05)
            rec["collect_sec"] = max(rec["collect_busy_sec"] - rec["update_sec"], 0.0) + 0.01
        else:
            rec["collect_sec"] = busy
        rows.append(rec)
        if step % eval_every == 0:
            add_eval(step)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def check_timing(tmp):
    log = tmp / "t.jsonl"
    make_log(log, "linear", games=10**6)
    up, ev = S.load_log(log)
    tm = S.timing(up)
    r, a1 = tm["pipeline/ramp"], tm["pipeline/alpha1"]
    assert r["bottleneck"] == "collect" and a1["bottleneck"] == "update", (r, a1)
    assert abs(r["busy"]["median"] - 8.0) < 0.02 and abs(a1["update"]["median"] - 6.5) < 0.02
    assert r["updates"] == a1["updates"] == 15_000
    assert abs(r["total"]["median"] - 8.01) < 0.03, r["total"]
    share = S.eval_share(up, ev)
    assert 0 < share["eval_share"] < 0.2
    print(f"1更新の秒: α<1 は集める（busy {r['busy']['median']:.2f}s > 更新 {r['update']['median']:.2f}s）、"
          f"α=1 は更新が律速。評価の割合 {100 * share['eval_share']:.1f}%")


def check_stages(tmp):
    log = tmp / "s.jsonl"
    make_log(log, "linear", games=10**6, start_eval=False, extra=3_000)
    up, ev = S.load_log(log)
    st = S.find_stages(up, ev)
    assert st[:3] == [(58000, 68000, True), (68000, 78000, True), (78000, 88000, True)], st
    assert st[3] == (88000, 98000, False), st
    assert S.find_stages(up, ev, dm=10_000, schedule_start=58_000)[:3] == st[:3]
    # 最初の reg_update より前が途中から始まるログ（区切りから始まっていない）なら、その段階は足さない
    up2 = [r for r in up if r["step"] > 60_000]
    ev2 = [r for r in ev if r["step"] > 60_000]
    assert S.find_stages(up2, ev2)[0] == (68000, 78000, True)
    print("段階の区切り: reg_update から 58000→68000→78000→88000（区切りから始まるログは最初の段階も）、"
          "最後は途中の段階")


def check_plateau(tmp):
    res = {}
    for curve in ("linear", "plateau"):
        log = tmp / f"{curve}.jsonl"
        make_log(log, curve, games=10**6)
        up, ev = S.load_log(log)
        st = S.find_stages(up, ev)
        t = S.eval_table(ev, "target")
        per, pooled = S.plateau(t, st)
        res[curve] = (per, pooled[10_000])
    per, pooled = res["linear"]
    for r in per:
        assert abs(r["g"] - 0.30) < 0.005, r
    assert abs(pooled["g"] - 0.30) < 0.005
    assert per[0]["how_start"] == "one-sided" and per[1]["how_start"] == "straddle"
    assert per[0]["how_end"] == "straddle" and per[-1]["how_end"] == "one-sided"
    per_b, pooled_b = res["plateau"]
    assert pooled_b["g"] < 0.15 and all(r["g"] < 0.15 for r in per_b), [r["g"] for r in per_b]
    assert S.verdict(pooled_b, 0.15, 0.25).startswith("g<") and S.verdict(pooled, 0.15, 0.25).startswith("g>")
    # 位置ごとの平均: 直線の勝率なら (0, 1000] の平均は 500.5 更新目の値に近い
    t = S.eval_table(S.load_log(tmp / "linear.jsonl")[1], "target")
    b = S.binned(t, [(68000, 78000, True)], 1000)[0]
    want = 0.50 + 0.04 * (68000 + 525 - 58000) / 10_000
    assert abs(b["cells"][1]["score"] - want) < 1e-4 and b["positions"][:3] == [0, 1000, 2000], b["cells"][1]
    print(f"段階の平らさ g: 直線 {pooled['g']:.3f}（各段階 0.30）、被覆率の模型 {pooled_b['g']:.3f}"
          f"（段階ごと {', '.join('%.3f' % r['g'] for r in per_b)}）")


def check_noise(tmp):
    """256局・50更新ごとの評価の雑音で、窓どうしの差の SE が docs の見積もり（約0.54pt）になり、g の SE も出る"""
    se_num, gs = [], []
    for seed in range(8):
        log = tmp / f"n{seed}.jsonl"
        make_log(log, "linear", games=256, seed=seed)
        up, ev = S.load_log(log)
        t = S.eval_table(ev, "target")
        per, pooled = S.plateau(t, S.find_stages(up, ev))
        se_num.append(per[1]["num_se"])
        gs.append((pooled[10_000]["g"], pooled[10_000]["se"]))
    assert 0.0045 < np.mean(se_num) < 0.0065, se_num
    g = np.array([x for x, _ in gs])
    se = np.mean([s for _, s in gs])
    assert abs(g.mean() - 0.30) < 3 * se, (g, se)
    print(f"256局の雑音: 窓どうしの差の SE {100 * np.mean(se_num):.2f}pt、合算 g = {g.mean():.2f}"
          f"（8本の SD {g.std():.2f}、デルタ法の SE {se:.2f}）")


def check_out_dir(tmp):
    d = tmp / "run"
    d.mkdir()
    rows = []
    for step in range(1, 401):
        rows.append({"step": step, "alpha": 1.0, "reg_update": step in (200, 400), "samples": 1000,
                     "collect_sec": 0.5, "update_sec": 1.5})
        if step % 50 == 0:
            rows.append({"step": step, "eval": "target", "eval_vs_anchor": 0.5, "w": 1, "l": 1, "d": 0, "sec": 4.0})
    (d / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    base = 1_800_000_000.0
    per = 1.05 * (100 * 2.0 + 2 * 4.0)  # 書き出しの間隔ごとの壁時計（5% がログに出ない秒）
    stop = 2 * 3600.0
    mt = {100: base, 200: base + per, 300: base + 2 * per + stop, 400: base + 3 * per + stop}
    for s, m in mt.items():
        p = d / f"policy_{s}.bin"
        p.write_bytes(b"x")
        os.utime(p, (m, m))
    q = d / "config_resume_200.json"
    q.write_text("{}")
    os.utime(q, (mt[200] + stop, mt[200] + stop))
    up, ev = S.load_log(d / "log.jsonl")
    st = S.find_stages(up, ev)
    assert st == [(0, 200, True), (200, 400, True)], st
    o = S.scan_out_dir(d, {0, 200, 400}, up, ev)
    (s0,) = o["stops"]
    assert s0["step"] == 200 and s0["boundary"] and s0["exact"] and abs(s0["hours"] - 2.0) < 1e-6, s0
    ov = o["overhead"]
    assert ov["intervals"] == 2 and abs(ov["other_share"] - (1 - 1 / 1.05)) < 1e-6, ov
    # コマンドとして（JSON を書いて表も出す）
    out = subprocess.run([sys.executable, str(HERE.parent / "stage_log.py"), str(d / "log.jsonl"), "--out-dir", str(d),
                          "--json", str(tmp / "o.json")], capture_output=True, text=True, check=True)
    assert "再開までの停止" in out.stdout and "区切り" in out.stdout, out.stdout
    j = json.loads((tmp / "o.json").read_text())
    assert j["out_dir"]["stops"][0]["step"] == 200 and set(j["g"]) == {"target"}
    print(f"--out-dir: 区切り 200 での停止 {s0['hours']:.1f}時間、ログに出ない秒 {100 * ov['other_share']:.1f}%")


def stage_rows(stages, eval_every=50, upd=1.0, score=0.5, games=1000):
    """長さの違う段階の列 [(始まり, 終わり)] を通した更新と評価の行（train_rnad.py と同じ α と reg_update）"""
    rows = []
    for a, b in stages:
        L = b - a
        for step in range(a + 1, b + 1):
            rows.append({"step": step, "alpha": min(2.0 * (step - 1 - a) / L, 1.0), "reg_update": step == b,
                         "samples": 1000, "collect_sec": 0.5, "update_sec": upd})
            if step % eval_every == 0:
                w = int(round(games * score))
                rows.append({"step": step, "eval": "target", "eval_vs_anchor": score, "w": w, "l": games - w, "d": 0,
                             "sec": 1.0})
    return rows


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def check_restart(tmp):
    """落ちて checkpoint 225 から再開した run: 226〜260 の行が2回（前の run は秒と評価が違う）。後の行を使う"""
    good = stage_rows([(0, 200), (200, 400)])
    bad = [dict(r, update_sec=99.0) if "update_sec" in r else dict(r, w=0, l=1000, eval_vs_anchor=0.0)
           for r in good if 225 < r["step"] <= 260]
    rows = [r for r in good if r["step"] <= 225] + bad + [r for r in good if r["step"] > 225]
    write_rows(tmp / "restart.jsonl", rows)
    up, ev = S.load_log(tmp / "restart.jsonl")
    assert [r["step"] for r in up] == list(range(1, 401)), "更新の行が step ごとに1行でない"
    assert max(r["update_sec"] for r in up) == 1.0 and all(e["w"] == 500 for e in ev), "前の run の行が残っている"
    assert [e["step"] for e in ev] == list(range(50, 401, 50))
    assert S.find_stages(up, ev) == [(0, 200, True), (200, 400, True)]
    # 同じ out で step 0 から回し直した（評価の行 step 0 から）: 前の run の行は全部捨てる
    fresh = [{"step": 0, "eval": "target", "eval_vs_anchor": 0.5, "w": 500, "l": 500, "d": 0, "sec": 1.0}]
    write_rows(tmp / "fresh.jsonl", bad + fresh + good[:120])
    up2, ev2 = S.load_log(tmp / "fresh.jsonl")
    assert max(r["update_sec"] for r in up2) == 1.0 and [e["step"] for e in ev2] == [0, 50, 100]
    print("再開: step が戻ったら前の run の行を捨てる（更新の行は step ごとに1行、評価は後の run のもの）")


def check_dm_change(tmp):
    """Δm=2,000 の段階のあと 58,000 から Δm=10,000（--schedule-start 58000）で、最後は途中の段階"""
    rows = stage_rows([(54000, 56000), (56000, 58000), (58000, 68000), (68000, 78000)])
    rows = [r for r in rows if r["step"] <= 71000]
    write_rows(tmp / "dm.jsonl", rows)
    up, ev = S.load_log(tmp / "dm.jsonl")
    assert S.find_stages(up, ev) == [(54000, 56000, True), (56000, 58000, True), (58000, 68000, True),
                                     (68000, 78000, False)], S.find_stages(up, ev)
    # 56,000 から始まるログ: 最初の段階の長さは次の段階（10,000）でなく α から 2,000
    up2, ev2 = [r for r in up if r["step"] > 56000], [r for r in ev if r["step"] > 56000]
    assert S.find_stages(up2, ev2)[0] == (56000, 58000, True), S.find_stages(up2, ev2)
    # 区切りの途中から始まるログは最初の段階を足さない
    up3, ev3 = [r for r in up if r["step"] > 57000], [r for r in ev if r["step"] > 57000]
    assert S.find_stages(up3, ev3) == [(58000, 68000, True), (68000, 78000, False)], S.find_stages(up3, ev3)
    # --dm / --schedule-start は train_rnad.py と同じ区切り。食い違えば警告する
    up4, ev4 = [r for r in up if r["step"] > 58000], [r for r in ev if r["step"] > 58000]
    assert S.find_stages(up4, ev4, dm=10_000, schedule_start=58_000) == [(58000, 68000, True), (68000, 78000, False)]
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        S.find_stages(up4, ev4, dm=10_000)
    assert "区切りに無い" in err.getvalue(), err.getvalue()
    print("段階の長さが変わる区切り: 2,000 → 10,000 を α から逆算（最初の段階・途中の最後の段階とも）、--dm の食い違いは警告")


def check_flat(tmp):
    """伸びが 0 の段階: g は読めないと出し、--json は NaN を書かない"""
    log = tmp / "flat.jsonl"
    make_log(log, "flat", games=10**6)
    up, ev = S.load_log(log)
    per, pooled = S.plateau(S.eval_table(ev, "target"), S.find_stages(up, ev))
    assert S.verdict(pooled[10_000], 0.15, 0.25).startswith("段階の伸びが誤差の内"), pooled
    out = subprocess.run([sys.executable, str(HERE.parent / "stage_log.py"), str(log), "--json", "-"],
                         capture_output=True, text=True, check=True)

    def no_nan(x):
        raise ValueError(f"厳密な JSON でない: {x}")
    json.loads(out.stdout, parse_constant=no_nan)
    print("伸びが 0: g は読めないと出し、--json は厳密な JSON")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        check_timing(tmp)
        check_stages(tmp)
        check_plateau(tmp)
        check_noise(tmp)
        check_out_dir(tmp)
        check_restart(tmp)
        check_dm_change(tmp)
        check_flat(tmp)
