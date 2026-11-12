"""離陸の判定と表（rnad/takeoff.py）の検査。torch も tsuitate_rl も使わない。

    .venv/bin/python rnad/tests/test_takeoff.py

- find_takeoff: 直近 window 更新の勝率がしきい値以上になった最初の更新。頭の window 更新に満たない区間では判定しない
- read_log: 同じ step が2回あれば後のもの・書きかけの行は飛ばす・相手が複数なら合算。再開で step が戻ったら
  前の run の後続の行を捨てる（stage_log.load_log と同じ規約）
- report: 離陸した run の数、打ち切りより前で止まった run は率に数えない、打ち切りを上限にした平均。
  離陸・打ち切りまでの局数は、その更新までの各更新の局数の和（途中で batch を変えた run でも後の batch が混ざらない）
"""

import io
import json
import pathlib
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import takeoff  # noqa: E402


def rows(wins, games=64):
    return [(i + 1, w, games) for i, w in enumerate(wins)]


def check_find():
    # 勝率 0 → 50更新目から毎更新 16勝（25%）: 窓 50 の勝率が 10% を超えるのは 20更新ぶん入った時（16×20 / 3200 = 10%）
    w = [0] * 100 + [16] * 100
    assert takeoff.find_takeoff(rows(w), 50, 0.10) == 120
    assert takeoff.find_takeoff(rows(w), 50, 0.30) is None
    assert takeoff.find_takeoff(rows([0] * 300), 50, 0.10) is None
    # 頭の window 更新に満たない区間では判定しない（最初から勝っていても 50更新目）
    assert takeoff.find_takeoff(rows([32] * 80), 50, 0.10) == 50
    assert takeoff.find_takeoff(rows([32] * 30), 50, 0.10) is None
    # 一度しきい値を超えた後で下がっても、最初の更新を返す
    assert takeoff.find_takeoff(rows([0] * 60 + [64] * 10 + [0] * 200), 50, 0.10) == 65
    print("K1 find_takeoff: 直近 50更新の勝率がしきい値以上になった最初の更新。頭の 50更新は判定しない")


def write_run(root, name, wins, games=64, dup=False, opponents=("x",)):
    """games は1更新の局数（更新ごとのリストも可。途中で batch を変えた run）"""
    run = root / name
    run.mkdir(parents=True)
    per = games if isinstance(games, (list, tuple)) else [games] * len(wins)
    games = per[-1]
    with open(run / "log.jsonl", "w") as f:
        for i, w in enumerate(wins):
            lg = {o: [0, 0, 0] for o in opponents}
            lg[opponents[0]] = [w, per[i] - w, 0]
            f.write(json.dumps({"step": i + 1, "league_games": lg}) + "\n")
            f.write(json.dumps({"step": i + 1, "eval": "target"}) + "\n")  # 評価の行（league_games なし）は読まない
        if dup:  # 再開で同じ step がもう1回書かれた（後のものを使う）
            f.write(json.dumps({"step": len(wins), "league_games": {opponents[0]: [games, 0, 0]}}) + "\n")
            f.write('{"step": 999, "league_games": {"x": [1, ')  # 書きかけの行
    return run


def check_read_and_report():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        a = write_run(root, "a_s1", [0] * 100 + [16] * 100)  # 離陸 120
        write_run(root, "a_s2", [0] * 400)  # 打ち切りまで離陸せず
        write_run(root, "a_s3", [0] * 150)  # 途中（率に数えない）
        write_run(root, "b_s1", [0] * 399 + [0], dup=True, opponents=("x", "y"))
        r = takeoff.read_log(a)
        assert len(r) == 200 and r[119] == (120, 16, 64)
        rb = takeoff.read_log(root / "b_s1")
        assert rb[-1] == (400, 64, 64), rb[-1]  # 後の行で上書き・書きかけの行は無視
        st = takeoff.run_status(a, 50, 0.10)
        assert st["takeoff"] == 120 and st["last"] == 200 and st["games_per_update"] == 64
        buf = io.StringIO()
        with redirect_stdout(buf):
            takeoff.cmd_report(SimpleNamespace(roots=[str(root)], extra=[], window=50, threshold=0.10, horizon=400))
        out = buf.getvalue()
        line_a = next(x for x in out.splitlines() if x.startswith("| `a`"))
        # 離陸 1 / 数える run 2（s3 は途中）。離陸までの局数 120 × 64 = 7.7k、打ち切りを上限にした平均 = (7.68k + 25.6k) / 2
        assert "| 1 / 2 |" in line_a and "**7.7k**" in line_a and "> 25.6k" in line_a and "（途中）" in line_a, line_a
        assert "| 7.7k | 16.6k |" in line_a, line_a
        line_b = next(x for x in out.splitlines() if x.startswith("| `b`"))
        assert "| 0 / 1 |" in line_b, line_b
        # 打ち切りを 100 にすると a_s1 の離陸（120）は打ち切りの後 = 離陸していない扱い
        buf = io.StringIO()
        with redirect_stdout(buf):
            takeoff.cmd_report(SimpleNamespace(roots=[str(root)], extra=[f"c={a}"], window=50, threshold=0.10, horizon=100))
        out = buf.getvalue()
        assert "| 0 / 3 |" in next(x for x in out.splitlines() if x.startswith("| `a`")), out
        assert next(x for x in out.splitlines() if x.startswith("| `c`")), out
    print("K2 read_log / report: 同じ step は後のもの・書きかけの行は飛ばす。離陸した run の数・途中の run は率に数えない・"
          "打ち切りを上限にした平均")


def check_batch_change():
    """途中で batch（1更新の局数）を変えた run: 離陸までの局数・打ち切りまでの局数は、その更新までの各更新の局数の和
    （run 全体の1更新あたりの平均 × 更新数だと、離陸より後の batch が混ざる）"""
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        # 前半 50更新は 64局（16勝 = 25%）、後半 50更新は 128局。step 50 で離陸 = 50 × 64 = 3,200局（平均を掛けると 4,800局）
        v = write_run(root, "v_s1", [16] * 50 + [0] * 50, games=[64] * 50 + [128] * 50)
        st = takeoff.run_status(v, 50, 0.10)
        assert st["takeoff"] == 50 and st["takeoff_games"] == 3200, st["takeoff_games"]
        # 離陸しない run: 打ち切り 60 更新までは 50 × 64 + 10 × 128 = 4,480局（平均を掛けると 5,760局）
        write_run(root, "w_s1", [0] * 100, games=[64] * 50 + [128] * 50)
        buf = io.StringIO()
        with redirect_stdout(buf):
            takeoff.cmd_report(SimpleNamespace(roots=[str(root)], extra=[], window=50, threshold=0.10, horizon=60))
        out = buf.getvalue()
        line_v = next(x for x in out.splitlines() if x.startswith("| `v`"))
        assert "**3.2k**" in line_v and "| 3.2k | 3.2k |" in line_v, line_v
        line_w = next(x for x in out.splitlines() if x.startswith("| `w`"))
        assert "> 4.5k" in line_w and "| — | 4.5k |" in line_w, line_w
    print("K4 report: 途中で batch を変えた run でも、離陸・打ち切りまでの局数はその更新までの局数の和")


def check_rollback():
    """read_log / run_status: checkpoint から再開して step が戻ったら、そこから先の前の run の行を捨てる。
    step 1, 2, 3, 2 なら今の経路は 1, 2 で、前の run の step 3 の勝ちで離陸を見つけない（last も 2）"""
    with tempfile.TemporaryDirectory() as d:
        run = pathlib.Path(d) / "r_s1"
        run.mkdir()
        rows = [{"step": s, "games": 0, "league_games": {"opp": [w, 8 - w, 0]}, "collect_sec": 6, "update_sec": 4}
                for s, w in [(1, 0), (2, 0), (3, 8), (2, 0)]]
        rows.insert(3, {"step": 3, "eval": "target", "eval_vs_anchor": 0.5})
        (run / "log.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
        r = takeoff.read_log(run)
        assert r == [(1, 0, 8), (2, 0, 8)], r
        assert takeoff.find_takeoff(r, window=1, threshold=0.5) is None
        st = takeoff.run_status(run, 1, 0.5)
        assert st["last"] == 2 and st["takeoff"] is None, st
        # 同じ out で step 0 から回し直した（評価の行 step 0 が先に出る）: 前の run の行を全部捨てる
        rows = [{"step": 1, "league_games": {"opp": [8, 0, 0]}}, {"step": 2, "league_games": {"opp": [8, 0, 0]}},
                {"step": 0, "eval": "target", "eval_vs_anchor": 0.5}, {"step": 1, "league_games": {"opp": [0, 8, 0]}}]
        (run / "log.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
        assert takeoff.read_log(run) == [(1, 0, 8)]
    print("K3 read_log: 再開で step が戻ったら前の run の後続の行を捨てる（前の run の勝ちで離陸を見つけない）")


if __name__ == "__main__":
    check_find()
    check_read_and_report()
    check_rollback()
    check_batch_change()
    print("ok")
