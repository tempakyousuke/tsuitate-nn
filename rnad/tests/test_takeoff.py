"""離陸の判定と表（rnad/takeoff.py）の検査。torch も tsuitate_rl も使わない。

    .venv/bin/python rnad/tests/test_takeoff.py

- find_takeoff: 直近 window 更新の勝率がしきい値以上になった最初の更新。頭の window 更新に満たない区間では判定しない
- read_log: 同じ step が2回あれば後のもの・書きかけの行は飛ばす・相手が複数なら合算
- report: 離陸した run の数、打ち切りより前で止まった run は率に数えない、打ち切りを上限にした平均
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
    run = root / name
    run.mkdir(parents=True)
    with open(run / "log.jsonl", "w") as f:
        for i, w in enumerate(wins):
            lg = {o: [0, 0, 0] for o in opponents}
            lg[opponents[0]] = [w, games - w, 0]
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


if __name__ == "__main__":
    check_find()
    check_read_and_report()
    print("ok")
