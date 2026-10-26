"""log_windows.py の検査（tsuitate_rl は要らない）。

    .venv/bin/python rnad/tests/test_log_windows.py

- 窓は (from + i·width, from + (i+1)·width] で、評価の行は数えない。無いキー・NaN は空欄
- 読めない行は飛ばし、再開で step が戻ったら後の行を使う（stage_log.load_log）
"""

import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import log_windows as W  # noqa: E402


def check_windows():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "log.jsonl"
        with open(path, "w") as f:
            for st in range(101, 301):
                rec = {"step": st, "collect_sec": 1.0, "update_sec": 1.0, "entropy_mean": st / 100}
                if st > 200:
                    rec["magnet_kl"] = 0.5
                f.write(json.dumps(rec) + "\n")
            f.write(json.dumps({"step": 150, "eval": "target", "eval_vs_anchor": 0.5, "entropy_mean": 99.0}) + "\n")
            f.write('{"step": 151, "collect_sec": 1.0, "upd\n')  # 落ちたときの書きかけの行
            # 250 の checkpoint から再開: 251〜300 がもう一度書かれる（前の行は捨て、後の行を使う）
            for st in range(251, 301):
                rec = {"step": st, "collect_sec": 1.0, "update_sec": 1.0, "entropy_mean": st / 100 + 1, "magnet_kl": 0.5}
                f.write(json.dumps(rec) + "\n")
            f.write(json.dumps({"step": 301, "collect_sec": 1.0, "update_sec": 1.0, "entropy_mean": float("nan")}) + "\n")
        rows = W.read_rows(path)
        assert len(rows) == 201  # 評価の行・書きかけの行・再開で重なった前の行は除く
        got = W.windows(rows, ["entropy_mean", "magnet_kl"], 100, 400, 100)
        assert [(b, n) for b, n, _ in got] == [(200, 100), (300, 100), (400, 1)]
        assert abs(got[0][2]["entropy_mean"] - 1.505) < 1e-12 and got[0][2]["magnet_kl"] is None
        # 201〜250 は前の行、251〜300 は再開の後の行（+1）: (2.255 + 3.755) / 2
        assert abs(got[1][2]["entropy_mean"] - 3.005) < 1e-12 and got[1][2]["magnet_kl"] == 0.5
        assert got[2][2]["entropy_mean"] is None  # NaN は数えない
        r = subprocess.run([sys.executable, str(HERE.parent / "log_windows.py"), str(path), "--from-step", "100",
                            "--to-step", "300", "--keys", "entropy_mean,magnet_kl"], capture_output=True, text=True)
        assert r.returncode == 0 and "| 200 | 100 | 1.505 |  |" in r.stdout, r.stdout + r.stderr
    print("log_windows: 窓の切り方・評価の行と書きかけの行の除外・再開で重なった行・無いキーと NaN の扱い")


if __name__ == "__main__":
    check_windows()
