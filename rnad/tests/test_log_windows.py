"""log_windows.py の検査（tsuitate_rl は要らない）。

    .venv/bin/python rnad/tests/test_log_windows.py

- 窓は (from + i·width, from + (i+1)·width] で、評価の行・パネルの行は数えない。無いキー・NaN は空欄
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
            f.write(json.dumps({"step": 150, "eval": {"x": 1}, "entropy_mean": 99.0}) + "\n")
            f.write(json.dumps({"step": 250, "collect_sec": 1.0, "update_sec": 1.0, "entropy_mean": float("nan")}) + "\n")
        rows = W.read_rows(path)
        assert len(rows) == 201  # 評価の行は除く
        got = W.windows(rows, ["entropy_mean", "magnet_kl"], 100, 300, 100)
        assert [(b, n) for b, n, _ in got] == [(200, 100), (300, 101)]
        assert abs(got[0][2]["entropy_mean"] - 1.505) < 1e-12 and got[0][2]["magnet_kl"] is None
        assert abs(got[1][2]["entropy_mean"] - 2.505) < 1e-12 and got[1][2]["magnet_kl"] == 0.5  # NaN は数えない
        r = subprocess.run([sys.executable, str(HERE.parent / "log_windows.py"), str(path), "--from-step", "100",
                            "--to-step", "300", "--keys", "entropy_mean,magnet_kl"], capture_output=True, text=True)
        assert r.returncode == 0 and "| 200 | 100 | 1.505 |  |" in r.stdout, r.stdout + r.stderr
    print("log_windows: 窓の切り方・評価の行の除外・無いキーと NaN の扱い")


if __name__ == "__main__":
    check_windows()
