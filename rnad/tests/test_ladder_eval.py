"""ハンデの梯子（rnad/ladder_eval.py）の評価済みの判定の検査。対局（`play`）は差し替えるので tsuitate_rl は使わない。

    .venv/bin/python rnad/tests/test_ladder_eval.py

- 同じ (which, step, 段) は、局数・seed・書き出しの重みの指紋が前の評価（最後の行）と同じなら測り直さない
- 違えば（局数を増やした・seed を変えた・再開で書き出しが変わった）測り直して行を足し、集計（load_rows）は後の行を使う
- `--redo` は同じ条件でも測り直す。指紋の無い前の行は局数・seed だけを確かめる
"""

import argparse
import contextlib
import io
import json
import pathlib
import sys
import tempfile

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ladder_eval as L  # noqa: E402
from export_weights import fingerprint, write  # noqa: E402
from model import PolicyValueNet  # noqa: E402


def fake_play(calls):
    def play(net, opponent, games, seed, planes, device):
        calls.append((opponent, games, seed))
        fin = [{"learner": g % 2, "plies": 10, "fouls": [0, 0], "reason": "checkmate"} for g in range(games)]
        return np.ones(games, dtype=np.int8), fin
    return play


def check_cache():
    torch.manual_seed(1)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        run = pathlib.Path(tmp) / "run_s1"
        run.mkdir()
        write(PolicyValueNet(8, 1).eval(), run / "current_5.bin")
        orig = L.play

        def ev(games, seed=7, **kw):
            calls = []
            L.play = fake_play(calls)
            try:
                L.cmd_eval(argparse.Namespace(runs=[run], opponent="rl_v15", eps="1,0", games=games, which="current",
                                              steps="", every=0, seed=seed, threads=0, device="cpu", **kw))
            finally:
                L.play = orig
            rows = [json.loads(s) for s in (run / "ladder_eval.jsonl").read_text().splitlines()]
            return len(calls), rows

        n, rows = ev(2)
        assert n == 2 and len(rows) == 2 and all(r["weights"] == fingerprint(run / "current_5.bin") for r in rows)
        assert ev(2)[0] == 0, "同じ条件で測り直した"
        n, rows = ev(4)
        assert n == 2 and len(rows) == 4 and {r["games"] for r in L.load_rows(run).values()} == {4}
        n, rows = ev(4, seed=8)
        assert n == 2 and len(rows) == 6
        write(PolicyValueNet(8, 1).eval(), run / "current_5.bin")  # 再開で同じ step の書き出しが変わった
        n, rows = ev(4, seed=8)
        assert n == 2 and len(rows) == 8
        assert ev(4, seed=8)[0] == 0
        assert ev(4, seed=8, redo=True)[0] == 2
        # 指紋の無い前の行: 局数・seed が同じなら使う
        (run / "ladder_eval.jsonl").write_text("".join(
            json.dumps({"which": "current", "step": 5, "opponent": o, "base": "rl_v15", "eps": e, "w": 4, "l": 0,
                        "d": 0, "games": 4, "seed": 7}) + "\n" for o, e in (("rl_v15@1", 1.0), ("rl_v15", 0.0))))
        assert ev(4)[0] == 0
        assert ev(6)[0] == 2
    print("ladder_eval: 局数・seed・書き出しの指紋が前の評価と同じなら飛ばし、違えば測り直して行を足す（集計は後の行）・"
          "--redo・指紋の無い前の行")


if __name__ == "__main__":
    check_cache()
    print("ok")
