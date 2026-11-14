"""固定パネルで、測られる側だけ序盤を定跡から指させる形（`--panel-open-p`、`speedlab.play_vs(open_kw=...)`）の検査（CPU）。

    .venv/bin/python rnad/tests/test_panel_open.py

- P1 `play_vs`: 相手が PyTorch のネットでも Rust の戦略でも、定跡から指すのは測られる側だけ。同じ seed なら同じラインの組。
  渡さなければ誰も強制されない
- P2 `train_rnad.py`: `--panel-open-p 1` で panel_eval.jsonl の行に open_p が付き、強制なしの記録がある run を
  測り方を変えて再開すると、その時点も測る
"""

import json
import pathlib
import sys
import tempfile

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import openings  # noqa: E402
import speedlab as S  # noqa: E402
from model import NUM_PLANES_DEDUCED  # noqa: E402
from test_openings import write_book  # noqa: E402
from test_shaping import run_train, small_net  # noqa: E402

PLANES = NUM_PLANES_DEDUCED


def check_play_vs(tmp):
    book = openings.load_book(write_book(tmp))
    kw = openings.env_kwargs(book, 1.0)
    net = small_net()
    games = 8
    for spec in ("random", "heuristic"):
        b = S.Member(spec, PLANES, "cpu")
        fin, fin2, fin0 = [], [], []
        o = S.play_vs(net, b, games, 77, PLANES, per_game=True, device="cpu", open_kw=kw, finished=fin)
        o2 = S.play_vs(net, b, games, 77, PLANES, per_game=True, device="cpu", open_kw=kw, finished=fin2)
        S.play_vs(net, b, games, 77, PLANES, per_game=True, device="cpu", finished=fin0)
        assert len(fin) == games and (o == o2).all()
        lines = {}
        for f in fin:
            a = f["game_no"] % 2  # 偶数番の局で測られる側が先手
            assert f["forced_line"][a] is not None and f["forced_moves"][a] >= 1, (spec, f["game_no"])
            assert f["forced_line"][1 - a] is None and f["forced_moves"][1 - a] == 0, (spec, f["game_no"])
            lines[f["game_no"]] = f["forced_line"][a]
        assert lines == {f["game_no"]: f["forced_line"][f["game_no"] % 2] for f in fin2}
        assert all(f["forced_line"] == [None, None] for f in fin0)
    print("P1 play_vs: 定跡から指すのは測られる側だけ（相手がネットでも Rust の戦略でも）。同じ seed なら同じラインの組")


def check_train(tmp):
    book = write_book(tmp)
    panel = tmp / "panel.json"
    panel.write_text(json.dumps({"planes": PLANES, "members": ["random", "heuristic"],
                                 "ratings": {"random": 0.0, "heuristic": 100.0}}))
    common = ["--open-book", str(book), "--open-p", "1.0", "--panel", str(panel), "--panel-games", "4",
              "--panel-every", "3", "--panel-which", "target,current"]
    out = tmp / "run"
    run_train(out, *common, steps=3)
    rows = [json.loads(x) for x in open(out / "panel_eval.jsonl")]
    assert [r["step"] for r in rows] == [0] * 4 + [3] * 4 and not any("open_p" in r for r in rows)
    # 測り方を変えて再開: 再開した時点（3）も新しい測り方で測り、以後の行に open_p が付く
    r = run_train(out, *common, "--panel-open-p", "1.0", steps=6)
    rows = [json.loads(x) for x in open(out / "panel_eval.jsonl")][8:]
    assert [x["step"] for x in rows] == [3] * 4 + [6] * 4 and all(x["open_p"] == 1.0 for x in rows), rows
    assert "序盤を強制 p=1" in r.stdout
    # 同じ測り方でもう一度再開しても、再開した時点は測り直さない
    run_train(out, *common, "--panel-open-p", "1.0", steps=7)
    steps = [json.loads(x)["step"] for x in open(out / "panel_eval.jsonl")][16:]
    assert steps == [7] * 4, steps
    r = run_train(tmp / "rej", "--panel", str(panel), "--panel-games", "4", "--panel-open-p", "1.0", steps=1, check=False)
    assert r.returncode != 0 and "--open-book" in (r.stderr + r.stdout)
    print("P2 train_rnad.py: --panel-open-p で行に open_p が付き、測り方を変えて再開した時点も測る")


if __name__ == "__main__":
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as d:
        check_play_vs(pathlib.Path(d))
        check_train(pathlib.Path(d))
    print("ok")
