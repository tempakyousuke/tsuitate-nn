"""固定パネルで、測られる側だけ序盤を定跡から指させる形（`--panel-open-p`、`speedlab.play_vs(open_kw=...)`）の検査（CPU）。

    .venv/bin/python rnad/tests/test_panel_open.py

- P1 `play_vs`: 相手が PyTorch のネットでも Rust の戦略でも、定跡から指すのは測られる側だけ。同じ seed なら同じラインの組。
  渡さなければ誰も強制されない
  `open_both` なら相手の側も定跡から指す（相手がネットのときだけ）
  `open_opponent` なら相手の側だけ定跡から指す（相手がネットのときだけ）
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
    # open_both: 相手（ネット）の側も定跡から指す。Rust の戦略の相手・open_kw なしでは使えない
    b = S.Member("random", PLANES, "cpu")
    fin = []
    S.play_vs(net, b, games, 77, PLANES, per_game=True, device="cpu", open_kw=kw, finished=fin, open_both=True)
    assert len(fin) == games
    assert all(f["forced_line"][c] is not None and f["forced_moves"][c] >= 1 for f in fin for c in (0, 1)), fin
    for bad_b, bad_kw in ((S.Member("heuristic", PLANES, "cpu"), kw), (b, None)):
        try:
            S.play_vs(net, bad_b, games, 77, PLANES, device="cpu", open_kw=bad_kw, open_both=True)
        except ValueError:
            pass
        else:
            raise AssertionError("open_both が通ってしまった")
    # open_opponent: 相手（ネット）の側だけ定跡から指す。Rust の戦略の相手・open_kw なし・open_both と一緒では使えない
    fin = []
    S.play_vs(net, b, games, 77, PLANES, per_game=True, device="cpu", open_kw=kw, finished=fin, open_opponent=True)
    assert len(fin) == games
    for f in fin:
        a = f["game_no"] % 2
        assert f["forced_line"][a] is None and f["forced_moves"][a] == 0, f
        assert f["forced_line"][1 - a] is not None and f["forced_moves"][1 - a] >= 1, f
    for bad_b, bad_kw, both in ((S.Member("heuristic", PLANES, "cpu"), kw, False), (b, None, False), (b, kw, True)):
        try:
            S.play_vs(net, bad_b, games, 77, PLANES, device="cpu", open_kw=bad_kw, open_opponent=True, open_both=both)
        except ValueError:
            pass
        else:
            raise AssertionError("open_opponent が通ってしまった")
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
    # レートの無い相手は勝敗だけ出して Elo には入れない。1体もレートが無ければ止まる
    panel2 = tmp / "panel2.json"
    panel2.write_text(json.dumps({"planes": PLANES, "members": ["random", "heuristic"], "ratings": {"random": 0.0}}))
    r = run_train(tmp / "unrated", "--panel", str(panel2), "--panel-games", "4", steps=1)
    assert "Elo はレートのある 1体から" in r.stdout and "heuristic " in r.stdout, r.stdout
    assert {json.loads(x)["opponent"] for x in open(tmp / "unrated" / "panel_eval.jsonl")} == {"random", "heuristic"}
    panel2.write_text(json.dumps({"planes": PLANES, "members": ["random"], "ratings": {}}))
    r = run_train(tmp / "norate", "--panel", str(panel2), "--panel-games", "4", steps=1, check=False)
    assert r.returncode != 0 and "レートのある相手" in (r.stderr + r.stdout)
    r = run_train(tmp / "rej", "--panel", str(panel), "--panel-games", "4", "--panel-open-p", "1.0", steps=1, check=False)
    assert r.returncode != 0 and "--open-book" in (r.stderr + r.stdout)
    # open_both に並べた相手の行には open_both が付く。Rust の戦略の名前・--panel-open-p なしは止まる
    panel3 = tmp / "panel3.json"
    panel3.write_text(json.dumps({"planes": PLANES, "members": ["random", "heuristic"],
                                  "ratings": {"random": 0.0, "heuristic": 100.0}, "open_both": ["random"]}))
    common3 = ["--open-book", str(book), "--open-p", "1.0", "--panel", str(panel3), "--panel-games", "4"]
    run_train(tmp / "both", *common3, "--panel-open-p", "1.0", steps=1)
    rows = [json.loads(x) for x in open(tmp / "both" / "panel_eval.jsonl")]
    assert rows and all(x.get("open_both", False) == (x["opponent"] == "random") for x in rows), rows
    r = run_train(tmp / "both_nop", *common3, steps=1, check=False)
    assert r.returncode != 0 and "open_both" in (r.stderr + r.stdout)
    panel3.write_text(json.dumps({"planes": PLANES, "members": ["random", "heuristic"],
                                  "ratings": {"random": 0.0, "heuristic": 100.0}, "open_both": ["heuristic"]}))
    r = run_train(tmp / "both_rust", *common3, "--panel-open-p", "1.0", steps=1, check=False)
    assert r.returncode != 0 and "open_both" in (r.stderr + r.stdout)
    # open_opponent に並べた相手の行には open_opponent が付く（学習は強制なし、定跡は --panel-open-book）。
    # 定跡なし・Rust の戦略の名前・open_both と重ねるのは止まる
    panel4 = tmp / "panel4.json"
    panel4.write_text(json.dumps({"planes": PLANES, "members": ["random", "heuristic"],
                                  "ratings": {"random": 0.0, "heuristic": 100.0}, "open_opponent": ["random"]}))
    common4 = ["--panel", str(panel4), "--panel-games", "4"]
    run_train(tmp / "opp", *common4, "--panel-open-book", str(book), steps=1)
    rows = [json.loads(x) for x in open(tmp / "opp" / "panel_eval.jsonl")]
    assert rows and all(x.get("open_opponent", False) == (x["opponent"] == "random") for x in rows), rows
    assert not any("open_p" in x or "open_both" in x for x in rows), rows
    r = run_train(tmp / "opp_nobook", *common4, steps=1, check=False)
    assert r.returncode != 0 and "open_opponent" in (r.stderr + r.stdout)
    panel4.write_text(json.dumps({"planes": PLANES, "members": ["random", "heuristic"],
                                  "ratings": {"random": 0.0, "heuristic": 100.0}, "open_opponent": ["heuristic"]}))
    r = run_train(tmp / "opp_rust", *common4, "--panel-open-book", str(book), steps=1, check=False)
    assert r.returncode != 0 and "open_opponent" in (r.stderr + r.stdout)
    panel4.write_text(json.dumps({"planes": PLANES, "members": ["random", "heuristic"], "ratings": {"random": 0.0, "heuristic": 100.0},
                                  "open_opponent": ["random"], "open_both": ["random"]}))
    r = run_train(tmp / "opp_both", "--open-book", str(book), "--open-p", "1.0", *common4, "--panel-open-p", "1.0", steps=1, check=False)
    assert r.returncode != 0 and "open_opponent" in (r.stderr + r.stdout)
    print("P2 train_rnad.py: --panel-open-p で行に open_p が付き、測り方を変えて再開した時点も測る")


if __name__ == "__main__":
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as d:
        check_play_vs(pathlib.Path(d))
        check_train(pathlib.Path(d))
    print("ok")
