"""入力の A/B の段取り（rnad/input_ab.py）の検査。tsuitate_rl は使わない（学習は回さない）。

    .venv/bin/python rnad/tests/test_input_ab.py

- ARMS: 版の名前の引数が、train_rnad.py の入力の組のフラグと 1 対 1（v5 は v4 に 4組、v4 は v3 に 1組を足したもの）
- parse_arms: 版の名前と `--arm 名前=引数`、重複・知らない名前・出力のディレクトリ名と紛れる名前は弾く
- plan_queue: seed を外側に回す。`--only` はその組だけを同じ順で残し、無い組は弾く
- pending: ログが `--steps` に届いていない run の数（ログが無い run も数える）。再開で同じ step が2回あれば後の行
- status: run ごとの表（ログが無い run も行を出す）
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
import input_ab  # noqa: E402
from widen_planes import EXTRA_FLAGS  # noqa: E402


def ns(**kw):
    base = dict(arms="v4,v5", arm=[], seeds="1,2,3", only="", steps=100, root="")
    base.update(kw)
    return SimpleNamespace(**base)


def must_fail(fn, *args):
    try:
        fn(*args)
    except SystemExit as e:
        return str(e)
    raise AssertionError("弾かれなかった")


def check_arms():
    flags = {a: {x for x in v.split() if x.startswith("--") and x != "--planes"} for a, v in input_ab.ARMS.items()}
    known = {"--" + n.replace("_", "-") for n in EXTRA_FLAGS}
    for a, f in flags.items():
        assert f <= known, (a, f - known)
    assert flags["v1"] == flags["v2"] == set()
    assert flags["v3"] < flags["v4"] < flags["v5"]
    assert flags["v4"] - flags["v3"] == {"--king-foul-planes"}
    assert len(flags["v5"] - flags["v4"]) == 4
    # v5 は今ある組の全部（組を足したら v6 の行を足すか、このテストを直す）
    assert flags["v5"] == known, known - flags["v5"]


def check_parse():
    arms = input_ab.parse_arms(ns(arms="v4,v5", arm=["v6k=--planes 91 --x=1"]))
    assert [n for n, _ in arms] == ["v4", "v5", "v6k"]
    assert arms[2][1] == "--planes 91 --x=1"  # 最初の = で分ける
    assert "v9" in must_fail(input_ab.parse_arms, ns(arms="v9"))
    assert "重複" in must_fail(input_ab.parse_arms, ns(arms="v4", arm=["v4=--planes 86"]))
    assert "腕が無い" in must_fail(input_ab.parse_arms, ns(arms=""))
    for bad in ("a_s1=--planes 86", "a:b=--planes 86", "a,b=--planes 86", "=--planes 86", "noequals"):
        must_fail(input_ab.parse_arms, ns(arms="", arm=[bad]))


def check_queue():
    args = ns()
    arms = input_ab.parse_arms(args)
    q = input_ab.plan_queue(args, arms)
    assert [(n, s) for n, _, s in q] == [("v4", 1), ("v5", 1), ("v4", 2), ("v5", 2), ("v4", 3), ("v5", 3)]
    q = input_ab.plan_queue(ns(only="v5:3,v4:1"), arms)
    assert [(n, s) for n, _, s in q] == [("v4", 1), ("v5", 3)]  # 順は元のまま
    assert "無い" in must_fail(input_ab.plan_queue, ns(only="v4:9"), arms)
    assert "無い" in must_fail(input_ab.plan_queue, ns(only="v3:1"), arms)
    must_fail(input_ab.plan_queue, ns(only="v4"), arms)


def write_log(run, steps):
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "log.jsonl", "w") as f:
        for s, plies in steps:
            f.write(json.dumps({"step": s, "update_sec": 1.0, "collect_sec": 2.0, "plies": plies,
                                "fouls_per_side": 7.0, "entropy_mean": 3.0}) + "\n")
        f.write('{"step": 999, "update_')  # 書きかけの行


def check_pending_status():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        write_log(root / "v4_s1", [(s, 80) for s in range(1, 101)])
        # 再開で 41 から書き直した run（後の行を使う）
        write_log(root / "v5_s1", [(s, 80) for s in range(1, 51)] + [(s, 120) for s in range(41, 61)])
        (root / "v4_s1" / "current_100.bin").write_bytes(b"x")
        (root / "v4_s1" / "current_0.bin").write_bytes(b"x")
        assert input_ab.last_step(root / "v4_s1") == 100
        assert input_ab.last_step(root / "v5_s1") == 60
        assert input_ab.read_log(root / "v5_s1")[44]["plies"] == 120
        assert input_ab.exported_steps(root / "v4_s1") == [0, 100]

        def pending(**kw):
            buf = io.StringIO()
            with redirect_stdout(buf):
                input_ab.cmd_pending(ns(root=str(root), **kw))
            return int(buf.getvalue())

        assert pending(seeds="1") == 1            # v5_s1 が届いていない
        assert pending(seeds="1,2") == 3          # seed 2 は 2本ともまだ無い
        assert pending(seeds="1,2", only="v4:1") == 0
        assert pending(seeds="1", steps=60) == 0
        buf = io.StringIO()
        with redirect_stdout(buf):
            input_ab.cmd_status(SimpleNamespace(root=str(root), window=10))
        table = buf.getvalue().splitlines()
        assert len(table) == 4 and table[2].startswith("| v4_s1 | 100 | 100 | 3.0 | 80 |"), table
        assert table[3].startswith("| v5_s1 | 60 | – | 3.0 | 120 |"), table
        assert input_ab.common_step(root) is None
        (root / "v5_s1" / "current_0.bin").write_bytes(b"x")
        assert input_ab.common_step(root) == 0


if __name__ == "__main__":
    check_arms()
    check_parse()
    check_queue()
    check_pending_status()
    print("ok")
