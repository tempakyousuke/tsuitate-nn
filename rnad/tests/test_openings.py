"""序盤の強制（`rnad/openings.py`、`train_rnad.py --open-book`、tsuitate_rl.VecEnv の `openings`）の検査（CPU）。
新しい tsuitate_rl が要る（tsuitate-bot の rl-env を maturin develop で作り直す）。

    .venv/bin/python rnad/tests/test_openings.py

- ブックの読み込み: 系統ごとの重み（系統の和が同じ）・ラインごとの重み・系統の切り出し・手数の範囲の解釈
- env: p = 0 はブックを渡さないのとビット単位で同じ局になる。p = 1 で両者の最初の手が定跡どおり・強制した手は
  観測の行に現れない。`set_opening_seed` の後の `reset` で予定が決まる（同じ乱数列なら同じ、局番号の起点によらない）。
  終局の記録に forced_* / contact_ply / fouls_precontact が付く。不正な指定は ValueError
- collect: 強制しても batch の不変量（局ごとに行が隙間なく並ぶ・終局の記録の数 = 局数・報酬の和 = 結果）が保たれる
- train_rnad.py: 強制ありで 4更新を続けた run と、2更新で止めて同じコマンドで 4 まで続けた run がログ・ネットで
  ビット単位で一致。checkpoint に集め方の設定が残り、再開で --open-* を付け忘れると止まる（--override-config で通す）。
  強制しない run の checkpoint には何も足さない。使えない組み合わせは起動時に止まる
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import openings  # noqa: E402
import tsuitate_rl  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import collect  # noqa: E402

TRAIN = HERE.parent / "train_rnad.py"

BOOK = {"lines": [
    {"name": "速攻", "category": "居飛車", "moves": ["2g2f", "2f2e", "2e2d", "2d2c+"]},
    {"name": "四間", "category": "振り飛車", "moves": ["7g7f", "6g6f", "2h6h", "5i4h", "4h3h"]},
    {"name": "中飛車", "category": "振り飛車", "moves": ["5g5f", "2h5h", "5f5e", "5i4h"]},
    {"name": "三間", "category": "振り飛車", "moves": ["7g7f", "2h7h", "5i4h", "4h3h"]},
]}


def write_book(tmp):
    path = tmp / "book.json"
    path.write_text(json.dumps(BOOK, ensure_ascii=False))
    return path


def check_book(tmp):
    book = openings.load_book(write_book(tmp))
    by_cat = {}
    for w, c in zip(book.weights, book.categories):
        by_cat[c] = by_cat.get(c, 0.0) + w
    assert by_cat == {"居飛車": 1.0, "振り飛車": 1.0}, by_cat
    assert openings.load_book(write_book(tmp), "line").weights == [1.0] * 4
    sub = book.subset("振り飛車")
    assert len(sub.lines) == 3 and sub.weights == [1.0] * 3 and sub.sha256 == book.sha256
    assert openings.parse_len("") is None and openings.parse_len("2:5") == (2, 5) and openings.parse_len("3") == (3, 3)
    for bad in ("0:2", "4:2"):
        try:
            openings.parse_len(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert openings.env_kwargs(book, 0.0) == {}
    real = HERE.parents[2] / "tsuitate-bot" / "joseki.json"
    if real.exists():
        jb = openings.load_book(real)
        tsuitate_rl.VecEnv(2, seed=0, auto_reset=False, **openings.env_kwargs(jb, 0.5))
        print(f"  （{real} の {len(jb.lines)} ラインも env が受け付ける）")
    print("B1 ブック: 系統ごとの重みの和がそろう・ラインごと・系統の切り出し・手数の範囲")


def play_random(env, seed, steps=10_000):
    """全局を一様な手で最後まで（観測の列を返す）"""
    rng = np.random.default_rng(seed)
    obs_seq = []
    for _ in range(steps):
        alive = env.alive()
        if not alive.any():
            break
        obs, mask, pl = env.observe()
        obs_seq.append((obs.copy(), mask.copy(), pl.copy()))
        acts = np.full(env.num_envs, -1, dtype=np.int64)
        for g in np.flatnonzero(alive):
            legal = np.flatnonzero(mask[g])
            acts[g] = legal[rng.integers(len(legal))]
        env.step(acts)
    return obs_seq


def same_seq(a, b):
    return len(a) == len(b) and all(all(np.array_equal(x, y) for x, y in zip(p, q)) for p, q in zip(a, b))


def check_env(tmp):
    book = openings.load_book(write_book(tmp))
    n = 16
    plain = tsuitate_rl.VecEnv(n, seed=3, auto_reset=False, planes=91)
    zero = tsuitate_rl.VecEnv(n, seed=3, auto_reset=False, planes=91, openings=book.lines, opening_prob=0.0)
    assert same_seq(play_random(plain, 1), play_random(zero, 1)), "p = 0 はブックを渡さないのと同じ"

    # p = 1・両者: 先手の最初の観測の時点で、先手も後手も1手ずつ以上を定跡で指している
    env = tsuitate_rl.VecEnv(n, seed=3, auto_reset=False, planes=91, **openings.env_kwargs(book, 1.0, sides="both"))
    fm = env.forced_moves()
    assert fm.shape == (n, 2) and (fm >= 1).all(), fm
    # 強制しない env の初期局面の観測とは違う（自分の手が履歴に載っている）
    o0, _, _ = tsuitate_rl.VecEnv(1, seed=3, auto_reset=False, planes=91).observe()
    o1, _, _ = env.observe()
    assert not np.array_equal(o0[0], o1[0])
    play_random(env, 2)
    fin = sorted(env.pop_finished(), key=lambda f: f["game_no"])
    assert len(fin) == n
    for f in fin:
        assert all(x is not None for x in f["forced_line"]), f
        assert all(s in ("contact", "length", "foul", "game_end") for s in f["forced_stop"]), f["forced_stop"]
        assert len(f["fouls_precontact"]) == 2 and (f["contact_ply"] is None or f["contact_ply"] >= 1)
    # 片側だけ（後手）
    env = tsuitate_rl.VecEnv(n, seed=3, auto_reset=False, planes=91, **openings.env_kwargs(book, 1.0, sides="gote"))
    fm = env.forced_moves()
    assert (fm[:, 0] == 0).all() and (fm[:, 1] == 0).all(), "後手の最初の手番はまだ来ていない（先手が指す前）"
    o, m, pl = env.observe()
    assert (pl == 0).all()
    acts = np.array([np.flatnonzero(m[g])[0] for g in range(n)])
    env.step(acts)
    assert (env.forced_moves()[:, 1] >= 1).all(), "先手が指すと後手は定跡で指して、また先手の手番になる"
    _, _, pl = env.observe()
    assert (pl == 0).all()

    # 予定は set_opening_seed の後の reset で決まり、局番号の起点によらない
    def lines_after(env, seed):
        env.set_opening_seed(seed)
        env.reset()
        play_random(env, 5)
        return [f["forced_line"] for f in sorted(env.pop_finished(), key=lambda f: f["game_no"])]
    kw = openings.env_kwargs(book, 0.5)
    a = tsuitate_rl.VecEnv(n, seed=0, auto_reset=False, planes=91, **kw)
    b = tsuitate_rl.VecEnv(n, seed=0, auto_reset=False, planes=91, **kw)
    a.pop_finished()
    first = lines_after(a, 77)
    lines_after(b, 5)  # b は局番号が先へ進んでいる
    assert lines_after(b, 77) == first, "同じ乱数列なら局番号の起点によらず同じ予定"
    assert lines_after(a, 78) != first
    flat = [x for pair in first for x in pair]
    assert 0 < sum(x is not None for x in flat) < len(flat), "p = 0.5 で強制する側としない側がある"

    for kwargs, msg in [
        (dict(opening_prob=0.5), "openings"),
        (dict(openings=[["7g7f+"]], opening_prob=0.5), "成れない"),
        (dict(openings=[["P*5e"]], opening_prob=0.5), "打ち"),
        (dict(openings=book.lines, opening_prob=1.5), "opening_prob"),
        (dict(openings=book.lines, opening_prob=0.5, opening_sides="learner"), "評価モード"),
        (dict(openings=book.lines, opening_prob=0.5, opening_sides="x"), "opening_sides"),
        (dict(openings=book.lines, opening_prob=0.5, opening_len=(3, 2)), "opening_len"),
        (dict(openings=book.lines, opening_prob=0.5, guard_bonus=0.5), "guard_bonus"),
        (dict(openings=book.lines, opening_prob=0.5, opening_weights=[1.0]), "opening_weights"),
    ]:
        try:
            tsuitate_rl.VecEnv(2, seed=0, auto_reset=False, **kwargs)
        except ValueError as e:
            assert msg in str(e), (kwargs, str(e))
            continue
        raise AssertionError(f"通ってしまった: {kwargs}")
    # 評価モードの学習側・相手側の指定は通る
    tsuitate_rl.VecEnv(2, opponent="heuristic", seed=0, auto_reset=False,
                       **openings.env_kwargs(book, 1.0, sides="opponent"))
    print("E1 env: p = 0 は従来と同じ局・p = 1 で両者が定跡で指す・片側だけ・予定は乱数列で決まる・不正な指定を止める")


def check_collect(tmp):
    book = openings.load_book(write_book(tmp))
    torch.manual_seed(0)
    net = PolicyValueNet(16, 1, 91).eval()
    env = tsuitate_rl.VecEnv(12, seed=1, auto_reset=False, planes=91, **openings.env_kwargs(book, 0.5))
    env.set_opening_seed(9)
    batch = collect(env, net, torch.device("cpu"), compact_legal=True)
    fin = env.pop_finished()
    valid = np.asarray(batch["valid"])
    T, B = valid.shape
    assert B == 12 and len(fin) == 12
    lengths = valid.sum(0)
    assert (valid == (np.arange(T)[:, None] < lengths[None, :])).all(), "局ごとに行が隙間なく並ぶ"
    assert np.allclose(np.asarray(batch["reward"]).sum(0), np.asarray(batch["outcome"])), "報酬の和 = 結果"
    # 試行 = 受理手 + 反則。強制した試行は受理された手と、反則でやめた側の1手
    forced = sum(sum(f["forced_moves"]) + f["forced_stop"].count("foul") for f in fin)
    fouls = sum(sum(f["fouls"]) for f in fin)
    plies = sum(f["plies"] for f in fin)
    assert int(lengths.sum()) + forced == plies + fouls, (int(lengths.sum()), forced, plies, fouls)
    s = openings.summarize(fin)
    assert 0 < s["forced_frac"] < 1 and s["forced_moves"] >= 1, s
    print(f"C1 collect: 行は隙間なし・記録 {len(fin)} 局・報酬の和 = 結果・学習の行 + 強制した手 = 受理手 + 反則"
          f"（強制した側 {s['forced_frac']:.0%}、平均 {s['forced_moves']:.1f} 手）")


def run_train(out, *extra, steps, check=True):
    cmd = [sys.executable, str(TRAIN), "--anchor", "none", "--init", "none", "--batch", "6", "--steps", str(steps),
           "--channels", "16", "--blocks", "1", "--planes", "91", "--eval-every", "0", "--export-every", "2",
           "--checkpoint-every", "2", "--out", str(out), "--threads", "2", "--device", "cpu", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"train_rnad.py が落ちた: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r


def read_log(path):
    return [{k: v for k, v in json.loads(line).items() if not k.endswith("_sec")}
            for line in open(path) if line.strip()]


def check_train(tmp):
    book = write_book(tmp)
    common = ["--seed", "3", "--open-book", str(book), "--open-p", "0.5", "--open-len", "1:4"]
    run_train(tmp / "straight", *common, steps=4)
    run_train(tmp / "resumed", *common, steps=2)
    run_train(tmp / "resumed", *common, steps=4)
    la, lb = read_log(tmp / "straight" / "log.jsonl"), read_log(tmp / "resumed" / "log.jsonl")
    assert la == lb, "強制ありの再開でログが変わる"
    assert all("opening" in x and 0 < x["opening"]["forced_frac"] <= 1 for x in la), la[0].get("opening")
    ca = torch.load(tmp / "straight" / "checkpoint.pt", map_location="cpu")
    cb = torch.load(tmp / "resumed" / "checkpoint.pt", map_location="cpu")
    for k in ca["net"]:
        assert torch.equal(ca["net"][k], cb["net"][k]), k
    assert ca["collect_cfg"]["opening"]["p"] == 0.5 and ca["collect_cfg"]["opening"]["len"] == [1, 4]
    cfg = json.loads((tmp / "straight" / "config.json").read_text())
    assert cfg["opening"]["book_sha256"] == openings.load_book(book).sha256

    # 再開で --open-* を付け忘れると止まる。--override-config なら通して記録に残す
    r = run_train(tmp / "resumed", "--seed", "3", steps=5, check=False)
    assert r.returncode != 0 and "opening" in r.stderr, r.stderr[-500:]
    run_train(tmp / "resumed", "--seed", "3", "--override-config", steps=5)
    rec = json.loads((tmp / "resumed" / "config_resume_4.json").read_text())
    assert rec["overridden"]["opening"][1] is None and "opening" not in json.loads(
        (tmp / "resumed" / "log.jsonl").read_text().splitlines()[-1])

    # 強制しない run の checkpoint には集め方の設定を足さない
    run_train(tmp / "off", "--seed", "3", steps=2)
    assert "collect_cfg" not in torch.load(tmp / "off" / "checkpoint.pt", map_location="cpu")

    for extra, msg in [
        (["--open-p", "0.5"], "--open-book"),
        (["--open-book", str(book)], "--open-p"),
        (["--open-book", str(book), "--open-p", "0.5", "--league", "heuristic"], "--league"),
        (["--open-book", str(book), "--open-p", "0.5", "--speed", "1.0"], "--speed"),
        (["--open-book", str(book), "--open-p", "0.5", "--nyugyoku", "1.0"], "--nyugyoku"),
        (["--open-book", str(book), "--open-p", "0.5", "--open-len", "0:3"], "手数"),
        (["--open-book", str(tmp / "missing.json"), "--open-p", "0.5"], "序盤の強制"),
    ]:
        r = run_train(tmp / "rej", *extra, steps=1, check=False)
        assert r.returncode != 0 and msg in (r.stderr + r.stdout), (extra, r.stderr[-500:])
    print("T1 train_rnad.py: 強制ありの再開がログ・ネットでビット単位で一致・checkpoint と config に設定が残る・"
          "付け忘れの再開は止まる（--override-config で通る）・強制しない run の checkpoint は従来どおり・"
          "使えない組み合わせを止める")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        check_book(tmp)
        check_env(tmp)
        check_collect(tmp)
        check_train(tmp)
    print("ok")
