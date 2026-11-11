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
- リーグの局の強制（`--league-open-p`。定跡に従う相手を搾取する学習）: 自己対局モードの env でも
  `opening_index_learner` で相手の側だけが強制される（偶数番目の局は先手が学習側）。相手のネットを Python 側で
  指させる collect で、学習側の行は全部自分の手・相手の側だけに強制の記録が付く。train_rnad.py は相手を Rust で
  指させる形・ネットで指させる形のどちらでも相手の側だけを強制し、報酬の整形と一緒に使える（学習側の強制は止める）
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
    for bad, msg in [("{}", "ラインが無い"), ('{"lines": [{"moves": [1]}]}', "moves"),
                     ('{"lines": [{"moves": ["7g7f"], "weight": -1}]}', "weight"), ("x", "JSON")]:
        (tmp / "bad.json").write_text(bad)
        try:
            openings.load_book(tmp / "bad.json")
        except ValueError as e:
            assert msg in str(e), (bad, str(e))
            continue
        raise AssertionError(f"通ってしまった: {bad}")
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
        assert all(e <= t for e, t in zip(f["fouls_early"], f["fouls"])), f
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
    assert 0 < s["forced_frac"] < 1 and s["forced_moves"] >= 1 and "fouls_early_per_side" in s, s
    print(f"C1 collect: 行は隙間なし・記録 {len(fin)} 局・報酬の和 = 結果・学習の行 + 強制した手 = 受理手 + 反則"
          f"（強制した側 {s['forced_frac']:.0%}、平均 {s['forced_moves']:.1f} 手）")


def check_league_env(tmp):
    """自己対局モードの env で、並びの偶奇で決めた学習側の相手だけを強制する（`--league-device gpu` の局）"""
    book = openings.load_book(write_book(tmp))
    kw = openings.env_kwargs(book, 1.0, sides="opponent")
    env = tsuitate_rl.VecEnv(8, seed=4, auto_reset=False, planes=91, opening_index_learner=True, **kw)
    for rnd in range(2):  # reset をはさんでも（局番号が進んでも）並びの偶奇で決まる
        env.reset()
        _, _, pl = env.observe()
        assert pl.tolist() == [0, 1] * 4, "最初の手番は学習側（奇数番目は相手の先手が env の中で指した後）"
        play_random(env, 7 + rnd)
        fin = sorted(env.pop_finished(), key=lambda f: f["game_no"])
        for i, f in enumerate(fin):
            lc = i % 2
            assert f["forced_line"][lc] is None and f["forced_moves"][lc] == 0, (i, f["forced_line"])
            assert f["forced_line"][1 - lc] is not None and f["forced_moves"][1 - lc] >= 1, (i, f["forced_moves"])
    # 学習側だけを強制する指定も同じ並びで読む
    env = tsuitate_rl.VecEnv(4, seed=4, auto_reset=False, planes=91, opening_index_learner=True,
                             **openings.env_kwargs(book, 1.0, sides="learner"))
    env.reset()
    play_random(env, 1)
    for i, f in enumerate(sorted(env.pop_finished(), key=lambda f: f["game_no"])):
        assert f["forced_line"][i % 2] is not None and f["forced_line"][1 - i % 2] is None
    for bad in (dict(auto_reset=True), dict(opponents=["heuristic"]), dict(opening_sides="both")):
        try:
            tsuitate_rl.VecEnv(4, seed=0, planes=91, opening_index_learner=True, **{**dict(auto_reset=False), **kw, **bad})
        except ValueError:
            continue
        raise AssertionError(f"opening_index_learner が通ってしまった: {bad}")
    try:
        tsuitate_rl.VecEnv(4, seed=0, planes=91, auto_reset=False, **kw)
    except ValueError:
        pass
    else:
        raise AssertionError("自己対局モードで opening_sides=opponent が（opening_index_learner なしで）通ってしまった")

    # 相手のネットを Python 側で指させる collect: 学習側の行だけが残り、強制は相手の側だけ
    torch.manual_seed(0)
    net, opp = PolicyValueNet(16, 1, 91).eval(), PolicyValueNet(16, 1, 91).eval()
    env = tsuitate_rl.VecEnv(8, seed=5, auto_reset=False, planes=91, opening_index_learner=True, **kw)
    games = [("opp", i % 2) for i in range(8)]
    batch = collect(env, net, torch.device("cpu"), compact_legal=True, opponents=games, opp_nets={"opp": opp})
    fin = sorted(env.pop_finished(), key=lambda f: f["game_no"])
    valid, player = np.asarray(batch["valid"]), np.asarray(batch["player"])
    for i, f in enumerate(fin):
        lc = i % 2
        assert (player[valid[:, i], i] == lc).all(), "記録する行は学習側の手番だけ"
        assert f["forced_line"][lc] is None and f["forced_line"][1 - lc] is not None
        assert f["forced_moves"][1 - lc] >= 1 and int(valid[:, i].sum()) >= f["fouls"][lc]
        r = np.asarray(batch["reward"])[:, i, lc].sum()
        want = 0.0 if f["winner"] is None else (1.0 if f["winner"] == lc else -1.0)
        assert abs(r - want) < 1e-6, (i, r, want)
    print("L1 リーグの局の強制: 自己対局モードの env で相手の側だけ（並びの偶奇・reset をはさんでも同じ）・"
          "相手をネットで指させる collect は学習側の行だけ・報酬の和 = 学習側の結果・使えない指定を止める")


def check_league_train(tmp):
    book = write_book(tmp)
    base = ["--seed", "5", "--open-book", str(book), "--league-open-p", "1.0", "--league-frac", "1",
            "--compact-legal"]
    for name, extra in [
        ("rust", ["--league", "heuristic"]),
        ("gpu", ["--algo", "mmd", "--league", "rl_v15", "--league-device", "gpu", "--mmd-magnet", "uniform",
                 "--shape", "d_capture_value=0.02,opp_fouls=0.05"]),
    ]:
        run_train(tmp / name, *base, *extra, steps=2)
        log = [json.loads(x) for x in open(tmp / name / "log.jsonl") if x.strip()]
        for x in log:
            lo = x["league_opening"]
            assert lo["forced_frac"] == 0.5 and lo["both_forced_frac"] == 0.0, (name, lo)
            assert sum(sum(v) for v in lo["by_line"].values()) == 6, (name, lo["by_line"])
            assert "opening" not in x, "自己対局の局は無い（--open-p は 0）"
        cfg = json.loads((tmp / name / "config.json").read_text())["opening"]
        assert cfg["p"] == 0.0 and cfg["league_p"] == 1.0 and cfg["league_sides"] == "opponent"
        # 再開で付け忘れると止まる
        r = run_train(tmp / name, *[a for a in base if a not in ("--league-open-p", "1.0")][:2], "--league-frac", "1",
                      "--compact-legal", *extra, steps=3, check=False)
        assert r.returncode != 0, name
    # --open-p だけなら、リーグの局は強制しない（自己対局の局だけ。tsuitate-nn #24）
    run_train(tmp / "selfonly", "--seed", "5", "--open-book", str(book), "--open-p", "1.0", "--league", "heuristic",
              "--league-frac", "0.5", steps=1)
    x = json.loads(open(tmp / "selfonly" / "log.jsonl").readline())
    assert x["opening"]["forced_frac"] == 1.0 and "league_opening" not in x
    for extra, msg in [
        (["--open-book", str(book), "--league-open-p", "1.0"], "--league"),
        (["--league-open-p", "1.0", "--league", "heuristic", "--league-frac", "1"], "--open-book"),
        (["--open-book", str(book), "--league-open-p", "1.0", "--league-open-sides", "learner", "--league", "heuristic",
          "--league-frac", "1", "--shape", "d_capture_value=0.02"], "--league-open-sides learner"),
    ]:
        r = run_train(tmp / "rej2", *extra, steps=1, check=False)
        assert r.returncode != 0 and msg in (r.stderr + r.stdout), (extra, r.stderr[-500:])
    print("L2 train_rnad.py --league-open-p: 相手を Rust で指させる形・ネットで指させる形（整形つき）とも相手の側だけを"
          "強制・ラインごとの勝敗がログに出る・設定が残る・使えない組み合わせを止める")


def check_adapt(tmp):
    """ラインの重みの付け替え（`--league-open-adapt`）"""
    ad = openings.LineAdapter([1.0, 1.0, 2.0, 2.0], gamma=1.0, window=2, min_games=4)
    assert ad.weights() == [1.0, 1.0, 2.0, 2.0], "記録が無ければ元の重み"
    ad.push({"0": [4, 0, 0], "1": [1, 3, 0], "2": [1, 0, 0]})  # ライン 2 は局数が足りない = 勝率 0 として扱う
    w = np.array(ad.weights())
    want = np.array([1.0 * 0.05, 1.0 * 0.75, 2.0, 2.0])
    assert np.allclose(w, want / want.sum() * 6.0), w
    ad.push({"0": [0, 4, 0]})
    ad.push({"3": [2, 0, 2]})  # 窓は 2更新: 最初の記録は落ちる。引き分けは 0.5 勝
    w = np.array(ad.weights())
    want = np.array([1.0, 1.0, 2.0, 2.0 * 0.25])
    assert np.allclose(w, want / want.sum() * 6.0), w
    assert openings.LineAdapter([1.0, 3.0], gamma=0.0).weights() == [1.0, 3.0]

    book = write_book(tmp)
    args = ["--seed", "6", "--open-book", str(book), "--league-open-p", "1.0", "--league-frac", "1", "--compact-legal",
            "--league", "heuristic", "--league-open-adapt", "2.0", "--league-open-window", "2"]
    run_train(tmp / "ad_a", *args, steps=4)
    run_train(tmp / "ad_b", *args, steps=2)
    run_train(tmp / "ad_b", *args, steps=4)
    la, lb = read_log(tmp / "ad_a" / "log.jsonl"), read_log(tmp / "ad_b" / "log.jsonl")
    assert la == lb, "重みの付け替えつきの再開でログが変わる"
    ws = [x["league_opening"]["line_weights"] for x in la]
    assert len(ws[0]) == 4 and abs(sum(ws[0]) - sum(openings.load_book(book).weights)) < 1e-3
    r = run_train(tmp / "rej3", "--open-book", str(book), "--open-p", "0.5", "--league-open-adapt", "1.0", steps=1,
                  check=False)
    assert r.returncode != 0 and "--league-open-adapt" in (r.stderr + r.stdout)
    print("A1 ラインの重みの付け替え: 勝てているラインを減らす・局数の足りないラインは減らさない・窓・引き分け・"
          "train_rnad.py の再開でログがビット単位で一致")


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
    # 各更新の batch は「その更新の step」の乱数列で集める（ログの step は更新の後の番号）
    assert [x["opening"]["seed"] for x in la] == [3 * 1_000_003 + 29 + x["step"] - 1 for x in la]
    cfg = json.loads((tmp / "straight" / "config.json").read_text())
    assert cfg["opening"]["book_sha256"] == openings.load_book(book).sha256

    # 再開で --open-* を付け忘れると止まる。--override-config なら通して記録に残す
    r = run_train(tmp / "resumed", "--seed", "3", steps=5, check=False)
    assert r.returncode != 0 and "opening" in r.stderr, r.stderr[-500:]
    run_train(tmp / "resumed", "--seed", "3", "--override-config", steps=5)
    rec = json.loads((tmp / "resumed" / "config_resume_4.json").read_text())
    assert rec["overridden"]["opening"][1] is None and "opening" not in json.loads(
        (tmp / "resumed" / "log.jsonl").read_text().splitlines()[-1])

    # --pipeline でも、batch は使う step の乱数列で集める（最初の2つが同じ乱数列にならない）。再開しても同じ
    pipe = [*common, "--pipeline", "--compact-legal", "--fast-collect", "py"]
    run_train(tmp / "pipe", *pipe, steps=4)
    run_train(tmp / "pipe_r", *pipe, steps=2)
    run_train(tmp / "pipe_r", *pipe, steps=4)
    for d in ("pipe", "pipe_r"):
        lg = read_log(tmp / d / "log.jsonl")
        assert [x["opening"]["seed"] for x in lg] == [3 * 1_000_003 + 29 + x["step"] - 1 for x in lg], d

    # 強制しない run の checkpoint には集め方の設定を足さない
    run_train(tmp / "off", "--seed", "3", steps=2)
    assert "collect_cfg" not in torch.load(tmp / "off" / "checkpoint.pt", map_location="cpu")

    for extra, msg in [
        (["--open-p", "0.5"], "--open-book"),
        (["--open-book", str(book)], "--open-p"),
        (["--open-book", str(book), "--open-p", "0.5", "--speed", "1.0"], "--speed"),
        (["--open-book", str(book), "--open-p", "0.5", "--nyugyoku", "1.0"], "--nyugyoku"),
        (["--open-book", str(book), "--open-p", "0.5", "--open-len", "0:3"], "手数"),
        (["--open-book", str(book), "--open-p", "0.5", "--c-ply", "40:0.5"], "--c-ply"),
        (["--open-len", "1:3"], "--open-book"),
        (["--open-book", str(tmp / "missing.json"), "--open-p", "0.5"], "序盤の強制"),
        (["--open-book", str(tmp / "bad.json"), "--open-p", "0.5"], "JSON"),
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
        check_league_env(tmp)
        check_league_train(tmp)
        check_adapt(tmp)
    print("ok")
