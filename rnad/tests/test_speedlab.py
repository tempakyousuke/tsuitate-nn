"""speedlab の鍵つきの乱数と Elo の当てはめの検査（`check_decode` だけ tsuitate_rl で数局指す）。

    .venv/bin/python rnad/tests/test_speedlab.py

- keyed_uniform は (salt, 局, 手番, 行動) だけで決まり、行の並びや一緒に作る行に依らない。一様に近い
- keyed_actions は合法手だけで乱数を作っても、全行動で作ったときと同じ手を選ぶ（マスク外は選ばない）
- fit_fixed はパネルのレートから離れた値でも発散しない
- report の A/B 用の比べ方（MMD の A/B。docs/rl-learning-speed.md 10節）: 対照の名前 `ab_rnad_s1` と `ab_mmd_s1`・`ab_mmd_s2` で
  対照の節が出る、`--pair-by wallclock` で秒が最も近い書き出しどうしが対になる、`--exclude-step` で起点が消える、
  `--control-any-seed` で seed 2 も対になる、`--vs-step` で同じ run の起点と対になる、`--groups` の小計、`--eval-file`
- 締まりを除いた比べ方: keyed_actions の温度（同じ乱数のまま softmax(logits/T) から引く。0 は greedy）、play_vs の温度が
  測られる側だけに掛かること、eval --decode の記録名、export-ck（checkpoint → 書き出し）、temp-match の値
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
import speedlab as S  # noqa: E402

DEV = torch.device("cpu")


def check_uniform():
    u = S.keyed_uniform(5, [0, 1, 2], [0, 0, 3], 300, DEV)
    v = S.keyed_uniform(5, [2], [3], 300, DEV)
    assert torch.equal(u[2], v[0])
    ids = torch.tensor([[7, 100, 299]])
    w = S.keyed_uniform(5, [2], [3], ids, DEV)
    assert torch.equal(w[0], u[2, [7, 100, 299]])
    x = S.keyed_uniform(1, np.arange(64), np.zeros(64, dtype=np.int64), 2000, DEV).flatten()
    assert 0 < float(x.min()) and float(x.max()) < 1
    assert abs(float(x.mean()) - 0.5) < 0.01 and abs(float(x.var()) - 1 / 12) < 0.002
    print("keyed_uniform: 鍵だけで決まり、一様に近い")


class FixedNet(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, x):
        return self.logits[: len(x)], torch.zeros(len(x))


def check_keyed_actions():
    torch.manual_seed(0)
    n, a = 40, 11259
    logits = torch.randn(n, a) * 3
    mask = np.random.default_rng(0).random((n, a)) < 0.01
    mask[:, 5] = True
    obs = np.zeros((n, 1, 9, 9), dtype=np.float32)
    games, decs = np.arange(n), np.arange(n) * 3
    got = S.keyed_actions(FixedNet(logits), obs, mask, games, decs, 11, DEV)
    # 全行動で乱数を作った場合
    u = S.keyed_uniform(11, games, decs, a, DEV)
    want = (logits.masked_fill(~torch.from_numpy(mask), float("-inf")) - torch.log(-torch.log(u)).float()).argmax(1)
    assert np.array_equal(got, want.numpy()), (got, want)
    assert mask[np.arange(n), got].all()
    print("keyed_actions: 合法手だけで乱数を作っても全行動の場合と同じ手")


def check_keyed_temp():
    """温度: 1 は従来と同じ手、0 は最大の手、T は同じ乱数のまま softmax(logits / T) から引く（多くの局で頻度を見る）"""
    torch.manual_seed(1)
    n, a = 40, 11259
    logits = torch.randn(n, a) * 3
    mask = np.random.default_rng(1).random((n, a)) < 0.01
    mask[:, 5] = True
    obs = np.zeros((n, 1, 9, 9), dtype=np.float32)
    games, decs = np.arange(n), np.arange(n) * 3
    net = FixedNet(logits)
    base = S.keyed_actions(net, obs, mask, games, decs, 11, DEV)
    assert np.array_equal(S.keyed_actions(net, obs, mask, games, decs, 11, DEV, temp=1.0), base)
    greedy = logits.masked_fill(~torch.from_numpy(mask), float("-inf")).argmax(1).numpy()
    assert np.array_equal(S.keyed_actions(net, obs, mask, games, decs, 11, DEV, temp=0.0), greedy)
    # 1つの局面を 20,000局の鍵で引くと、頻度が softmax(logits / T) に合う
    g = 20_000
    l1 = torch.tensor([[1.0, 0.5, 0.0, -0.5, -1.0] + [0.0] * (a - 5)])
    m1 = np.zeros((1, a), dtype=bool)
    m1[0, :5] = True
    one = FixedNet(l1.expand(g, a))
    for t in (0.5, 2.0):
        got = S.keyed_actions(one, np.zeros((g, 1, 9, 9), dtype=np.float32), np.repeat(m1, g, 0), np.arange(g),
                              np.zeros(g, dtype=np.int64), 3, DEV, temp=t)
        freq = np.bincount(got, minlength=5)[:5] / g
        want = torch.softmax(l1[0, :5] / t, -1).numpy()
        assert np.abs(freq - want).max() < 0.012, (t, freq, want)
    print("keyed_actions の温度: 1 は従来と同じ手、0 は最大の手、T は頻度が softmax(logits/T) に合う")


def check_decode():
    """play_vs の温度は測られる側だけに掛かる・eval --decode の記録名・export-ck・temp-match（tsuitate_rl で数局指す）"""
    from model import NUM_PLANES, PolicyValueNet
    from export_weights import tiny_net
    planes = NUM_PLANES
    torch.manual_seed(2)
    a_net = PolicyValueNet(8, 1, planes).eval()
    b = S.Member("random", planes)
    # 温度の配線: keyed_actions を包んで、どちらのネットにどの温度が渡ったかを記録する
    seen = []
    orig = S.keyed_actions

    def spy(net, *args, temp=1.0, **kw):
        seen.append((net is a_net, temp))
        return orig(net, *args, temp=temp, **kw)

    S.keyed_actions = spy
    try:
        o_greedy = S.play_vs(a_net, b, 6, 99, planes, per_game=True, a_temp=0.0)
    finally:
        S.keyed_actions = orig
    assert {t for is_a, t in seen if is_a} == {0.0} and {t for is_a, t in seen if not is_a} == {1.0}, set(seen)
    assert np.array_equal(S.play_vs(a_net, b, 6, 99, planes, per_game=True, a_temp=0.0), o_greedy)  # 決定的
    assert np.array_equal(S.play_vs(a_net, b, 6, 99, planes, per_game=True),
                          S.play_vs(a_net, b, 6, 99, planes, per_game=True, a_temp=1.0))
    assert S.parse_decode("sample") == (1.0, "") and S.parse_decode("greedy") == (0.0, "@greedy")
    assert S.parse_decode("temp:0.7") == (0.7, "@t0.7")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        # export-ck: checkpoint の net → current_<step>.bin、target → policy_<step>.bin（BatchNorm を畳んでも出力は同じ）
        net, tgt = tiny_net(3), tiny_net(4)
        if net.config["planes"] != planes:
            net, tgt = PolicyValueNet(8, 1, planes).eval(), PolicyValueNet(8, 1, planes).eval()
        torch.save({"net": net.state_dict(), "target": tgt.state_dict(), "config": dict(net.config), "step": 123},
                   tmp / "ck.pt")
        run = tmp / "run_s1"
        S.cmd_export_ck(argparse.Namespace(checkpoint=str(tmp / "ck.pt"), run=str(run), step=None, force=False))
        x = torch.rand(3, net.config["planes"], 9, 9)
        for name, src in (("current_123.bin", net), ("policy_123.bin", tgt)):
            back = S.read_weights(run / name).eval()
            with torch.no_grad():
                d = max(float((u - v).abs().max()) for u, v in zip(back(x), src.eval()(x)))
            assert d < 1e-4, (name, d)
        try:
            S.cmd_export_ck(argparse.Namespace(checkpoint=str(tmp / "ck.pt"), run=str(run), step=None, force=False))
        except SystemExit:
            pass
        else:
            raise AssertionError("書き出しを黙って上書きした")
        # policy_<step>.bin だけが既にあるフォルダ（--export-current なしの書き出し）でも、current を書く前に止まる
        half = tmp / "half_s1"
        half.mkdir()
        (half / "policy_123.bin").write_bytes(b"old")
        try:
            S.cmd_export_ck(argparse.Namespace(checkpoint=str(tmp / "ck.pt"), run=str(half), step=None, force=False))
        except SystemExit:
            pass
        else:
            raise AssertionError("書き出しを黙って上書きした")
        assert not (half / "current_123.bin").exists() and (half / "policy_123.bin").read_bytes() == b"old"
        # --step が checkpoint の step と違えば注意を出す（写した checkpoint.pt が書き直された後のこともある）
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            S.cmd_export_ck(argparse.Namespace(checkpoint=str(tmp / "ck.pt"), run=str(tmp / "lbl_s1"), step=100, force=False))
        assert "checkpoint の step は 123" in err.getvalue() and (tmp / "lbl_s1" / "current_100.bin").exists(), err.getvalue()
        # eval --decode greedy: which に @greedy を付けて記録し、sample の行とは別に持つ
        (tmp / "panel.json").write_text(json.dumps({"planes": planes, "members": ["random"], "ratings": {"random": 0.0}}))
        for dec in ("greedy", "sample"):
            S.cmd_eval(argparse.Namespace(runs=[str(run)], panel=str(tmp / "panel.json"), games=4, which=["current"],
                                          every=0, only=None, device="cpu", out_file="panel_eval.jsonl", steps="",
                                          decode=dec))
        rows = [json.loads(line) for line in open(run / "panel_eval.jsonl")]
        assert sorted(r["which"] for r in rows) == ["current", "current@greedy"], rows
        # temp-match: 温度 0 は π>0.999 が全部、温度を下げるとエントロピーが下がる。値はそれぞれの温度の自己対局の局面で
        # 測るので、乱数初期化のほぼ平らな方策（エントロピー ≈ log 合法手数）では温度の差が局面の違いに埋もれる。
        # ロジットを大きくした方策で測る
        class Peaked(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                logits, v = self.net(x)
                return logits * 8, v
        peaked = Peaked(a_net).eval()
        e1, h1, n1 = S.self_play_stats(peaked, 6, 5, planes, 1.0)
        e5, h5, _ = S.self_play_stats(peaked, 6, 5, planes, 0.5)
        e0, h0, _ = S.self_play_stats(peaked, 6, 5, planes, 0.0)
        # π>0.999 の割合は温度 1 と 0.5 ではどちらもほぼ 0 なので比べない（6局の揺れで逆転する）
        assert n1 > 0 and e1 > e5 > 0 and h0 == 1.0, (e1, e5, h1, h5, h0)
    print("締まりを除いた比べ方: play_vs の温度は測られる側だけ・greedy は決定的・eval --decode の記録名・"
          f"export-ck（出力の差 < 1e-4）・temp-match（エントロピー T1 {e1:.2f} > T0.5 {e5:.2f}、T0 は π>0.999 が 100%）")


def check_seed_salt():
    """keyed_uniform は salt と局番号を XOR で混ぜるので、局数より小さい桁だけが違う salt は同じ鍵の組を使う（学習中のパネルの
    記録と対にするため変えない）。seed を振るときは seed_salt を通し、違う seed で違う局になる（temp-match の --seed）"""
    games = torch.arange(16)
    zero = torch.zeros(16, dtype=torch.int64)
    raw = [sorted(S.keyed_uniform(s, games, zero, 4, "cpu")[:, 0].tolist()) for s in (1, 2)]
    assert raw[0] == raw[1]  # 同じ組（並びが違うだけ）。この性質を変えると記録と対にならない
    spread = [sorted(S.keyed_uniform(S.seed_salt(s), games, zero, 4, "cpu")[:, 0].tolist()) for s in (1, 2)]
    assert spread[0] != spread[1]
    # int64 に収まる（大きい seed でも落ちない）
    S.keyed_uniform(S.seed_salt(202610261200), games, zero, 4, "cpu")
    # パネルの乱数の値そのものを固定する（混ぜ方・定数・名前の salt が変わると学習中のパネルの記録と対にならない）
    assert S.EVAL_SEED == 777_000 and S.stable_seed("rl_deduce_v4") == 56079
    got = S.keyed_uniform(S.EVAL_SEED + S.stable_seed("rl_v30"), [0, 1, 1023], [0, 3, 7], 4, "cpu").tolist()
    want = [[0.5534499549817595, 0.5610912487390687, 0.5745616191889509, 0.657698376267789],
            [0.5667035880278402, 0.4564721114720928, 0.1657893042420357, 0.7460447611434959],
            [0.032622780917235084, 0.7279467137855173, 0.32516869738446635, 0.14315671828934756]]
    assert got == want, got
    from model import NUM_PLANES, PolicyValueNet
    torch.manual_seed(5)
    net = PolicyValueNet(8, 1, NUM_PLANES).eval()
    # self_play_stats が seed_salt を通した salt を keyed_actions に渡す（局の並べ替えだけでは値がほぼ同じになり、値の比較では見分けられない）
    salts = []
    orig = S.keyed_actions

    def spy(net_, obs, mask, games_, decs, salt, *args, **kw):
        salts.append(salt)
        return orig(net_, obs, mask, games_, decs, salt, *args, **kw)

    S.keyed_actions = spy
    try:
        S.self_play_stats(net, 4, 3, NUM_PLANES, 1.0)
    finally:
        S.keyed_actions = orig
    assert salts and set(salts) == {S.seed_salt(3)}, set(salts)
    # π>0.999 の割合: 入力によらない固定のロジットで、記録した合法手から直接数えた値と一致する（温度 1 と 0.5）
    base = torch.randn(139 * 81, generator=torch.Generator().manual_seed(9)) * 10

    class Fixed(torch.nn.Module):
        def forward(self, x):
            return base.expand(x.shape[0], -1).clone(), torch.zeros(x.shape[0])
    fixed = Fixed().eval()
    for temp in (1.0, 0.5):
        masks = []

        def spy2(net_, obs, mask, *args, **kw):
            masks.append(torch.from_numpy(mask))
            return orig(net_, obs, mask, *args, **kw)
        S.keyed_actions = spy2
        try:
            e, h, n = S.self_play_stats(fixed, 6, 4, NUM_PLANES, temp)
        finally:
            S.keyed_actions = orig
        m = torch.cat(masks)
        p_ = torch.softmax((base / temp).expand(m.shape[0], -1).masked_fill(~m, float("-inf")), -1)
        want_h = int((p_.max(-1).values > 0.999).sum()) / m.shape[0]
        assert n == m.shape[0] and abs(h - want_h) < 1e-12 and 0 < want_h < 1, (temp, h, want_h, n)
    print("seed_salt: 生の salt 1 と 2 は同じ鍵の組（変えない性質。パネルの乱数の値も固定）、seed_salt を通すと違う局になり、"
          "self_play_stats はそれを使う。π>0.999 の割合は合法手から直接数えた値と一致")


def check_fit():
    ratings = {"a": 1431.0, "b": 1490.0, "c": 1539.0}
    r, se = S.fit_fixed([("a", 50, 14, 0), ("b", 40, 24, 0), ("c", 30, 34, 0)], ratings)
    assert 1400 < r < 1700 and 0 < se < 100, (r, se)
    r2, _ = S.fit_fixed([("a", 64, 0, 0)], ratings)
    assert r2 > 1431 and np.isfinite(r2)
    print(f"fit_fixed: {r:.0f}±{se:.0f}、全勝でも有限（{r2:.0f}）")


def write_run(d, sec, outcomes, fname="panel_eval.jsonl"):
    """作った run: log.jsonl（1更新 sec 秒）と評価（{step: {相手: 結果の配列}}）"""
    d.mkdir(parents=True)
    steps = sorted(outcomes)
    with open(d / "log.jsonl", "w") as f:
        for st in range(steps[0] + 1, steps[-1] + 1):
            f.write(json.dumps({"step": st, "collect_sec": sec / 2, "update_sec": sec / 2, "games": 8}) + "\n")
    with open(d / fname, "w") as f:
        for st in steps:
            for opp, o in outcomes[st].items():
                f.write(json.dumps({"step": st, "which": "target", "opponent": opp, "w": int((o > 0).sum()),
                                    "l": int((o < 0).sum()), "d": int((o == 0).sum()), "games": len(o),
                                    "outcomes": o.tolist()}) + "\n")


def check_report():
    n = 8
    win, lose, draw = np.ones(n, int), -np.ones(n, int), np.zeros(n, int)
    opps = ("in1", "out1")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        # 対照（R-NaD）: 1更新 5.5秒。起点 100 は引き分け、110 は全敗、120 は全勝
        write_run(tmp / "ab_rnad_s1", 5.5, {100: {o: draw for o in opps}, 110: {o: lose for o in opps},
                                             120: {o: win for o in opps}})
        # MMD: 1更新 2.75秒（倍速）。120（壁時計 55秒 = 対照の 110）は全勝、140（110秒 = 対照の 120）は引き分け
        for sd in (1, 2):
            write_run(tmp / f"ab_mmd_s{sd}", 2.75, {100: {o: draw for o in opps}, 120: {o: win for o in opps},
                                                   140: {o: draw for o in opps}})
        # 対照の log は分けた点より前（本線の 1〜100、1更新 1秒）も持つ（要約は起点 100 からの差で数える）
        log = tmp / "ab_rnad_s1" / "log.jsonl"
        pre = "".join(json.dumps({"step": st, "collect_sec": 0.5, "update_sec": 0.5, "games": 8}) + "\n"
                      for st in range(1, 101))
        log.write_text(pre + log.read_text())
        (tmp / "groups.json").write_text(json.dumps({"系列内": ["in1"], "系列外": ["out1"]}))
        ratings = {"in1": 1500.0, "out1": 1500.0}
        base = dict(which="target", gain=50.0, control="ab_rnad", out="", json="", eval_file="panel_eval.jsonl",
                    from_step=100, exclude_step=[100], pair_by="wallclock", sec_per_update=None,
                    control_any_seed=True, vs_step=None, groups=str(tmp / "groups.json"))
        runs = [tmp / r for r in ("ab_rnad_s1", "ab_mmd_s1", "ab_mmd_s2")]
        arms = {}
        for r in runs:
            arms.setdefault(S.arm_of(r), []).append((r, S.load_eval(r, ratings), S.load_log(r)))
        assert set(arms) == {"ab_rnad", "ab_mmd"}
        lines = S.paired_section(arms, "ab_rnad", "target", [100, 110, 120, 140], argparse.Namespace(**base))
        text = "\n".join(lines)
        row = next(x for x in lines if x.startswith("| ab_mmd |"))
        cells = [c.strip() for c in row.split("|")[1:-1]]
        # 列は 110 / 120 / 140（100 は外す）。120 は対照の 110（全敗）と対: +100、140 は対照の 120（全勝）と対: −50
        assert cells[3].startswith("+100.0") and cells[4].startswith("-50.0"), row  # cells[0] は腕の名前
        assert "2 seed" in cells[5] and "ab_mmd 120→110" in text and "ab_mmd 140→120" in text, text
        assert "| ab_mmd | +25.0" in text  # 組ごとの小計（両方の組で同じ）
        # step で対にすると 120 どうし（全勝と全勝）: 0、140 は対照に無い
        lines = S.paired_section(arms, "ab_rnad", "target", [100, 110, 120, 140],
                                 argparse.Namespace(**{**base, "pair_by": "step"}))
        row = next(x for x in lines if x.startswith("| ab_mmd |"))
        cells = [c.strip() for c in row.split("|")[1:-1]]
        assert cells[3].startswith("+0.0") and cells[4] == "", row
        # --control-any-seed なしなら seed 2 は対にならない（1 seed）
        lines = S.paired_section(arms, "ab_rnad", "target", [100, 110, 120, 140],
                                 argparse.Namespace(**{**base, "control_any_seed": False}))
        assert "1 seed" in next(x for x in lines if x.startswith("| ab_mmd |"))
        # 秒/更新を省くと log.jsonl の中央値（5.5 と 2.75）
        pr = S.Pairing(arms, "ab_rnad", argparse.Namespace(**base))
        assert pr.spu == {"ab_rnad": 5.5, "ab_mmd": 2.75}, pr.spu
        # --sec-per-update は腕ごとに上書きする（カンマ区切りでも、フラグを並べても）
        for spu in (["ab_rnad=5.5,ab_mmd=1.0"], ["ab_rnad=5.5", "ab_mmd=1.0"]):
            pr = S.Pairing(arms, "ab_rnad", argparse.Namespace(**{**base, "sec_per_update": spu}))
            assert pr.spu == {"ab_rnad": 5.5, "ab_mmd": 1.0}, pr.spu
        # --vs-step 100: 同じ run の起点（引き分け）と対
        lines = S.vs_step_section(arms, "target", [100, 110, 120, 140], argparse.Namespace(**{**base, "vs_step": 100}))
        row = next(x for x in lines if x.startswith("| ab_mmd |"))
        assert row.split("|")[2].strip() == "" and row.split("|")[3].strip().startswith("+50.0"), row
        # --vs-step と --groups: 組ごとの表（系列内の相手にだけ伸びた run なら、系列内 +50・系列外 +0、全体 +25）
        write_run(tmp / "g_s1", 1.0, {100: {o: draw for o in opps}, 120: {"in1": win, "out1": draw}})
        garms = {"g": [(tmp / "g_s1", S.load_eval(tmp / "g_s1", ratings), S.load_log(tmp / "g_s1"))]}
        lines = S.vs_step_section(garms, "target", [100, 120], argparse.Namespace(**{**base, "vs_step": 100}))
        cell = lambda label: next(x for x in lines if x.startswith(f"| {label} |")).split("|")[2].strip()  # noqa: E731
        assert cell("g").startswith("+25.0") and cell("g（系列内）").startswith("+50.0"), lines
        assert cell("g（系列外）").startswith("+0.0"), lines
        # --vs-which: 起点の側だけ別の記録名（締まりをそろえた起点 target@t0.9 = 両方の相手に全敗）と対にする
        with open(tmp / "g_s1" / "panel_eval.jsonl", "a") as f:
            for opp in opps:
                f.write(json.dumps({"step": 100, "which": "target@t0.9", "opponent": opp, "w": 0, "l": n, "d": 0,
                                    "games": n, "outcomes": lose.tolist()}) + "\n")
        lines = S.vs_step_section(garms, "target", [100, 120],
                                  argparse.Namespace(**{**base, "vs_step": 100, "vs_which": "target@t0.9"}))
        row = [c.strip() for c in next(x for x in lines if x.startswith("| g |")).split("|")[1:-1]]
        # 列は 100（起点そのもの〔引き分け〕と温度をそろえた起点〔全敗〕の差 +50）と 120（勝ち・引き分けと全敗の差 +75）
        assert row[1].startswith("+50.0") and row[2].startswith("+75.0"), row
        assert cell("g（系列内）").startswith("+50.0"), lines  # 100 の列（系列内: 引き分け − 全敗）
        # 局数が起点の側と違う組は対にできないので、数を出して飛ばす（黙って空欄にしない）
        with open(tmp / "g_s1" / "panel_eval.jsonl", "a") as f:
            for opp in opps:
                f.write(json.dumps({"step": 100, "which": "target@t0.8", "opponent": opp, "w": 0, "l": 4, "d": 0,
                                    "games": 4, "outcomes": [-1] * 4}) + "\n")
        lines = S.vs_step_section(garms, "target", [100, 120],
                                  argparse.Namespace(**{**base, "vs_step": 100, "vs_which": "target@t0.8"}))
        assert any("比べなかった組: 4" in x for x in lines), lines
        # コマンドとして（cmd_report の引数の配線）
        (tmp / "panel.json").write_text(json.dumps({"planes": 91, "members": [], "ratings": ratings}))
        # --sec-per-update で MMD を 5.5秒に上書きする（log の中央値 2.75 と違う値: 120 は対照の 120 と対になる）
        r = __import__("subprocess").run(
            [sys.executable, str(HERE.parent / "speedlab.py"), "report", *map(str, runs), "--panel", str(tmp / "panel.json"),
             "--control", "ab_rnad", "--which", "target", "--pair-by", "wallclock", "--sec-per-update", "ab_rnad=5.5,ab_mmd=5.5",
             "--from-step", "100", "--exclude-step", "100", "--control-any-seed", "--vs-step", "100",
             "--groups", str(tmp / "groups.json")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        out = r.stdout
        assert "ab_mmd 120→120" in out and "step 100 との局ごとのペア差" in out and "系列内" in out, out
        # 要約: 起点 100（表からは外した引き分け）から +50 に届くのは 120 = 20更新後。局数・壁時計・1更新の秒も起点から
        # （対照の log の分けた点より前は数えない。数えると 1更新の秒は 210/120 = 1.75 になる）
        summ = out[out.index("## 要約"):]
        row = [c.strip() for c in next(x for x in summ.splitlines() if x.startswith("| ab_mmd |")).split("|")[1:-1]]
        # 壁時計は 20更新 × 2.75秒 = 0.9分
        assert (row[4].startswith("20") and "（2/2）" in row[4] and row[5] == "160±0" and row[6] == "0.9±0.0"
                and row[7] == "2.8±0.0"), row
        row = [c.strip() for c in next(x for x in summ.splitlines() if x.startswith("| ab_rnad |")).split("|")[1:-1]]
        # 平均 Elo は起点 100 から（起点の Elo は表から外していても使う）: 100〜120 の台形の面積 / 20
        E = {st: e for (st, w), (e, *_) in S.load_eval(runs[0], ratings).items()}
        auc = ((E[100] + E[110]) / 2 * 10 + (E[110] + E[120]) / 2 * 10) / 20
        assert row[3] == f"{auc:.0f}" and row[4] == "20（1/1）" and row[5] == "160" and row[6] == "1.8" and row[7] == "5.5", row
        # Elo の差の表も同じ壁時計で対にする: 120 は対照の 120（全勝どうしで 0）、140（壁時計 220秒）も対照の 120 と対
        # （同じ step で対にすると 140 は対照に無く空になる）
        diff = out[out.index("との差（同じ seed どうしの Elo の差"):]
        assert "同じ壁時計の書き出しと" in diff.splitlines()[0], diff.splitlines()[0]
        row = [c.strip() for c in next(x for x in diff.splitlines() if x.startswith("| ab_mmd |")).split("|")[1:-1]]
        Em = {st: e for (st, w), (e, *_) in S.load_eval(runs[1], ratings).items()}
        assert row[2] == "+0±0" and row[3] == f"{Em[140] - E[120]:+.0f}±0", row  # 列は 110 / 120 / 140
        # 全 step 平均（seed ごとに step の差を平均）も同じ対応: (0 + (140 の差)) / 2
        assert row[4] == f"{(Em[140] - E[120]) / 2:+.0f}±0（2 seed）", row
        # 起点しか測っていない run（始めたばかり）の平均 Elo は起点の Elo（面積 0 にしない）
        write_run(tmp / "solo_s1", 2.75, {100: {o: draw for o in opps}})
        r = __import__("subprocess").run(
            [sys.executable, str(HERE.parent / "speedlab.py"), "report", str(runs[0]), str(tmp / "solo_s1"),
             "--panel", str(tmp / "panel.json"), "--which", "target", "--from-step", "100"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        summ = r.stdout[r.stdout.index("## 要約"):]
        row = [c.strip() for c in next(x for x in summ.splitlines() if x.startswith("| solo |")).split("|")[1:-1]]
        Es = S.load_eval(tmp / "solo_s1", ratings)[(100, "target")][0]
        assert row[3] == f"{Es:.0f}", row
        # 起点を step 0 とする壁時計の対応では、腕の step 10 を対照の step 0（出発点）とは対にしない
        # （対照 6秒/更新、腕 2秒/更新: 腕の 10 = 20秒は対照の 0 = 0秒のほうが近いが、出発点は出発点とだけ対にする）
        write_run(tmp / "w0" / "c_s1", 6.0, {0: {o: draw for o in opps}, 10: {o: lose for o in opps}, 20: {o: win for o in opps}})
        write_run(tmp / "w0" / "a_s1", 2.0, {0: {o: draw for o in opps}, 10: {o: win for o in opps}, 20: {o: draw for o in opps}})
        r = __import__("subprocess").run(
            [sys.executable, str(HERE.parent / "speedlab.py"), "report", str(tmp / "w0" / "c_s1"), str(tmp / "w0" / "a_s1"),
             "--panel", str(tmp / "panel.json"), "--which", "target", "--control", "c", "--pair-by", "wallclock"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        diff = r.stdout[r.stdout.index("との差（同じ seed どうしの Elo の差"):]
        row = [c.strip() for c in next(x for x in diff.splitlines() if x.startswith("| a |")).split("|")[1:-1]]
        Ec = {st: e for (st, w), (e, *_) in S.load_eval(tmp / "w0" / "c_s1", ratings).items()}
        Ea = {st: e for (st, w), (e, *_) in S.load_eval(tmp / "w0" / "a_s1", ratings).items()}
        assert row[1] == "+0" and row[2] == f"{Ea[10] - Ec[10]:+.0f}", (row, Ea, Ec)  # 列は 0 / 10 / 20
        # log.jsonl の無いフォルダ（export-ck で起点だけ書き出して eval したもの）も落とさない。どちらも無ければ名前を出して飛ばす
        write_run(tmp / "nolog_s1", 1.0, {100: {o: draw for o in opps}, 120: {o: win for o in opps}})
        (tmp / "nolog_s1" / "log.jsonl").unlink()
        (tmp / "empty_s1").mkdir()
        r = __import__("subprocess").run(
            [sys.executable, str(HERE.parent / "speedlab.py"), "report", str(tmp / "nolog_s1"), str(tmp / "empty_s1"),
             "--panel", str(tmp / "panel.json"), "--which", "target", "--from-step", "100", "--vs-step", "100"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        vs = r.stdout[r.stdout.index("との局ごとのペア差"):]
        row = [c.strip() for c in next(x for x in vs.splitlines() if x.startswith("| nolog |")).split("|")[1:-1]]
        assert row[1].startswith("+50.0"), (row, r.stdout)  # 列は 120（起点 100 の引き分けと全勝の差）
        assert "empty_s1" in r.stderr and "| empty |" not in r.stdout, (r.stdout, r.stderr)
        # --eval-file: 別のファイルの評価だけを読む
        write_run(tmp / "x_s1", 1.0, {5: {"in1": win}}, fname="panel_eval_final.jsonl")
        assert S.load_eval(tmp / "x_s1", ratings) == {} and S.load_eval(tmp / "x_s1", ratings, "panel_eval_final.jsonl")
    print("report: 壁時計の対応・起点の除外・対照の seed の共用・起点とのペア差・組ごとの小計・評価のファイル名・"
          "秒/更新の上書き・要約は起点から・起点とのペア差の組ごとの表・局数違いの組の数・log.jsonl の無いフォルダ")


if __name__ == "__main__":
    check_uniform()
    check_keyed_actions()
    check_keyed_temp()
    check_decode()
    check_seed_salt()
    check_fit()
    check_report()
