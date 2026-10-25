"""speedlab の鍵つきの乱数と Elo の当てはめの検査（tsuitate_rl は要らない）。

    .venv/bin/python rnad/tests/test_speedlab.py

- keyed_uniform は (salt, 局, 手番, 行動) だけで決まり、行の並びや一緒に作る行に依らない。一様に近い
- keyed_actions は合法手だけで乱数を作っても、全行動で作ったときと同じ手を選ぶ（マスク外は選ばない）
- fit_fixed はパネルのレートから離れた値でも発散しない
- report の A/B 用の比べ方（MMD の A/B。docs/rl-learning-speed.md 10節）: 対照の名前 `ab_rnad_s1` と `ab_mmd_s1`・`ab_mmd_s2` で
  対照の節が出る、`--pair-by wallclock` で秒が最も近い書き出しどうしが対になる、`--exclude-step` で起点が消える、
  `--control-any-seed` で seed 2 も対になる、`--vs-step` で同じ run の起点と対になる、`--groups` の小計、`--eval-file`
"""

import argparse
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
        # --vs-step 100: 同じ run の起点（引き分け）と対
        lines = S.vs_step_section(arms, "target", [100, 110, 120, 140], argparse.Namespace(**{**base, "vs_step": 100}))
        row = next(x for x in lines if x.startswith("| ab_mmd |"))
        assert row.split("|")[2].strip() == "" and row.split("|")[3].strip().startswith("+50.0"), row
        # コマンドとして（cmd_report の引数の配線）
        (tmp / "panel.json").write_text(json.dumps({"planes": 91, "members": [], "ratings": ratings}))
        r = __import__("subprocess").run(
            [sys.executable, str(HERE.parent / "speedlab.py"), "report", *map(str, runs), "--panel", str(tmp / "panel.json"),
             "--control", "ab_rnad", "--which", "target", "--pair-by", "wallclock", "--sec-per-update", "ab_rnad=5.5,ab_mmd=2.75",
             "--from-step", "100", "--exclude-step", "100", "--control-any-seed", "--vs-step", "100",
             "--groups", str(tmp / "groups.json")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
        assert "ab_mmd 120→110" in r.stdout and "step 100 との局ごとのペア差" in r.stdout and "系列内" in r.stdout
        # --eval-file: 別のファイルの評価だけを読む
        write_run(tmp / "x_s1", 1.0, {5: {"in1": win}}, fname="panel_eval_final.jsonl")
        assert S.load_eval(tmp / "x_s1", ratings) == {} and S.load_eval(tmp / "x_s1", ratings, "panel_eval_final.jsonl")
    print("report: 壁時計の対応・起点の除外・対照の seed の共用・起点とのペア差・組ごとの小計・評価のファイル名")


if __name__ == "__main__":
    check_uniform()
    check_keyed_actions()
    check_fit()
    check_report()
