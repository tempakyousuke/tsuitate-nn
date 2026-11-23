"""エンコーダ v3 以降の入力プレーンの組（`model.EXTRA_PLANES`、`--opp-hand-planes` / `--opp-foul-planes` /
`--king-foul-planes` / v5 の `--lost-fresh-planes` / `--lost-role-planes` / `--cap-role-planes` / `--pawn-file-planes`）の検査。

    .venv/bin/python rnad/tests/test_extra_planes.py

- 組の並びが tsuitate-bot の `rl::encode::EXTRA_PLANE_GROUPS` と同じ（tsuitate_rl が入っていれば照合する）
- 使わない組は入力を読まない: 値を変えても出力が変わらず、stem のその列に勾配が来ず 0 のまま
- v2 → v3 / v4 / v5 に広げた checkpoint（R-NaD・MMD）は、組の選び方によらず元と同じ関数。optimizer の状態も読める
- v3 の checkpoint に組を後から足しても（v4 の組なら 102枚、v5 の組なら 124枚へ広げても）同じ関数で、足した列の
  2次モーメントは入れ直される。外す・何も変えない指定と、後の版の組を前の版のネットに入れる指定は止める
- 書き出した重みは使わない組の列が 0 で、読み戻すと組の選び方も戻る
- 学習のフラグ（train_rnad.with_extras）: 足りない組は足し（組が入る版まで広げ）、外すのと v1 からは止める
- フラグの名前（`widen_planes.EXTRA_FLAGS`）は組の全部を1つずつ指す
"""

import argparse
import pathlib
import sys
import tempfile

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from export_weights import read as read_weights  # noqa: E402
from export_weights import write as write_weights  # noqa: E402
from mmd import MMDLearner, make_config  # noqa: E402
from model import (DROPPABLE_PLANES, EXTRA_PLANES, NUM_PLANES, NUM_PLANES_DEDUCED, NUM_PLANES_V3,  # noqa: E402
                   NUM_PLANES_V4, NUM_PLANES_V5, NUM_PLANES_V6, PolicyValueNet, planes_for, unused_planes)
from rnad import RNaDConfig, RNaDLearner  # noqa: E402
from widen_planes import EXTRA_FLAGS, add_extra_flags, extras_from_args, widen_checkpoint  # noqa: E402

DEV = torch.device("cpu")
CFG = RNaDConfig(eta=0.2, beta=16.0, lr=2e-4, schedule_sizes=(200,), schedule_repeats=(1,))
V6_GROUPS = ["king_tight", "check_clock", "checker_memo", "own_attacks", "king_lines", "opp_counts"]
# (プレーン数, 使う組)
CHOICES = [
    (NUM_PLANES_V3, []), (NUM_PLANES_V3, ["opp_hand"]), (NUM_PLANES_V3, ["opp_fouls"]),
    (NUM_PLANES_V3, ["opp_fouls", "opp_hand"]),
    (NUM_PLANES_V4, []), (NUM_PLANES_V4, ["king_fouls"]), (NUM_PLANES_V4, ["king_fouls", "opp_hand"]),
    (NUM_PLANES_V4, ["king_fouls", "opp_fouls", "opp_hand"]),
    (NUM_PLANES_V5, []), (NUM_PLANES_V5, ["lost_fresh"]), (NUM_PLANES_V5, ["lost_roles"]), (NUM_PLANES_V5, ["cap_roles"]),
    (NUM_PLANES_V5, ["pawn_files"]), (NUM_PLANES_V5, ["cap_roles", "king_fouls", "lost_roles", "opp_hand"]),
    (NUM_PLANES_V5, sorted(set(EXTRA_PLANES) - set(V6_GROUPS))),
    (NUM_PLANES_V6, []), (NUM_PLANES_V6, ["king_tight"]), (NUM_PLANES_V6, ["check_clock", "own_attacks"]),
    (NUM_PLANES_V6, ["checker_memo", "king_lines", "opp_hand", "pawn_files"]), (NUM_PLANES_V6, ["opp_counts"]),
    (NUM_PLANES_V6, sorted(EXTRA_PLANES)),
]
V5_GROUPS = ["lost_fresh", "lost_roles", "cap_roles", "pawn_files"]


def span(name):
    s, n = EXTRA_PLANES[name]
    return list(range(s, s + n))


KING_WIDE, KING_TIGHT = DROPPABLE_PLANES["king_wide"], EXTRA_PLANES["king_tight"][0]


def as_before(x, old_planes, extras):
    """広げる前のネットに渡す入力: 先頭 `old_planes` 枚。`king_tight` を足したネットは元の相手玉の候補（89番）の代わりに
    締めた候補（124番）を読むので、比べる相手の 89番には 124番の値を入れる"""
    xb = x[:, :old_planes].clone()
    if "king_tight" in extras:
        xb[:, KING_WIDE] = x[:, KING_TIGHT]
    return xb


def step(learner, planes, seed=0):
    """適当な損失で1回更新する（optimizer の状態を作る）"""
    torch.manual_seed(seed)
    x = torch.rand(4, planes, 9, 9)
    logits, v = learner.net(x)
    learner.opt.zero_grad()
    (logits.square().mean() + v.square().mean()).backward()
    learner.opt.step()


def rnad_learner(planes=NUM_PLANES_DEDUCED, extras=()):
    torch.manual_seed(0)
    lr = RNaDLearner(PolicyValueNet(8, 1, planes, extras), CFG, DEV)
    step(lr, planes)
    return lr


def same(a, b, x_a, x_b, what):
    a, b = a.eval(), b.eval()
    with torch.no_grad():
        la, va = a(x_a)
        lb, vb = b(x_b)
    assert torch.allclose(la, lb, atol=1e-5) and torch.allclose(va, vb, atol=1e-6), what


def check_layout():
    spans = sorted(EXTRA_PLANES.values())
    assert spans[0][0] == NUM_PLANES_DEDUCED and sum(n for _, n in spans) == NUM_PLANES_V6 - NUM_PLANES_DEDUCED
    for (s, n), (t, _) in zip(spans, spans[1:]):
        assert s + n == t, "組は隙間なく並ぶ"
    assert planes_for([]) == NUM_PLANES_DEDUCED
    assert planes_for(["opp_hand"]) == planes_for(["opp_fouls", "opp_hand"]) == NUM_PLANES_V3
    assert planes_for(["king_fouls"]) == planes_for(["king_fouls", "opp_hand"]) == NUM_PLANES_V4
    for name in V5_GROUPS:
        assert planes_for([name]) == planes_for([name, "opp_hand"]) == NUM_PLANES_V5, name
    for name in V6_GROUPS:
        assert planes_for([name]) == planes_for([name, "pawn_files"]) == NUM_PLANES_V6, name
    assert sorted(V5_GROUPS + V6_GROUPS + ["opp_hand", "opp_fouls", "king_fouls"]) == sorted(EXTRA_PLANES)
    # フラグは組の全部を1つずつ指し、フラグから組の名前が取れる
    assert sorted(EXTRA_FLAGS.values()) == sorted(EXTRA_PLANES)
    ap = argparse.ArgumentParser()
    add_extra_flags(ap)
    assert extras_from_args(ap.parse_args([])) == []
    for flag, name in EXTRA_FLAGS.items():
        assert extras_from_args(ap.parse_args(["--" + flag.replace("_", "-")])) == [name], flag
    try:
        import tsuitate_rl
    except ImportError:
        print("tsuitate_rl が無いので Rust 側との照合は飛ばす")
        return
    assert tsuitate_rl.NUM_PLANES_V3 == NUM_PLANES_V3 and tsuitate_rl.NUM_PLANES_V4 == NUM_PLANES_V4
    assert tsuitate_rl.NUM_PLANES_V5 == NUM_PLANES_V5 and tsuitate_rl.NUM_PLANES_V6 == NUM_PLANES_V6
    assert dict(tsuitate_rl.EXTRA_PLANE_GROUPS) == {k: tuple(v) for k, v in EXTRA_PLANES.items()}, \
        (tsuitate_rl.EXTRA_PLANE_GROUPS, EXTRA_PLANES)


def check_mask():
    for planes, extras in CHOICES:
        what = f"{planes}枚 {extras}"
        lr = rnad_learner(planes, extras)
        off = unused_planes(planes, extras)
        net = lr.net
        assert net.config["extras"] == sorted(extras)
        assert (net.stem.weight[:, off] == 0).all(), what
        torch.manual_seed(3)
        x = torch.rand(8, planes, 9, 9)
        y = x.clone()
        y[:, off] = torch.rand(8, len(off), 9, 9) * 3
        same(net, net, x, y, f"{what}: 使わない組の値で出力が変わった")
        # 勾配も来ない（何回更新しても 0 のまま）
        net.train()
        logits, v = net(y)
        lr.opt.zero_grad()
        (logits.square().mean() + v.square().mean()).backward()
        g = net.stem.weight.grad
        assert (g[:, off] == 0).all() and (g.abs().sum() > 0), what
        lr.opt.step()
        assert (net.stem.weight[:, off] == 0).all(), what
        for name in extras:
            assert g[:, span(name)].abs().sum() > 0, f"{what}: 使う組 {name} に勾配が来ない"
    # v1・v2 は組を持てない・後の版の組は前の版のネットに入らない・知らない組は止める
    for bad in ((NUM_PLANES_DEDUCED, ["opp_hand"]), (NUM_PLANES_V3, ["king_fouls"]), (NUM_PLANES_V3, ["nope"]),
                (NUM_PLANES_V4, ["pawn_files"]), (NUM_PLANES_V3, ["lost_fresh"])):
        try:
            PolicyValueNet(8, 1, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} を受け付けた")


def check_widen_rnad():
    base = rnad_learner()
    ck = base.state_dict()
    for planes, extras in CHOICES:
        wide = widen_checkpoint(ck, planes, extras)
        assert wide["config"]["planes"] == planes and wide["config"]["extras"] == sorted(extras)
        assert ck["config"]["planes"] == NUM_PLANES_DEDUCED and "extras" not in ck["config"], "元を書き換えた"
        other = RNaDLearner(PolicyValueNet.from_config(wide["config"]), CFG, DEV)
        other.load_state_dict(wide)
        torch.manual_seed(1)
        x = torch.rand(16, planes, 9, 9) * 2
        for name in ("net", "target", "prev", "prev_"):
            same(getattr(base, name), getattr(other, name), as_before(x, NUM_PLANES_DEDUCED, extras), x,
                 f"{planes} {extras} {name}")
        st = other.opt.state_dict()["state"][0]
        assert st["exp_avg_sq"].shape[1] == planes
        assert (st["exp_avg_sq"][:, NUM_PLANES_DEDUCED:] > 0).all()
        # 足した列の 1次モーメントは 0 から。締めた相手玉の候補だけは元の候補の列から移したもの
        zero = [i for i in range(NUM_PLANES_DEDUCED, planes) if not ("king_tight" in extras and i == KING_TIGHT)]
        assert (st["exp_avg"][:, zero] == 0).all()
        step(other, planes, seed=2)


def check_widen_mmd():
    torch.manual_seed(0)
    base = MMDLearner(PolicyValueNet(8, 1, NUM_PLANES_DEDUCED), make_config(), DEV)
    step(base, NUM_PLANES_DEDUCED)
    ck = base.state_dict()
    for planes, extras in [c for c in CHOICES if c[1]]:
        wide = widen_checkpoint(ck, planes, extras)
        other = MMDLearner(PolicyValueNet.from_config(wide["config"]), make_config(), DEV)
        other.load_state_dict(wide)
        torch.manual_seed(1)
        x = torch.rand(16, planes, 9, 9) * 2
        for name in ("net", "target", "magnet_net"):
            same(getattr(base, name), getattr(other, name), as_before(x, NUM_PLANES_DEDUCED, extras), x,
                 f"MMD {extras} {name}")
        assert (other.opt.state_dict()["state"][0]["exp_avg_sq"][:, NUM_PLANES_DEDUCED:] > 0).all()
        step(other, planes, seed=2)
    # --init から始めたばかり（Adam の状態が空）の checkpoint も広げられる
    empty = MMDLearner(PolicyValueNet(8, 1, NUM_PLANES_DEDUCED), make_config(), DEV).state_dict()
    assert widen_checkpoint(empty, NUM_PLANES_V4, ["king_fouls"])["config"]["extras"] == ["king_fouls"]
    assert widen_checkpoint(empty, NUM_PLANES_V5, ["pawn_files"])["config"]["extras"] == ["pawn_files"]


def check_add_later():
    """v3 で opp_hand だけ使って学習した後に、opp_fouls を足す（101枚のまま）・king_fouls を足す（102枚へ広げる）・
    v5 の組を足す（124枚へ広げる）"""
    lr = rnad_learner(NUM_PLANES_V3, ["opp_hand"])
    for s in range(3):
        step(lr, NUM_PLANES_V3, seed=10 + s)
    ck = lr.state_dict()
    fouls = span("opp_fouls")
    before = ck["opt"]["state"][0]["exp_avg_sq"][:, fouls]
    torch.manual_seed(4)
    x = torch.rand(16, NUM_PLANES_V5, 9, 9) * 2
    for planes, extras, added in ((NUM_PLANES_V3, ["opp_hand", "opp_fouls"], fouls),
                                  (NUM_PLANES_V4, ["opp_hand", "king_fouls"], span("king_fouls")),
                                  (NUM_PLANES_V5, ["opp_hand", "lost_roles", "pawn_files"],
                                   span("lost_roles") + span("pawn_files")),
                                  (NUM_PLANES_V5, ["opp_hand", *V5_GROUPS], [i for n in V5_GROUPS for i in span(n)])):
        wide = widen_checkpoint(ck, planes, extras)
        assert wide["config"]["planes"] == planes and wide["config"]["extras"] == sorted(extras)
        other = RNaDLearner(PolicyValueNet.from_config(wide["config"]), CFG, DEV)
        other.load_state_dict(wide)
        xa = x[:, :planes].clone()
        xa[:, added] = 0
        same(lr.net, other.net, x[:, :NUM_PLANES_V3], x[:, :planes], f"{extras}: 足した直後は元と同じ関数")
        same(lr.net, other.net, xa[:, :NUM_PLANES_V3], x[:, :planes], f"{extras}: 足した組の列は 0 から始まる")
        after = other.opt.state_dict()["state"][0]["exp_avg_sq"][:, added]
        assert (after > 0).all(), f"{extras}: 足した組の 2次モーメントが 0"
        step(other, planes, seed=20)
    after = widen_checkpoint(ck, NUM_PLANES_V3, ["opp_hand", "opp_fouls"])["opt"]["state"][0]["exp_avg_sq"][:, fouls]
    assert not torch.equal(before, after), "使っていなかった組の 2次モーメントを入れ直していない"
    for planes, bad in ((NUM_PLANES_V3, ["opp_fouls"]), (NUM_PLANES_V3, ["opp_hand"]),
                        (NUM_PLANES_V3, ["opp_hand", "king_fouls"]), (NUM_PLANES_V4, ["opp_hand", "cap_roles"]),
                        (NUM_PLANES_V5, ["cap_roles"])):
        try:
            widen_checkpoint(ck, planes, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{planes} {bad}: 組を外す・何も変えない・入らない指定を受け付けた")


def check_export():
    for planes, extras in CHOICES:
        torch.manual_seed(5)
        net = PolicyValueNet(8, 1, planes, extras).eval()
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "w.bin"
            write_weights(net, path)
            back = read_weights(path).eval()
        assert back.config == net.config, (back.config, net.config)
        off = unused_planes(planes, extras)
        assert (back.stem.weight[:, off] == 0).all()
        torch.manual_seed(6)
        x = torch.rand(8, planes, 9, 9)
        same(net, back, x, x, f"書き出し {planes} {extras}")
        # 読み戻したネットを、入力を消さないネット（Rust 側と同じ = 観測を全部読む）に移しても同じ
        every = [n for n, (s, k) in EXTRA_PLANES.items() if s + k <= planes]
        raw = PolicyValueNet(8, 1, planes, every, drop=())
        raw.load_state_dict(back.state_dict())
        same(back, raw, x, x, f"使わない組の列が 0 でない {planes} {extras}")


def check_replace():
    """締めた相手玉の候補（king_tight）は、元の候補（確定事実の 89番）の代わりに読む:
    - 組を選んで作ったネットは 89番を読まない（config の drop。入力を変えても出力が同じ・列は 0・勾配が来ない）
    - 組を選ばないネット・drop を空にしたネット（全部読む = pad_planes と Rust 側）は 89番を読む
    - 既存の checkpoint に足すと、89番の列の重みと Adam の状態を 124番へ移す。広げた直後は「89番の代わりに 124番を
      読む」ネットで、89番の値にはよらない。もう一度別の組を足しても移し直さない
    - 書き出した重みは 89番の列が 0 で、ヘッダに drop が残る"""
    from rnad import pad_planes
    torch.manual_seed(7)
    x = torch.rand(8, NUM_PLANES_V6, 9, 9)
    y = x.clone()
    y[:, KING_WIDE] = torch.rand(8, 9, 9) * 3
    tight = rnad_learner(NUM_PLANES_V6, ["king_tight"]).net
    assert tight.config["drop"] == ["king_wide"] and KING_WIDE in unused_planes(NUM_PLANES_V6, ["king_tight"])
    assert (tight.stem.weight[:, KING_WIDE] == 0).all()
    same(tight, tight, x, y, "king_tight のネットが 89番を読んだ")
    plain = rnad_learner(NUM_PLANES_V6, ["check_clock"]).net
    assert "drop" not in plain.config and KING_WIDE not in unused_planes(NUM_PLANES_V6, ["check_clock"])
    assert (plain.stem.weight[:, KING_WIDE] != 0).any()
    every = PolicyValueNet(8, 1, NUM_PLANES_V6, sorted(EXTRA_PLANES), drop=())
    assert "drop" not in every.config and every.in_off is None
    # 89番を読む v5 のネットを 136枚へ積み直しても（相手のネットをまとめて推論するとき）、89番を読んだまま
    old = rnad_learner(NUM_PLANES_V5, ["opp_hand", "pawn_files"])
    for s_ in range(3):
        step(old, NUM_PLANES_V5, seed=30 + s_)
    padded = pad_planes(old.net, NUM_PLANES_V6)
    same(old.net, padded, y[:, :NUM_PLANES_V5], y, "積み直したネットが 89番を読まなくなった")
    with torch.no_grad():
        a, b = old.net.eval()(x[:, :NUM_PLANES_V5])[0], old.net(y[:, :NUM_PLANES_V5])[0]
    assert not torch.allclose(a, b), "検査の入力で 89番が効いていない"
    # 既存の checkpoint に足す
    ck = old.state_dict()
    wide = widen_checkpoint(ck, NUM_PLANES_V6, ["opp_hand", "pawn_files", "king_tight"])
    assert wide["config"]["drop"] == ["king_wide"] and "drop" not in ck["config"]
    other = RNaDLearner(PolicyValueNet.from_config(wide["config"]), CFG, DEV)
    other.load_state_dict(wide)
    for name in ("net", "target", "prev", "prev_"):
        a, b = getattr(old, name), getattr(other, name)
        same(a, b, as_before(x, NUM_PLANES_V5, ["king_tight"]), x, f"移し替え {name}")
        same(b, b, x, y, f"移し替えた後も 89番を読んでいる {name}")
        assert torch.equal(wide[name]["stem.weight"][:, KING_TIGHT], ck[name]["stem.weight"][:, KING_WIDE])
        assert (wide[name]["stem.weight"][:, KING_WIDE] == 0).all()
    st0, st1 = ck["opt"]["state"][0], wide["opt"]["state"][0]
    assert torch.equal(st1["exp_avg"][:, KING_TIGHT], st0["exp_avg"][:, KING_WIDE])
    assert torch.equal(st1["exp_avg_sq"][:, KING_TIGHT], st0["exp_avg_sq"][:, KING_WIDE])
    assert (st1["exp_avg"][:, KING_WIDE] == 0).all()
    step(other, NUM_PLANES_V6, seed=40)
    assert (other.net.stem.weight[:, KING_WIDE] == 0).all(), "読まない列が動いた"
    # もう一度別の組を足しても、移し直さない（124番は学習した重みのまま、drop も残る）
    again = widen_checkpoint(other.state_dict(), NUM_PLANES_V6, ["opp_hand", "pawn_files", "king_tight", "king_lines"])
    assert again["config"]["drop"] == ["king_wide"]
    assert torch.equal(again["net"]["stem.weight"][:, KING_TIGHT], other.net.stem.weight[:, KING_TIGHT])
    # v1・v2 の版が違うだけの checkpoint（89番を読んでいる 91枚）からも移す
    v2 = rnad_learner()
    w2 = widen_checkpoint(v2.state_dict(), NUM_PLANES_V6, ["king_tight"])
    assert torch.equal(w2["net"]["stem.weight"][:, KING_TIGHT], v2.net.stem.weight[:, KING_WIDE])
    # 書き出し
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "w.bin"
        write_weights(other.net.eval(), path)
        back = read_weights(path).eval()
    assert back.config["drop"] == ["king_wide"] and (back.stem.weight[:, KING_WIDE] == 0).all()
    same(other.net, back, y, y, "書き出し（king_tight）")


def check_with_extras():
    try:
        import train_rnad
    except ImportError as e:
        print(f"train_rnad を読めない（{e}）ので with_extras は飛ばす")
        return
    ck = rnad_learner().state_dict()
    assert train_rnad.with_extras(ck, [], "v2") is ck
    wide = train_rnad.with_extras(ck, ["opp_fouls"], "v2")
    assert wide["config"]["planes"] == NUM_PLANES_V3 and wide["config"]["extras"] == ["opp_fouls"]
    assert train_rnad.with_extras(wide, ["opp_fouls"], "v3") is wide
    v3 = train_rnad.with_extras(wide, ["opp_fouls", "opp_hand"], "v3")
    assert v3["config"]["planes"] == NUM_PLANES_V3 and v3["config"]["extras"] == ["opp_fouls", "opp_hand"]
    v4 = train_rnad.with_extras(wide, ["opp_fouls", "king_fouls"], "v3")
    assert v4["config"]["planes"] == NUM_PLANES_V4 and v4["config"]["extras"] == ["king_fouls", "opp_fouls"]
    only_king = train_rnad.with_extras(ck, ["king_fouls"], "v2")
    assert only_king["config"]["planes"] == NUM_PLANES_V4 and only_king["config"]["extras"] == ["king_fouls"]
    # v4 の checkpoint に v5 の組を足す（124枚へ広がる）・v2 から v5 の組だけ
    v5 = train_rnad.with_extras(v4, ["opp_fouls", "king_fouls", "pawn_files", "lost_fresh"], "v4")
    assert v5["config"]["planes"] == NUM_PLANES_V5
    assert v5["config"]["extras"] == ["king_fouls", "lost_fresh", "opp_fouls", "pawn_files"]
    assert train_rnad.with_extras(v5, ["opp_fouls", "king_fouls", "pawn_files", "lost_fresh"], "v5") is v5
    only_cap = train_rnad.with_extras(ck, ["cap_roles"], "v2")
    assert only_cap["config"]["planes"] == NUM_PLANES_V5 and only_cap["config"]["extras"] == ["cap_roles"]
    v1 = RNaDLearner(PolicyValueNet(8, 1, NUM_PLANES), CFG, DEV).state_dict()
    for c, want in ((wide, []), (wide, ["opp_hand"]), (v4, ["opp_fouls"]), (v1, ["opp_hand"]),
                    (v5, ["opp_fouls", "king_fouls", "pawn_files"]), (v1, ["lost_roles"])):
        try:
            train_rnad.with_extras(c, want, "x")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"{c['config']} → {want} を受け付けた")
    for extras, planes in ((["opp_hand"], NUM_PLANES_V3), (["king_fouls"], NUM_PLANES_V4),
                           (["lost_fresh"], NUM_PLANES_V5), (["king_tight"], NUM_PLANES_V6),
                           (sorted(EXTRA_PLANES), NUM_PLANES_V6)):
        net = train_rnad.load_net("none", 8, 1, extras=extras)
        assert net.config["planes"] == planes and net.config["extras"] == extras


def main():
    check_layout()
    check_mask()
    check_replace()
    check_widen_rnad()
    check_widen_mmd()
    check_add_later()
    check_export()
    check_with_extras()
    print("ok")


if __name__ == "__main__":
    main()
