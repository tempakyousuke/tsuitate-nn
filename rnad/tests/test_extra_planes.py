"""エンコーダ v3 / v4 の入力プレーンの組（`model.EXTRA_PLANES`、`--opp-hand-planes` / `--opp-foul-planes` /
`--king-foul-planes`）の検査。

    .venv/bin/python rnad/tests/test_extra_planes.py

- 組の並びが tsuitate-bot の `rl::encode::EXTRA_PLANE_GROUPS` と同じ（tsuitate_rl が入っていれば照合する）
- 使わない組は入力を読まない: 値を変えても出力が変わらず、stem のその列に勾配が来ず 0 のまま
- v2 → v3 / v4 に広げた checkpoint（R-NaD・MMD）は、組の選び方によらず元と同じ関数。optimizer の状態も読める
- v3 の checkpoint に組を後から足しても（v4 の組なら 102枚へ広げても）同じ関数で、足した列の 2次モーメントは入れ直される。
  外す・何も変えない指定と、v4 の組を v3 のネットに入れる指定は止める
- 書き出した重みは使わない組の列が 0 で、読み戻すと組の選び方も戻る
- 学習のフラグ（train_rnad.with_extras）: 足りない組は足し（組が入る版まで広げ）、外すのと v1 からは止める
"""

import pathlib
import sys
import tempfile

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from export_weights import read as read_weights  # noqa: E402
from export_weights import write as write_weights  # noqa: E402
from mmd import MMDLearner, make_config  # noqa: E402
from model import (EXTRA_PLANES, NUM_PLANES, NUM_PLANES_DEDUCED, NUM_PLANES_V3, NUM_PLANES_V4,  # noqa: E402
                   PolicyValueNet, planes_for, unused_planes)
from rnad import RNaDConfig, RNaDLearner  # noqa: E402
from widen_planes import widen_checkpoint  # noqa: E402

DEV = torch.device("cpu")
CFG = RNaDConfig(eta=0.2, beta=16.0, lr=2e-4, schedule_sizes=(200,), schedule_repeats=(1,))
# (プレーン数, 使う組)
CHOICES = [
    (NUM_PLANES_V3, []), (NUM_PLANES_V3, ["opp_hand"]), (NUM_PLANES_V3, ["opp_fouls"]),
    (NUM_PLANES_V3, ["opp_fouls", "opp_hand"]),
    (NUM_PLANES_V4, []), (NUM_PLANES_V4, ["king_fouls"]), (NUM_PLANES_V4, ["king_fouls", "opp_hand"]),
    (NUM_PLANES_V4, ["king_fouls", "opp_fouls", "opp_hand"]),
]


def span(name):
    s, n = EXTRA_PLANES[name]
    return list(range(s, s + n))


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
    assert spans[0][0] == NUM_PLANES_DEDUCED and sum(n for _, n in spans) == NUM_PLANES_V4 - NUM_PLANES_DEDUCED
    for (s, n), (t, _) in zip(spans, spans[1:]):
        assert s + n == t, "組は隙間なく並ぶ"
    assert planes_for([]) == NUM_PLANES_DEDUCED
    assert planes_for(["opp_hand"]) == planes_for(["opp_fouls", "opp_hand"]) == NUM_PLANES_V3
    assert planes_for(["king_fouls"]) == planes_for(["king_fouls", "opp_hand"]) == NUM_PLANES_V4
    try:
        import tsuitate_rl
    except ImportError:
        print("tsuitate_rl が無いので Rust 側との照合は飛ばす")
        return
    assert tsuitate_rl.NUM_PLANES_V3 == NUM_PLANES_V3 and tsuitate_rl.NUM_PLANES_V4 == NUM_PLANES_V4
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
    # v1・v2 は組を持てない・v4 の組は v3 のネットに入らない・知らない組は止める
    for bad in ((NUM_PLANES_DEDUCED, ["opp_hand"]), (NUM_PLANES_V3, ["king_fouls"]), (NUM_PLANES_V3, ["nope"])):
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
            same(getattr(base, name), getattr(other, name), x[:, :NUM_PLANES_DEDUCED], x, f"{planes} {extras} {name}")
        st = other.opt.state_dict()["state"][0]
        assert st["exp_avg_sq"].shape[1] == planes
        assert (st["exp_avg_sq"][:, NUM_PLANES_DEDUCED:] > 0).all()
        assert (st["exp_avg"][:, NUM_PLANES_DEDUCED:] == 0).all()
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
            same(getattr(base, name), getattr(other, name), x[:, :NUM_PLANES_DEDUCED], x, f"MMD {extras} {name}")
        assert (other.opt.state_dict()["state"][0]["exp_avg_sq"][:, NUM_PLANES_DEDUCED:] > 0).all()
        step(other, planes, seed=2)
    # --init から始めたばかり（Adam の状態が空）の checkpoint も広げられる
    empty = MMDLearner(PolicyValueNet(8, 1, NUM_PLANES_DEDUCED), make_config(), DEV).state_dict()
    assert widen_checkpoint(empty, NUM_PLANES_V4, ["king_fouls"])["config"]["extras"] == ["king_fouls"]


def check_add_later():
    """v3 で opp_hand だけ使って学習した後に、opp_fouls を足す（101枚のまま）・king_fouls を足す（102枚へ広げる）"""
    lr = rnad_learner(NUM_PLANES_V3, ["opp_hand"])
    for s in range(3):
        step(lr, NUM_PLANES_V3, seed=10 + s)
    ck = lr.state_dict()
    fouls = span("opp_fouls")
    before = ck["opt"]["state"][0]["exp_avg_sq"][:, fouls]
    torch.manual_seed(4)
    x = torch.rand(16, NUM_PLANES_V4, 9, 9) * 2
    for planes, extras, added in ((NUM_PLANES_V3, ["opp_hand", "opp_fouls"], fouls),
                                  (NUM_PLANES_V4, ["opp_hand", "king_fouls"], span("king_fouls"))):
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
                        (NUM_PLANES_V3, ["opp_hand", "king_fouls"])):
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
        raw = PolicyValueNet(8, 1, planes, every)
        raw.load_state_dict(back.state_dict())
        same(back, raw, x, x, f"使わない組の列が 0 でない {planes} {extras}")


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
    v1 = RNaDLearner(PolicyValueNet(8, 1, NUM_PLANES), CFG, DEV).state_dict()
    for c, want in ((wide, []), (wide, ["opp_hand"]), (v4, ["opp_fouls"]), (v1, ["opp_hand"])):
        try:
            train_rnad.with_extras(c, want, "x")
        except SystemExit:
            pass
        else:
            raise AssertionError(f"{c['config']} → {want} を受け付けた")
    for extras, planes in ((["opp_hand"], NUM_PLANES_V3), (["king_fouls"], NUM_PLANES_V4)):
        net = train_rnad.load_net("none", 8, 1, extras=extras)
        assert net.config["planes"] == planes and net.config["extras"] == extras


def main():
    check_layout()
    check_mask()
    check_widen_rnad()
    check_widen_mmd()
    check_add_later()
    check_export()
    check_with_extras()
    print("ok")


if __name__ == "__main__":
    main()
