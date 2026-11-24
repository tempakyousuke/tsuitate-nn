"""実験用の入力「序盤の時計」（`--exp-opp-clock`、`model.EXP_OPP_CLOCK_PLANE`）の検査（CPU）。

    .venv/bin/python rnad/tests/test_exp_opp_clock.py

tsuitate-bot の docs/rl-fixed-opponent.md 8.8節の腕（`check_clock` の組が離陸を早くしたのは王手の情報か、細かい時計か）に使う。

- C1 入口で作る値: 139番（相手の指した手 ÷ 100）から「相手の指した手 ÷ 12・上限 1」を作って 139番の位置で読む
  （同じ重みで、139番を手で作り替えた入力を `opp_counts` を読むネットに渡したときと同じ出力）
- C2 読むのは 139番だけ: `opp_counts` の残りの 3枚・選んでいない組を変えても出力は変わらず、その列は 0 のまま勾配も来ない。
  139番の列には勾配が来る
- C3 書き出し: Python 専用の magic（Rust の推論は読まない）で書き、読み戻すと形も出力も同じ。ふつうのネットは元の magic のまま
- C4 止める指定: `opp_counts` と一緒・v5 以下の枚数・相手の束（`pad_planes`）・`widen_planes`
- C5 実際の観測（tsuitate_rl があれば）: 自分がまだ王手をかけていない手番では、作った値が `check_clock` の
  「最後にかけてからの相手の手数」（126番。王手がまだ無ければ初手から数える）と一致する = この腕の前提
- C6 `train_rnad.py`: `--exp-opp-clock` で回り、checkpoint の形に残り、重みは Python 専用の magic。フラグを外した再開と
  `--init` との併用は止まる
"""

import pathlib
import subprocess
import sys
import tempfile

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from export_weights import MAGIC, MAGIC_PY  # noqa: E402
from export_weights import read as read_weights  # noqa: E402
from export_weights import write as write_weights  # noqa: E402
from model import (EXP_OPP_CLOCK_DIV, EXP_OPP_CLOCK_PLANE, EXP_OPP_CLOCK_SRC, EXTRA_PLANES, NUM_PLANES_V5,  # noqa: E402
                   NUM_PLANES_V6, PolicyValueNet)
from rnad import pad_planes  # noqa: E402
from widen_planes import widen_checkpoint  # noqa: E402

TRAIN = HERE.parent / "train_rnad.py"
P = EXP_OPP_CLOCK_PLANE
COUNTS = EXTRA_PLANES["opp_counts"]
assert COUNTS[0] + COUNTS[1] - 1 == P, "139番は opp_counts の最後の 1枚（相手の指した手 ÷ 100）のはず"


def obs_like(n: int, seed: int) -> torch.Tensor:
    """観測に似せた入力: 乱数の盤に、139番だけ「相手の指した手 ÷ 100」（全面・fp16 に丸めた値）を入れる"""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((n, NUM_PLANES_V6, 9, 9), generator=g)
    moves = torch.arange(n) % 41  # 0〜40手（上限の 12手をまたぐ）
    x[:, P] = (moves.float() / EXP_OPP_CLOCK_SRC).half().float()[:, None, None]
    return x, moves


def clock_net(seed=0, extras=()):
    torch.manual_seed(seed)
    return PolicyValueNet(8, 1, NUM_PLANES_V6, extras, exp_opp_clock=True).eval()


def out(net, x):
    with torch.no_grad():
        return net(x)


def check_value():
    net = clock_net()
    assert net.config["exp_opp_clock"] is True and net.config["planes"] == NUM_PLANES_V6
    x, moves = obs_like(64, 1)
    # 同じ重みを、opp_counts を読む（= 139番をそのまま読む）ネットへ移し、139番を手で作り替えた入力を渡す
    plain = PolicyValueNet(8, 1, NUM_PLANES_V6, ["opp_counts"]).eval()
    plain.load_state_dict(net.state_dict())
    want = x.clone()
    want[:, P] = (moves.float() / EXP_OPP_CLOCK_DIV).clamp(max=1.0)[:, None, None]
    a, b = out(net, x), out(plain, want)
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-6)
    # 作り替えなければ違う出力（139番の値が 8倍ほど違う）
    c = out(plain, x)
    assert (a[0] - c[0]).abs().max() > 1e-3
    # 12手より先は上限に張り付く: 13手目以降の手数を入れ替えても出力は同じ
    y = x.clone()
    late = moves >= 12
    y[late, P] = 0.5
    d = out(net, y)
    assert torch.equal(a[0][late], d[0][late])
    # 入力は書き換えない
    x2, _ = obs_like(64, 1)
    assert torch.equal(x, x2)


def check_reads_only_clock():
    net = clock_net(seed=2, extras=["check_clock"])  # ほかの組と一緒でも同じ
    x, _ = obs_like(32, 3)
    base = out(net, x)
    off = net.in_off.view(-1).nonzero().view(-1).tolist()
    assert P not in off and all(i in off for i in range(COUNTS[0], P))
    y = x.clone()
    y[:, off] = torch.rand((32, len(off), 9, 9), generator=torch.Generator().manual_seed(4))
    other = out(net, y)
    assert torch.equal(base[0], other[0]) and torch.equal(base[1], other[1])
    assert (net.stem.weight[:, off] == 0).all() and (net.stem.weight[:, P] != 0).any()
    net.train()
    logits, v = net(x)
    (logits.sum() + v.sum()).backward()
    g = net.stem.weight.grad
    assert (g[:, off] == 0).all() and (g[:, P] != 0).any()


def check_export():
    net = clock_net(seed=5)
    x, _ = obs_like(16, 6)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "w.bin"
        write_weights(net, path)
        assert path.read_bytes()[:8] == MAGIC_PY
        back = read_weights(path).eval()
        assert back.config == net.config and back.exp_opp_clock
        a, b = out(net, x), out(back, x)
        assert torch.allclose(a[0], b[0], atol=1e-4) and torch.allclose(a[1], b[1], atol=1e-5)
        # magic とヘッダが食い違う重みは読まない（Python 専用の重みを元の magic に書き換えても Rust へ通さない）
        raw = path.read_bytes()
        path.write_bytes(MAGIC + raw[8:])
        try:
            read_weights(path)
        except ValueError:
            pass
        else:
            raise AssertionError("magic を書き換えた重みを読んだ")
        torch.manual_seed(7)
        write_weights(PolicyValueNet(8, 1, NUM_PLANES_V6, ["opp_counts"]).eval(), path)
        assert path.read_bytes()[:8] == MAGIC


def raises(f, exc=ValueError):
    try:
        f()
    except exc:
        return
    raise AssertionError("止まるはずの指定が通った")


def check_guards():
    raises(lambda: PolicyValueNet(8, 1, NUM_PLANES_V6, ["opp_counts"], exp_opp_clock=True))
    raises(lambda: PolicyValueNet(8, 1, NUM_PLANES_V5, exp_opp_clock=True))
    raises(lambda: PolicyValueNet(8, 1, 91, exp_opp_clock=True))
    net = clock_net()
    raises(lambda: pad_planes(net, NUM_PLANES_V6))
    ck = {"config": dict(net.config), "state_dict": net.state_dict()}
    raises(lambda: widen_checkpoint(ck, NUM_PLANES_V6, ["opp_counts"]))


def check_real_observation():
    try:
        import tsuitate_rl
    except ImportError:
        print("  （tsuitate_rl が無いので C5 は飛ばす）")
        return
    from rnad import collect
    torch.manual_seed(11)
    learner = PolicyValueNet(8, 1, NUM_PLANES_V6, ["check_clock"]).eval()
    env = tsuitate_rl.VecEnv(48, seed=21, auto_reset=False, planes=NUM_PLANES_V6)
    b = collect(env, learner, torch.device("cpu"), compact_legal=True)
    obs = b["obs"].float()
    moves = torch.round(obs[:, P, 0, 0] * EXP_OPP_CLOCK_SRC)
    clock = (moves / EXP_OPP_CLOCK_DIV).clamp(max=1.0)
    cc = EXTRA_PLANES["check_clock"][0]
    gave, since_gave = obs[:, cc, 0, 0] > 0.5, obs[:, cc + 1, 0, 0]
    got, since_got = obs[:, cc + 3, 0, 0] > 0.5, obs[:, cc + 4, 0, 0]
    assert (obs[:, P] == obs[:, P, :1, :1]).all(), "139番は全面に同じ値"
    for name, flag, since in (("かけた側", gave, since_gave), ("受けた側", got, since_got)):
        none = ~flag
        assert none.sum() > 200, (name, int(none.sum()))
        err = (since[none] - clock[none]).abs().max()
        assert err < 2e-3, (name, float(err))  # 観測は fp16（k/12 の丸め）
    assert (moves > 12).any() and (moves < 12).any()  # 上限の前後の手番がどちらも入っている
    assert gave.any() and got.any()
    # 王手があった後は別の値（王手から数え直す）= 時計だけの腕は王手の情報を運ばない
    assert (since_gave[gave] - clock[gave]).abs().max() > 0.05


def run_train(outdir, *extra, steps=2):
    cmd = [sys.executable, str(TRAIN), "--init", "none", "--anchor", "none", "--batch", "6", "--steps", str(steps),
           "--channels", "8", "--blocks", "1", "--planes", "91", "--compact-legal", "--eval-every", "0",
           "--export-every", "2", "--export-current", "--checkpoint-every", "2", "--out", str(outdir), "--threads", "2",
           "--device", "cpu", "--algo", "mmd", "--mmd-chunk", "64", "--league", "heuristic", "--league-frac", "1", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def check_train():
    with tempfile.TemporaryDirectory() as d:
        outdir = pathlib.Path(d) / "run"
        r = run_train(outdir, "--exp-opp-clock")
        assert r.returncode == 0, r.stderr[-2500:]
        ck = torch.load(outdir / "checkpoint.pt", map_location="cpu")
        cfg = ck["config"]
        assert cfg["exp_opp_clock"] is True and cfg["planes"] == NUM_PLANES_V6 and cfg["extras"] == []
        bins = sorted(outdir.glob("*.bin"))
        assert bins and all(p.read_bytes()[:8] == MAGIC_PY for p in bins), bins
        assert read_weights(bins[0]).exp_opp_clock
        # 続きはフラグを付けたまま回る。外した再開は止まる
        r = run_train(outdir, "--exp-opp-clock", steps=4)
        assert r.returncode == 0, r.stderr[-2500:]
        r = run_train(outdir, steps=6)
        assert r.returncode != 0 and "--exp-opp-clock" in r.stderr, r.stderr[-800:]
        # フラグなしで始めた run にフラグを付けた再開も止まる
        plain = pathlib.Path(d) / "plain"
        r = run_train(plain)
        assert r.returncode == 0, r.stderr[-2500:]
        r = run_train(plain, "--exp-opp-clock", steps=4)
        assert r.returncode != 0 and "--exp-opp-clock" in r.stderr, r.stderr[-800:]
        # 既存の重みからは始められない
        r = run_train(pathlib.Path(d) / "init", "--exp-opp-clock", "--init", str(plain / "checkpoint.pt"))
        assert r.returncode != 0 and "--exp-opp-clock" in r.stderr, r.stderr[-800:]
        # opp_counts の組とは一緒に使えない
        r = run_train(pathlib.Path(d) / "both", "--exp-opp-clock", "--opp-count-planes")
        assert r.returncode != 0 and "opp_counts" in r.stderr, r.stderr[-800:]


def main():
    for name, f in (("C1 入口で作る値", check_value), ("C2 読むのは 139番だけ", check_reads_only_clock),
                    ("C3 書き出し", check_export), ("C4 止める指定", check_guards),
                    ("C5 実際の観測", check_real_observation), ("C6 train_rnad.py", check_train)):
        f()
        print(f"ok {name}", flush=True)


if __name__ == "__main__":
    main()
