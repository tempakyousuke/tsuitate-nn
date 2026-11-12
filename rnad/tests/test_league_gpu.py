"""リーグの相手を GPU（PyTorch）で指させる経路の検査。tsuitate_rl が要る。

    .venv/bin/python rnad/tests/test_league_gpu.py

- export_weights.read: write で書いた TSRLPV01 を読み戻すと、元のネットと同じ出力（BatchNorm は恒等に置き換え）。
  本線の checkpoint と、そこから書き出した policy_<step>.bin が手元にあれば、それでも突き合わせる
- OpponentPool: 積み重ねた推論が相手ごとの推論と一致する。入力の版・ブロック数・チャネル数が違う相手を混ぜても
  （形ごとの組に分けて積む）、並びに依らず一致する
- 推論にだけ使うネットの構築（read・pad_planes・OpponentPool・speedlab.Member）はグローバルの乱数を引かない
  （checkpoint の乱数状態を戻した後に作っても、中断なしの学習と再開した学習の対局がずれない）
- collect(opponents=...): 相手を固定した局は学習側の手番だけが記録され（valid・player）、学習側の列の報酬の
  合計が勝敗と一致する。相手の列には何も入らない。自己対局の局は従来どおり両方の手番が記録される
"""

import pathlib
import sys
import tempfile

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from export_weights import read, tiny_net, write  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import MultiEnv, OpponentPool, collect  # noqa: E402

DEV = torch.device("cpu")
ROOT = HERE.parent.parent


def same_outputs(a, b, x):
    with torch.no_grad():
        la, va = a.eval()(x)
        lb, vb = b.eval()(x)
    return float((la - lb).abs().max()), float((va - vb).abs().max())


def check_roundtrip():
    net = tiny_net(3)
    x = torch.rand((4, net.config["planes"], 9, 9), generator=torch.Generator().manual_seed(1))
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "w.bin"
        write(net, path)
        back = read(path)
    dl, dv = same_outputs(net, back, x)
    assert dl < 1e-4 and dv < 1e-5, (dl, dv)
    print(f"OK: tiny ネットの書き出し→読み戻しで出力が一致（ロジット差 {dl:.1e}・価値差 {dv:.1e}）")

    ck_path = ROOT / "out_rnad" / "rnad_gpu_b256" / "checkpoint_55000.pt"
    bin_path = ROOT / "out_rnad" / "rnad_gpu_b256" / "policy_55000.bin"
    if ck_path.exists() and bin_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        tgt = PolicyValueNet(ck["config"]["channels"], ck["config"]["blocks"], ck["config"]["planes"])
        tgt.load_state_dict(ck["target"])
        env = tsuitate_rl.VecEnv(8, seed=2, auto_reset=False)
        obs = torch.from_numpy(env.observe()[0]).float()
        dl, dv = same_outputs(tgt, read(bin_path), obs)
        assert dl < 1e-3 and dv < 1e-4, (dl, dv)
        print(f"OK: 本線 55,000更新の policy_55000.bin を読むとターゲットネットと一致（ロジット差 {dl:.1e}）")
    else:
        print("（本線の checkpoint_55000.pt / policy_55000.bin が無いので実物での突き合わせは省略）")


def check_pool():
    """OpponentPool の積み重ねた推論が、相手ごとに推論した結果と一致し、サンプルはマスク内に収まる"""
    nets = {f"n{i}": tiny_net(10 + i) for i in range(3)}
    pool = OpponentPool(nets)
    env = tsuitate_rl.VecEnv(7, seed=8, auto_reset=False)
    obs, mask, _ = env.observe()
    groups = {"n0": np.array([0, 1, 2]), "n2": np.array([3, 4, 5, 6])}  # n1 は局なし（埋めるだけ）
    m = 4
    x = torch.zeros((3, m, *obs.shape[1:]))
    for name, bs in groups.items():
        x[pool.index[name], :len(bs)] = torch.from_numpy(obs[bs]).half().float()
    with torch.no_grad():
        stacked = pool.forward(pool.params, pool.buffers, x)
        for name, bs in groups.items():
            want = nets[name].eval()(torch.from_numpy(obs[bs]).half().float())[0]
            got = stacked[pool.index[name], :len(bs)]
            assert float((got - want).abs().max()) < 1e-4, name
    acts = pool.sample(groups, obs, mask, DEV)
    for name, bs in groups.items():
        assert all(mask[b, a] for b, a in zip(bs, acts[name])), "マスク外の手を選んだ"
    print("OK: OpponentPool の積み重ねた推論 = 相手ごとの推論、サンプルはマスク内")


def check_pool_mixed_shapes():
    """ブロック数・チャネル数が違う相手を1つの pool に入れても、相手ごとの推論と一致する（どの並びでも）。
    以前は先頭のネットの構造で全員を推論していたので、先頭が浅いと深い相手の余分なブロックが黙って落ち、
    逆の並びでは KeyError、チャネル数が違うと形の不一致で落ちた。env は使わない"""
    torch.manual_seed(123)
    specs = {"one_block": (8, 1), "two_blocks": (8, 2), "wide": (16, 1), "one_block_b": (8, 1)}
    nets = {name: PolicyValueNet(ch, bl).eval() for name, (ch, bl) in specs.items()}
    x = torch.randn(len(nets), 3, 86, 9, 9).half().float()
    for order in (list(nets), list(reversed(nets)), ["two_blocks", "one_block", "one_block_b", "wide"]):
        pool = OpponentPool({n: nets[n] for n in order})
        assert len(pool.groups) == 3, [ks for ks, _ in pool.groups]  # (8, 1) の2体は同じ組に積む
        xs = torch.stack([x[list(nets).index(n)] for n in order])
        with torch.no_grad():
            stacked = pool.forward(pool.params, pool.buffers, xs)
            for name in order:
                k = pool.index[name]
                want = nets[name](xs[k])[0]
                assert float((stacked[k] - want).abs().max()) < 1e-4, (order, name)
        rng = np.random.default_rng(4)
        mask = rng.random((8, 11259)) < 0.01
        mask[:, 0] = True
        obs = rng.random((8, 86, 9, 9), dtype=np.float32)
        groups = {"one_block": np.array([0, 1]), "two_blocks": np.array([2, 3, 4]), "wide": np.array([5, 6, 7])}
        acts = pool.sample(groups, obs, mask, DEV)
        for name, bs in groups.items():
            assert all(mask[b, a] for b, a in zip(bs, acts[name])), "マスク外の手を選んだ"
    # 形が全員同じなら組は1つ（従来どおり1回の推論）
    assert len(OpponentPool({"a": tiny_net(1), "b": tiny_net(2)}).groups) == 1
    print("OK: ブロック数・チャネル数の違う相手を1つの pool に入れても、相手ごとの推論と一致する（並びに依らない）")


def check_rng_untouched():
    """推論にだけ使うネットの構築（相手の重みの読込 read・入力を広げる pad_planes・OpponentPool・パネルの相手）が
    CPU のグローバル乱数を引かない。学習の再開は checkpoint の乱数状態を戻した後でリーグの相手・パネルを作るので、
    ここで乱数を引くと中断なしの学習と再開した学習の対局がずれる（--league-device gpu を CPU で回したとき）。env は使わない"""
    import speedlab
    from model import NUM_PLANES_DEDUCED, NUM_PLANES_V4
    from rnad import pad_planes
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "opp.bin"
        write(tiny_net(5), path)
        torch.manual_seed(42)
        state = torch.get_rng_state()
        net86 = PolicyValueNet(8, 1).eval()  # 学習するネットの初期化は乱数を引く（検査の前提）
        assert not torch.equal(torch.get_rng_state(), state)
        state = torch.get_rng_state()
        a = read(path)
        wide = pad_planes(net86, NUM_PLANES_V4)
        pool = OpponentPool({"a": a, "wide": wide, "v1": net86, "deep": read(path)})
        speedlab.Member("random", NUM_PLANES_DEDUCED)
        speedlab.Member(f"opp={path}", NUM_PLANES_DEDUCED)
        assert torch.equal(torch.get_rng_state(), state), "推論用のネットの構築がグローバルの乱数を引いた"
        x = torch.rand(4, NUM_PLANES_V4, 9, 9)
        with torch.no_grad():
            assert float((wide(x)[0] - net86(x)[0]).abs().max()) < 1e-5  # 乱数を引かなくても同じ関数
            assert float((a(x)[0] - tiny_net(5).eval()(x)[0]).abs().max()) < 1e-4
        assert len(pool.names) == 4
    print("OK: 相手の重みの読込・pad_planes・OpponentPool・パネルの相手の構築はグローバルの乱数を引かない")


def check_resume_cpu_league():
    """`--device cpu --league-device gpu`（相手も PyTorch の CPU 推論）の学習: 3更新を続けて回したものと、1更新で止めて
    同じ出力先で3更新まで再開したものの最終 checkpoint のネットが一致する（相手の重みの読込が、checkpoint から戻した
    乱数状態をずらさない）"""
    import subprocess
    train = HERE.parent / "train_rnad.py"
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        opp = tmp / "opp.bin"
        torch.manual_seed(0)
        write(PolicyValueNet(8, 1).eval(), opp)
        common = ["--init", "none", "--anchor", "none", "--channels", "8", "--blocks", "1", "--batch", "2",
                  "--eval-every", "0", "--export-every", "0", "--checkpoint-every", "0", "--threads", "1",
                  "--device", "cpu", "--seed", "3", "--league", f"rl_policy:{opp}", "--league-frac", "1",
                  "--league-device", "gpu", "--packed", "--compact-legal"]

        def run(out, steps):
            cmd = [sys.executable, str(train), *common, "--steps", str(steps), "--out", str(out)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            assert r.returncode == 0, f"train_rnad.py が落ちた: {' '.join(cmd)}\n{r.stderr[-2000:]}"

        run(tmp / "a", 3)
        run(tmp / "b", 1)
        run(tmp / "b", 3)
        ca = torch.load(tmp / "a" / "checkpoint.pt", map_location="cpu", weights_only=False)
        cb = torch.load(tmp / "b" / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert ca["net"].keys() == cb["net"].keys()
        bad = [k for k in ca["net"] if not torch.equal(ca["net"][k], cb["net"][k])]
        assert not bad, f"続けて回した学習と再開した学習のネットが違う: {bad[:5]}（{len(bad)} 個）"
    print("OK: CPU で相手を PyTorch で指させる学習を1更新で止めて再開しても、続けて回したものとネットが一致する")


def check_pool_mixed_planes():
    """入力の版が違う相手（v1 86枚・v2 91枚・v3 101枚・v4 102枚〔組の選び方も違う〕）を1つの pool に積んでも、
    相手ごとの推論と一致する。v3 / v4 の組のプレーンには 0 でない値を入れて確かめる（初期局面では全部 0 で、
    組を読まなくなる間違いが見えない）"""
    from model import NUM_PLANES_DEDUCED, NUM_PLANES_V3, NUM_PLANES_V4
    from rnad import pad_planes
    for top, specs in [(NUM_PLANES_V3, [(86, ()), (NUM_PLANES_DEDUCED, ()), (NUM_PLANES_V3, ("opp_hand",)),
                                        (NUM_PLANES_V3, ("opp_hand", "opp_fouls"))]),
                       (NUM_PLANES_V4, [(NUM_PLANES_DEDUCED, ()), (NUM_PLANES_V3, ("opp_hand",)),
                                        (NUM_PLANES_V4, ("king_fouls",)), (NUM_PLANES_V4, ("opp_hand", "king_fouls"))])]:
        _check_pool_mixed(top, specs, pad_planes)
    print("OK: 入力の版が違う相手（86 / 91 / 101 / 102枚）を1つの pool に積んでも、相手ごとの推論と一致する")


def _check_pool_mixed(top, specs, pad_planes):
    NUM_PLANES_V3 = top
    nets = {}
    for i, (planes, extras) in enumerate(specs):
        torch.manual_seed(20 + i)
        nets[f"n{i}"] = PolicyValueNet(8, 1, planes, extras).eval()
    pool = OpponentPool(nets)
    assert pool.planes == NUM_PLANES_V3
    env = tsuitate_rl.VecEnv(8, seed=9, auto_reset=False, planes=NUM_PLANES_V3)
    obs, mask, _ = env.observe()
    obs = obs.copy()
    obs[:, 91:] = np.random.default_rng(3).random(obs[:, 91:].shape, dtype=np.float32)
    x = torch.from_numpy(obs).half().float()
    assert float(x[:, 91:].abs().sum()) > 0.0
    groups = {name: np.array([2 * i, 2 * i + 1]) for i, name in enumerate(nets)}
    xs = torch.zeros((len(nets), 2, *obs.shape[1:]))
    for name, bs in groups.items():
        xs[pool.index[name]] = x[bs]
    with torch.no_grad():
        stacked = pool.forward(pool.params, pool.buffers, xs)
        for name, bs in groups.items():
            want = nets[name](x[bs])[0]
            got = stacked[pool.index[name]]
            assert float((got - want).abs().max()) < 1e-4, name
            # 広げたネット単体でも同じ（価値も）
            lw, vw = pad_planes(nets[name], NUM_PLANES_V3).eval()(x[bs])
            l0, v0 = nets[name](x[bs])
            assert float((lw - l0).abs().max()) < 1e-5 and float((vw - v0).abs().max()) < 1e-6, name
    acts = pool.sample(groups, obs, mask, DEV)
    for name, bs in groups.items():
        assert all(mask[b, a] for b, a in zip(bs, acts[name]))
    try:
        pad_planes(nets["n3"], 91)
    except ValueError:
        pass
    else:
        raise AssertionError("狭める指定が通った")


def check_collect_opponents():
    torch.manual_seed(0)
    learner = PolicyValueNet(8, 1).eval()
    torch.manual_seed(1)
    opp = PolicyValueNet(8, 1).eval()
    selfplay = tsuitate_rl.VecEnv(4, seed=5, auto_reset=False)
    league = tsuitate_rl.VecEnv(6, seed=6, auto_reset=False)
    games = [("opp", i % 2) for i in range(6)]
    torch.manual_seed(2)
    batch = collect(MultiEnv([selfplay, league]), learner, DEV,
                    opponents=[None] * 4 + games, opp_nets=OpponentPool({"opp": opp}))
    fin = sorted(league.pop_finished(), key=lambda f: f["game_no"])
    assert len(fin) == 6
    valid, player, reward = batch["valid"], batch["player"], batch["reward"]
    for j, ((_, color), f) in enumerate(zip(games, fin)):
        b = 4 + j
        steps = valid[:, b].nonzero().squeeze(1)
        assert len(steps) > 0
        assert (player[steps, b] == color).all(), "相手の手番が記録された"
        want = 0.0 if f["winner"] is None else (1.0 if f["winner"] == color else -1.0)
        got = float(reward[:, b, color].sum())
        assert got == want, (j, got, want)
        assert float(reward[:, b, 1 - color].abs().sum()) == 0.0, "相手の列に報酬が入った"
        # 報酬は学習側の手番にだけ置かれる（相手の手で終局しても学習側の直前の手番へ）
        nz = reward[:, b, color].nonzero().squeeze(1)
        assert all(bool(valid[int(t), b]) and int(player[int(t), b]) == color for t in nz)
    assert set(player[valid[:, 0], 0].tolist()) == {0, 1}, "自己対局の局で片側しか記録されていない"
    print(f"OK: 自己対局4局 + GPU 相手6局（学習側 {[c for _, c in games]}）: 相手の手番は記録されず、"
          "学習側の報酬 = 勝敗")


if __name__ == "__main__":
    check_roundtrip()
    check_pool()
    check_pool_mixed_shapes()
    check_rng_untouched()
    check_pool_mixed_planes()
    check_collect_opponents()
    check_resume_cpu_league()
