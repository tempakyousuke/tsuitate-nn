"""リーグの相手を GPU（PyTorch）で指させる経路の検査。tsuitate_rl が要る。

    .venv/bin/python rnad/tests/test_league_gpu.py

- export_weights.read: write で書いた TSRLPV01 を読み戻すと、元のネットと同じ出力（BatchNorm は恒等に置き換え）。
  本線の checkpoint と、そこから書き出した policy_<step>.bin が手元にあれば、それでも突き合わせる
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
from rnad import MultiEnv, collect  # noqa: E402

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
                    opponents=[None] * 4 + games, opp_nets={"opp": opp})
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
    check_collect_opponents()
