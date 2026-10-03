"""決着の速さのボーナス（tsuitate_rl.VecEnv の speed_bonus）が collect の報酬に正しく載るかと、
監視値の集計（rnad/speed.py）を検査する。tsuitate_rl（tsuitate-bot PR #65 以降）が要る。

    .venv/bin/python rnad/tests/test_speed.py

- 自己対局: 同じ乱数で λ=0 と λ>0 を回すと対局は同じで、局ごとの報酬の差は
  own = 勝った側に λ × max(0, (H − 手数) / H)、zero = さらに負けた側から同じ額を引く
- 相手固定（heuristic、リーグが使う opponents=[...]）: 学習側の列の報酬 = 勝敗 + （勝っていれば）λ × 速さ
- ボーナス込みの各局・各側の報酬を 1 + λ で割ると −1〜1 に収まる（train_rnad の reward_scale）
- summarize: 自己対局は決着した局をすべて学習側の勝ちとして数え、相手固定の局は勝ち負けを分ける
"""

import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import MultiEnv, collect  # noqa: E402
from speed import speed_of, summarize  # noqa: E402

LAM, H = 2.0, 120
DEV = torch.device("cpu")


def run(env):
    torch.manual_seed(0)
    model = PolicyValueNet(8, 1).eval()
    torch.manual_seed(1)
    batch = collect(MultiEnv([env]), model, DEV)
    fin = sorted(env.pop_finished(), key=lambda f: f["game_no"])
    assert len(fin) == batch["outcome"].shape[0]
    return batch, fin


def check_selfplay():
    plain, fin0 = run(tsuitate_rl.VecEnv(16, seed=3, auto_reset=False))
    for mode in ("own", "zero"):
        sped, fin1 = run(tsuitate_rl.VecEnv(16, seed=3, auto_reset=False, speed_bonus=LAM,
                                            speed_horizon=H, speed_sum=mode))
        assert [f["plies"] for f in fin0] == [f["plies"] for f in fin1], "ボーナスで対局が変わった"
        diff = sped["reward"].sum(dim=0) - plain["reward"].sum(dim=0)  # [B, 2]
        decided = 0
        for b, f in enumerate(fin1):
            want = [0.0, 0.0]
            if f["winner"] is not None:
                decided += 1
                bonus = LAM * speed_of(f["plies"], H)
                want[f["winner"]] = bonus
                if mode == "zero":
                    want[1 - f["winner"]] = -bonus
            assert torch.allclose(diff[b], torch.tensor(want), atol=1e-5), (mode, b, diff[b], want)
        assert decided > 0, "決着した局が無く、検査になっていない"
        ret = sped["reward"].sum(dim=0)
        assert (ret.abs() / (1.0 + LAM)).max() <= 1.0 + 1e-6
        print(f"OK: 自己対局 {mode}: 報酬の差 = 勝った側に λ × 速さ{'（負けた側から同額）' if mode == 'zero' else ''}"
              f"（決着 {decided}局）")


def check_league():
    names = ["heuristic"] * 6
    batch, fin = run(tsuitate_rl.VecEnv(6, opponents=names, seed=7, auto_reset=False,
                                        speed_bonus=LAM, speed_horizon=H))
    ret = batch["reward"].sum(dim=0)
    for b, f in enumerate(fin):
        me = f["learner"]
        want = 0.0 if f["winner"] is None else (1.0 if f["winner"] == me else -1.0)
        if f["winner"] == me:
            want += LAM * speed_of(f["plies"], H)
        assert abs(float(ret[b, me]) - want) < 1e-5, (b, float(ret[b, me]), want)
    print("OK: 相手固定（opponents=[heuristic]）で学習側の報酬 = 勝敗 + 勝ちなら λ × 速さ")


def check_summarize():
    fin = [{"winner": 0, "learner": None, "plies": 60, "reason": "checkmate"},
           {"winner": 1, "learner": None, "plies": 100, "reason": "foul_limit"},
           {"winner": None, "learner": None, "plies": 200, "reason": "max_plies"}]
    s = summarize(fin, 120)
    assert s["wins"] == 2 and s["draws"] == 1 and s["win_plies"] == 80.0 and s["win_mate"] == 0.5
    assert abs(s["win_speed"] - (0.5 + 20 / 120) / 2) < 1e-4
    fin = [{"winner": 0, "learner": 0, "plies": 60, "reason": "checkmate"},
           {"winner": 1, "learner": 0, "plies": 90, "reason": "checkmate"}]
    s = summarize(fin, 120)
    assert s["wins"] == 1 and s["losses"] == 1 and s["win_plies"] == 60 and s["loss_plies"] == 90
    assert summarize([], 120) == {}
    print("OK: summarize の勝ち負けの振り分け")


if __name__ == "__main__":
    check_summarize()
    check_selfplay()
    check_league()
