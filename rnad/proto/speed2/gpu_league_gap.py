"""F9（本題の外）: `--league-device gpu` の batch では、相手の手番が valid=False の隙間になる。v_trace_scalars は
無効なステップで carry を 0 に戻すので、学習側の手番の価値の目標 vt に終局の報酬が届かないことを確かめる
（docs/rl-learning-speed.md「さらに速くする」の 9.8。学習には使っていない）。

    python rnad/proto/speed2/gpu_league_gap.py

小さいネット（8ch×1）・η=0・v=0・π=μ で、局ごとに「vt が終局の報酬 G と一致した学習側の手番の数」を数える。
自己対局の局は全手番で vt=G、GPU リーグの局は最後の数手番だけになれば不具合が再現している
（作った環境での結果: 自己対局 4局は全手番、GPU リーグ 6局は 33〜72手番のうち 1〜3手番だけ）。
直したら（無効なステップで carry をそのまま通す案）、GPU リーグの局も全手番で vt=G になるはず
"""

import torch

import proto_common  # noqa: F401  （rnad を読めるようにする）
import tsuitate_rl  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import MultiEnv, OpponentPool, collect, v_trace_scalars  # noqa: E402


def main():
    torch.set_num_threads(2)
    dev = torch.device("cpu")
    torch.manual_seed(0)
    learner = PolicyValueNet(8, 1).eval()
    torch.manual_seed(1)
    opp = PolicyValueNet(8, 1).eval()
    selfplay = tsuitate_rl.VecEnv(4, seed=5, auto_reset=False)
    league = tsuitate_rl.VecEnv(6, seed=6, auto_reset=False)
    games = [("opp", i % 2) for i in range(6)]
    torch.manual_seed(2)
    batch = collect(MultiEnv([selfplay, league]), learner, dev, opponents=[None] * 4 + games,
                    opp_nets=OpponentPool({"opp": opp}))
    valid, player, reward = batch["valid"], batch["player"], batch["reward"]
    T, B = valid.shape
    v0 = torch.zeros(T, B)
    ones = torch.ones(T, B)
    ent0 = torch.zeros(T, B)
    gap = 0
    for b in range(B):
        k = int(player[valid[:, b], b][0]) if b >= 4 else 0
        vt, s = v_trace_scalars(v0, valid, player, ones, ones, ent0, reward[:, :, k], k, eta=0.0)
        steps = (valid[:, b] & (player[:, b] == k)).nonzero().squeeze(1)
        G = float(reward[:, b, k].sum())
        vts = vt[steps, b]
        hit = int((vts == G).sum())
        gap += b >= 4 and hit < len(steps)
        print(f"局{b} {'自己対局' if b < 4 else 'GPUリーグ'} 学習側{k}: 終局の報酬 {G:+.0f}, 手番 {len(steps)}, "
              f"vt=G の手番 {hit}, vt=0 の手番 {int((vts == 0).sum())}")
    print(f"GPU リーグの局のうち vt に終局の報酬が届かない手番がある局: {gap} / {B - 4}")


if __name__ == "__main__":
    main()
