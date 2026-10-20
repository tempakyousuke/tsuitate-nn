"""合法手に詰めた更新（`RNaDLearner(packed=True)`）が従来の更新と同じ損失・勾配になることの検査。
tsuitate_rl が要る。

    .venv/bin/python rnad/tests/test_packed.py

- 実際の自己対局の batch で、従来・詰めた版の損失と全パラメータの勾配が浮動小数の誤差の範囲で一致する
  （α=0 と α>0 の両方。補助ヘッドつきも）
- 詰めた添字は行ごとに合法手を昇順に並べ、選んだ手の位置を指す。マスク外の手は止める
"""

import copy
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import RNaDConfig, RNaDLearner, collect, pack_legal  # noqa: E402

DEV = torch.device("cpu")


def check_pack():
    legal = torch.zeros(3, 10, dtype=torch.bool)
    legal[0, [1, 4, 7]] = True
    legal[1, [0]] = True
    legal[2, [2, 3, 5, 9]] = True
    action = torch.tensor([7, 0, 3])
    idx, pmask, apos = pack_legal(legal, action)
    assert idx.shape == (3, 4)
    assert idx[0, :3].tolist() == [1, 4, 7] and pmask[0].tolist() == [True, True, True, False]
    assert idx[1, :1].tolist() == [0] and pmask[1].sum() == 1
    assert idx[2].tolist() == [2, 3, 5, 9] and pmask[2].all()
    assert apos.tolist() == [2, 0, 1]
    try:
        pack_legal(legal, torch.tensor([6, 0, 3]))
    except ValueError:
        pass
    else:
        raise AssertionError("マスク外の手を通した")
    print("pack_legal: 並びと選んだ手の位置が正しい")


def grads(learner):
    out = [p.grad.detach().clone() for p in learner.net.parameters()]
    if learner.aux is not None:
        out += [p.grad.detach().clone() for p in learner.aux.parameters()]
    return out


def check_equal(step, aux):
    torch.manual_seed(0)
    net = PolicyValueNet(16, 2).eval()
    cfg = RNaDConfig(beta=16.0, lr=0.0, schedule_sizes=(10,), schedule_repeats=(1,), chunk=300,
                     aux_truth=(1.0 if aux else 0.0))
    base = RNaDLearner(net, cfg, DEV)
    # 正則化方策・ターゲットを現在のネットと違う値にする（同じだと lpr が 0 になり η の項を検査できない）
    with torch.no_grad():
        for m, scale in ((base.prev, 0.3), (base.prev_, -0.2), (base.target, 0.1)):
            for p in m.parameters():
                p.add_(scale * torch.randn_like(p) * p.abs().mean())
    base.step_count = step
    if aux:
        base._ensure_aux()  # 補助ヘッドは初回の update で作られるので、写す前に作って同じ重みにする
    packed = copy.deepcopy(base)
    packed.packed = True
    torch.manual_seed(1)
    batch = collect(tsuitate_rl.VecEnv(12, seed=3, auto_reset=False), base.net, DEV, truth=aux)
    a = base.update(batch)
    b = packed.update(batch)
    for k in ("loss_v", "loss_nerd", "v_target_mean", "s_abs_mean", "ratio_dev_max", "loss_aux"):
        if k in a:
            np.testing.assert_allclose(b[k], a[k], rtol=1e-4, atol=1e-6, err_msg=k)
    ga, gb = grads(base), grads(packed)
    worst = 0.0
    for x, y in zip(ga, gb):
        np.testing.assert_allclose(y.numpy(), x.numpy(), rtol=1e-3, atol=1e-6)
        worst = max(worst, float((x - y).abs().max() / (x.abs().max() + 1e-12)))
    print(f"step {step}（α {a['alpha']:.2f}）{'補助ヘッドつき ' if aux else ''}: 損失・勾配が一致"
          f"（標本 {a['samples']}、勾配の相対差 最大 {worst:.1e}）")


def check_skip_zero_weight():
    """α が 0 / 1 のとき重みが 0 の正則化ネットを順伝播しなくても、混ぜた値とビット単位で同じ"""
    torch.manual_seed(0)
    net = PolicyValueNet(8, 1).eval()
    learner = RNaDLearner(net, RNaDConfig(), DEV)
    with torch.no_grad():
        for p in learner.prev_.parameters():
            p.add_(0.3 * torch.randn_like(p))
    torch.manual_seed(1)
    batch = collect(tsuitate_rl.VecEnv(4, seed=5, auto_reset=False), net, DEV)
    obs = batch["obs"][:200].float()
    legal = batch["legal"][:200]
    from rnad import forward, legal_log_policy
    with torch.no_grad():
        lp = legal_log_policy(forward(learner.prev, obs)[0], legal)
        lp_ = legal_log_policy(forward(learner.prev_, obs)[0], legal)
        for alpha, want in ((1.0, 1.0 * lp + 0.0 * lp_), (0.0, 0.0 * lp + 1.0 * lp_)):
            got = learner._lpr_mix(obs, legal, alpha)
            assert torch.equal(got, want), alpha
    print("重み 0 の正則化ネットを省いても α 混合とビット単位で同じ")


if __name__ == "__main__":
    check_skip_zero_weight()
    check_pack()
    check_equal(0, aux=False)
    check_equal(3, aux=False)
    check_equal(3, aux=True)
