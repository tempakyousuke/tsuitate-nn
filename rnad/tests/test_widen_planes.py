"""入力プレーンを広げる（widen_planes.py、エンコーダ v1 → v2）の検査。tsuitate_rl は要らない。

    .venv/bin/python rnad/tests/test_widen_planes.py

- 広げた checkpoint の4つのネットは、v2 の観測（後ろのプレーンに何が入っていても）に対して元と同じ出力
- v1 のネットに v2 の観測を渡すと先頭の 86枚だけを読む（錨・リーグの相手が v1 のまま使える）
- optimizer の状態も広がって読み込め、足した列の 2次モーメントは 0 でない
"""

import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from model import NUM_PLANES, NUM_PLANES_DEDUCED, PolicyValueNet  # noqa: E402
from rnad import RNaDConfig, RNaDLearner  # noqa: E402
from widen_planes import widen_checkpoint  # noqa: E402

DEV = torch.device("cpu")
CFG = RNaDConfig(eta=0.2, beta=16.0, lr=2e-4, schedule_sizes=(200,), schedule_repeats=(1,))


def learner(planes=NUM_PLANES):
    torch.manual_seed(0)
    lr = RNaDLearner(PolicyValueNet(8, 1, planes), CFG, DEV)
    # optimizer の状態を作る（1回だけ適当な損失で進める）
    x = torch.rand(4, planes, 9, 9)
    logits, v = lr.net(x)
    lr.opt.zero_grad()
    (logits.square().mean() + v.square().mean()).backward()
    lr.opt.step()
    return lr


def main():
    base = learner()
    ck = base.state_dict()
    wide = widen_checkpoint(ck)
    assert wide["config"]["planes"] == NUM_PLANES_DEDUCED and ck["config"]["planes"] == NUM_PLANES
    other = RNaDLearner(PolicyValueNet(8, 1, NUM_PLANES_DEDUCED), CFG, DEV)
    other.load_state_dict(wide)

    torch.manual_seed(1)
    x = torch.rand(16, NUM_PLANES_DEDUCED, 9, 9) * 2
    for name in ("net", "target", "prev", "prev_"):
        a, b = getattr(base, name).eval(), getattr(other, name).eval()
        with torch.no_grad():
            la, va = a(x[:, :NUM_PLANES])
            lb, vb = b(x)
            lc, vc = a(x)  # v1 のネットに v2 の観測 = 先頭だけ読む
        assert torch.allclose(la, lb, atol=1e-5) and torch.allclose(va, vb, atol=1e-6), name
        assert torch.equal(la, lc) and torch.equal(va, vc), name

    st = other.opt.state_dict()["state"][0]
    assert st["exp_avg_sq"].shape[1] == NUM_PLANES_DEDUCED
    assert (st["exp_avg_sq"][:, NUM_PLANES:] > 0).all()
    assert torch.equal(st["exp_avg"][:, NUM_PLANES:], torch.zeros_like(st["exp_avg"][:, NUM_PLANES:]))
    # 広げた後も更新できる（形が optimizer と合っている）
    logits, v = other.net(x)
    other.opt.zero_grad()
    (logits.square().mean() + v.square().mean()).backward()
    other.opt.step()

    try:
        widen_checkpoint(wide)
    except ValueError:
        pass
    else:
        raise AssertionError("同じプレーン数へ広げられた")
    print("ok")


if __name__ == "__main__":
    main()
