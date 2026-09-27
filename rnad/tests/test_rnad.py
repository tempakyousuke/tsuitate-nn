"""rnad.py を OpenSpiel の参照実装（JAX）の出力と突き合わせる。

フィクスチャ `openspiel_rnad_ref*.npz` は `make_reference.py` が参照実装（OpenSpiel
d1dcdf5 の rnad.py。R-NaD は 2025-05 に OpenSpiel から削除された）で乱数入力から計算したもの。
反則で同じプレイヤーが続けて指す並び・長さの違う局・引き分けを含む。

    .venv/bin/python rnad/tests/test_rnad.py
"""

import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rnad import EntropySchedule, legal_log_policy, legal_policy, loss_terms, v_trace_scalars  # noqa: E402


def check_fixture(path):
    f = np.load(path)
    T, B, A = int(f["T"]), int(f["B"]), int(f["A"])
    eta, beta, alpha = float(f["eta"]), float(f["beta"]), float(f["alpha"])
    valid = torch.from_numpy(f["valid"]).bool()
    player = torch.from_numpy(f["player"]).long()
    legal = torch.from_numpy(f["legal"]).bool()
    actions = torch.from_numpy(f["actions"])
    rewards = torch.from_numpy(f["rewards"])
    logit = torch.from_numpy(f["logit"]).clone().requires_grad_(True)
    v = torch.from_numpy(f["v"]).squeeze(-1).clone().requires_grad_(True)
    v_tgt = torch.from_numpy(f["v_tgt"]).squeeze(-1)
    mu = torch.from_numpy(f["mu"])

    # rnad.py の learner と同じ手順（1回目: 目標、2回目: 損失）
    with torch.no_grad():
        log_pi = legal_log_policy(logit, legal)
        mix = (alpha * legal_log_policy(torch.from_numpy(f["logit_prev"]), legal)
               + (1 - alpha) * legal_log_policy(torch.from_numpy(f["logit_prev_"]), legal))
        lpr = log_pi - mix
        pi = legal_policy(logit, legal)
        pi_a = pi.gather(-1, actions[..., None]).squeeze(-1)
        mu_a = torch.where(valid, mu.gather(-1, actions[..., None]).squeeze(-1), torch.ones(T, B))
        ent = (pi * lpr).sum(-1)
    vts, ss = [], []
    for k in (0, 1):
        vt_k, s_k = v_trace_scalars(v_tgt, valid, player, pi_a, mu_a, ent, rewards[:, :, k], k, eta)
        vts.append(vt_k)
        ss.append(s_k)
        # 価値の目標は参照の v_trace の出力と一致（k の手番のステップ）
        want = f[f"v_target{k}"].squeeze(-1)
        mine = (valid & (player == k)).numpy()
        np.testing.assert_allclose(vt_k.numpy()[mine], want[mine], rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(mine.astype(np.float32), f[f"has_played{k}"])

    vt = torch.where(player == 0, vts[0], vts[1])
    s = torch.where(player == 0, ss[0], ss[1])
    flat = valid.reshape(-1)
    p_flat = player.reshape(-1)[flat]
    counts = [int((p_flat == 0).sum()), int((p_flat == 1).sum())]
    lv, ln = loss_terms(logit.reshape(-1, A)[flat], v.reshape(-1)[flat], legal.reshape(-1, A)[flat],
                        actions.reshape(-1)[flat], p_flat, lpr.reshape(-1, A)[flat],
                        vt.reshape(-1)[flat], s.reshape(-1)[flat], counts, eta, beta)
    np.testing.assert_allclose(lv.item(), float(f["loss_v"]), rtol=1e-4)
    np.testing.assert_allclose(ln.item(), float(f["loss_nerd"]), rtol=1e-4, atol=1e-5)
    (lv + ln).backward()
    np.testing.assert_allclose(logit.grad.numpy(), f["g_logit"], rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(v.grad.numpy(), f["g_v"].squeeze(-1), rtol=1e-3, atol=1e-5)
    print(f"{path.name}: loss_v {lv.item():.5f} / loss_nerd {ln.item():.5f} と勾配が参照と一致")


def check_schedule():
    s = EntropySchedule([3, 5, 10], [2, 4, 1])
    assert list(s.schedule) == [0, 3, 6, 11, 16, 21, 26, 36]
    # 各繰り返しの最後のステップで差し替える
    updates = [t for t in range(60) if s(t)[1]]
    assert updates == [2, 5, 10, 15, 20, 25, 35, 45, 55], updates
    assert s(0)[0] == 0.0 and s(1)[0] == 2 / 3 and s(2)[0] == 1.0
    # 予定表の後ろ（36〜）は最後の大きさ 10 で繰り返す: 46 で α=0、50 で 0.8、51 で 1
    assert s(46)[0] == 0.0 and abs(s(50)[0] - 0.8) < 1e-12 and s(51)[0] == 1.0
    print("EntropySchedule: 参照と同じ差し替え位置と α")


if __name__ == "__main__":
    for p in sorted(HERE.glob("openspiel_rnad_ref*.npz")):
        check_fixture(p)
    check_schedule()
    print("OK")
