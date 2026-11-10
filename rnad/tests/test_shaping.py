"""報酬の整形（rnad/shaping.py）と、相手を固定した学習の配線の検査（tsuitate-bot の docs/rl-fixed-opponent.md）。
tsuitate_rl（VecEnv.signals のある rl-env）が要る。

    .venv/bin/python rnad/tests/test_shaping.py

- 重みの書式（自分 / opp_ / d_）と、知らない鍵・重複の拒否
- ShapeRun（合成の信号）: plain は 1局の合計が clip(Φ(終局) − Φ(初期)) × 倍率、potential は合計 0、負けの長さは負けた側だけ、
  終局した局には以後何も足さない
- v-trace: 局の途中の無効な行（相手の手番を記録しない局の隙間）で carry が通る = 隙間を詰めた batch と同じ値。
  自己対局の形（隙間なし・後ろだけ埋め草）の値は変わらない
- collect: 整形を付けても対局（行動・勝敗）は付けないときと同じ。学習側の報酬の合計 = 勝敗 + 整形の合計
  （Rust の評価モード / 相手のネットに指させる形 / 自己対局）。potential は勝敗だけと合計が同じ
- MMD: 相手を固定した batch で λ-return が終局まで届く。reward_scale で報酬と正則化が同じ倍率で縮む
- ハンデつきの相手（<名前>@<ε>）: ε = 0 は元の相手と同じ局、ε = 1 の相手は反則しない（合法手の乱数手）。
  Rust の評価モードと、相手のネットに指させる形の両方
- train_rnad.py: --algo mmd / R-NaD とも --league ＋ --shape ＋ --lose-len で回り、log に signals が出る
"""

import json
import math
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from mmd import MMDLearner, make_config, player_lambda_returns  # noqa: E402
from model import NUM_PLANES_DEDUCED, PolicyValueNet  # noqa: E402
from rnad import MultiEnv, collect, split_noise, v_trace_scalars  # noqa: E402
from shaping import SNAP_PLIES, ShapeRun, Shaping, parse_weights, summarize  # noqa: E402

DEV = torch.device("cpu")
TRAIN = HERE.parent / "train_rnad.py"
NAMES = list(tsuitate_rl.SIGNAL_NAMES)
K = len(NAMES)


def check_parse():
    own, opp = parse_weights("d_capture_value=0.02, fouls=-0.03,opp_fouls=0.01", NAMES)
    cv, fo = NAMES.index("capture_value"), NAMES.index("fouls")
    assert own[cv] == 0.02 and opp[cv] == -0.02 and own[fo] == -0.03 and opp[fo] == 0.01
    assert np.count_nonzero(own) == 2 and np.count_nonzero(opp) == 2
    for bad, msg in (("nope=1", "知らない"), ("fouls", "鍵=重み"), ("fouls=x", "数でない"), ("fouls=1,fouls=2", "2回")):
        try:
            parse_weights(bad, NAMES)
        except ValueError as e:
            assert msg in str(e), (bad, e)
        else:
            raise AssertionError(f"{bad} が通った")
    sh = Shaping.from_spec("d_capture_value=0.02", "potential", 0.5, names=NAMES)
    assert sh.zero_sum and sh.reward_scale == 1.5 and sh.multiplier(10 ** 6) == 1.0
    sh = Shaping.from_spec("fouls=-0.03", lose_len="0.5:100", decay="10:30", names=NAMES)
    assert not sh.zero_sum and sh.reward_scale == 2.5
    assert [sh.multiplier(s) for s in (0, 10, 20, 30, 99)] == [1.0, 1.0, 0.5, 0.0, 0.0]
    assert Shaping.from_spec("", lose_len="0.3", names=NAMES).reward_scale == 1.3
    for kw in ({"lose_len": "1.0"}, {"decay": "5"}, {"decay": "9:3"}, {"mode": "x"}, {"cap": 0.0}):
        try:
            Shaping.from_spec("fouls=-1", names=NAMES, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} が通った")
    print("S1 重みの書式: 自分 / opp_ / d_ を読み、知らない鍵・重複・範囲外の設定を止める")


def synth(B, steps, seed):
    """合成の信号の列 [steps + 1, B, 2, K]（累積の信号は増えるだけ）と、局ごとの終局の step"""
    rng = np.random.default_rng(seed)
    inc = rng.integers(0, 3, size=(steps, B, 2, K)).astype(np.float32)
    sig = np.concatenate([np.zeros((1, B, 2, K), np.float32), np.cumsum(inc, 0)])
    sig += rng.normal(size=(1, B, 2, K)).astype(np.float32)  # 初期局面の値（状態の信号は 0 でない）
    end = rng.integers(2, steps + 1, size=B)
    return sig, end


def run_synth(sh, sig, end, mult=1.0, loser=None):
    steps, B = len(sig) - 1, sig.shape[1]
    run = ShapeRun(NAMES, sig[0], sh, mult)
    total = np.zeros((B, 2))
    cur = sig[0].copy()
    for t in range(1, steps + 1):
        alive = end >= t
        cur = np.where(alive[:, None, None], sig[t], cur)  # 終局した局の信号は止まる（auto_reset=False）
        done = end == t
        rew = np.zeros((B, 2), np.float32)
        if loser is not None:
            rew[done, loser[done]] = -1.0
            rew[done, 1 - loser[done]] = 1.0
        add = run.step(cur, done, rew)
        assert (add[~alive] == 0).all(), "終局した局に足している"
        total += add
    return run, total


def check_shape_run():
    B, steps = 32, 12
    sig, end = synth(B, steps, 0)
    final = sig[end, np.arange(B)]
    sh = Shaping.from_spec("d_capture_value=0.05,fouls=-0.04,opp_checks=0.03", "plain", 0.6, names=NAMES)
    want = np.clip(sh.potential(final) - sh.potential(sig[0]), -0.6, 0.6)
    run, total = run_synth(sh, sig, end)
    assert np.allclose(total, want, atol=1e-5), np.abs(total - want).max()
    assert np.allclose(run.total, total, atol=1e-5)
    assert (np.abs(want) >= 0.6 - 1e-9).any() and (np.abs(want) < 0.6).any(), "上限に当たる局と当たらない局の両方が要る"
    assert np.array_equal(run.sig, final), "終局の信号が残っていない"
    # 倍率
    _, half = run_synth(sh, sig, end, mult=0.5)
    assert np.allclose(half, 0.5 * want, atol=1e-5)
    # potential: 1局の合計は 0。途中の報酬は plain と同じで、終局の遷移だけが累計の差し戻しになる
    shp = Shaping.from_spec("d_capture_value=0.05,fouls=-0.04,opp_checks=0.03", "potential", 0.6, names=NAMES)
    _, total_p = run_synth(shp, sig, end)
    assert np.allclose(total_p, 0.0, atol=1e-5), np.abs(total_p).max()
    # 負けの長さ: 負けた側だけ λ × min(手数, H) / H
    rng = np.random.default_rng(1)
    loser = rng.integers(0, 2, size=B)
    shl = Shaping.from_spec("", lose_len="0.5:10", names=NAMES)
    mv = NAMES.index("moves")
    plies = final[:, 0, mv] + final[:, 1, mv]
    _, total_l = run_synth(shl, sig, end, loser=loser)
    want_l = np.zeros((B, 2))
    want_l[np.arange(B), loser] = 0.5 * np.minimum(plies, 10) / 10
    assert np.allclose(total_l, want_l, atol=1e-5)
    assert (plies > 10).any() and (plies < 10).any()
    # 監視値: 終局と途中（SNAP_PLIES）の信号を、学習側から見た並び [自分, 相手] で、勝ち負けに分ける
    outcome = np.zeros((B, 2), np.float32)
    outcome[np.arange(B), loser] = -1.0
    outcome[np.arange(B), 1 - loser] = 1.0
    run, _ = run_synth(shl, sig, end, loser=loser)
    learner = np.where(np.arange(B) % 2 == 0, 0, 1)
    s = summarize(run, outcome, learner)
    won = outcome[np.arange(B), learner] > 0
    assert s["win"]["final"]["n"] == int(won.sum()) and s["loss"]["final"]["n"] == int((~won).sum())
    own = final[np.arange(B), learner]
    assert np.allclose(s["win"]["final"]["mean"][:K], own[won].mean(0), atol=1e-3)
    assert np.allclose(s["loss"]["final"]["mean"][K:2 * K], final[np.arange(B), 1 - learner][~won].mean(0), atol=1e-3)
    diff = own - final[np.arange(B), 1 - learner]
    assert np.allclose(s["win"]["final"]["mean"][2 * K:], diff[won].mean(0), atol=1e-3)
    assert np.allclose(s["win"]["final"]["sd"][2 * K:], diff[won].std(0), atol=1e-3)
    assert set(s["win"]["at"]) == {str(t) for t in SNAP_PLIES} and s["draw"]["final"]["n"] == 0
    # 自己対局（学習側なし）は両方の側を1標本ずつ数える
    s2 = summarize(run, outcome, None)
    assert s2["win"]["final"]["n"] == B and s2["loss"]["final"]["n"] == B
    assert abs(s2["shaped_total"]["mean"] - total_l.mean()) < 1e-6
    print("S2 ShapeRun: plain の合計 = clip(Φ(終局) − Φ(初期)) × 倍率、potential の合計 = 0、負けの長さは負けた側だけ、"
          "終局した局には足さない。監視値は学習側から見た並びで勝ち負けに分かれる")


def check_vtrace_gap():
    """局の途中の無効な行で carry が通る: 相手の手番を記録しない局（隙間あり）= 隙間を詰めた batch"""
    g = torch.Generator().manual_seed(0)
    T, B = 14, 6
    # 詰めた形: 局 b は学習側（色 b % 2）の行が n_b 個続く
    n = torch.tensor([14, 9, 5, 12, 3, 7])
    valid = torch.arange(T)[:, None] < n[None, :]
    player = (torch.arange(B) % 2)[None, :].expand(T, B).clone()
    v = torch.rand(T, B, generator=g) * 2 - 1
    pi = torch.rand(T, B, generator=g) * 0.8 + 0.1
    mu = torch.rand(T, B, generator=g) * 0.8 + 0.1
    ent = torch.rand(T, B, generator=g)
    rew = (torch.rand(T, B, 2, generator=g) - 0.5) * valid[:, :, None]
    # 隙間あり: 各行の後ろに 0〜2 個の無効な行をはさむ（相手の手番。報酬は学習側の行に載せたまま）
    gaps = torch.randint(0, 3, (T, B), generator=g)
    T2 = int((valid * (1 + gaps)).sum(0).max())
    pos = torch.zeros(T, B, dtype=torch.long)
    for b in range(B):
        t2 = 0
        for t in range(int(n[b])):
            pos[t, b] = t2
            t2 += 1 + int(gaps[t, b])

    def spread(x, fill=0.0):
        out = torch.full((T2, B) + x.shape[2:], fill, dtype=x.dtype)
        for b in range(B):
            out[pos[:int(n[b]), b], b] = x[:int(n[b]), b]
        return out

    valid2 = spread(valid, False)
    assert int(valid2.sum()) == int(valid.sum()) and T2 > T
    for eta, c, cap in ((0.0, 1.0, 0.0), (0.2, 1.0, 0.0), (0.0, 0.6, 0.0), (0.2, 0.8, 100.0)):
        for k in (0, 1):
            vt, s = v_trace_scalars(v, valid, player, pi, mu, ent, rew[:, :, k], k, eta, c=c, inv_mu_cap=cap)
            vt2, s2 = v_trace_scalars(spread(v), valid2, spread(player), spread(pi, 1.0), spread(mu, 1.0), spread(ent),
                                      spread(rew)[:, :, k], k, eta, c=c, inv_mu_cap=cap)
            assert torch.equal(spread(vt), vt2) and torch.equal(spread(s), s2), (eta, c, k)
    # η=0・π=μ・c=1 なら vt は終局までの報酬の和（学習側の色の列）
    ones = torch.ones(T2, B)
    for k in (0, 1):
        vt2, _ = v_trace_scalars(spread(v), valid2, spread(player), ones, ones, torch.zeros(T2, B), spread(rew)[:, :, k], k, 0.0)
        r2 = spread(rew)[:, :, k]
        togo = torch.flip(torch.cumsum(torch.flip(r2, [0]), 0), [0])
        mine = valid2 & (spread(player) == k)
        assert torch.allclose(vt2[mine], togo[mine], atol=1e-5)
    print("S3 v-trace: 局の途中の無効な行で carry が通り、隙間のある batch が隙間を詰めた batch とビット単位で同じ値になる")


def small_net(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(8, 1, NUM_PLANES_DEDUCED).eval()


def run_collect(env, net, seed=1, **kw):
    torch.manual_seed(seed)
    return collect(MultiEnv([env]), net, DEV, compact_legal=True, **kw)


def learner_sum(batch, color):
    """局ごとの、学習側の列の報酬の合計 [B]"""
    r = batch["reward"]
    return np.array([float(r[:, b, int(color[b])].sum()) for b in range(r.shape[1])])


def check_env_signals():
    env = tsuitate_rl.VecEnv(8, seed=0, auto_reset=False, planes=NUM_PLANES_DEDUCED)
    sig = env.signals()
    assert sig.shape == (8, 2, K) and sig.dtype == np.float32
    assert (sig[:, 0] == sig[:, 1]).all(), "初期局面の信号は先後で同じ"
    assert (sig[:, :, NAMES.index("material")] == 60.5).all()
    net = small_net()
    b = run_collect(env, net, watch_signals=True)
    run = b["shape_run"]
    fin = sorted(env.pop_finished(), key=lambda f: f["game_no"])
    mv, fo = NAMES.index("moves"), NAMES.index("fouls")
    for i, f in enumerate(fin):
        assert run.sig[i, :, mv].sum() == f["plies"], (i, f)
        assert list(run.sig[i, :, fo]) == f["fouls"], (i, f)
    print(f"S4 env の信号: [局, 2, {K}]、初期局面は先後で同じ、終局の moves / fouls は pop_finished の手数・反則数と一致")


def check_collect():
    net = small_net()
    spec = "d_capture_value=0.03,fouls=-0.02,opp_fouls=0.01,d_king_attack=0.01"

    def envs():
        return tsuitate_rl.VecEnv(16, opponent="heuristic", seed=3, auto_reset=False, planes=NUM_PLANES_DEDUCED)

    env = envs()
    base = run_collect(env, net)
    color = env.learner_colors()
    z = base["outcome"][np.arange(16), color]
    assert np.allclose(learner_sum(base, color), z)
    for mode in ("plain", "potential"):
        sh = Shaping.from_spec(spec, mode, 0.5, lose_len="0.4:150", names=NAMES)
        env = envs()
        b = run_collect(env, net, shaping=sh, shape_mult=0.5)
        # 整形は乱数を使わず、対局も変えない
        for key in ("action", "mu_a", "t", "b", "valid", "player", "legal_cols", "legal_counts"):
            assert torch.equal(b[key], base[key]), key
        assert np.array_equal(b["outcome"], base["outcome"])
        run = b["shape_run"]
        tot = run.total[np.arange(16), color]
        assert np.allclose(learner_sum(b, color), z + tot, atol=1e-5)
        mv = NAMES.index("moves")
        lose = 0.5 * 0.4 * np.minimum(run.sig[:, :, mv].sum(1), 150) / 150 * (z < 0)
        phi = np.clip(sh.potential(run.sig) - run.phi0, -0.5, 0.5)[np.arange(16), color] * 0.5
        want = lose + (phi if mode == "plain" else 0.0)
        assert np.allclose(tot, want, atol=1e-5), (mode, np.abs(tot - want).max())
        assert (np.abs(learner_sum(b, color)) <= sh.reward_scale + 1e-6).all()
    # 自己対局: 反対称な重みなら両者の報酬の和は各時刻で 0（零和のまま）
    sh = Shaping.from_spec("d_capture_value=0.03,d_checks=0.02", "plain", 0.5, names=NAMES)
    env = tsuitate_rl.VecEnv(12, seed=5, auto_reset=False, planes=NUM_PLANES_DEDUCED)
    b = run_collect(env, net, shaping=sh)
    assert float(b["reward"].sum(-1).abs().max()) < 1e-6
    assert float(b["reward"].abs().sum()) > 12 * 2, "整形が載っていない"
    # 相手のネットに指させる形（--league-device gpu）: 学習側の列だけが、学習側の手番へ載る
    opp = small_net(7)
    games = [("opp", i % 2) for i in range(12)]
    sh = Shaping.from_spec(spec, "plain", 0.5, lose_len="0.4:150", names=NAMES)
    env = tsuitate_rl.VecEnv(12, seed=6, auto_reset=False, planes=NUM_PLANES_DEDUCED)
    torch.manual_seed(2)
    bg = collect(MultiEnv([env]), net, DEV, compact_legal=True, opponents=games, opp_nets={"opp": opp}, shaping=sh,
                 actor_extras=True)
    color = np.array([c for _, c in games])
    zg = bg["outcome"][np.arange(12), color]
    assert np.allclose(learner_sum(bg, color), zg + bg["shape_run"].total[np.arange(12), color], atol=1e-5)
    other = np.array([float(bg["reward"][:, i, 1 - color[i]].abs().sum()) for i in range(12)])
    assert (other == 0).all(), "相手の列に報酬が載っている"
    assert (bg["player"][bg["t"], bg["b"]].numpy() == color[bg["b"].numpy()]).all()
    assert len(bg["v_actor"]) == len(bg["action"])
    print("S5 collect: 整形を付けても対局は同じ。学習側の報酬の合計 = 勝敗 + 整形の合計（Rust の評価モード・相手のネットに"
          "指させる形）、potential は負けの長さだけが残る、自己対局の反対称な重みは零和のまま")
    return net, bg, color


def check_mmd(net, bg, color):
    """相手を固定した batch で λ-return が終局まで届く（λ=1 なら、その手番から後ろの学習側の報酬の和）"""
    n = len(bg["action"])
    g = player_lambda_returns(bg, torch.zeros(n), 1.0)
    r = bg["reward"]
    t_idx, b_idx = bg["t"].numpy(), bg["b"].numpy()
    for i in range(n):
        want = float(r[t_idx[i]:, b_idx[i], color[b_idx[i]]].sum())
        assert abs(float(g[i]) - want) < 1e-4, (i, float(g[i]), want)
    # λ < 1: 素直なループ（学習側の行だけを順に）
    v = torch.rand(n, generator=torch.Generator().manual_seed(0)) * 2 - 1
    g5 = player_lambda_returns(bg, v, 0.5)
    for b in range(bg["valid"].shape[1]):
        rows = np.flatnonzero(b_idx == b)
        nxt_g, nxt_v, nxt_t = 0.0, 0.0, bg["valid"].shape[0]
        for i in rows[::-1]:
            ri = float(r[t_idx[i]:nxt_t, b, color[b]].sum())
            gi = ri + (0.5 * nxt_v + 0.5 * nxt_g if nxt_t < bg["valid"].shape[0] else 0.0)
            assert abs(float(g5[i]) - gi) < 1e-4, (b, i)
            nxt_g, nxt_v, nxt_t = gi, float(v[i]), t_idx[i]
    # reward_scale: 報酬・磁石の温度・近い方策への KL の係数が同じ倍率で縮む
    cfg = make_config("tsuitate", magnet="uniform", temp=(0.02, 0.0, 0.02, 0.02), steps=2, chunk=64)
    outs = {}
    for s in (1.0, 2.0):
        L = MMDLearner(small_net(), cfg, DEV, seed=0)
        outs[s] = L.update({k: (x.clone() if torch.is_tensor(x) else x) for k, x in bg.items() if k != "shape_run"},
                           reward_scale=s)
        assert L._reg_scale == 1.0 / s
        for key in ("loss_v", "loss_nerd", "loss_kl", "loss_mag", "grad_norm_max"):
            assert math.isfinite(outs[s][key]), (s, key)
    assert abs(outs[2.0]["temperature"] - 0.01) < 1e-12 and abs(outs[1.0]["temperature"] - 0.02) < 1e-12
    # 価値の目標は、報酬を 1/s にした batch の TD(λ)（収集時の価値はそのまま）
    half = {**bg, "reward": bg["reward"] / 2.0}
    want = float(player_lambda_returns(half, bg["v_actor"].float(), cfg.td_lambda).mean())
    assert abs(outs[2.0]["v_target_mean"] - want) < 1e-5 and abs(outs[1.0]["v_target_mean"] - want) > 1e-3
    # アドバンテージの正規化（--mmd-adv-norm）: A を max(std(A), 下限) で割る。下限が大きければ割る数は下限
    adv = player_lambda_returns(bg, bg["v_actor"].float(), cfg.gae_lambda) - bg["v_actor"].float()
    std = float(adv.std())
    for floor, div in ((1e-4, std), (10.0 * std, 10.0 * std)):
        L = MMDLearner(small_net(), make_config("tsuitate", magnet="uniform", steps=2, chunk=64, adv_norm=floor), DEV, seed=0)
        o = L.update({k: (x.clone() if torch.is_tensor(x) else x) for k, x in bg.items() if k != "shape_run"})
        assert abs(o["adv_std_raw"] - std) < 1e-6 and abs(o["adv_abs_mean_all"] - float(adv.abs().mean()) / div) < 1e-4
    assert abs(outs[1.0]["adv_abs_mean_all"] - float(adv.abs().mean())) < 1e-5, "既定（0）は正規化しない"
    try:
        MMDLearner(small_net(), cfg, DEV).update(bg, reward_scale=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("reward_scale < 1 が通った")
    print("S6 MMD: 相手を固定した batch で λ-return が終局まで届く（λ=1 は後ろの報酬の和、λ=0.5 は素直なループと一致）。"
          "reward_scale で報酬・温度・KL の係数が同じ倍率で縮む。--mmd-adv-norm は A を max(std, 下限) で割る")


def check_handicap(net):
    assert split_noise("rl_v15@0.3") == ("rl_v15", 0.3) and split_noise("rl_v15") == ("rl_v15", 0.0)
    assert split_noise("rl_policy:/a@b/x.bin") == ("rl_policy:/a@b/x.bin", 0.0)
    assert split_noise("rl_policy:/a@b/x.bin@1") == ("rl_policy:/a@b/x.bin", 1.0)
    assert split_noise("x@1.5") == ("x@1.5", 0.0)

    # Rust の評価モード
    def rust(opp):
        env = tsuitate_rl.VecEnv(12, opponent=opp, seed=4, auto_reset=False, planes=NUM_PLANES_DEDUCED)
        b = run_collect(env, net)
        return b, sorted(env.pop_finished(), key=lambda f: f["game_no"])

    b0, f0 = rust("heuristic")
    bz, fz = rust("heuristic@0")
    assert torch.equal(b0["action"], bz["action"]) and np.array_equal(b0["outcome"], bz["outcome"])
    assert all(f["noise_moves"] == 0 for f in fz) and fz[0]["opponent"] == "heuristic@0"
    b1, f1 = rust("heuristic@1")
    assert all(f["fouls"][1 - f["learner"]] == 0 for f in f1), "雑音の手は反則にならない"
    assert all(f["noise_moves"] > 0 for f in f1)
    assert any(f["fouls"][1 - f["learner"]] > 0 for f in f0), "heuristic は反則する（比べる意味がある）"
    bh, fh = rust("heuristic@0.5")
    frac = sum(f["noise_moves"] for f in fh) / max(sum(f["plies"] for f in fh) / 2, 1)
    assert 0.2 < frac < 0.8, frac

    # 相手のネットに指させる形（--league-device gpu）
    opp = small_net(7)

    def gpu(spec, seed=0):
        games = [(spec, i % 2) for i in range(12)]
        env = tsuitate_rl.VecEnv(12, seed=6, auto_reset=False, planes=NUM_PLANES_DEDUCED)
        torch.manual_seed(2)
        b = collect(MultiEnv([env]), net, DEV, compact_legal=True, opponents=games, opp_nets={"opp": opp},
                    noise_rng=np.random.default_rng(seed))
        fin = sorted(env.pop_finished(), key=lambda f: f["game_no"])
        return b, [f["fouls"][1 - i % 2] for i, f in enumerate(fin)]

    g0, fo0 = gpu("opp")
    gz, _ = gpu("opp@0")
    assert torch.equal(g0["action"], gz["action"]) and np.array_equal(g0["outcome"], gz["outcome"])
    g1, fo1 = gpu("opp@1")
    assert fo1 == [0] * 12 and sum(fo0) > 0, (fo0, fo1)
    ga, _ = gpu("opp@0.5", seed=1)
    gb, _ = gpu("opp@0.5", seed=1)
    assert torch.equal(ga["action"], gb["action"])
    try:
        collect(MultiEnv([tsuitate_rl.VecEnv(2, seed=0, auto_reset=False, planes=NUM_PLANES_DEDUCED)]), net, DEV,
                compact_legal=True, opponents=[("opp@0.5", 0), ("opp@0.5", 1)], opp_nets={"opp": opp})
    except ValueError as e:
        assert "noise_rng" in str(e)
    else:
        raise AssertionError("noise_rng なしのハンデが通った")
    print(f"S8 ハンデつきの相手: ε = 0 は元の相手と同じ局、ε = 1 は相手が反則しない（Rust の評価モード・相手のネットに"
          f"指させる形）、ε = 0.5 は相手の手の {frac:.0%} が雑音")


def run_train(out, *extra, algo="mmd", steps=3, check=True):
    cmd = [sys.executable, str(TRAIN), "--init", "none", "--anchor", "none", "--batch", "6", "--steps", str(steps),
           "--channels", "8", "--blocks", "1", "--planes", "91", "--compact-legal", "--eval-every", "0",
           "--export-every", "3", "--checkpoint-every", "3", "--out", str(out), "--threads", "2", "--device", "cpu",
           *(["--algo", "mmd", "--mmd-chunk", "64"] if algo == "mmd" else ["--schedule", "2"]), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"train_rnad.py が落ちた: {' '.join(cmd)}\n{r.stderr[-2500:]}")
    return r


def check_train(tmp):
    shape = ["--league", "heuristic", "--league-frac", "1", "--shape", "d_capture_value=0.03,fouls=-0.02",
             "--shape-cap", "0.5", "--lose-len", "0.3:100", "--shape-decay", "1:3"]
    for algo in ("mmd", "rnad"):
        for mode in ("plain", "potential"):
            out = tmp / f"{algo}_{mode}"
            run_train(out, *shape, "--shape-mode", mode, algo=algo)
            lines = [json.loads(x) for x in open(out / "log.jsonl") if x.strip()]
            assert len(lines) == 3
            assert [x["shape_mult"] for x in lines] == [1.0, 1.0, 0.5]
            for x in lines:
                assert x["reward_scale"] == 1.8 and math.isfinite(x["loss_v"]) and math.isfinite(x["loss_nerd"])
                sg = x["signals"]
                n = sum(sg[g]["final"]["n"] for g in ("win", "loss", "draw"))
                assert n == 6 and sum(x["league_games"]["heuristic"]) == 6, (n, x["league_games"])
                for g in ("win", "loss"):
                    if sg[g]["final"]["n"]:
                        assert len(sg[g]["final"]["mean"]) == 3 * K
            cfg = json.loads((out / "config.json").read_text())
            assert cfg["signal_names"] == NAMES and cfg["shaping"]["mode"] == mode
            assert cfg["shaping"]["own"] == {"capture_value": 0.03, "fouls": -0.02}
            assert cfg["shaping"]["opp"] == {"capture_value": -0.03}
    # 監視だけ（報酬は変えない）: 自己対局でも出る。reward_scale は 1 のまま
    out = tmp / "watch"
    run_train(out, "--watch-signals", steps=1)
    x = json.loads(open(out / "log.jsonl").readline())
    assert x["reward_scale"] == 1.0 and "shape_mult" not in x
    assert x["signals"]["win"]["final"]["n"] + x["signals"]["draw"]["final"]["n"] // 2 == 6
    # 既定（付けない）なら signals は出ない = 収集は従来と同じ経路
    run_train(tmp / "off", steps=1)
    assert "signals" not in json.loads(open(tmp / "off" / "log.jsonl").readline())
    # 相手のネットに指させる形（--league-device gpu）でも MMD が回る
    run_train(tmp / "src", steps=3)
    out = tmp / "gpu"
    run_train(out, "--league", f"rl_policy:{tmp / 'src' / 'policy_3.bin'}", "--league-frac", "1", "--league-device", "gpu",
              "--shape", "d_capture_value=0.03", "--shape-mode", "potential", steps=2)
    lines = [json.loads(x) for x in open(out / "log.jsonl") if x.strip()]
    assert len(lines) == 2 and all(math.isfinite(x["loss_nerd"]) for x in lines)
    # ハンデの梯子（勝率が五分前後の段に多く当てる）。Rust の評価モードと、相手のネットに指させる形
    weights = tmp / "src" / "policy_3.bin"
    for name, extra in (("ladder_rust", ["--league", "heuristic@1,heuristic@0.5,heuristic"]),
                        ("ladder_gpu", ["--league", f"rl_policy:{weights}@1,rl_policy:{weights}@0.5,rl_policy:{weights}",
                                        "--league-device", "gpu"])):
        out = tmp / name
        run_train(out, *extra, "--league-frac", "1", "--league-weighting", "even", "--watch-signals", steps=2)
        lines = [json.loads(x) for x in open(out / "log.jsonl") if x.strip()]
        assert len(lines) == 2
        for x in lines:
            assert set(x["league"]) == set(extra[1].split(",")) and sum(sum(v) for v in x["league_games"].values()) == 6
    # 使えない組み合わせ
    for extra, msg in ((["--shape", "nope=1"], "知らない"), (["--shape-decay", "1:3"], "--shape-decay"),
                       (["--shape", "fouls=-0.1", "--pipeline"], "--pipeline"),
                       (["--shape", "fouls=-0.1", "--fast-collect", "py"], "--fast-collect"),
                       (["--league", "heuristic", "--mmd-lambda-unit", "turn"], "turn"),
                       (["--lose-len", "1.5"], "λ")):
        r = run_train(tmp / "rej", *extra, steps=1, check=False)
        assert r.returncode != 0 and msg in (r.stderr + r.stdout), (extra, r.stderr[-600:])
    print("S7 train_rnad.py: MMD・R-NaD とも --league ＋ --shape（plain / potential）＋ --lose-len ＋ --shape-decay で回り、"
          "log に signals・shape_mult・reward_scale、config に設定が残る。監視だけなら報酬は変わらない。使えない組み合わせは止まる")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_parse()
    check_shape_run()
    check_vtrace_gap()
    check_env_signals()
    net, bg, color = check_collect()
    check_mmd(net, bg, color)
    check_handicap(net)
    with tempfile.TemporaryDirectory() as d:
        check_train(pathlib.Path(d))
    print("ok")
