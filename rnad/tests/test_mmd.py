"""MMD 学習器（`rnad/mmd.py`、`train_rnad.py --algo mmd`）の検査（CPU）。tsuitate_rl が要る。

    .venv/bin/python rnad/tests/test_mmd.py

- 手番側ごとの λ-return（`player_lambda_returns`）を、反則で同じ側が続く手番・途中の報酬・長さ1の局を含む合成格子で
  局ごと・プレイヤーごとの素直なループと突き合わせる（λ を試行ごと / 受理された手ごと）。λ=1・λ=0・反則の無い交互の列の
  閉形式（教科書どおりの Σλ^j·δ）とも合わせる
- +check の単位（自分が王手をかけた手でも λ をかけない）: 観測のスカラーから読む反則・王手の印（`own_next_flags`）が、
  序盤の強制で同じ側の行が続く合成格子でも真の印と一致し、λ-return が素直なループと一致する。実際の自己対局（強制なし・
  あり）でも、印を別の観測（行の並び・累計の反則数・相手の次の行の「王手されている」）と突き合わせる。train_rnad.py の
  `--mmd-gae-lambda-unit turn+check` が序盤の強制と一緒に回り、設定が checkpoint に残る
- 収集の extras（`v_actor`・`mu_logp`）: 付けても既存のキーがビット単位で同じ。3経路（従来・py・rust）で extras が同じ。
  行ごとの Σexp = 1、選んだ手の exp(mu_logp) = μ、v_actor = 同じ観測での順伝播
- CSR の行の拾い直し（`csr_positions`）が全体を詰めたものの同じ行と一致
- 損失と勾配: Ataraxos の rl.py の式を文字どおり写した密な参照（非合法手の古い対数確率は −1e10、磁石は合法手に一様 /
  混合）と、decoupled PPO（近い方策を中心・μ との比を外側の重み）の密な参照
- 端から端: 絞り込みなし・1歩・ミニバッチ = 全行の勾配を、batch の dict から行ごとに直接組んだ参照（素直なループの GAE）と比べる。
  8個のミニバッチの「歩の前の損失」の重み付き和が全体の損失と一致
- π=μ（pipeline なし）の1歩目は比 = 1・KL = 0・クリップ 0・外側の重要度 1
- 行の選び方: quantile の数・下限、pps の mean p = 割合と、乱数を変えた平均が全行の平均に一致（不偏）
- 予定: Ataraxos の power_schedule と一致。t のずらし・刻み
- 再開: 6更新を続けたものと 3更新 → 保存 → 読み込み → 3更新がビット単位で同じ（学習器・pipeline なしの train_rnad.py の両方）。
  学習率の warmup は MMD を始めた step から数え、再開で走り直さない
- R-NaD の checkpoint からの始め方（--init-which both --init-step）と、使えない組み合わせの停止
- train_rnad.py の煙テスト（pipeline・fast py・device_batch、preset 3つ）と stage_log.py が読めること
"""

import copy
import io
import json
import math
import pathlib
import subprocess
import sys
import tempfile
from argparse import Namespace

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from mmd import (MMDLearner, csr_positions, make_config, player_lambda_returns, power_schedule,  # noqa: E402
                 pps_probs, select_rows)
from rnad import _pack_csr, collect, forward  # noqa: E402

DEV = torch.device("cpu")
TRAIN = HERE.parent / "train_rnad.py"
HAS_RUST = hasattr(tsuitate_rl.VecEnv, "observe_alive_f16_csr")


def bits(t):
    return t.detach().contiguous().view(-1).view(torch.int32) if t.dtype == torch.float32 else t.detach().view(-1)


def same_tensor(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))


# ---------------------------------------------------------------- 合成格子と素直なループ


def synthetic_batch(g, B=None, foul_p=None):
    """反則で同じ側が続く手番・途中の報酬・長さ1の局を含む [T, B] の格子と、平らにした (t, b)"""
    B = B or int(torch.randint(1, 10, (1,), generator=g))
    lengths = torch.randint(1, 16, (B,), generator=g)
    T = int(lengths.max())
    valid = torch.arange(T)[:, None] < lengths[None, :]
    fp = foul_p if foul_p is not None else float(torch.rand(1, generator=g) * 0.5)
    player = torch.zeros(T, B, dtype=torch.long)
    for b in range(B):
        p = int(torch.randint(0, 2, (1,), generator=g))
        for t in range(int(lengths[b])):
            player[t, b] = p
            if float(torch.rand(1, generator=g)) >= fp:  # 反則でなければ手番が替わる
                p = 1 - p
    reward = torch.randn(T, B, 2, generator=g) * (torch.rand(T, B, 1, generator=g) < 0.3) * valid[..., None]
    t_idx, b_idx = valid.nonzero(as_tuple=True)
    return {"valid": valid, "player": player, "reward": reward, "t": t_idx, "b": b_idx}


def naive_returns(batch, v, lam, unit):
    """局ごと・プレイヤーごとに、自分の決定点の列を時刻の順に並べて後ろから G を作る（v_trace_scalars を使わない参照）"""
    valid, player, reward = batch["valid"], batch["player"], batch["reward"]
    t_idx, b_idx = batch["t"].tolist(), batch["b"].tolist()
    flat = {(t, b): i for i, (t, b) in enumerate(zip(t_idx, b_idx))}
    out = torch.zeros(len(t_idx), dtype=torch.float64)
    T, B = valid.shape
    for b in range(B):
        Tb = int(valid[:, b].sum())
        for k in (0, 1):
            ts = [t for t in range(Tb) if int(player[t, b]) == k]
            G = None
            for j in range(len(ts) - 1, -1, -1):
                t0 = ts[j]
                t1 = ts[j + 1] if j + 1 < len(ts) else Tb
                r = float(reward[t0:t1, b, k].double().sum())
                if j + 1 == len(ts):
                    G = r
                else:
                    foul = ts[j + 1] == t0 + 1
                    lj = 1.0 if (unit == "turn" and foul) else lam
                    v_next = float(v[flat[(ts[j + 1], b)]])
                    G = r + (1 - lj) * v_next + lj * G
                out[flat[(t0, b)]] = G
    return out


def synthetic_obs_batch(g, forced):
    """`synthetic_batch` の形に、反則数・王手のスカラーだけを入れた観測 [N, 86, 1, 1] と真の印の格子を足したもの。
    各側の決定点は「反則の確率 fp・受理された手は確率 0.3 で王手」。`forced` なら各側の最初の 0〜3手を行にしない
    （序盤の強制。強制した手は受理されて王手でない）ので、同じ側の行が反則なしで続く。
    戻り値の "foul_true" / "check_true" は行ごとの真の印 [T, B]（王手は受理された手だけ）"""
    from mmd import FOULS_THIS_TURN_PLANE, GAVE_CHECK_LAST_PLANE
    B = int(torch.randint(1, 10, (1,), generator=g))
    fp = float(torch.rand(1, generator=g) * 0.5)
    rnd = lambda: float(torch.rand(1, generator=g))  # noqa: E731
    games = []
    for b in range(B):
        n_dec = int(torch.randint(1, 20, (1,), generator=g))
        n_forced = [int(torch.randint(0, 4, (1,), generator=g)) if forced else 0 for _ in range(2)]
        p = int(torch.randint(0, 2, (1,), generator=g))
        ftt, gcl, accepted = 0, [False, False], [0, 0]
        rows = []
        for _ in range(n_dec):
            is_forced = accepted[p] < n_forced[p]
            foul = (not is_forced) and rnd() < fp
            check = (not is_forced) and (not foul) and rnd() < 0.3
            if not is_forced:
                rows.append({"p": p, "foul": foul, "check": check, "ftt": ftt, "gcl": gcl[p]})
            if foul:
                ftt += 1
            else:
                gcl[p], ftt = check, 0
                accepted[p] += 1
                p = 1 - p
        games.append(rows)
    if not any(games):
        games[0].append({"p": 0, "foul": False, "check": False, "ftt": 0, "gcl": False})
    T = max(len(r) for r in games)
    valid = torch.zeros(T, B, dtype=torch.bool)
    player = torch.zeros(T, B, dtype=torch.long)
    foul_true = torch.zeros(T, B, dtype=torch.bool)
    check_true = torch.zeros(T, B, dtype=torch.bool)
    for b, rows in enumerate(games):
        for t, r in enumerate(rows):
            valid[t, b] = True
            player[t, b] = r["p"]
            foul_true[t, b] = r["foul"]
            check_true[t, b] = r["check"]
    reward = torch.randn(T, B, 2, generator=g) * (torch.rand(T, B, 1, generator=g) < 0.3) * valid[..., None]
    t_idx, b_idx = valid.nonzero(as_tuple=True)
    obs = torch.zeros(len(t_idx), 86, 1, 1, dtype=torch.float16)
    for i, (t, b) in enumerate(zip(t_idx.tolist(), b_idx.tolist())):
        r = games[b][t]
        obs[i, FOULS_THIS_TURN_PLANE] = r["ftt"] / 10
        obs[i, GAVE_CHECK_LAST_PLANE] = float(r["gcl"])
    return {"valid": valid, "player": player, "reward": reward, "t": t_idx, "b": b_idx, "obs": obs,
            "foul_true": foul_true, "check_true": check_true}


def has_next_own(batch):
    """[T, B] の bool: 同じ局で同じ側の決定点がこの後にある"""
    valid, player = batch["valid"], batch["player"]
    T, B = valid.shape
    out = torch.zeros(T, B, dtype=torch.bool)
    for k in (0, 1):
        seen = torch.zeros(B, dtype=torch.bool)
        for t in range(T - 1, -1, -1):
            mine = valid[t] & (player[t] == k)
            out[t] |= mine & seen
            seen |= mine
    return out


def naive_returns_skip(batch, v, lam, skip):
    """`naive_returns` の λ_j を「skip[t_j, b] なら 1、それ以外は λ」にしたもの（反則は真の印で与える）"""
    valid, player, reward = batch["valid"], batch["player"], batch["reward"]
    flat = {(t, b): i for i, (t, b) in enumerate(zip(batch["t"].tolist(), batch["b"].tolist()))}
    out = torch.zeros(len(flat), dtype=torch.float64)
    for b in range(valid.shape[1]):
        Tb = int(valid[:, b].sum())
        for k in (0, 1):
            ts = [t for t in range(Tb) if int(player[t, b]) == k]
            G = None
            for j in range(len(ts) - 1, -1, -1):
                t0 = ts[j]
                t1 = ts[j + 1] if j + 1 < len(ts) else Tb
                r = float(reward[t0:t1, b, k].double().sum())
                if j + 1 == len(ts):
                    G = r
                else:
                    lj = 1.0 if bool(skip[t0, b]) else lam
                    G = r + (1 - lj) * float(v[flat[(ts[j + 1], b)]]) + lj * G
                out[flat[(t0, b)]] = G
    return out


def check_lambda_returns_check():
    """+check の単位: 観測のスカラーから読む反則・王手の印（`own_next_flags`）が真の印と一致し（序盤の強制で同じ側の行が
    反則なしで続いても）、λ-return が「その行だけ λ = 1」の素直なループと一致する"""
    from mmd import foul_rows, own_next_flags
    g = torch.Generator().manual_seed(17)
    cases, worst, n_check, n_forced_mismatch = 0, 0.0, 0, 0
    for trial in range(60):
        forced = trial % 2 == 1
        batch = synthetic_obs_batch(g, forced)
        nxt = has_next_own(batch)
        foul, check = own_next_flags(batch)
        assert torch.equal(foul, batch["foul_true"] & nxt), f"反則の印が違う（試行 {trial}）"
        assert torch.equal(check, batch["check_true"] & nxt), f"王手の印が違う（試行 {trial}）"
        n_check += int(check.sum())
        if forced:
            n_forced_mismatch += int((foul_rows(batch["valid"], batch["player"]) != foul).sum())
        v = torch.rand(len(batch["t"]), generator=g) * 2 - 1
        for lam in (0.0, 0.5, 0.8, 1.0):
            for unit, skip in (("attempt+check", batch["check_true"]),
                               ("turn+check", batch["check_true"] | batch["foul_true"])):
                got = player_lambda_returns(batch, v, lam, unit).double()
                ref = naive_returns_skip(batch, v, lam, skip)
                err = float((got - ref).abs().max()) if len(ref) else 0.0
                worst = max(worst, err)
                assert err <= 1e-5, f"λ-return が違う（試行 {trial}、λ {lam}、{unit}、差 {err:.2e}）"
                cases += 1
            if not forced:
                # 強制なしなら turn+check の反則の側は turn と同じ。王手の無い batch では +check は元の単位と同じ
                nocheck = {**batch, "_own_next_flags": (foul, torch.zeros_like(check))}
                for base in ("attempt", "turn"):
                    a = player_lambda_returns(nocheck, v, lam, base + "+check")
                    b = player_lambda_returns(batch, v, lam, base)
                    assert float((a - b).abs().max()) <= 1e-6, f"王手の無い {base}+check が {base} と違う"
    assert n_check > 0 and n_forced_mismatch > 0, (n_check, n_forced_mismatch)
    print(f"T1c +check の λ-return: 合成格子 {cases} 通りで素直なループと一致（最大差 {worst:.1e}）。観測から読む反則・王手の印は"
          f"真の印と一致（王手 {n_check} 行。序盤の強制あり の格子で行の並びの反則判定 foul_rows は {n_forced_mismatch} 行を取り違える）")


def check_own_flags_env(net):
    """実際の自己対局（tsuitate_rl）で `own_next_flags` を別の観測と突き合わせる:
    - 強制なし: 反則の印 = 行の並びの反則判定（foul_rows）、王手の印 = 相手の次の行の「王手されている」
    - 序盤の強制あり: 反則の印 = 次の自分の決定点で累計の反則数（S_MY_FOULS）が増えた、王手の印 = 次の時刻が相手の行なら
      その行の「王手されている」（強制した手が挟まるときは比べない）"""
    from mmd import IN_CHECK_PLANE, foul_rows, own_next_flags
    my_fouls_plane = 77
    book = {"lines": [{"name": "速攻", "category": "居飛車", "moves": ["2g2f", "2f2e", "2e2d"]},
                      {"name": "四間", "category": "振り飛車", "moves": ["7g7f", "6g6f", "2h6h", "5i4h"]}]}
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "book.json"
        path.write_text(json.dumps(book, ensure_ascii=False))
        import openings
        bk = openings.load_book(path)
    totals = {}
    for forced in (False, True):
        kw = openings.env_kwargs(bk, 0.75) if forced else {}
        env = tsuitate_rl.VecEnv(24, seed=29, auto_reset=False, planes=91, **kw)
        gen = torch.Generator().manual_seed(31)
        batch = collect(env, net, DEV, generator=gen, compact_legal=True, actor_extras=True)
        valid, player = batch["valid"], batch["player"]
        T, B = valid.shape
        t_idx, b_idx = batch["t"], batch["b"]
        grid = lambda plane: torch.zeros(T, B).index_put_((t_idx, b_idx), batch["obs"][:, plane, 0, 0].float())  # noqa
        in_check, my_fouls = grid(IN_CHECK_PLANE), grid(my_fouls_plane)
        foul, check = own_next_flags(batch)
        nxt_valid = torch.zeros_like(valid)
        nxt_valid[:-1] = valid[1:]
        nxt_player = torch.zeros_like(player)
        nxt_player[:-1] = player[1:]
        nxt_in_check = torch.zeros(T, B)
        nxt_in_check[:-1] = in_check[1:]
        opp_next = valid & nxt_valid & (nxt_player != player)
        nxt = has_next_own(batch)
        # 次の自分の決定点の累計反則数
        nxt_fouls = torch.zeros(T, B)
        for k in (0, 1):
            cur = torch.zeros(B)
            for t in range(T - 1, -1, -1):
                mine = valid[t] & (player[t] == k)
                nxt_fouls[t] = torch.where(mine, cur, nxt_fouls[t])
                cur = torch.where(mine, my_fouls[t], cur)
        ref_foul = nxt & (nxt_fouls > my_fouls + 1e-4)
        assert torch.equal(foul, ref_foul), f"反則の印が累計の反則数の増え方と違う（強制 {forced}）"
        if not forced:
            assert torch.equal(foul, foul_rows(valid, player)), "強制なしで反則の印が foul_rows と違う"
        cmp = opp_next & nxt & ~foul
        assert torch.equal(check[cmp], nxt_in_check[cmp] > 0), f"王手の印が相手の次の行の「王手されている」と違う（強制 {forced}）"
        assert not bool((check & foul).any())
        totals[forced] = (int(valid.sum()), int(foul.sum()), int(check.sum()),
                          int((foul_rows(valid, player) != foul).sum()))
    assert all(v[1] > 0 and v[2] > 0 for v in totals.values()), totals
    assert totals[True][3] > 0, "序盤の強制で同じ側の行が続く局が無い（検査が空振り）"
    print("T2c 観測から読む反則・王手の印（実際の自己対局 24局ずつ）: "
          + " / ".join(f"{'強制あり' if f else '強制なし'} 行 {n}・反則 {nf}・王手 {nc}"
                       + (f"（行の並びの反則判定は {mm} 行を取り違える）" if f else "")
                       for f, (n, nf, nc, mm) in totals.items()))


def check_lambda_returns():
    g = torch.Generator().manual_seed(3)
    cases = 0
    worst = 0.0
    for trial in range(40):
        batch = synthetic_batch(g)
        v = torch.rand(len(batch["t"]), generator=g) * 2 - 1
        for lam in (0.0, 0.5, 0.8, 1.0):
            for unit in ("attempt", "turn"):
                got = player_lambda_returns(batch, v, lam, unit).double()
                ref = naive_returns(batch, v, lam, unit)
                err = float((got - ref).abs().max()) if len(ref) else 0.0
                worst = max(worst, err)
                assert err <= 1e-5, f"λ-return が違う（試行 {trial}、λ {lam}、{unit}、差 {err:.2e}）"
                cases += 1
    print(f"T1 λ-return: 合成格子 {cases} 通りで素直なループと一致（最大差 {worst:.1e}、反則・途中の報酬・長さ1の局あり）")

    # 閉形式: λ=1 なら G = 自分の決定点から局の終わりまでの自分の報酬の和。λ=0 なら r + V'
    for trial in range(20):
        batch = synthetic_batch(g)
        v = torch.rand(len(batch["t"]), generator=g) * 2 - 1
        g1 = player_lambda_returns(batch, v, 1.0)
        pl = batch["player"][batch["t"], batch["b"]]
        tail = batch["reward"].flip(0).cumsum(0).flip(0)  # tail[t, b, k] = Σ_{t' ≥ t} reward
        ref = tail[batch["t"], batch["b"], pl]
        assert float((g1 - ref).abs().max()) <= 1e-5, "λ=1 が「終わりまでの報酬の和」と違う"
    # 反則の無い交互の列: A = Σ_j λ^j·δ_{i+j}、δ_i = r_i + V_{i+1} − V_i（教科書の形）
    for trial in range(20):
        batch = synthetic_batch(g, foul_p=0.0)
        v = torch.rand(len(batch["t"]), generator=g) * 2 - 1
        lam = 0.7
        a = (player_lambda_returns(batch, v, lam) - v).double()
        valid, player, reward = batch["valid"], batch["player"], batch["reward"]
        flat = {(t, b): i for i, (t, b) in enumerate(zip(batch["t"].tolist(), batch["b"].tolist()))}
        for b in range(valid.shape[1]):
            Tb = int(valid[:, b].sum())
            for k in (0, 1):
                ts = [t for t in range(Tb) if int(player[t, b]) == k]
                deltas = []
                for j, t0 in enumerate(ts):
                    t1 = ts[j + 1] if j + 1 < len(ts) else Tb
                    r = float(reward[t0:t1, b, k].sum())
                    vn = float(v[flat[(ts[j + 1], b)]]) if j + 1 < len(ts) else 0.0
                    deltas.append(r + vn - float(v[flat[(t0, b)]]))
                for j, t0 in enumerate(ts):
                    ref = sum(lam ** i * deltas[j + i] for i in range(len(ts) - j))
                    assert abs(float(a[flat[(t0, b)]]) - ref) <= 1e-5, "交互の列の GAE が Σλ^j·δ と違う"
    print("T1b 閉形式: λ=1 は終わりまでの報酬の和、λ=0・交互の列は Σλ^j·δ（教科書の形）と一致")


# ---------------------------------------------------------------- 収集の extras


def batch_bits_equal(a, b, keys):
    bad = []
    for k in keys:
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray):
            ok = x.dtype == y.dtype and np.array_equal(x, y)
        else:
            iv = {torch.float16: torch.int16, torch.float32: torch.int32}
            ok = (x.dtype == y.dtype and x.shape == y.shape
                  and torch.equal(x.view(iv[x.dtype]) if x.dtype in iv else x, y.view(iv[y.dtype]) if y.dtype in iv else y))
        if not ok:
            bad.append(k)
    return bad


def check_extras():
    torch.manual_seed(0)
    net = PolicyValueNet(16, 2, 91).eval()

    def run(fast, extras):
        gen = torch.Generator().manual_seed(5)
        env = tsuitate_rl.VecEnv(10, seed=11, auto_reset=False, planes=91)
        return collect(env, net, DEV, generator=gen, compact_legal=True, fast=fast, actor_extras=extras)

    base = run("", False)
    keys = sorted(base)
    ref = run("", True)
    assert not batch_bits_equal(base, ref, keys), "extras を付けると既存のキーが変わる"
    for fast in ["py"] + (["rust"] if HAS_RUST else []):
        x = run(fast, True)
        bad = batch_bits_equal(base, x, keys) + batch_bits_equal(ref, x, ["v_actor", "mu_logp"])
        assert not bad, f"fast={fast} の extras 付きの batch が違う: {bad}"
    n = len(ref["action"])
    counts = ref["legal_counts"].long()
    rowid = torch.repeat_interleave(torch.arange(n), counts)
    psum = torch.zeros(n, dtype=torch.float64).index_add_(0, rowid, ref["mu_logp"].double().exp())
    assert float((psum - 1).abs().max()) <= 1e-5, "mu_logp の行ごとの Σexp が 1 でない"
    hit = (ref["legal_cols"].long() == ref["action"][rowid]).nonzero().squeeze(1)
    assert len(hit) == n
    assert float((ref["mu_logp"][hit].exp() - ref["mu_a"]).abs().max()) <= 1e-6, "選んだ手の exp(mu_logp) が μ と違う"
    with torch.no_grad():
        v = forward(net, ref["obs"].float())[1]
    assert float((v - ref["v_actor"]).abs().max()) <= 1e-5, "v_actor が同じ観測での価値と違う"
    print(f"T2 収集の extras: 既存のキーはビット単位で同じ、extras は {'従来・py・rust' if HAS_RUST else '従来・py'} で同じ、"
          f"Σexp(mu_logp)=1・exp(mu_logp[a])=μ・v_actor=順伝播（決定点 {n}）")
    return net, ref


# ---------------------------------------------------------------- CSR の拾い直し


def check_csr(batch):
    g = torch.Generator().manual_seed(1)
    n = len(batch["action"])
    counts = batch["legal_counts"].long()
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, 0)])
    k = int(counts.max())
    idx_all, pm_all, ap_all, ok = _pack_csr(batch["legal_cols"], counts, batch["action"], k)
    assert bool(ok)
    for _ in range(5):
        rows = torch.randperm(n, generator=g)[: int(torch.randint(1, n, (1,), generator=g))]
        pos = csr_positions(ptr, counts, rows, int(counts[rows].sum()))
        kk = int(counts[rows].max())
        idx, pm, ap, ok = _pack_csr(batch["legal_cols"][pos], counts[rows], batch["action"][rows], kk)
        assert bool(ok)
        assert torch.equal(idx, idx_all[rows][:, :kk]) and torch.equal(pm, pm_all[rows][:, :kk])
        assert torch.equal(ap, ap_all[rows])
    print("T3 CSR の行の拾い直し: 無作為な行の並びで、全体を詰めたものの同じ行と完全に一致")


# ---------------------------------------------------------------- 損失と勾配（密な参照）


def dense(batch, rows, flat):
    """CSR の値（平らな [Σcounts]）を、行 rows の密な [m, 11259] に（非合法手は fill）"""
    counts = batch["legal_counts"].long()
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, 0)])
    pos = csr_positions(ptr, counts, rows, int(counts[rows].sum()))
    rep = torch.repeat_interleave(torch.arange(len(rows)), counts[rows])
    legal = torch.zeros(len(rows), 139 * 81, dtype=torch.bool)
    legal[rep, batch["legal_cols"][pos].long()] = True
    vals = torch.zeros(len(rows), 139 * 81)
    if flat is not None:
        vals[rep, batch["legal_cols"][pos].long()] = flat[pos].float()
    return legal, vals


def ataraxos_loss(net, obs, legal, action, old_lp, adv, returns, temp, clip, kl_coef, vf, magnet_lp=None,
                  prox_lp=None, prox_cap=None):
    """Ataraxos の pyengine/core/rl.py:543-579 を文字どおり写したもの（スカラーの価値・合法手に一様な磁石）。
    `magnet_lp` を渡せばその磁石、`prox_lp` を渡せば decoupled PPO（クリップと KL の中心を prox、μ との比を外側の重み）"""
    logits, values_pred = net(obs)
    log_probs = logits.masked_fill(~legal, torch.finfo(logits.dtype).min).log_softmax(-1)
    old_log_probs = torch.where(legal, old_lp, torch.full_like(old_lp, -1e10))
    center = old_log_probs if prox_lp is None else torch.where(legal, prox_lp, torch.full_like(old_lp, -1e10))
    old_log_prob = center.gather(-1, action[:, None]).squeeze(-1)
    log_prob = log_probs.gather(-1, action[:, None]).squeeze(-1)
    ratio = torch.exp(log_prob - old_log_prob)
    outer = torch.ones_like(ratio)
    if prox_lp is not None:
        outer = (old_log_prob - old_log_probs.gather(-1, action[:, None]).squeeze(-1)).exp().clamp(max=prox_cap)
    policy_loss_1 = adv * ratio
    policy_loss_2 = adv * torch.clamp(ratio, 1 - clip, 1 + clip)
    policy_loss = -(outer * torch.min(policy_loss_1, policy_loss_2)).mean()
    probs = log_probs.exp() * legal
    kl_loss = (probs * (log_probs - center)).sum(dim=-1).mean()
    value_loss = torch.nn.functional.mse_loss(values_pred, returns, reduction="none").mean()
    entropy = -(probs * log_probs).sum(-1)
    if magnet_lp is None:
        magnet = legal.float() / legal.sum(-1, keepdim=True)
        xe = -(probs * torch.log(magnet.clamp(min=1e-10))).sum(dim=-1)
    else:
        xe = -(probs * torch.where(legal, magnet_lp, torch.zeros_like(magnet_lp))).sum(dim=-1)
    magnet_kl = -(entropy - xe).mean()
    return policy_loss + temp * magnet_kl + vf * value_loss + kl_coef * kl_loss


def grads_of(net):
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in net.parameters()]


def max_rel(ga, gb):
    num = max(float((a - b).abs().max()) for a, b in zip(ga, gb))
    den = max(float(b.abs().max()) for b in gb)
    return num / max(den, 1e-12)


def check_loss_dense(net0, batch):
    g = torch.Generator().manual_seed(4)
    n = len(batch["action"])
    adv = torch.randn(n, generator=g) * 0.3
    ret = torch.rand(n, generator=g) * 2 - 1
    v_act = batch["v_actor"].float()
    rows = torch.arange(n)
    legal, mu_dense = dense(batch, rows, batch["mu_logp"])
    obs = batch["obs"].float()
    act = batch["action"]
    temp = 0.05
    for magnet in ("uniform", "mix:init,0.2"):
        # 学習する方策を μ から動かしておく（比が 1 でない・クリップが効く・KL が 0 でない）
        torch.manual_seed(7)
        net = copy.deepcopy(net0)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(1.5 * torch.randn_like(p) * p.abs().mean())
        cfg = make_config("tsuitate", filter="none", magnet=magnet, chunk=10_000)
        L = MMDLearner(net, cfg, DEV)
        L.magnet_net.load_state_dict(net0.state_dict())
        keep = torch.ones(n, dtype=torch.bool)
        st, _ = L.prepare(batch, adv, ret, v_act, keep, torch.ones(n), keep, torch.ones(n))
        L.net.zero_grad(set_to_none=True)
        loss, stats = L._chunk_loss(st, rows, n, n, temp, False, True)
        loss.backward()
        ours = grads_of(L.net)
        ref_net = copy.deepcopy(net)
        mag_lp = None
        if magnet != "uniform":
            with torch.no_grad():
                lm = net0(obs)[0].masked_fill(~legal, float("-inf")).log_softmax(-1).exp()
                mag_lp = torch.log(0.8 * lm + 0.2 / legal.sum(-1, keepdim=True))
        ref = ataraxos_loss(ref_net, obs, legal, act, mu_dense, adv, ret, temp, cfg.clip, cfg.kl, cfg.vf, mag_lp)
        ref.backward()
        rel = max_rel(ours, grads_of(ref_net))
        loss, ref = float(loss.detach()), float(ref.detach())
        assert abs(loss - ref) <= 1e-5 * max(1.0, abs(ref)), (magnet, loss, ref)
        assert rel <= 1e-4, f"勾配が Ataraxos の式と違う（磁石 {magnet}、相対差 {rel:.2e}）"
        clip_share = float(stats[5] / stats[6])
        assert 0.05 < clip_share < 0.95, f"クリップが効く状態で検査できていない（{clip_share:.0%}）"
        print(f"T4 損失と勾配（磁石 {magnet}）: Ataraxos の式の密な参照と一致（損失 {loss:.6f}、勾配の相対差 {rel:.1e}、"
              f"クリップされた行 {clip_share:.0%}）")

    # decoupled PPO: 近い方策（学習器の今のネット）を中心にし、μ との比を外側の重みにする
    torch.manual_seed(8)
    net = copy.deepcopy(net0)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.5 * torch.randn_like(p) * p.abs().mean())
    cfg = make_config("tsuitate", filter="none", magnet="uniform", chunk=10_000, prox_cap=1.5)
    L = MMDLearner(net, cfg, DEV, stale_actor=True)
    keep = torch.ones(n, dtype=torch.bool)
    st, capped = L.prepare(batch, adv, ret, v_act, keep, torch.ones(n), keep, torch.ones(n))
    with torch.no_grad():
        prox_dense = net(obs)[0].masked_fill(~legal, float("-inf")).log_softmax(-1)
    # 1歩目は中心 = 今の方策なので比は 1、KL は 0
    L.net.zero_grad(set_to_none=True)
    loss, stats = L._chunk_loss(st, rows, n, n, temp, False, True)
    loss.backward()
    ours = grads_of(L.net)
    assert float(stats[5]) == 0 and abs(float(stats[1])) <= 1e-6, "decoupled の1歩目で比が 1 でない・KL が 0 でない"
    # 動かした後の勾配を参照と比べる（中心は更新の頭の方策のまま）
    with torch.no_grad():
        for p in L.net.parameters():
            p.add_(0.3 * torch.randn_like(p) * p.abs().mean())
    moved = copy.deepcopy(L.net)
    L.net.zero_grad(set_to_none=True)
    loss, stats = L._chunk_loss(st, rows, n, n, temp, False, True)
    loss.backward()
    ours = grads_of(L.net)
    ref = ataraxos_loss(moved, obs, legal, act, mu_dense, adv, ret, temp, cfg.clip, cfg.kl, cfg.vf,
                        prox_lp=prox_dense, prox_cap=1.5)
    ref.backward()
    rel = max_rel(ours, grads_of(moved))
    loss, ref = float(loss.detach()), float(ref.detach())
    assert abs(loss - ref) <= 1e-5 * max(1.0, abs(ref)), (loss, ref)
    assert rel <= 1e-4, f"decoupled PPO の勾配が参照と違う（相対差 {rel:.2e}）"
    clip_share = float(stats[5] / stats[6])
    assert 0.05 < capped < 0.95 and 0.05 < clip_share < 0.95, (capped, clip_share)
    print(f"T5b decoupled PPO: 1歩目は比 1・KL 0、動いた後は密な参照と一致（勾配の相対差 {rel:.1e}、"
          f"外側の重みが上限に当たった行 {capped:.0%}、クリップされた行 {clip_share:.0%}）")


def check_end_to_end(net0, batch):
    """絞り込みなし・1歩・ミニバッチ = 全行の勾配を、batch の dict から行ごとに直接組んだ参照と比べる"""
    n = len(batch["action"])
    cfg = make_config("tsuitate", filter="none", magnet="uniform", steps=1, chunk=10_000, max_grad_norm=0.0,
                      lr=(0.0, 0.0, 0.0, 0.0), gae_lambda=0.9, td_lambda=0.6)
    L = MMDLearner(copy.deepcopy(net0), cfg, DEV)
    L.update(batch)
    ours = grads_of(L.net)
    v = batch["v_actor"].float()
    adv = (naive_returns(batch, v, 0.9, "attempt") - v.double()).float()
    ret = naive_returns(batch, v, 0.6, "attempt").float()
    legal, mu_dense = dense(batch, torch.arange(n), batch["mu_logp"])
    ref_net = copy.deepcopy(net0)
    ref = ataraxos_loss(ref_net, batch["obs"].float(), legal, batch["action"], mu_dense, adv, ret,
                        L.temperature(0), cfg.clip, cfg.kl, cfg.vf)
    ref.backward()
    rel = max_rel(ours, grads_of(ref_net))
    assert rel <= 1e-4, f"端から端の勾配が参照と違う（相対差 {rel:.2e}）"

    # 8個のミニバッチ: 各ミニバッチの損失（÷ その行の数）に行の数の割合を掛けて足すと、全体の損失
    cfg8 = make_config("tsuitate", filter="none", magnet="uniform", chunk=37, steps=8)
    L8 = MMDLearner(copy.deepcopy(net0), cfg8, DEV)
    keep = torch.ones(n, dtype=torch.bool)
    st, _ = L8.prepare(batch, adv, ret, v, keep, torch.ones(n), keep, torch.ones(n))
    with torch.no_grad():
        full = sum(float(L8._chunk_loss(st, torch.arange(n)[i:i + 37], n, n, 0.01, False, False)[0])
                   for i in range(0, n, 37))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(2))
        tot = 0.0
        for mb in torch.tensor_split(perm, 8):
            m = len(mb)
            part = sum(float(L8._chunk_loss(st, mb[i:i + 37], m, m, 0.01, False, False)[0]) for i in range(0, m, 37))
            tot += part * m / n
    assert abs(tot - full) <= 1e-5 * max(1.0, abs(full)), (tot, full)
    print(f"T4b 端から端: 1歩の勾配が batch から直接組んだ参照と一致（相対差 {rel:.1e}）。"
          f"8個のミニバッチの損失の重み付き和 = 全体の損失（{tot:.6f} / {full:.6f}）")


def check_first_step(net0, batch):
    cfg = make_config("tsuitate", filter="none", steps=1, lr=(0.0, 0.0, 0.0, 0.0))
    L = MMDLearner(copy.deepcopy(net0), cfg, DEV)
    out = L.update(batch)
    assert out["clip_frac"] == 0.0 and out["prox_is_capped"] == 0.0
    assert out["ratio_dev_first"] <= 1e-5, out["ratio_dev_first"]
    assert abs(out["loss_kl"]) <= 1e-6, out["loss_kl"]
    print(f"T5 π=μ の1歩目: |π/μ − 1| 平均 {out['ratio_dev_first']:.1e}・KL {out['loss_kl']:.1e}・クリップ 0・外側の重み 1")


# ---------------------------------------------------------------- 行の選び方・予定


def check_selection():
    g = torch.Generator().manual_seed(9)
    a = torch.cat([torch.randn(4000, generator=g).abs() * 0.2, torch.zeros(300), torch.randn(700, generator=g).abs()])
    # quantile: 残す数は ⌈(1−q)·n⌉ 前後（同値を除く）、下限が効く
    adv = torch.randn(5001, generator=g)
    kp, wp, kv, wv, info = select_rows(adv, ("quantile", 0.75, 0.0), ("all",), g)
    assert abs(int(kp.sum()) - math.ceil(0.25 * 5001)) <= 1 and torch.equal(wp, kp.float())
    kp, *_ = select_rows(adv * 1e-3, ("quantile", 0.75, 0.01), ("all",), g)
    assert int(kp.sum()) == int((adv.abs() * 1e-3 >= 0.01).sum())
    # topk: 同値が多くてもちょうど ⌈割合·n⌉ 行、残した行の |A| は残さなかった行以上、同値は乱数で切る
    tied = torch.tensor([0.5] * 10 + [0.1] * 90)
    picks = set()
    for _ in range(20):
        kp, wp, *_ = select_rows(tied, ("topk", 0.05), ("all",), g)
        assert int(kp.sum()) == 5 and bool((tied[kp] == 0.5).all()) and torch.equal(wp, kp.float())
        picks.add(tuple(kp.nonzero().squeeze(1).tolist()))
    assert len(picks) > 1, "topk の同値が乱数で切られていない"
    # pps: mean p = 割合、p ∈ [floor, 1]
    for rate, floor in ((0.25, 0.05), (0.5, 0.05), (0.1, 0.02)):
        p, c = pps_probs(a, rate, floor)
        assert abs(float(p.mean()) - rate) <= 1e-6 and float(p.min()) >= floor - 1e-12 and float(p.max()) <= 1.0
    # 不偏: 乱数を変えた 1,000回の Σ_残す w·f / 残す数 の平均が、全行の平均 f に一致（3σ 以内）
    f = torch.randn(len(a), generator=g) + 2 * a
    ests = []
    for _ in range(1000):
        kp, wp, kv, wv, _ = select_rows(a, ("pps", 0.25, 0.05), ("filtered",), g)
        ests.append(float((wp * f).sum() / kp.sum()))
        assert torch.equal(kv, kp) and torch.equal(wv, wp)
    ests = torch.tensor(ests, dtype=torch.float64)
    se = float(ests.std()) / math.sqrt(len(ests))
    dev_ = abs(float(ests.mean()) - float(f.mean()))
    assert dev_ <= 3 * se, f"pps の重み付き平均が全行の平均からずれる（差 {dev_:.4f}、SE {se:.4f}）"
    # 価値の行: uniform は約その割合、重み 1
    kp, wp, kv, wv, _ = select_rows(a, ("quantile", 0.75, 0.0), ("uniform", 0.25), g)
    assert abs(float(kv.float().mean()) - 0.25) < 0.03 and torch.equal(wv, kv.float())
    print(f"T6 行の選び方: quantile の数と下限、topk の数と同値の切り方、pps の mean p = 割合、pps の重み付き平均は不偏"
          f"（1,000回の平均と全行の差 {dev_:.4f}、SE {se:.4f}）")


def ataraxos_power_schedule(coef, step, decay, ceil, floor):
    # AtaraxosAI/stratego pyengine/core/rl.py:756-760 の写し
    x = coef / ((step + 1) ** decay)
    x = max(x, floor)
    x = min(x, ceil)
    return x


def check_schedules():
    for t in (0, 1, 99, 1000, 2303, 2304, 10_000, 35_111, 42_400):
        for spec in ((0.05, 0.3, 0.1, 0.001), (0.5, 1.1, 1e-4, 5e-6)):
            assert power_schedule(spec[0], t, *spec[1:]) == ataraxos_power_schedule(spec[0], t, *spec[1:])
    net = PolicyValueNet(8, 1, 91)
    cfg = make_config("tsuitate", temp=(0.05, 0.3, 0.1, 0.001), temp_t_off=2000, t_scale=0.5,
                      lr=(0.5, 1.1, 1e-4, 5e-6), lr_t_off=100)
    L = MMDLearner(net, cfg, DEV)
    L.start_from(net.state_dict(), net.state_dict(), 98_000)
    for step in (98_000, 98_010, 99_000):
        t = (step - 98_000) * 0.5
        assert L.temperature(step) == ataraxos_power_schedule(0.05, t + 2000, 0.3, 0.1, 0.001)
        assert L.scheduled_lr(step) == ataraxos_power_schedule(0.5, t + 100, 1.1, 1e-4, 5e-6)
    assert abs(make_config("ataraxos").temp_t_off) == 0 and make_config("tsuitate").lr == (1e-4, 0.0, 1e-4, 1e-4)
    # 既定の λ_A・k（10.4 で変えたもの）と、設計の初版・Ataraxos の preset は回したときの値のまま
    d, mi, at = make_config("tsuitate"), make_config("mixinit"), make_config("ataraxos")
    assert (d.gae_lambda, d.td_lambda, d.steps) == (0.5, 0.8, 4)
    assert (mi.gae_lambda, mi.steps, mi.filter, mi.magnet) == (0.8, 8, "pps:0.25,floor=0.05", "mix:init,0.03")
    assert (at.gae_lambda, at.td_lambda, at.steps) == (0.5, 0.8, 8)
    # 既定の温度は 0.05 固定（t によらない）
    Ld = MMDLearner(PolicyValueNet(8, 1, 91), make_config("tsuitate"), DEV)
    assert Ld.temperature(0) == Ld.temperature(10_000) == 0.05
    print("T7 予定: Ataraxos の power_schedule と一致（t のずらし・刻み・MMD を始めた step からの数え方）")


def check_moving_magnet(net0, batch):
    """動く磁石: magnet_avg > 0 なら磁石（magnet_net）だけが別の EMA で動き、書き出しのターゲットは target_avg で動く。
    磁石の対数確率は magnet_net から作る"""
    cfg = make_config("tsuitate", magnet="target", magnet_avg=0.5, target_avg=0.1, filter="none", chunk=10_000)
    L = MMDLearner(copy.deepcopy(net0), cfg, DEV)
    m0 = [p.detach().clone() for p in L.magnet_net.parameters()]
    t0 = [p.detach().clone() for p in L.target.parameters()]
    L.update(batch)
    for p, q, a, b in zip(L.net.parameters(), L.magnet_net.parameters(), m0, t0):
        assert torch.allclose(q, a + 0.5 * (p - a), atol=1e-7)
    for p, q, b in zip(L.net.parameters(), L.target.parameters(), t0):
        assert torch.allclose(q, b + 0.1 * (p - b), atol=1e-7)
    # 2回目の更新の頭の磁石は magnet_net（ターゲットではない）
    n = len(batch["action"])
    keep = torch.ones(n, dtype=torch.bool)
    st, _ = L.prepare(batch, torch.zeros(n), torch.zeros(n), batch["v_actor"].float(), keep, torch.ones(n), keep, torch.ones(n))
    rows = torch.arange(min(n, 50))
    obs, pmask, apos, idx, pos = st.view(rows)
    with torch.no_grad():
        want = torch.log_softmax(L.magnet_net(obs)[0].gather(1, idx).masked_fill(~pmask, float("-inf")), -1)[pmask]
    assert float((st.lrho[pos] - want).abs().max()) < 1e-5
    for bad in (dict(magnet="mix:init,0.03", magnet_avg=0.1), dict(magnet="uniform", magnet_avg=0.1)):
        try:
            make_config("tsuitate", **bad)
        except ValueError:
            continue
        raise AssertionError(f"magnet_avg を {bad['magnet']} と通した")
    # 既定（magnet target・magnet_avg 0）の磁石は書き出しのターゲット。1更新でターゲットが動き、magnet_net（始めたときの
    # 写し）とずれた状態で、磁石の対数確率がターゲットから作られる（magnet_net から作ると固定の磁石に戻ってしまう）
    for magnet in ("target", "mix:target,0.2"):
        cfg = make_config("tsuitate", magnet=magnet, target_avg=0.1, filter="none", chunk=10_000)
        assert make_config("tsuitate").magnet == "target" and cfg.magnet_avg == 0
        L = MMDLearner(copy.deepcopy(net0), cfg, DEV)
        L.update(batch)
        assert any(not torch.equal(a, b) for a, b in zip(L.target.parameters(), L.magnet_net.parameters()))
        st, _ = L.prepare(batch, torch.zeros(n), torch.zeros(n), batch["v_actor"].float(), keep, torch.ones(n), keep,
                          torch.ones(n))
        obs, pmask, apos, idx, pos = st.view(rows)
        with torch.no_grad():
            def lp_of(m):
                return torch.log_softmax(m(obs)[0].gather(1, idx).masked_fill(~pmask, float("-inf")), -1)
            lt, lm = lp_of(L.target), lp_of(L.magnet_net)
        if magnet != "target":
            cnt = pmask.sum(1, keepdim=True).float()
            lt, lm = torch.log(0.8 * lt.exp() + 0.2 / cnt), torch.log(0.8 * lm.exp() + 0.2 / cnt)
        assert float((st.lrho[pos] - lt[pmask]).abs().max()) < 1e-5, magnet
        assert float((st.lrho[pos] - lm[pmask]).abs().max()) > 1e-4, magnet
    print("動く磁石: magnet_avg の EMA で磁石だけが動き（書き出しのターゲットとは別）、磁石の対数確率はそのネットから作る。"
          "既定（magnet_avg 0）と mix:target はターゲットから作る")


def check_apply_cfg(net0, batch):
    """再開時の設定の上書き: 別の EMA の磁石へ切り替えると現在のターゲットから始める、mix:init へは途中から切り替えない、
    分位の窓は新しい長さに切り詰める。train_rnad.py と同じ順（新しい設定で作る → checkpoint を読む → 上書き）で見る"""
    cfg = make_config("tsuitate", target_avg=0.1, filter="quantile:0.5", filt_window=3, chunk=10_000)
    L = MMDLearner(copy.deepcopy(net0), cfg, DEV)
    L.update(batch)
    L.update(batch)
    assert len(L.filt_hist) == 2
    ck = L.state_dict()
    new = make_config("tsuitate", target_avg=0.1, filter="quantile:0.5", filt_window=2, chunk=10_000, magnet_avg=0.5)

    def resumed(with_old):
        L2 = MMDLearner(copy.deepcopy(net0), new, DEV)
        L2.load_state_dict(ck)
        L2.apply_cfg(new, **({"old": ck["cfg"]} if with_old else {}))
        return L2

    def same(a, b):
        return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))

    L2 = resumed(True)
    assert len(L2.filt_hist) == 1 and same(L2.magnet_net, L2.target)
    # 検査の検査: 再開では学習器が新しい設定で作られているので、old を渡さないと切り替えに気づかない
    assert not same(resumed(False).magnet_net, L2.target)
    L2.apply_cfg(make_config("tsuitate", target_avg=0.1, filter="quantile:0.5", filt_window=1, chunk=10_000))
    assert L2.filt_hist == []
    try:
        L2.apply_cfg(make_config("tsuitate", magnet="mix:init,0.03"))
    except ValueError:
        pass
    else:
        raise AssertionError("途中から mix:init へ切り替えた")
    print("設定の上書き: 別の EMA の磁石はターゲットから始め（train_rnad.py と同じ順で、checkpoint の設定と比べる）、"
          "mix:init へは途中から切り替えず、分位の窓を切り詰める")


# ---------------------------------------------------------------- 再開


def check_resume(net0):
    batches = []
    for i in range(6):
        gen = torch.Generator().manual_seed(40 + i)
        env = tsuitate_rl.VecEnv(6, seed=50 + i, auto_reset=False, planes=91)
        batches.append(collect(env, net0, DEV, generator=gen, compact_legal=True, actor_extras=True))
    cfg = make_config("tsuitate", chunk=64, steps=3)

    def fresh():
        L = MMDLearner(copy.deepcopy(net0), cfg, DEV, seed=3)
        L.start_from(net0.state_dict(), net0.state_dict(), 500)
        return L

    def strip(lg):
        return {k: v for k, v in lg.items() if not k.endswith("_sec")}

    a = fresh()
    logs_a = [strip(a.update(b)) for b in batches]
    b = fresh()
    logs_b = [strip(b.update(x)) for x in batches[:3]]
    buf = io.BytesIO()
    torch.save(b.state_dict(), buf)
    buf.seek(0)
    c = MMDLearner(PolicyValueNet(16, 2, 91), cfg, DEV, seed=99)
    c.load_state_dict(torch.load(buf, map_location="cpu"))
    logs_b += [strip(c.update(x)) for x in batches[3:]]
    for name in ("net", "target", "magnet_net"):
        for (k, x), (_, y) in zip(getattr(a, name).state_dict().items(), getattr(c, name).state_dict().items()):
            assert same_tensor(x, y), f"再開すると {name}.{k} が違う"
    for sa, sb in zip(a.opt.state.values(), c.opt.state.values()):
        for k in sa:
            if torch.is_tensor(sa[k]):
                assert same_tensor(sa[k], sb[k]), f"再開すると Adam の {k} が違う"
    assert logs_a == logs_b, [k for k in logs_a[-1] if logs_a[-1][k] != logs_b[-1].get(k)]
    assert a.step_count == c.step_count == 506 and c.mmd_t0 == 500
    print("T8 再開（学習器）: 6更新を続けたものと 3更新→保存→読み込み→3更新が、ネット・ターゲット・磁石・Adam・ログで"
          "ビット単位で一致")


def run_train(out, *extra, steps, check=True):
    cmd = [sys.executable, str(TRAIN), "--algo", "mmd", "--anchor", "none", "--batch", "6", "--steps", str(steps),
           "--channels", "16", "--blocks", "1", "--planes", "91", "--compact-legal", "--eval-every", "0",
           "--export-every", "3", "--checkpoint-every", "3", "--out", str(out), "--threads", "2", "--device", "cpu",
           *extra]
    if "--init" not in extra:
        cmd += ["--init", "none"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"train_rnad.py が落ちた: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r


def read_log(path):
    return [{k: v for k, v in json.loads(line).items() if not k.endswith("_sec")}
            for line in open(path) if line.strip()]


def check_resume_train(tmp):
    """pipeline なしの train_rnad.py: 6更新を続けた run と、3更新で止めて同じコマンドで 6 まで続けた run が同じ"""
    # 温度の予定（t で下げる・起点のずらし）の配線も見るため、初版の予定を明示する
    common = ["--seed", "4", "--lr-warmup", "4", "--mmd-chunk", "50", "--mmd-steps", "3",
              "--mmd-temp", "0.05,0.3,0.1,0.001", "--mmd-temp-t-off", "2000"]
    run_train(tmp / "straight", *common, steps=6)
    run_train(tmp / "resumed", *common, steps=3)
    run_train(tmp / "resumed", *common, steps=6)
    la, lb = read_log(tmp / "straight" / "log.jsonl"), read_log(tmp / "resumed" / "log.jsonl")
    assert la == lb, "train_rnad.py の再開でログが変わる"
    lrs = [x["lr"] for x in la]
    want = [1e-4 * min(1.0, (s + 1) / 4) for s in range(6)]
    assert all(abs(x - y) <= 1e-12 for x, y in zip(lrs, want)), (lrs, want)
    assert [x["temperature"] for x in la] == [power_schedule(0.05, s + 2000, 0.3, 0.1, 0.001) for s in range(6)]
    ca = torch.load(tmp / "straight" / "checkpoint.pt", map_location="cpu")
    cb = torch.load(tmp / "resumed" / "checkpoint.pt", map_location="cpu")
    for name in ("net", "target", "magnet_net"):
        for k in ca[name]:
            assert same_tensor(ca[name][k], cb[name][k]), f"train_rnad.py の再開で {name}.{k} が違う"
    print(f"T8b 再開（train_rnad.py、pipeline なし）: ログ・ネットがビット単位で一致、lr は warmup どおり "
          f"{[f'{x:.1e}' for x in lrs]}（再開で走り直さない）、温度は予定どおり")


def check_value_warmup_lr(tmp):
    """価値の温めの後に幹と方策の頭が初めて Adam の歩を踏むので、学習率の warmup はそこから数え直す（2026-10-26。
    3070 の A/B では warmup を温めの間に使い切り、方策は Adam の状態が空のまま学習率いっぱいで動き始めた）。再開しても同じ"""
    common = ["--seed", "6", "--mmd-chunk", "50", "--lr-warmup", "2", "--mmd-value-warmup", "2"]
    run_train(tmp / "vw", *common, steps=5)
    lg = read_log(tmp / "vw" / "log.jsonl")
    assert [x["value_only"] for x in lg] == [True, True, False, False, False], [x["value_only"] for x in lg]
    want = [5e-5, 1e-4, 5e-5, 1e-4, 1e-4]
    assert all(abs(x["lr"] - y) <= 1e-12 for x, y in zip(lg, want)), [x["lr"] for x in lg]
    run_train(tmp / "vw_r", *common, steps=3)
    run_train(tmp / "vw_r", *common, steps=5)
    assert read_log(tmp / "vw_r" / "log.jsonl") == lg, "価値の温めつきの再開でログが変わる"
    # 温めの後で --reset-adam を付けて再開すると、Adam が空になるのでその頭から温め直す（付けなければ走り直さない）
    run_train(tmp / "vw_reset", *common, steps=3)
    run_train(tmp / "vw_reset", *common, "--reset-adam", steps=5)
    got = [x["lr"] for x in read_log(tmp / "vw_reset" / "log.jsonl")]
    assert got[:3] == [x["lr"] for x in lg[:3]] and abs(got[3] - 5e-5) <= 1e-12 and abs(got[4] - 1e-4) <= 1e-12, got
    # 温めの途中で --reset-adam: 温めの終わり（方策が動き始める step）でもう一度数え直す
    run_train(tmp / "vw_reset1", *common, steps=1)
    run_train(tmp / "vw_reset1", *common, "--reset-adam", steps=5)
    got = [x["lr"] for x in read_log(tmp / "vw_reset1" / "log.jsonl")]
    assert all(abs(x - y) <= 1e-12 for x, y in zip(got, [5e-5, 5e-5, 5e-5, 1e-4, 1e-4])), got
    print("価値の温めと学習率の warmup: 温めの後に warmup を数え直す（学習率 "
          f"{[f'{x:.0e}' for x in want]}）、再開しても同じ。--reset-adam の再開はその頭から温め直す")


def check_override_magnet(tmp):
    """train_rnad.py の再開で --override-config --mmd-magnet-avg 0.5 にすると、磁石の EMA は再開した時点のターゲットから始まる
    （始めたときの写しからではない）。train_rnad.py が checkpoint の設定を apply_cfg に渡す配線を見る"""
    out = tmp / "override"
    run_train(out, "--seed", "5", "--mmd-chunk", "50", "--target-avg", "0.2", steps=3)
    ck3 = torch.load(out / "checkpoint.pt", map_location="cpu")
    r = run_train(out, "--seed", "5", "--mmd-chunk", "50", "--target-avg", "0.2", "--mmd-magnet-avg", "0.5",
                  "--override-config", "--checkpoint-every", "1", steps=4)
    assert "再開時に設定を上書き" in r.stdout and "magnet_avg" in r.stdout, r.stdout[-800:]
    ck4 = torch.load(out / "checkpoint.pt", map_location="cpu")
    assert ck4["step"] == 4
    for k, t3 in ck3["target"].items():
        if not t3.is_floating_point() or k not in dict(PolicyValueNet(16, 1, 91).named_parameters()):
            continue
        want = t3 + 0.5 * (ck4["net"][k] - t3)
        assert torch.allclose(ck4["magnet_net"][k], want, atol=1e-6), k
        stale = ck3["magnet_net"][k] + 0.5 * (ck4["net"][k] - ck3["magnet_net"][k])
        assert not torch.allclose(stale, want, atol=1e-6) or torch.equal(ck3["magnet_net"][k], t3), k
    print("T13 再開で磁石の EMA に切り替える（--override-config --mmd-magnet-avg）: 再開した時点のターゲットから始まる")


def check_init_from_rnad(tmp):
    """R-NaD の checkpoint から --init-which both --init-step で始める"""
    r = subprocess.run([sys.executable, str(TRAIN), "--init", "none", "--anchor", "none", "--batch", "4", "--steps", "2",
                        "--channels", "16", "--blocks", "1", "--planes", "91", "--compact-legal", "--eval-every", "0",
                        "--export-every", "2", "--checkpoint-every", "2", "--out", str(tmp / "rnad_src"),
                        "--threads", "2", "--schedule", "1", "--device", "cpu"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    src = tmp / "rnad_src" / "checkpoint.pt"
    ck = torch.load(src, map_location="cpu")
    sys.path.insert(0, str(TRAIN.parent))
    import train_rnad
    net = PolicyValueNet(16, 1, 91)
    net.load_state_dict(ck["net"])
    L = MMDLearner(net, make_config("tsuitate"), DEV)
    train_rnad.start_mmd_from_init(L, Namespace(init=str(src), init_which="both", init_step=True))
    for name, key in (("net", "net"), ("target", "target"), ("magnet_net", "target")):
        for k, x in getattr(L, name).state_dict().items():
            assert same_tensor(x, ck[key][k]), f"{name}.{k} が checkpoint の {key} と違う"
    assert L.step_count == L.mmd_t0 == ck["step"] == 2 and not L.opt.state
    assert L.init_from["which"] == "both" and L.init_from["algo"] == "rnad"
    # 目に見える違い: R-NaD の net と target は違う（同じだと both の検査にならない）
    assert any(not torch.equal(ck["net"][k], ck["target"][k]) for k in ck["net"] if ck["net"][k].is_floating_point())
    # train_rnad.py で1更新進める（番号が続く）
    run_train(tmp / "from_rnad", "--init", str(src), "--init-which", "both", "--init-step", "--export-current", steps=3)
    lg = read_log(tmp / "from_rnad" / "log.jsonl")
    assert [x["step"] for x in lg] == [3] and lg[0]["mmd_t"] == 0
    # 起点の書き出しは始めた番号（step 2）で書く（speedlab eval --decode で起点を同じ指し方で測り直せるように）
    assert (tmp / "from_rnad" / "current_2.bin").exists() and (tmp / "from_rnad" / "policy_2.bin").exists()
    cfg = json.loads((tmp / "from_rnad" / "config.json").read_text())
    assert cfg["init_from"]["step"] == 2 and cfg["mmd"]["filter"] == "pps:0.5,floor=0.05" and cfg["mmd"]["magnet"] == "target"
    # checkpoint があれば --init* は使わない（表示して続ける）
    r = run_train(tmp / "from_rnad", "--init", str(src), "--init-which", "both", "--init-step", steps=4)
    assert "--init / --init-which / --init-step は使わない" in r.stdout
    # 再開の記録にも出どころ（checkpoint から戻した init_from）が残る
    rec = json.loads((tmp / "from_rnad" / "config_resume_3.json").read_text())
    assert rec["init_from"] == cfg["init_from"], rec["init_from"]
    print("T12 R-NaD の checkpoint から: net・ターゲット・磁石・step が ck と同じ、Adam は空、番号が続く、起点を始めた番号で"
          "書き出す。再開では --init* を使わず、再開の記録に出どころが残る")


def check_rejects(tmp):
    cases = [
        (["--fused"], "使えない"),
        (["--lr", "1e-4"], "--lr"),
        (["--schedule", "1000"], "--schedule"),
        (["--reuse", "2"], "--reuse"),
        # 相手を固定した局（--league）は使える（tests/test_shaping.py）。turn の単位とだけ使えない
        (["--league", "heuristic", "--mmd-lambda-unit", "turn"], "--league"),
        (["--mmd-magnet", "twostage"], "twostage"),
        (["--mmd-filter", "pps:0.01,floor=0.05"], "floor"),
    ]
    for extra, msg in cases:
        r = run_train(tmp / "rej", *extra, steps=1, check=False)
        assert r.returncode != 0 and msg in (r.stderr + r.stdout), (extra, r.stderr[-500:])
    r = subprocess.run([sys.executable, str(TRAIN), "--algo", "mmd", "--init", "none", "--anchor", "none",
                        "--device", "cpu", "--out", str(tmp / "rej2")], capture_output=True, text=True)
    assert r.returncode != 0 and "--compact-legal" in r.stderr
    r = subprocess.run([sys.executable, str(TRAIN), "--mmd-filter", "none", "--init", "none", "--anchor", "none",
                        "--device", "cpu", "--out", str(tmp / "rej3")], capture_output=True, text=True)
    assert r.returncode != 0 and "--algo mmd のときだけ" in r.stderr
    # R-NaD の out に MMD で再開しない
    run_train(tmp / "x", steps=1)
    r = subprocess.run([sys.executable, str(TRAIN), "--init", "none", "--anchor", "none", "--batch", "4",
                        "--steps", "2", "--channels", "16", "--blocks", "1", "--planes", "91", "--device", "cpu",
                        "--out", str(tmp / "x")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "別の学習器" in r.stderr
    print(f"T11 使えない組み合わせ: {len(cases) + 3} 通りとも起動時に止まる")


def write_small_book(tmp):
    path = tmp / "book.json"
    path.write_text(json.dumps({"lines": [
        {"name": "速攻", "category": "居飛車", "moves": ["2g2f", "2f2e", "2e2d"]},
        {"name": "四間", "category": "振り飛車", "moves": ["7g7f", "6g6f", "2h6h", "5i4h"]}]}, ensure_ascii=False))
    return str(path)


def check_check_unit_train(tmp):
    """train_rnad.py の配線: --mmd-gae-lambda-unit turn+check は序盤の強制と一緒に回り、ログに印の見張りが出て、設定が
    checkpoint に残る（付けずに再開すると設定の差で止まる）。turn は序盤の強制と一緒に使えない"""
    book = write_small_book(tmp)
    out = tmp / "check_unit"
    run_train(out, "--mmd-gae-lambda-unit", "turn+check", "--open-book", book, "--open-p", "0.5", "--pipeline",
              "--fast-collect", "py", "--device-batch", "--mmd-chunk", "64", steps=3)
    raw = [json.loads(line) for line in open(out / "log.jsonl") if line.strip()]
    assert len(raw) == 3
    for x in raw:
        assert 0.0 <= x["lam_check_frac"] < 0.5 and 0.0 < x["lam_foul_frac"] < 1.0, x
        assert math.isfinite(x["loss_nerd"]) and math.isfinite(x["loss_v"])
    cfg = json.loads((out / "config.json").read_text())["mmd"]
    assert cfg["gae_lambda_unit"] == "turn+check" and cfg["lambda_unit"] == "attempt", cfg
    r = run_train(out, "--open-book", book, "--open-p", "0.5", "--pipeline", "--fast-collect", "py", "--device-batch",
                  "--mmd-chunk", "64", steps=4, check=False)
    assert r.returncode != 0 and "gae_lambda_unit" in r.stderr, r.stderr[-800:]
    # 既定（付けない）なら見張りは出ない = 収集・損失は従来と同じ経路
    run_train(tmp / "check_unit_off", "--mmd-chunk", "64", steps=1)
    x = json.loads(open(tmp / "check_unit_off" / "log.jsonl").readline())
    assert "lam_check_frac" not in x
    for unit in ("turn",):
        for flag in ("--mmd-lambda-unit", "--mmd-gae-lambda-unit"):
            r = run_train(tmp / "rej_turn", flag, unit, "--open-book", book, "--open-p", "0.5", steps=1, check=False)
            assert r.returncode != 0 and f"{flag} turn" in r.stderr, (flag, r.stderr[-500:])
    print(f"T14 +check の配線: --mmd-gae-lambda-unit turn+check は序盤の強制と一緒に回り（王手の行 "
          f"{sum(x['lam_check_frac'] for x in raw) / len(raw):.1%}・反則の行 {sum(x['lam_foul_frac'] for x in raw) / len(raw):.1%}）、"
          "設定が checkpoint に残る。turn は序盤の強制と使えない")


def check_smoke(tmp):
    """pipeline・fast py・device_batch で3つの preset を回し、NaN なし・ログのキーあり・stage_log.py が読める"""
    for preset in ("tsuitate", "mixinit", "ataraxos"):
        out = tmp / f"smoke_{preset}"
        run_train(out, "--mmd-preset", preset, "--pipeline", "--fast-collect", "py", "--device-batch",
                  "--mmd-chunk", "64", "--lr-warmup", "2", steps=4)
        lg = read_log(out / "log.jsonl")
        raw = [json.loads(line) for line in open(out / "log.jsonl") if line.strip()]
        assert len(lg) == 4
        for x, xr in zip(lg, raw):
            for k in ("loss_v", "loss_nerd", "loss_kl", "loss_mag", "temperature", "clip_frac", "kl_mu_unkept",
                      "grad_norm_p50", "calib_band", "resid_band", "entropy_mean"):
                assert k in x, (preset, k)
            # 時間のキーは read_log が落とすので生の行で見る（律速の判定に使う）
            for k in ("collect_busy_sec", "collect_sec", "update_sec", "train_sec"):
                assert k in xr, (preset, k)
            for k in ("loss_v", "loss_nerd", "loss_kl", "loss_mag", "grad_norm_max"):
                assert math.isfinite(x[k]), (preset, k, x[k])
        r = subprocess.run([sys.executable, str(HERE.parent / "stage_log.py"), str(out / "log.jsonl"), "--dm", "2",
                            "--out-dir", str(out / "stage")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-1500:]
    print("T10 煙テスト: --pipeline --fast-collect py --device-batch で preset 3つとも NaN なし・ログのキーあり・"
          "stage_log.py が読める")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_lambda_returns()
    check_lambda_returns_check()
    check_selection()
    check_schedules()
    net0, batch = check_extras()
    check_own_flags_env(net0)
    check_csr(batch)
    check_loss_dense(net0, batch)
    check_end_to_end(net0, batch)
    check_first_step(net0, batch)
    check_moving_magnet(net0, batch)
    check_apply_cfg(net0, batch)
    check_resume(net0)
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        check_resume_train(tmp)
        check_value_warmup_lr(tmp)
        check_override_magnet(tmp)
        check_init_from_rnad(tmp)
        check_rejects(tmp)
        check_check_unit_train(tmp)
        check_smoke(tmp)
