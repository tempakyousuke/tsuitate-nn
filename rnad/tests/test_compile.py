"""更新の torch.compile（`RNaDLearner(compile=...)` / `MMDLearner(compile=...)`、`rnad.compiled`）の検査（CPU）。
tsuitate_rl と C++ コンパイラ（inductor の CPU 版が使う）が要る。

    .venv/bin/python rnad/tests/test_compile.py

- 実際の自己対局の batch で、compile した学習器と素の学習器を同じ状態から2回ずつ更新し、勾配・ログの値が許容の差
  （浮動小数の足し順の違い程度）に収まる。R-NaD は packed / fused / device_batch の組み合わせと補助ヘッド、
  MMD は行の間引きあり・なし・pipeline の decoupled・補助ヘッドと価値だけの更新
- 版は「形（`rnad.compile_rows` = チャンクの 1/4 の倍数）× 勾配の扱い」の数で抑えられ、batch の大きさが変わっても増えない
- 自己対局（`collect`）は compile した版を使わない（素の `forward` のまま）
"""

import copy
import pathlib
import sys

import torch
import torch._dynamo
from torch._dynamo.utils import counters

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
import rnad as R  # noqa: E402
from mmd import MMDLearner, make_config as make_mmd_config  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import RNaDConfig, RNaDLearner, collect  # noqa: E402

DEV = torch.device("cpu")
# 勾配の相対誤差（compile は畳み込みの BN・ReLU を融合して足し順が変わる）。更新量では比べない: Adam の状態が新しいと
# 1歩目は勾配の符号に近く、0 付近の成分の符号の揺れで更新量が大きく違って見える（bench_update.py の説明と同じ）
TOL = 1e-4


def grad_vec(m):
    return torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()) for p in m.parameters()])


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def log_rel(a, b):
    worst, key = 0.0, None
    for k, x in a.items():
        y = b.get(k)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            if k.endswith("_sec"):
                continue
            assert x == y, f"ログの {k} が違う: {x} / {y}"
            continue
        if k.endswith("_sec"):
            continue
        if k == "kl_mu_unkept":
            # 更新した後のネットで測る見張り。Adam の状態が新しいと1歩目の更新量は勾配の符号に近く、0 付近の成分の
            # 符号の揺れで大きく動く（勾配で比べるのと同じ理由）ので比べない
            continue
        # 分母の下限 1e-3: |π/μ − 1| や μ への KL のように、値そのものが行動時と学習時の数値誤差（1e-7 の大きさ）の
        # 見張りは相対誤差に意味がないので、差の大きさで見る
        e = abs(x - y) / max(abs(x), abs(y), 1e-3)
        if e > worst:
            worst, key = e, k
    return worst, key


def batches(net, n, truth=False, move_class=False, king_escape=False, sizes=(10, 7, 13)):
    out = []
    for i in range(n):
        torch.manual_seed(10 + i)
        env = tsuitate_rl.VecEnv(sizes[i % len(sizes)], seed=20 + i, auto_reset=False, planes=91)
        out.append(collect(env, net, DEV, truth=truth, compact_legal=True, actor_extras=True,
                           move_class=move_class, king_escape=king_escape))
    return out


def perturbed(learner, nets):
    with torch.no_grad():
        for name, scale in nets:
            for p in getattr(learner, name).parameters():
                p.add_(scale * torch.randn_like(p) * p.abs().mean())


def compare(name, base, bs, mode="default", log_tol=1e-3):
    """base を deepcopy して素と compile の2本で bs を順に更新し、更新ごとの勾配（最後の歩）とログを比べる"""
    ref, x = copy.deepcopy(base), copy.deepcopy(base)
    x.compile = mode
    worst, worst_log = 0.0, (0.0, None)
    for b in bs:
        lr = ref.update(b)
        lx = x.update(b)
        if getattr(ref.cfg, "aux_class", 0) > 0 or getattr(ref.cfg, "aux_truth", 0) > 0:
            assert "loss_aux" in lr and "loss_aux" in lx, f"{name}: 補助ヘッドが使われていない"
        if getattr(ref.cfg, "aux_escape", 0) > 0:
            assert any(k.startswith("loss_esc") or "escape" in k for k in lx), f"{name}: 逃げ場の補助ヘッドが使われていない"
        worst = max(worst, rel(grad_vec(x.net), grad_vec(ref.net)))
        e = log_rel(lr, lx)
        worst_log = max(worst_log, e, key=lambda t: t[0])
        # 次の更新の出発点をそろえる（差が積もって比べられなくなるのを防ぐ）
        x.net.load_state_dict(ref.net.state_dict())
        for a in ("target", "prev", "prev_", "magnet_net"):
            if hasattr(ref, a):
                getattr(x, a).load_state_dict(getattr(ref, a).state_dict())
        x.opt.load_state_dict(ref.opt.state_dict())
        for h, o in (("aux", "aux_opt"), ("aux_esc", "aux_esc_opt")):
            if getattr(ref, h, None) is not None:
                getattr(x, h).load_state_dict(getattr(ref, h).state_dict())
                getattr(x, o).load_state_dict(getattr(ref, o).state_dict())
        if hasattr(ref, "gen"):
            x.gen.set_state(ref.gen.get_state())
    assert worst < TOL, f"{name}: 勾配の相対誤差 {worst:.2e}"
    assert worst_log[0] < log_tol, f"{name}: ログの {worst_log[1]} の相対誤差 {worst_log[0]:.2e}"
    print(f"{name}: 勾配の相対誤差 最大 {worst:.1e} / ログの値 最大 {worst_log[0]:.1e}（{worst_log[1]}）", flush=True)


def small_net():
    """小さいネット。BN の統計とバイアスを乱数にする: 作った直後のバイアス 0・平均 0 だと、入力が 0 のマス（盤の外の
    詰め物・0 のプレーン）で ReLU の手前がちょうど 0 や ±(Adam の1歩) に並び、素と compile の 1e-7 の差で ReLU の
    向きが入れ替わる行が多くなって、勾配の差が 1e-2 まで膨らむ（学習済みのネットでは起きない形）"""
    torch.manual_seed(0)
    net = PolicyValueNet(16, 2, 91).eval()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.bias.add_(0.2 * torch.randn_like(m.bias))
                m.running_mean.add_(0.2 * torch.randn_like(m.running_mean))
                m.running_var.mul_(torch.rand_like(m.running_var) + 0.5)
    return net


def rnad_base(step, cfg):
    L = RNaDLearner(small_net(), cfg, DEV, packed=True)
    perturbed(L, (("prev", 0.3), ("prev_", -0.2), ("target", 0.1)))
    L.step_count = step
    if cfg.aux_truth > 0:
        L._ensure_aux()
    return L


def mmd_base(**over):
    # 1歩（歩が2つ以上だと2歩目の勾配が1歩目の更新量の揺れを拾う）。ミニバッチの中のチャンクは複数
    cfg = make_mmd_config("tsuitate", chunk=64, steps=1, **over)
    L = MMDLearner(small_net(), cfg, DEV, seed=3)
    perturbed(L, (("target", 0.1), ("magnet_net", -0.1)))
    return L


def check_rnad():
    common = dict(beta=16.0, lr=1e-3, schedule_sizes=(10,), schedule_repeats=(1,), chunk=97)
    base = rnad_base(13, RNaDConfig(**common))
    bs = batches(base.net, 3)
    for flags in ({}, {"fused": True}, {"device_batch": True}, {"fused": True, "device_batch": True}):
        b = copy.deepcopy(base)
        for k, v in flags.items():
            setattr(b, k, v)
        compare("R-NaD packed" + "".join(f"+{k}" for k in flags), b, bs)
    compare("R-NaD packed なし", (lambda b: (setattr(b, "packed", False), b)[1])(copy.deepcopy(base)), bs)
    base = rnad_base(18, RNaDConfig(**common, aux_truth=1.0))
    compare("R-NaD α=1・差し替え・補助ヘッド（fused）",
            (lambda b: (setattr(b, "fused", True), b)[1])(copy.deepcopy(base)),
            batches(base.net, 2, truth=True))


def check_mmd():
    base = mmd_base()
    bs = batches(base.net, 3)
    compare("MMD 既定（行の間引きあり）", base, bs)
    compare("MMD 絞り込みなし", mmd_base(filter="none"), bs)
    b = mmd_base()
    b.stale_actor = True
    compare("MMD pipeline の decoupled", b, bs)
    b = mmd_base()
    b.device_batch = True
    compare("MMD device_batch", b, bs)
    b = mmd_base(aux_class=0.5)
    compare("MMD 補助ヘッド", b, batches(b.net, 2, move_class=True))
    b = mmd_base(aux_class=0.5, aux_escape=0.5)
    compare("MMD 補助ヘッド2つ（手の狙い・玉の逃げ場）", b, batches(b.net, 2, move_class=True, king_escape=True))
    b = mmd_base(value_warmup=10)
    compare("MMD 価値だけの更新", b, bs[:2])


def check_no_recompile():
    """版は「形（チャンクの 1/4 の倍数の4つ）× 勾配の扱い（3通り）」で抑えられ、batch を何度通しても増えない"""
    common = dict(beta=16.0, lr=1e-3, schedule_sizes=(10,), schedule_repeats=(1,), chunk=97)
    torch._dynamo.reset()
    R._COMPILED.clear()
    counters.clear()
    L = rnad_base(13, RNaDConfig(**common))
    L.fused = True
    L.device_batch = True
    L.compile = "default"
    bs = batches(L.net, 4, sizes=(10, 7, 13, 5))
    for b in bs + bs:
        L.update(b)
    n_rnad = counters["stats"]["unique_graphs"]
    assert n_rnad <= 4 * 3, f"R-NaD の版が多すぎる（{n_rnad}）"
    M = mmd_base()
    M.compile = "default"
    for b in bs + bs:
        M.update(b)
    n_all = counters["stats"]["unique_graphs"]
    # MMD はチャンク 64（形は 16・32・48・64）。版は同じ forward の別の形として足される
    assert n_all - n_rnad <= 4 * 3, f"MMD の版が多すぎる（{n_all - n_rnad}）"
    print(f"作り直し: 版の数 R-NaD {n_rnad}・MMD {n_all - n_rnad}（どちらも上限 12 以内。batch 8回ずつ）", flush=True)


def check_collect_untouched():
    """自己対局は compile した版を通らない（compile した学習器を作って更新しても、collect で版が増えない）"""
    L = rnad_base(13, RNaDConfig(beta=16.0, chunk=97))
    L.compile = "default"
    before = counters["stats"]["unique_graphs"]
    for n in (3, 9, 4):
        collect(tsuitate_rl.VecEnv(n, seed=n, auto_reset=False, planes=91), L.net, DEV, compact_legal=True)
    assert counters["stats"]["unique_graphs"] == before, "collect が compile した版を通った"
    print("collect は compile した版を通らない", flush=True)


def check_rejects():
    try:
        RNaDLearner(PolicyValueNet(8, 1, 91), RNaDConfig(), DEV, compile="reduce-overhead")
    except ValueError:
        pass
    else:
        raise AssertionError("知らない mode を通した")
    print("知らない mode を止める", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_rejects()
    check_rnad()
    check_mmd()
    check_no_recompile()
    check_collect_untouched()
