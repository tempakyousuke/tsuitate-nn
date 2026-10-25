"""CPU の仕事を減らした収集（`collect(fast="py" / "rust")`）が従来の `collect(compact_legal=True)` と
**ビット単位で同じ batch** を返すことの検査。tsuitate_rl が要る（"rust" は rl-env を作り直した版。無ければ飛ばす）。

    .venv/bin/python rnad/tests/test_fast_collect.py

- 自己対局の VecEnv・複数の env をつないだ MultiEnv・評価モード（相手は Rust の戦略）の env で、全キーがビット単位で一致
  （観測は fp16 のビット、合法手の添字・個数、行動、μ、時刻・局、valid・player・reward、入玉・前進・勝敗）
- Rust 側の fp16 への丸め（`f32_to_f16_bits`）が numpy の float16 と全部一致（境界・非正規化数・あふれ・NaN・乱数）
- 対象外の設定（相手のネット・入玉ボーナス・truth・密なマスク）は止める
"""

import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from model import PolicyValueNet  # noqa: E402
from rnad import MultiEnv, collect  # noqa: E402

DEV = torch.device("cpu")
HAS_RUST = hasattr(tsuitate_rl.VecEnv, "observe_alive_f16_csr")


def same(a, b):
    """batch どうしがビット単位で同じか（違うキーの一覧を返す）"""
    bad = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            bad.append(k)
        elif isinstance(x, np.ndarray):
            if not (x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)):
                bad.append(k)
        else:
            iv = {torch.float16: torch.int16, torch.float32: torch.int32, torch.float64: torch.int64}
            xv = x.view(iv[x.dtype]) if x.dtype in iv else x
            yv = y.view(iv[y.dtype]) if y.dtype in iv else y
            if not (x.dtype == y.dtype and x.shape == y.shape and torch.equal(xv, yv)):
                bad.append(k)
    return bad


def run(make_env, net, fast):
    gen = torch.Generator().manual_seed(5)
    return collect(make_env(), net, DEV, generator=gen, compact_legal=True, fast=fast)


def check_equal(name, make_env, planes):
    torch.manual_seed(0)
    net = PolicyValueNet(16, 2, planes).eval()
    ref = run(make_env, net, "")
    modes = ["py"] + (["rust"] if HAS_RUST else [])
    for fast in modes:
        bad = same(ref, run(make_env, net, fast))
        assert not bad, f"{name} fast={fast}: 違うキー {bad}"
    alive_share = len(ref["action"]) / ref["valid"].numel()
    print(f"{name}（{planes}枚）: fast={'/'.join(modes)} が従来とビット単位で一致"
          f"（決定点 {len(ref['action'])}、T {ref['valid'].shape[0]}、生きている枠の割合 {alive_share:.2f}）")


def check_f16_bits():
    if not HAS_RUST:
        print("f32_to_f16_bits: rl-env が古いので飛ばす")
        return
    rng = np.random.default_rng(0)
    # f16 の全ビット列の値と、隣どうしの中点（丸めの境目）とその前後1ulp（f32 の）
    h = np.arange(0, 0x10000, dtype=np.uint32).astype(np.uint16).view(np.float16)
    h = h[np.isfinite(h)].astype(np.float32)
    h.sort()
    mid = ((h[:-1].astype(np.float64) + h[1:]) / 2).astype(np.float32)
    near = np.concatenate([mid, np.nextafter(mid, np.float32(np.inf)), np.nextafter(mid, np.float32(-np.inf))])
    special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 65504, 65519.996, 65520, 1e9, -1e9, 1e-10, 2.0 ** -24,
                        2.0 ** -25, 2.0 ** -25 * 1.0000001, 2.0 ** -14, 0.95 ** 7, 1 / 3], dtype=np.float32)
    # 指数の広い範囲の乱数（float32 のビット列を一様に引く）
    rand = rng.integers(0, 2 ** 32, size=1_000_000, dtype=np.uint64).astype(np.uint32).view(np.float32)
    x = np.concatenate([h, near, special, rand])
    got = tsuitate_rl.f32_to_f16_bits(x)
    with np.errstate(over="ignore", invalid="ignore"):
        want = x.astype(np.float16).view(np.uint16)
    nan = np.isnan(x)
    # NaN はどちらも NaN になればよい（仮数のビットの残し方は実装による）
    assert np.isnan(got[nan].view(np.float16)).all()
    bad = np.flatnonzero(got[~nan] != want[~nan])
    assert not len(bad), f"fp16 の丸めが numpy と違う: {x[~nan][bad[:5]]} → {got[~nan][bad[:5]]} 対 {want[~nan][bad[:5]]}"
    print(f"f32_to_f16_bits: {len(x):,} 個の値で numpy の float16 と一致（NaN は {int(nan.sum())} 個とも NaN）")


def check_rejects():
    net = PolicyValueNet(8, 1).eval()
    env = tsuitate_rl.VecEnv(2, seed=0, auto_reset=False)
    for kw in ({"compact_legal": False}, {"compact_legal": True, "truth": True},
               {"compact_legal": True, "nyugyoku_lambda": 1.0},
               {"compact_legal": True, "opponents": [None, ("x", 0)], "opp_nets": {}},
               {"compact_legal": True, "fast": "gpu"}):
        kw = {"fast": "py", **kw}
        try:
            collect(env, net, DEV, **kw)
        except ValueError:
            continue
        raise AssertionError(f"対象外の設定を通した: {kw}")
    print("collect(fast=...): 対象外の設定は止める")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_f16_bits()
    check_rejects()
    check_equal("自己対局", lambda: tsuitate_rl.VecEnv(12, seed=3, auto_reset=False, planes=91), 91)
    check_equal("自己対局", lambda: tsuitate_rl.VecEnv(8, seed=4, auto_reset=False), tsuitate_rl.NUM_PLANES)
    check_equal("MultiEnv", lambda: MultiEnv([tsuitate_rl.VecEnv(5, seed=6, auto_reset=False, planes=91),
                                              tsuitate_rl.VecEnv(7, seed=7, auto_reset=False, planes=91)]), 91)
    check_equal("評価モード（相手 heuristic）", lambda: MultiEnv([
        tsuitate_rl.VecEnv(4, seed=8, auto_reset=False, planes=91),
        tsuitate_rl.VecEnv(6, opponents=["heuristic"], seed=9, auto_reset=False, planes=91)]), 91)
    if not HAS_RUST:
        print("（rl-env が古いので fast='rust' は確かめていない）")
