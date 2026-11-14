"""報酬の整形（`--shape`）を、序盤の強制（`--open-p`）・速い収集（`--fast-collect`）・`--pipeline` と一緒に使う形の検査（CPU）。

    .venv/bin/python rnad/tests/test_shape_forced.py

- F1 `ShapeRun`: 強制の手が指された step の Φ の変化は報酬にならず、基準へ繰り入れられる（合成の信号）
- F2 `collect`: 強制つきの自己対局で、速い収集（py / rust）が従来の収集と同じ batch（整形込みの報酬も）を返す。
  反対称な重みは零和のまま、1局の整形の合計は「終局の Φ − 繰り入れ後の基準」を切ったもの
- F3 `train_rnad.py`: `--shape` ＋ `--open-p 1` ＋ `--pipeline` ＋ `--fast-collect` で回り、log に signals・shape_mult が残る
"""

import json
import math
import pathlib
import sys
import tempfile

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import tsuitate_rl  # noqa: E402
import openings  # noqa: E402
from model import NUM_PLANES_DEDUCED  # noqa: E402
from rnad import MultiEnv, collect  # noqa: E402
from shaping import ShapeRun, Shaping  # noqa: E402
from test_openings import write_book  # noqa: E402
from test_shaping import DEV, K, NAMES, run_train, small_net, synth  # noqa: E402

HAS_RUST = hasattr(tsuitate_rl.VecEnv, "observe_alive_f16_csr")
SPEC = "d_capture_value=0.05,d_king_attack=0.03"


def check_shape_run():
    B, steps, cap, mult = 32, 12, 0.6, 0.5
    sig, end = synth(B, steps, 3)
    rng = np.random.default_rng(4)
    # 強制の手は局の頭のほうに寄せる（step ごとに、どちらかの側の数が増えるかどうか）
    inc = (rng.random((steps, B, 2)) < np.linspace(0.8, 0.0, steps)[:, None, None]).astype(np.int64)
    forced = np.concatenate([rng.integers(0, 3, size=(1, B, 2)), np.zeros((steps, B, 2), np.int64)])
    forced = forced[:1] + np.concatenate([np.zeros((1, B, 2), np.int64), np.cumsum(inc, 0)])
    sh = Shaping.from_spec("d_capture_value=0.05,fouls=-0.04,opp_checks=0.03", "plain", cap, names=NAMES)
    run = ShapeRun(NAMES, sig[0], sh, mult, forced0=forced[0])
    total = np.zeros((B, 2))
    acc = np.zeros((B, 2))  # 強制の手が無い step の Φ の変化だけの累計（切る前）
    cur, cur_f = sig[0].copy(), forced[0].copy()
    hit = 0
    for t in range(1, steps + 1):
        alive = end >= t
        prev = sh.potential(cur)
        cur = np.where(alive[:, None, None], sig[t], cur)
        moved = alive & (forced[t].sum(1) > cur_f.sum(1))
        cur_f = np.where(alive[:, None], forced[t], cur_f)
        add = run.step(cur, end == t, np.zeros((B, 2), np.float32), forced=cur_f)
        assert (add[moved] == 0).all(), "強制の手が指された step に整形が付いた"
        assert (add[~alive] == 0).all()
        acc += np.where((alive & ~moved)[:, None], sh.potential(cur) - prev, 0.0)
        total += add
        hit += int(moved.sum())
    want = np.clip(acc, -cap, cap) * mult
    assert np.allclose(total, want, atol=1e-5), np.abs(total - want).max()
    assert hit > B and (np.abs(want) > 1e-3).any(), "強制のある step・整形の付く局の両方が要る"
    # 基準へ繰り入れた後の関係: 合計 = 切った（終局の Φ − 基準）
    assert np.allclose(total, np.clip(sh.potential(run.sig) - run.phi0, -cap, cap) * mult, atol=1e-5)
    # 強制の数を渡さなければ従来と同じ（初期値からの差）
    plain = ShapeRun(NAMES, sig[0], sh, mult)
    cur = sig[0].copy()
    tot0 = np.zeros((B, 2))
    for t in range(1, steps + 1):
        cur = np.where((end >= t)[:, None, None], sig[t], cur)
        tot0 += plain.step(cur, end == t, np.zeros((B, 2), np.float32))
    assert np.allclose(tot0, np.clip(sh.potential(plain.sig) - sh.potential(sig[0]), -cap, cap) * mult, atol=1e-5)
    assert not np.allclose(tot0, total, atol=1e-4)
    # 作るときと step で渡し方が違えば止める
    for run_, kw in ((ShapeRun(NAMES, sig[0], sh, forced0=forced[0]), {}),
                     (ShapeRun(NAMES, sig[0], sh), {"forced": forced[1]})):
        try:
            run_.step(sig[1], end == 1, np.zeros((B, 2), np.float32), **kw)
        except ValueError:
            continue
        raise AssertionError("強制の数の渡し方の食い違いを通した")
    print("F1 ShapeRun: 強制の手が指された step の Φ の変化は報酬にせず基準へ繰り入れる。渡さなければ従来と同じ")


def check_collect(tmp):
    book = openings.load_book(write_book(tmp))
    net = small_net()
    n, cap = 24, 0.5
    sh = Shaping.from_spec(SPEC, "plain", cap, names=NAMES)

    def make():
        return tsuitate_rl.VecEnv(n, seed=5, auto_reset=False, planes=NUM_PLANES_DEDUCED,
                                  **openings.env_kwargs(book, 1.0, sides="both"))

    def run(fast, wrap=False, **kw):
        env = make()
        gen = torch.Generator().manual_seed(11)
        return collect(MultiEnv([env]) if wrap else env, net, DEV, generator=gen, compact_legal=True, fast=fast,
                       shaping=sh, shape_mult=0.7, actor_extras=True, **kw)

    twin = make()
    twin.reset()
    forced_at_reset = twin.forced_moves().sum(1)
    base = run("")
    r = base["shape_run"]
    assert (forced_at_reset >= 2).all(), "p = 1・両者なので、収集の頭で両方の側が1手以上を定跡で指している"
    assert (r.forced_n > forced_at_reset).any(), "方策の手番が始まった後にも強制の手が要る（片方の側だけが先にやめた局）"
    # 反対称な重みは零和のまま。1局の整形の合計 = 切った（終局の Φ − 繰り入れ後の基準）× 倍率
    assert float(base["reward"].sum(-1).abs().max()) < 1e-6
    assert np.allclose(r.total, np.clip(sh.potential(r.sig) - r.phi0, -cap, cap) * 0.7, atol=1e-5)
    assert float(np.abs(r.total).sum()) > 0, "整形が載っていない"
    # 強制の手を見ない形（従来の式）とは違う報酬になる = 繰り入れが効いている
    old = np.clip(sh.potential(r.sig) - sh.potential(twin.signals()), -cap, cap) * 0.7
    assert not np.allclose(old, r.total, atol=1e-4)
    for fast in ["py"] + (["rust"] if HAS_RUST else []):
        for wrap in (False, True):
            b = run(fast, wrap)
            bad = [k for k in base if k != "shape_run" and not (
                torch.equal(b[k], base[k]) if torch.is_tensor(base[k]) else np.array_equal(b[k], base[k]))]
            assert not bad and set(b) == set(base), (fast, wrap, bad)
            assert np.array_equal(b["shape_run"].total, r.total) and np.array_equal(b["shape_run"].sig, r.sig)
    # 整形なしの監視だけでも速い収集で回り、対局は同じ
    w = run("py", watch_signals=True)
    assert torch.equal(w["action"], base["action"]) and "shape_run" in w
    print(f"F2 collect: 強制つきの自己対局で fast={'/'.join(['py'] + (['rust'] if HAS_RUST else []))} が従来と同じ batch"
          "（整形込みの報酬も）。零和のまま、強制の手で動いた Φ は報酬にならない"
          + ("" if HAS_RUST else "（rl-env が古いので fast='rust' は確かめていない）"))


def check_train(tmp):
    book = write_book(tmp)
    common = ["--shape", SPEC, "--shape-decay", "1:3", "--open-book", str(book), "--open-p", "1.0"]
    logs = {}
    for name, extra in (("plain", []), ("fast", ["--fast-collect", "py"]),
                        ("pipe", ["--pipeline", "--fast-collect", "rust" if HAS_RUST else "py"])):
        out = tmp / name
        run_train(out, *common, *extra)
        lines = [json.loads(x) for x in open(out / "log.jsonl") if x.strip()]
        assert len(lines) == 3 and [x["shape_mult"] for x in lines] == [1.0, 1.0, 0.5], name
        for x in lines:
            assert x["reward_scale"] == 2.0 and math.isfinite(x["loss_v"]) and math.isfinite(x["loss_nerd"])
            assert x["signals"]["win"]["final"]["n"] + x["signals"]["loss"]["final"]["n"] \
                + x["signals"]["draw"]["final"]["n"] == 12, "自己対局は両方の側を数える（6局 × 2）"
            assert "shaped_total" in x["signals"] and "opening" in x
        logs[name] = lines
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["shaping"]["own"] and len(cfg["signal_names"]) == K
    # 速い収集は従来と同じ batch なので、ログも同じ（時間の項を除く）
    strip = lambda rows: [{k: v for k, v in x.items() if not k.endswith("_sec")} for x in rows]  # noqa: E731
    assert strip(logs["plain"]) == strip(logs["fast"])
    # 使えない組み合わせ: --pipeline のリーグ（監視値の学習側の色が取れない）
    r = run_train(tmp / "rej", "--shape", "fouls=-0.1", "--pipeline", "--league", "heuristic", steps=1, check=False)
    assert r.returncode != 0 and "--pipeline" in (r.stderr + r.stdout), r.stderr[-600:]
    print("F3 train_rnad.py: --shape ＋ --open-p 1 が、従来の収集・--fast-collect・--pipeline のどれでも回り、"
          "log に signals・shape_mult・reward_scale が残る。速い収集のログは従来と同じ")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_shape_run()
    with tempfile.TemporaryDirectory() as d:
        check_collect(pathlib.Path(d))
        check_train(pathlib.Path(d))
    print("ok")
