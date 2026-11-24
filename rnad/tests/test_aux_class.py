"""手の区分ごとの期待値を予測する補助ヘッド（MMDConfig.aux_class、`--aux-class`）の検査（CPU）。
tsuitate_rl（VecEnv.move_classes がある版）が要る。

    .venv/bin/python rnad/tests/test_aux_class.py

- collect(move_class=True) は記録した手番ごとに [N] の区分（0〜11）を返し、ほかのキーは move_class=False と
  ビット単位で同じ（乱数を消費しない）
- aux_class=0 の update は補助ヘッドを作らず、move_class の有無で結果が変わらない（従来と同じ）
- aux_class>0 の update は同じ batch で繰り返すと補助損失が下がる。ネットの形と書き出しの state_dict は変わらない
- state_dict は補助ヘッドと optimizer を持ち回り、3更新→保存→読み込み→3更新が6更新を続けたものと一致する
- checkpoint に無い設定（aux_class）は既定値として比べる（古い checkpoint の再開で差にならない）
- train_rnad.py: --algo mmd --aux-class で回り、ログに loss_aux が出る。R-NaD・--fast-collect・--pipeline とは起動時に止まる
"""

import copy
import io
import json
import pathlib
import subprocess
import sys
import tempfile

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsuitate_rl  # noqa: E402
from mmd import MMDConfig, MMDLearner, make_config  # noqa: E402
from model import NUM_MOVE_CLASSES, PolicyValueNet  # noqa: E402
from rnad import cfg_diff, collect  # noqa: E402

DEV = torch.device("cpu")
TRAIN = HERE.parent / "train_rnad.py"


def net():
    torch.manual_seed(0)
    return PolicyValueNet(16, 1, 91).eval()


def batch(move_class, seed=7):
    gen = torch.Generator().manual_seed(seed)
    env = tsuitate_rl.VecEnv(8, seed=seed, auto_reset=False, planes=91)
    return collect(env, net(), DEV, generator=gen, compact_legal=True, actor_extras=True, move_class=move_class)


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def check_collect():
    a, b = batch(False), batch(True)
    for k, v in a.items():
        if torch.is_tensor(v):
            assert same(v, b[k]), f"move_class=True で {k} が変わった"
    mc = b["move_class"]
    assert mc.shape == b["action"].shape and mc.dtype == torch.int8
    assert int(mc.min()) >= 0 and int(mc.max()) < NUM_MOVE_CLASSES, (int(mc.min()), int(mc.max()))
    assert len(tsuitate_rl.MOVE_CLASS_NAMES) == NUM_MOVE_CLASSES
    counts = torch.bincount(mc.long(), minlength=NUM_MOVE_CLASSES).tolist()
    print(f"T1 collect(move_class=True): {len(mc)} 行に区分が付き、ほかのキーはビット単位で同じ。区分ごとの数 "
          + " ".join(f"{n}={c}" for n, c in zip(tsuitate_rl.MOVE_CLASS_NAMES, counts)))
    return b


def learner(cfg, seed=3):
    n0 = net()
    L = MMDLearner(copy.deepcopy(n0), cfg, DEV, seed=seed)
    L.start_from(n0.state_dict(), n0.state_dict(), 0)
    return L


def strip(lg):
    return {k: v for k, v in lg.items() if not k.endswith("_sec")}


def check_off(bt):
    cfg = make_config("tsuitate", chunk=64, steps=2)
    plain = {k: v for k, v in bt.items() if k != "move_class"}
    a, b = learner(cfg), learner(cfg)
    la, lb = strip(a.update(dict(plain))), strip(b.update(dict(bt)))
    assert a.aux is None and b.aux is None and "loss_aux" not in lb
    assert la == lb
    for (k, x), (_, y) in zip(a.net.state_dict().items(), b.net.state_dict().items()):
        assert same(x, y), k
    assert "aux" not in a.state_dict()
    print("T2 aux_class=0: 補助ヘッドを作らず、move_class の有無で更新がビット単位で同じ")


def check_on(bt):
    cfg = make_config("tsuitate", chunk=64, steps=2, filter="none", aux_class=1.0)
    L = learner(cfg)
    keys = set(L.net.state_dict())
    losses = [L.update(dict(bt))["loss_aux"] for _ in range(25)]
    assert losses[-1] < losses[0] * 0.8, losses
    assert set(L.net.state_dict()) == keys  # 補助ヘッドはネットの外（書き出しに入らない）
    sd = L.state_dict()
    assert "aux" in sd and "aux_opt" in sd
    # 区分の付いていない batch では止まる
    plain = {k: v for k, v in bt.items() if k != "move_class"}
    try:
        L.update(plain)
    except ValueError:
        pass
    else:
        raise AssertionError("move_class の無い batch で aux_class>0 の update が通った")
    print(f"T3 aux_class=1: 同じ batch で25更新して補助損失 {losses[0]:.3f} → {losses[-1]:.3f}、ネットの形は不変")


def check_resume():
    batches = [batch(True, seed=20 + i) for i in range(4)]
    cfg = make_config("tsuitate", chunk=64, steps=2, aux_class=0.5)
    a = learner(cfg)
    logs_a = [strip(a.update(dict(x))) for x in batches]
    b = learner(cfg)
    logs_b = [strip(b.update(dict(x))) for x in batches[:2]]
    buf = io.BytesIO()
    torch.save(b.state_dict(), buf)
    buf.seek(0)
    c = MMDLearner(PolicyValueNet(16, 1, 91), cfg, DEV, seed=99)
    c.load_state_dict(torch.load(buf, map_location="cpu"))
    logs_b += [strip(c.update(dict(x))) for x in batches[2:]]
    assert logs_a == logs_b, [k for k in logs_a[-1] if logs_a[-1][k] != logs_b[-1].get(k)]
    for (k, x), (_, y) in zip(a.aux.state_dict().items(), c.aux.state_dict().items()):
        assert same(x, y), f"再開すると aux.{k} が違う"
    for (k, x), (_, y) in zip(a.net.state_dict().items(), c.net.state_dict().items()):
        assert same(x, y), f"再開すると net.{k} が違う"
    old = dict(MMDConfig().__dict__)
    del old["aux_class"]
    assert cfg_diff(old, MMDConfig()) == {}
    assert cfg_diff(old, MMDConfig(aux_class=0.1)) == {"aux_class": (0.0, 0.1)}
    print("T4 再開: 4更新を続けたものと 2更新→保存→読み込み→2更新が、ネット・補助ヘッド・ログで一致。"
          "古い checkpoint の設定は差にならない")


def run_train(out, *extra, algo="mmd"):
    cmd = [sys.executable, str(TRAIN), "--algo", algo, "--init", "none", "--anchor", "none", "--batch", "4",
           "--steps", "2", "--channels", "16", "--blocks", "1", "--planes", "91", "--compact-legal",
           "--eval-every", "0", "--export-every", "2", "--checkpoint-every", "2", "--out", str(out),
           "--threads", "2", "--device", "cpu", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def check_train(tmp):
    r = run_train(tmp / "ok", "--aux-class", "0.5")
    assert r.returncode == 0, r.stderr[-1500:]
    recs = [json.loads(line) for line in open(tmp / "ok" / "log.jsonl") if line.strip()]
    assert all("loss_aux" in json.dumps(rec) for rec in recs), recs[-1].keys()
    ck = torch.load(tmp / "ok" / "checkpoint.pt", map_location="cpu")
    assert "aux" in ck
    # 再開も通る
    r = run_train(tmp / "ok", "--aux-class", "0.5", "--steps", "4")
    assert r.returncode == 0, r.stderr[-1500:]
    for extra, msg in [(["--fast-collect", "py"], "--fast-collect"), (["--pipeline"], "--pipeline")]:
        r = run_train(tmp / "rej", "--aux-class", "0.5", *extra)
        assert r.returncode != 0 and msg in (r.stderr + r.stdout), (extra, r.stderr[-500:])
    r = run_train(tmp / "rej2", "--aux-class", "0.5", algo="rnad")
    assert r.returncode != 0 and "--algo mmd のときだけ" in r.stderr, r.stderr[-500:]
    print("T5 train_rnad.py: --algo mmd --aux-class で回り（再開も）、ログと checkpoint に補助ヘッドが残る。"
          "R-NaD・--fast-collect・--pipeline とは止まる")


if __name__ == "__main__":
    torch.set_num_threads(2)
    bt = check_collect()
    check_off(bt)
    check_on(bt)
    check_resume()
    with tempfile.TemporaryDirectory() as d:
        check_train(pathlib.Path(d))
