"""自玉の逃げ場を当てる補助ヘッド（MMDConfig.aux_escape、`--aux-escape`）の検査（CPU）。
tsuitate_rl（VecEnv.king_escape がある版）が要る。

    .venv/bin/python rnad/tests/test_aux_escape.py

- collect(king_escape=True) は記録した手番ごとに [N, 8] のラベル（−1 / 0 / 1）と受理手の数 escape_ply を返し、ほかのキーは king_escape=False と
  ビット単位で同じ（乱数を消費しない）。収集の速い経路（fast="py" / "rust"）でも同じ
- aux_escape=0 の update は補助ヘッドを作らず、king_escape の有無で結果が変わらない（従来と同じ）
- aux_escape>0 の update は同じ batch で繰り返すと補助損失が下がる。ネットの形と書き出しの state_dict は変わらない。
  --aux-class と一緒でも回る。手数帯別の正解率（acc_escape_p<下限>）と「動ける」の割合（escape_pos_p<下限>）が出る
- state_dict は補助ヘッドと optimizer を持ち回り、2更新→保存→読み込み→2更新が4更新を続けたものと一致する
- train_rnad.py: --algo mmd --aux-escape（値なし = 既定の重み）で回り、ログに loss_escape / acc_escape が出る。
  --fast-collect・--pipeline とも回る。R-NaD とは起動時に止まる
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
from model import NUM_KING_ESCAPE, PolicyValueNet  # noqa: E402
from rnad import aux_escape_loss, cfg_diff, collect  # noqa: E402

DEV = torch.device("cpu")
TRAIN = HERE.parent / "train_rnad.py"


def net():
    torch.manual_seed(0)
    return PolicyValueNet(16, 1, 91).eval()


def batch(king_escape, seed=7, fast="", move_class=False):
    gen = torch.Generator().manual_seed(seed)
    env = tsuitate_rl.VecEnv(8, seed=seed, auto_reset=False, planes=91)
    return collect(env, net(), DEV, generator=gen, compact_legal=True, actor_extras=True, king_escape=king_escape,
                   move_class=move_class, fast=fast)


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def check_collect():
    a, b = batch(False), batch(True)
    for k, v in a.items():
        if torch.is_tensor(v):
            assert same(v, b[k]), f"king_escape=True で {k} が変わった"
    ke = b["king_escape"]
    assert ke.shape == (len(b["action"]), NUM_KING_ESCAPE) and ke.dtype == torch.int8
    assert set(ke.unique().tolist()) <= {-1, 0, 1}
    assert len(tsuitate_rl.KING_ESCAPE_NAMES) == NUM_KING_ESCAPE
    # 初手（先手）: 前左・前・前右 だけ動ける（左右は金、後ろは盤外）
    first = (b["t"] == 0).nonzero()[0, 0]
    assert ke[first].tolist() == [1, 1, 1, -1, -1, -1, -1, -1], ke[first].tolist()
    ply = b["escape_ply"]
    assert ply.shape == b["action"].shape and int(ply[first]) == 0 and int(ply.min()) >= 0
    # 受理手の数は同じ局の中で t とともに減らない（反則では増えない）
    for g in b["b"].unique().tolist():
        rows = (b["b"] == g).nonzero().squeeze(1)
        p = ply[rows[b["t"][rows].argsort()]]
        assert bool((p[1:] >= p[:-1]).all()), g
    n_ok = int((ke >= 0).sum())
    print(f"T1 collect(king_escape=True): {len(ke)} 行、数えるマス {n_ok}（うち動ける {int((ke == 1).sum())}）。"
          "ほかのキーはビット単位で同じ")
    return b


def check_fast(ref):
    paths = ["py"] + (["rust"] if hasattr(tsuitate_rl.VecEnv, "observe_alive_f16_csr") else [])
    for fast in paths:
        b = batch(True, fast=fast)
        for k, v in ref.items():
            if torch.is_tensor(v):
                assert same(v, b[k]), f"fast={fast} で {k} が従来の経路と違う"
        assert set(b) == set(ref), (set(b) ^ set(ref))
    print(f"T1b 収集の速い経路（{' / '.join(paths)}）: ラベルを含む batch が従来の経路とビット単位で同じ")


def check_loss_fn():
    logits = torch.tensor([[0.0, 2.0, -2.0, 5.0], [0.0, 0.0, 0.0, 0.0]])
    label = torch.tensor([[1, 0, -1, -1], [-1, -1, -1, -1]])
    got = aux_escape_loss(logits, label)
    want0 = (torch.nn.functional.softplus(torch.tensor(-0.0)) + torch.nn.functional.softplus(torch.tensor(2.0))) / 2
    assert torch.allclose(got, torch.stack([want0, torch.tensor(0.0)])), got
    print("T0 aux_escape_loss: −1 のマスを数えず、数えるマスの無い行は 0")


def learner(cfg, seed=3):
    n0 = net()
    L = MMDLearner(copy.deepcopy(n0), cfg, DEV, seed=seed)
    L.start_from(n0.state_dict(), n0.state_dict(), 0)
    return L


def strip(lg):
    return {k: v for k, v in lg.items() if not k.endswith("_sec")}


def check_off(bt):
    cfg = make_config("tsuitate", chunk=64, steps=2)
    plain = {k: v for k, v in bt.items() if k != "king_escape"}
    a, b = learner(cfg), learner(cfg)
    la, lb = strip(a.update(dict(plain))), strip(b.update(dict(bt)))
    assert a.aux_esc is None and b.aux_esc is None and "loss_escape" not in lb
    assert la == lb
    for (k, x), (_, y) in zip(a.net.state_dict().items(), b.net.state_dict().items()):
        assert same(x, y), k
    assert "aux_esc" not in a.state_dict()
    print("T2 aux_escape=0: 補助ヘッドを作らず、king_escape の有無で更新がビット単位で同じ")


def check_on(bt):
    cfg = make_config("tsuitate", chunk=64, steps=2, filter="none", aux_escape=1.0)
    L = learner(cfg)
    keys = set(L.net.state_dict())
    logs = [L.update(dict(bt)) for _ in range(25)]
    losses = [lg["loss_escape"] for lg in logs]
    assert losses[-1] < losses[0] * 0.9, losses  # 学習率が小さいので下がり方はゆっくり（25更新で約 0.69 → 0.59）
    assert 0.0 <= logs[-1]["acc_escape"] <= 1.0
    # 手数帯別: 帯の数えるマスの合計が全体と一致し、正解率と「動ける」の割合は [0, 1]
    from mmd import ESCAPE_PLY_BANDS
    lg = logs[-1]
    assert sum(lg[f"escape_cells_p{lo}"] for lo in ESCAPE_PLY_BANDS) == int((bt["king_escape"] >= 0).sum())
    bands = [lo for lo in ESCAPE_PLY_BANDS if lg[f"escape_cells_p{lo}"]]
    assert bands and all(0.0 <= lg[f"acc_escape_p{lo}"] <= 1.0 and 0.0 <= lg[f"escape_pos_p{lo}"] <= 1.0 for lo in bands)
    # 「動ける」の割合は正解率と同じ行・同じ重み: filter="none"（全行が重み 1）なら、数えるマスのある行の
    # 「行ごとの動けるの割合」の平均に一致する（マスをプールした割合ではない）
    ke = bt["king_escape"]
    okc = ke >= 0
    has = okc.sum(1) > 0
    want = float(((ke == 1).sum(1)[has] / okc.sum(1)[has]).mean())
    assert abs(lg["escape_pos"] - want) < 1e-5, (lg["escape_pos"], want)
    assert min(lg[f"acc_escape_p{lo}"] for lo in bands) - 1e-6 <= lg["acc_escape"] <= max(
        lg[f"acc_escape_p{lo}"] for lo in bands) + 1e-6
    band_txt =" ".join(f"p{lo}: {lg[f'acc_escape_p{lo}']:.3f}/{lg[f'escape_pos_p{lo}']:.3f}" for lo in bands)
    assert set(L.net.state_dict()) == keys  # 補助ヘッドはネットの外（書き出しに入らない）
    sd = L.state_dict()
    assert "aux_esc" in sd and "aux_esc_opt" in sd
    plain = {k: v for k, v in bt.items() if k != "king_escape"}
    try:
        L.update(plain)
    except ValueError:
        pass
    else:
        raise AssertionError("king_escape の無い batch で aux_escape>0 の update が通った")
    # 手の区分の補助ヘッドと一緒でも回る（幹は1回だけ通す）
    both = batch(True, move_class=True)
    M = learner(make_config("tsuitate", chunk=64, steps=2, aux_escape=0.1, aux_class=0.5))
    lg = M.update(dict(both))
    assert "loss_aux" in lg and "loss_escape" in lg
    print(f"T3 aux_escape=1: 同じ batch で25更新して補助損失 {losses[0]:.3f} → {losses[-1]:.3f}"
          f"（正解率 {logs[0]['acc_escape']:.3f} → {logs[-1]['acc_escape']:.3f}。手数帯別の正解率/動けるの割合 {band_txt}）、"
          "ネットの形は不変。--aux-class と併用可")


def check_resume():
    batches = [batch(True, seed=20 + i) for i in range(4)]
    cfg = make_config("tsuitate", chunk=64, steps=2, aux_escape=0.5)
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
    for (k, x), (_, y) in zip(a.aux_esc.state_dict().items(), c.aux_esc.state_dict().items()):
        assert same(x, y), f"再開すると aux_esc.{k} が違う"
    for (k, x), (_, y) in zip(a.net.state_dict().items(), c.net.state_dict().items()):
        assert same(x, y), f"再開すると net.{k} が違う"
    # 設定の上書き（--override-config）: Adam の β₁ を変えると、checkpoint から読んだ補助ヘッドの optimizer にも入る
    buf.seek(0)
    both_cfg = make_config("tsuitate", chunk=64, steps=2, aux_escape=0.5, aux_class=0.5)
    d = learner(both_cfg)
    d.update(dict(batch(True, seed=30, move_class=True)))
    new = make_config("tsuitate", chunk=64, steps=2, aux_escape=0.5, aux_class=0.5, adam_b1=0.5)
    e = MMDLearner(PolicyValueNet(16, 1, 91), new, DEV, seed=99)
    sd = d.state_dict()
    e.load_state_dict(sd)
    assert e.aux_esc_opt.param_groups[0]["betas"][0] == 0.9  # 読んだ直後は checkpoint の値
    e.apply_cfg(new, old=sd["cfg"])
    for name, opt in (("net", e.opt), ("aux", e.aux_opt), ("aux_esc", e.aux_esc_opt)):
        assert all(g["betas"][0] == 0.5 for g in opt.param_groups), name
    # escape_ply の無い batch は更新を始める前に止まる（学習の状態を動かさない）
    step0 = e.step_count
    try:
        e.update({k: v for k, v in batch(True, seed=30, move_class=True).items() if k != "escape_ply"})
    except ValueError:
        assert e.step_count == step0
    else:
        raise AssertionError("escape_ply の無い batch で aux_escape>0 の update が通った")
    old = dict(MMDConfig().__dict__)
    del old["aux_escape"]
    assert cfg_diff(old, MMDConfig()) == {}
    print("T4 再開: 4更新を続けたものと 2更新→保存→読み込み→2更新が、ネット・補助ヘッド・ログで一致。"
          "古い checkpoint の設定は差にならない")


def run_train(out, *extra, algo="mmd"):
    cmd = [sys.executable, str(TRAIN), "--algo", algo, "--init", "none", "--anchor", "none", "--batch", "4",
           "--steps", "2", "--channels", "16", "--blocks", "1", "--planes", "91", "--compact-legal",
           "--eval-every", "0", "--export-every", "2", "--checkpoint-every", "2", "--out", str(out),
           "--threads", "2", "--device", "cpu", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def check_train(tmp):
    r = run_train(tmp / "ok", "--aux-escape")
    assert r.returncode == 0, r.stderr[-1500:]
    recs = [json.loads(line) for line in open(tmp / "ok" / "log.jsonl") if line.strip()]
    assert all("loss_escape" in json.dumps(rec) and "acc_escape_p0" in json.dumps(rec) for rec in recs), recs[-1].keys()
    ck = torch.load(tmp / "ok" / "checkpoint.pt", map_location="cpu")
    assert "aux_esc" in ck
    assert ck["cfg"]["aux_escape"] == 0.1, ck["cfg"].get("aux_escape")  # 値なしの --aux-escape = 既定の重み
    r = run_train(tmp / "ok", "--aux-escape", "--steps", "4")
    assert r.returncode == 0, r.stderr[-1500:]
    for i, extra in enumerate([["--fast-collect", "py"], ["--pipeline"], ["--pipeline", "--fast-collect", "py"]]):
        r = run_train(tmp / f"fast{i}", "--aux-escape", "0.2", *extra)
        assert r.returncode == 0, (extra, r.stderr[-1500:])
        recs = [json.loads(line) for line in open(tmp / f"fast{i}" / "log.jsonl") if line.strip()]
        assert recs and all("loss_escape" in json.dumps(rec) for rec in recs), extra
    r = run_train(tmp / "rej", "--aux-escape", algo="rnad")
    assert r.returncode != 0 and "--algo mmd のときだけ" in r.stderr, r.stderr[-500:]
    print("T5 train_rnad.py: --algo mmd --aux-escape で回り（再開も）、ログと checkpoint に補助ヘッドが残る。"
          "--fast-collect・--pipeline（とその両方）でも回る。R-NaD とは止まる")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_loss_fn()
    bt = check_collect()
    check_fast(bt)
    check_off(bt)
    check_on(bt)
    check_resume()
    with tempfile.TemporaryDirectory() as d:
        check_train(pathlib.Path(d))
