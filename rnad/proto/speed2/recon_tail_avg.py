"""F8: 段階の後半の一様な平均（`--tail-avg` の tail_<step>.bin）を、既存の書き出し（policy_<step>.bin = ターゲット）
から遡って作る（docs/rl-learning-speed.md「さらに速くする」の T3。学習には使っていない）。

    python rnad/proto/speed2/recon_tail_avg.py --self-test
    # 段階の後半に e 更新ごとの書き出しが残っている run で（e は --export-every。250 なら 21本）
    python rnad/proto/speed2/recon_tail_avg.py --dir out_rnad/deduce_s10000 --stage-start 78000 --dm 10000 \\
        --every 250 --out tail_recon_88000.bin

ターゲットは更新ごとに target ← target + τ(現在 − target) なので、e 更新離れた2つの書き出しから
    W_t = (target_t − r·target_{t−e}) / (1 − r)、r = (1 − τ)^e（τ=0.001・e=250 なら r = 0.779）
が (t − e, t] の現在のネットのほぼ一様な平均（重みの端どうしの比は r）になる。段階の後半 (start + Δm/2, start + Δm] を
e ごとに区切った W_t を平均すれば `--tail-avg` の平均に近い。
- `--self-test`: (1) float32 のベクトルの列（遅い漂流 + 相関 300更新の雑音）で、再構成と本当の一様平均の差を
  |最後の書き出し − 一様平均| と比べる（調べたときは 0.35%）。(2) 小さいネット（8ch×1）の重みで同じことをし、
  書き出しのファイル（BatchNorm を畳み込みへ畳んだ形）を通しても近いことを見る。畳んだ重みは w·γ の積なので
  厳密には線形でなく、γ が段階の中で動く分だけ近似になる
- 判定: 再構成と tail の差が |最後の書き出し − tail| の数% 以内なら、遡って作った平均を固定パネルで測って T3
  （書き出し・凍結を段階後半の平均で作るか）を判断してよい。`--compare` に本物の tail_<step>.bin を渡せば
  実際の run で確かめられる
"""

import argparse
import pathlib
import re
import sys
import tempfile

import numpy as np
import torch

import proto_common  # noqa: F401  （rnad を読めるようにする）
from export_weights import read, write  # noqa: E402
from model import PolicyValueNet  # noqa: E402


def vec(net):
    return torch.nn.utils.parameters_to_vector(net.parameters()).detach().double()


def reconstruct(files, start, dm, every, tau):
    """files = {step: 書き出しのパス}。(再構成のベクトル, 最後の書き出しのベクトル, 型にするネット) を返す"""
    half = dm // 2
    if (dm - half) % every:
        raise SystemExit(f"段階の後半 {dm - half} 更新が書き出しの間隔 {every} で割り切れない")
    need = list(range(start + half, start + dm + 1, every))
    missing = [s for s in need if s not in files]
    if missing:
        raise SystemExit(f"書き出しが足りない: {missing[:5]}…（{len(missing)}個）")
    r = (1.0 - tau) ** every
    vecs = {s: vec(read(files[s])) for s in need}
    ws = [(vecs[t] - r * vecs[t - every]) / (1 - r) for t in need[1:]]
    return torch.stack(ws).mean(0), vecs[need[-1]], read(files[need[-1]])


def rel(a, b, last):
    return float((a - b).norm() / (last - b).norm())


def synthetic_vectors():
    """遅い漂流 + OU 雑音（相関 300更新）の列で、rnad.py と同じ順（更新 → target += τ(p − target)）に float32 で回す"""
    rng = np.random.default_rng(0)
    D, tau, L, S, every = 2000, 0.001, 10000, 60000, 250
    cur = np.zeros(D, np.float32)
    tgt = np.zeros(D, np.float32)
    ou = np.zeros(D, np.float32)
    drift = rng.standard_normal(D).astype(np.float32) * 1e-4
    exports = {}
    tail_sum = np.zeros(D, np.float64)
    n = 0
    rho = np.exp(-1 / 300)
    for step in range(1, S + L + 1):  # step = 更新の後の step_count
        ou = (rho * ou + np.sqrt(1 - rho ** 2) * rng.standard_normal(D) * 0.01).astype(np.float32)
        cur = (drift * step + ou).astype(np.float32)
        tgt += np.float32(tau) * (cur - tgt)
        if step - 1 >= S + L // 2:  # train_rnad.py の update_tail と同じ条件（step_count − 1 − start ≥ size//2）
            tail_sum += cur
            n += 1
        if step % every == 0 and step >= S + L // 2:
            exports[step] = tgt.copy()
    tail = tail_sum / n
    r = (1 - tau) ** every
    ts = sorted(exports)
    recon = np.mean([(exports[t].astype(np.float64) - r * exports[t - every]) / (1 - r) for t in ts[1:]], axis=0)
    naive = np.mean([exports[t] for t in ts], axis=0)
    last = exports[S + L]
    e = lambda a: np.linalg.norm(a - tail) / np.linalg.norm(last - tail)  # noqa: E731
    print(f"(1) ベクトル: 書き出し {len(ts)} 本、tail の更新数 {n}。|再構成 − tail| / |最後 − tail| = {e(recon):.4f}"
          f"（素朴な書き出しの平均なら {e(naive):.4f}）")
    return e(recon)


def synthetic_net(tmp):
    """小さいネットの重みで、書き出しのファイル（BatchNorm を畳んだ形）を通して再構成する"""
    torch.manual_seed(0)
    tau, L, S, every = 0.001, 2000, 3000, 50
    net = PolicyValueNet(8, 1).eval()
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 1.5)
    theta0 = vec(net).float()
    D = len(theta0)
    g = torch.Generator().manual_seed(1)
    drift = torch.randn(D, generator=g) * 2e-5
    ou = torch.zeros(D)
    rho = float(np.exp(-1 / 300))
    tgt = theta0.clone()
    tail_sum = torch.zeros(D, dtype=torch.float64)
    n = 0

    def save(v, path):
        torch.nn.utils.vector_to_parameters(v.float(), net.parameters())
        write(net, path)

    for step in range(1, S + L + 1):
        ou = rho * ou + (1 - rho ** 2) ** 0.5 * torch.randn(D, generator=g) * 0.003
        cur = theta0 + drift * step + ou
        tgt += tau * (cur - tgt)
        if step - 1 >= S + L // 2:
            tail_sum += cur.double()
            n += 1
        if step % every == 0 and step >= S + L // 2:
            save(tgt, tmp / f"policy_{step}.bin")
    save(tail_sum / n, tmp / f"tail_{S + L}.bin")
    files = {int(re.fullmatch(r"policy_(\d+)\.bin", p.name).group(1)): p for p in tmp.glob("policy_*.bin")}
    recon, last, _ = reconstruct(files, S, L, every, tau)
    tail = vec(read(tmp / f"tail_{S + L}.bin"))
    e = rel(recon, tail, last)
    print(f"(2) 小さいネット（{D} パラメータ、書き出し {len(files)} 本、BatchNorm を畳んだファイル）: "
          f"|再構成 − tail| / |最後 − tail| = {e:.4f}")
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true", help="合成データで再構成の誤差を見る")
    ap.add_argument("--dir", default="", help="run の出力ディレクトリ")
    ap.add_argument("--prefix", default="policy")
    ap.add_argument("--stage-start", type=int, default=0, help="段階の始まりの step")
    ap.add_argument("--dm", type=int, default=10000)
    ap.add_argument("--every", type=int, default=250, help="使う書き出しの間隔（--export-every の倍数）")
    ap.add_argument("--tau", type=float, default=0.001, help="ターゲットの平均化係数（--target-avg）")
    ap.add_argument("--out", default="", help="再構成した平均の書き出し（TSRLPV01）")
    ap.add_argument("--compare", default="", help="本物の tail_<step>.bin（--tail-avg の run）と比べる")
    args = ap.parse_args()
    if args.self_test or not args.dir:
        e1 = synthetic_vectors()
        with tempfile.TemporaryDirectory() as td:
            e2 = synthetic_net(pathlib.Path(td))
        ok = e1 < 0.01 and e2 < 0.05
        print("再構成は一様平均に近い" if ok else "再構成の誤差が大きい")
        sys.exit(0 if ok else 1)
    d = pathlib.Path(args.dir)
    files = {}
    for p in d.glob(f"{args.prefix}_*.bin"):
        m = re.fullmatch(rf"{re.escape(args.prefix)}_(\d+)\.bin", p.name)
        if m:
            files[int(m.group(1))] = p
    recon, last, net = reconstruct(files, args.stage_start, args.dm, args.every, args.tau)
    print(f"{args.stage_start + args.dm // 2}〜{args.stage_start + args.dm} の書き出し "
          f"{(args.dm - args.dm // 2) // args.every + 1} 本から再構成。|再構成 − 最後の書き出し| / |最後| = "
          f"{float((recon - last).norm() / last.norm()):.2e}")
    if args.compare:
        tail = vec(read(args.compare))
        print(f"|再構成 − tail| / |最後 − tail| = {rel(recon, tail, last):.4f}")
    if args.out:
        torch.nn.utils.vector_to_parameters(recon.float(), net.parameters())
        write(net, pathlib.Path(args.out))
        print(f"書き出し: {args.out}")


if __name__ == "__main__":
    main()
