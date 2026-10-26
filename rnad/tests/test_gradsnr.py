"""`gradsnr.u_stat_jackknife` の検査（CPU、数秒）。

- U 統計量が、同じ局どうしの項を除いた内積の素直な平均（i ≠ j の全組）と一致
- 1組抜きの U が、その組を除いた局だけから組み直した U と一致し、標準誤差がジャックナイフの式どおり
- 雑音が勝つ模擬（B_noise ≈ 局数）で、標準誤差が繰り返しの U の標準偏差の 0.9〜1.6倍に入る
"""

import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gradsnr import u_stat_jackknife  # noqa: E402


def blocks(g, K):
    """局ごとの勾配 g（局 × 変種 × 次元）を、局番号 mod K の組に積む（gradsnr.py の main と同じ分け方）"""
    N, V, P = g.shape
    Sblk = torch.zeros(K, V, P, dtype=torch.float64)
    cross = torch.zeros(K, V, V, dtype=torch.float64)
    for i in range(N):
        Sblk[i % K] += g[i]
        cross[i % K] += g[i] @ g[i].T
    nk = torch.bincount(torch.arange(N) % K, minlength=K).double()
    return Sblk, cross, nk


def u_direct(g):
    """i ≠ j の全組の g_a,i·g_b,j の平均"""
    N = len(g)
    tot = torch.zeros(g.shape[1], g.shape[1], dtype=torch.float64)
    for i in range(N):
        for j in range(N):
            if i != j:
                tot += g[i] @ g[j].T
    return tot / (N * (N - 1))


def check_exact():
    gen = torch.Generator().manual_seed(0)
    N, V, P, K = 23, 3, 5, 6
    g = torch.randn(N, V, P, generator=gen, dtype=torch.float64) + torch.randn(1, V, P, generator=gen, dtype=torch.float64)
    U, se = u_stat_jackknife(*blocks(g, K))
    assert torch.allclose(U, u_direct(g), atol=1e-12)
    dels = torch.stack([u_direct(g[[i for i in range(N) if i % K != k]]) for k in range(K)])
    want = torch.sqrt((K - 1) / K * ((dels - dels.mean(0)) ** 2).sum(0))
    assert torch.allclose(se, want, atol=1e-12)
    print(f"U 統計量と1組抜きのジャックナイフ: 素直な計算と一致（{N}局・{V}変種・{K}組）")


def check_coverage():
    """雑音が勝つ模擬: |G|² = 1、B_noise = 局数（等方な雑音）。標準誤差の平均と U の標準偏差を比べる"""
    gen = torch.Generator().manual_seed(1)
    N, P, K, reps = 256, 64, 32, 300
    G = torch.zeros(P, dtype=torch.float64)
    G[0] = 1.0
    sd = (N / P) ** 0.5  # trΣ = P·sd² = N → B_noise = N
    Us, ses = [], []
    for _ in range(reps):
        g = (G + sd * torch.randn(N, P, generator=gen, dtype=torch.float64))[:, None, :]
        U, se = u_stat_jackknife(*blocks(g, K))
        Us.append(float(U[0, 0]))
        ses.append(float(se[0, 0]))
    ratio = (sum(ses) / reps) / float(torch.tensor(Us).std())
    assert 0.9 <= ratio <= 1.6, ratio
    print(f"雑音が勝つ模擬（B_noise = 局数 {N}、{K}組）: 標準誤差 / 真の標準偏差 = {ratio:.2f}")


if __name__ == "__main__":
    check_exact()
    check_coverage()
