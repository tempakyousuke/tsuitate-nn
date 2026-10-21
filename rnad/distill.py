"""方策ネットを小さいネットへ蒸留する（学習の速さの代理 P2 の出発点。tsuitate-bot の docs/rl-learning-speed.md）。

    python rnad/distill.py --teacher ../tsuitate-bot/models/rl_deduce_v4.bin --channels 32 --blocks 3 \\
        --games 768 --epochs 4 --out lab/distill_32x3.pt

先生（TSRLPV01 の重み）の自己対局から決定点を集め、生徒の方策を先生の方策へ（合法手の上の KL）、価値を
先生の価値へ（2乗誤差）合わせる。出力は train_rnad.py の `--init` / `--anchor` で読める checkpoint
（`config` と `state_dict`）と、同じ重みの TSRLPV01（`.bin`、speedlab のパネルに使う）。

なぜ要るか: ゼロからの代理（P1）は方策がほぼ一様なまま進むので、本線の後半で効いている
「鋭い方策でまれに選ばれた手の 1/μ が勾配の分散を占める」状況が再現しない。先生の鋭い方策を写した
小さいネットから R-NaD を続ければ、その状況を CPU で回せる大きさで作れる。
BatchNorm は学習でも推論でも eval モード（RNaDLearner と同じ。統計は初期値のまま、アフィンとして学ぶ）
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tsuitate_rl  # noqa: E402
from export_weights import read as read_weights  # noqa: E402
from export_weights import write as write_weights  # noqa: E402
from model import NUM_PLANES, PolicyValueNet  # noqa: E402
from rnad import legal_log_policy, pack_legal, sample_actions  # noqa: E402

DEV = torch.device("cpu")


@torch.no_grad()
def gather(teacher, games, seed):
    """先生の自己対局の決定点: 観測（fp16）・合法手の詰めた添字とマスク・先生の詰めた対数確率・先生の価値"""
    planes = teacher.config["planes"]
    kw = {"planes": planes} if planes != NUM_PLANES else {}
    env = tsuitate_rl.VecEnv(games, seed=seed, auto_reset=False, **kw)
    env.reset()
    obs_l, idx_l, pm_l, lp_l, v_l = [], [], [], [], []
    for _ in range(260):
        alive = env.alive()
        if not alive.any():
            break
        obs, mask, _ = env.observe()
        idx = np.flatnonzero(alive)
        x = torch.from_numpy(obs[idx]).half().float()
        legal = torch.from_numpy(mask[idx])
        logits, v = teacher(x)
        a, _ = sample_actions(teacher, obs[idx], mask[idx], DEV)
        pidx, pm, _ = pack_legal(legal, a)
        lp = legal_log_policy(logits.gather(1, pidx), pm)
        actions = np.full(games, -1, dtype=np.int64)
        actions[idx] = a.numpy()
        env.step(actions)
        obs_l.append(x.half())
        idx_l.append(pidx)
        pm_l.append(pm)
        lp_l.append(lp)
        v_l.append(v)
    return obs_l, idx_l, pm_l, lp_l, v_l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--games", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cache", default="", help="先生の自己対局の決定点を保存・再利用するファイル")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    teacher = read_weights(args.teacher).eval()
    planes = teacher.config["planes"]
    t0 = time.time()
    cache = pathlib.Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        obs_l, idx_l, pm_l, lp_l, v_l = torch.load(cache)
    else:
        obs_l, idx_l, pm_l, lp_l, v_l = gather(teacher, args.games, args.seed + 1)
        if cache is not None:
            torch.save((obs_l, idx_l, pm_l, lp_l, v_l), cache)
    # 手番ごとにバッチの幅（合法手の数の最大）が違うので、決定点を平らにして幅を揃える
    k = max(p.shape[1] for p in pm_l)

    def pad(x, fill):
        return torch.nn.functional.pad(x, (0, k - x.shape[1]), value=fill)

    obs = torch.cat(obs_l)
    pidx = torch.cat([pad(x, 0) for x in idx_l])
    pmask = torch.cat([pad(x, False) for x in pm_l])
    tlp = torch.cat([pad(x, 0.0) for x in lp_l])
    tv = torch.cat(v_l)
    n = len(obs)
    print(f"先生の自己対局 {args.games}局・{n}決定点（{time.time() - t0:.0f}s）", flush=True)

    student = PolicyValueNet(args.channels, args.blocks, planes).eval()
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    perm_gen = torch.Generator().manual_seed(args.seed)
    # 無作為な 5% を検証にする（決定点は手番の順に並んでいるので、先頭を取ると序盤の局面だけになる。
    # 同じ局の決定点が学習側と混ざるが、蒸留の進み具合を見るだけ）
    shuffled = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + 7))
    val, tr = shuffled[: n // 20], shuffled[n // 20:]
    steps = 0
    total = args.epochs * (len(tr) // args.batch)
    for ep in range(args.epochs):
        order = tr[torch.randperm(len(tr), generator=perm_gen)]
        for i in range(0, len(order) - args.batch + 1, args.batch):
            # 学習率は最後へ向けて線形に下げる
            for g in opt.param_groups:
                g["lr"] = args.lr * (1 - steps / total)
            b = order[i:i + args.batch]
            logits, v = student(obs[b].float())
            lp = legal_log_policy(logits.gather(1, pidx[b]), pmask[b])
            p_t = tlp[b].exp() * pmask[b]
            kl = (p_t * (tlp[b] - lp)).sum(1).mean()
            lv = ((v - tv[b]) ** 2).mean()
            loss = kl + args.value_weight * lv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
        with torch.no_grad():
            logits, v = student(obs[val].float())
            lp = legal_log_policy(logits.gather(1, pidx[val]), pmask[val])
            p_t = tlp[val].exp() * pmask[val]
            kl = float((p_t * (tlp[val] - lp)).sum(1).mean())
            agree = float((lp.masked_fill(~pmask[val], -1e9).argmax(1)
                           == tlp[val].masked_fill(~pmask[val], -1e9).argmax(1)).float().mean())
            lv = float(((v - tv[val]) ** 2).mean())
        print(f"epoch {ep + 1}: 検証 KL {kl:.4f} 最善手の一致 {agree:.3f} 価値の2乗誤差 {lv:.4f}"
              f"（{time.time() - t0:.0f}s）", flush=True)
    out = pathlib.Path(args.out)
    torch.save({"config": student.config, "state_dict": student.state_dict(), "teacher": args.teacher}, out)
    write_weights(student, out.with_suffix(".bin"))
    print(f"書き出し: {out} / {out.with_suffix('.bin')}")


if __name__ == "__main__":
    main()
