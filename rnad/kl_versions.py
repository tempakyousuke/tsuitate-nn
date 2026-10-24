"""凍結版どうしの方策の KL を、共通の局面の上で総当たりに測る（tsuitate-bot の docs/rl-learning-speed.md「さらに速くする」）。
段階の長さ Δm=2,000 と 10,000 で、1段階あたりに方策がどれだけ動いたか（書き出し = ターゲットネット）を比べる。

    python rnad/kl_versions.py --games 128 --threads 2 --out kl_versions.json
    python rnad/kl_versions.py --versions rl_deduce_v6:78000,rl_deduce_v7:88000 --sources rl_deduce_v7 --games 4

局面は `--sources` の版ごとの自己対局（`--games` 局）から作る（版ごとに別の局面集合。どちらの版の局面で測るかで
KL が変わるので、古い版と新しい版の両方を元にして並べる）。版の並びは `--versions`（名前:更新数、既定は
rl_selfplay_v1〜rl_deduce_v7）で、重みは `--models/<名前>.bin`（既定は tsuitate-bot の models/）。
エンコーダ v1（86枚）の版は v2（91枚）の観測の先頭だけを読む（PolicyValueNet.trunk が切る）。

出すもの（局面集合ごと・版の組ごと）: KL(前‖後) の平均・SE・中央値、逆向きの KL、全変動距離、最善手の一致率と、
版ごとの方策のエントロピー
"""

import argparse
import itertools
import json
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tsuitate_rl  # noqa: E402
from export_weights import read as read_weights  # noqa: E402
from model import NUM_PLANES  # noqa: E402
from rnad import PRECISIONS, collect, pack_csr, set_precision  # noqa: E402

MODELS = HERE.parent.parent / "tsuitate-bot" / "models"
VERSIONS = (  # 名前:更新数
    "rl_selfplay_v1:42000,rl_selfplay_v2:46000,rl_selfplay_v3:48000,rl_selfplay_v4:50000,"
    "rl_deduce_v1:52000,rl_deduce_v2:54000,rl_deduce_v3:56000,rl_deduce_v4:58000,"
    "rl_deduce_v5:68000,rl_deduce_v6:78000,rl_deduce_v7:88000")


def parse_versions(spec):
    out = []
    for item in spec.split(","):
        name, _, step = item.strip().partition(":")
        if not name:
            continue
        out.append((name, int(step) if step else 0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--versions", default=VERSIONS, help="比べる版（名前:更新数 のカンマ区切り。更新数は表示用）")
    ap.add_argument("--models", default=str(MODELS), help="凍結版の重み <名前>.bin のディレクトリ")
    ap.add_argument("--sources", default="rl_deduce_v4,rl_deduce_v7", help="局面を作る自己対局の版（カンマ区切り）")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", choices=PRECISIONS, default="fp32")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default="kl_versions.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    set_precision(args.precision)
    dev = torch.device(args.device)
    versions = parse_versions(args.versions)
    nets = {n: read_weights(pathlib.Path(args.models) / f"{n}.bin").to(dev).eval() for n, _ in versions}
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    for s in sources:
        if s not in nets:
            nets[s] = read_weights(pathlib.Path(args.models) / f"{s}.bin").to(dev).eval()
    planes = max(net.config["planes"] for net in nets.values())
    kw = {"planes": planes} if planes != NUM_PLANES else {}
    sets = {}
    for si, src in enumerate(sources):
        t0 = time.time()
        env = tsuitate_rl.VecEnv(args.games, seed=args.seed + si, auto_reset=False, **kw)
        b = collect(env, nets[src], dev, compact_legal=True)
        ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(b["legal_counts"].long(), 0)])
        n = len(b["action"])
        print(f"{src}: {n} 決定点 / {args.games} 局 ({time.time() - t0:.0f}s)", flush=True)
        packs = []
        for i in range(0, n, 2048):
            sl = slice(i, min(i + 2048, n))
            cols = b["legal_cols"][int(ptr[sl.start]):int(ptr[sl.stop])]
            idx, pmask, _ = pack_csr(cols, b["legal_counts"][sl], b["action"][sl])
            packs.append((sl, idx.to(dev), pmask.to(dev)))
        K = max(p[1].shape[1] for p in packs)
        mask = torch.zeros(n, K, dtype=torch.bool)
        for sl, _, pmask in packs:
            mask[sl, :pmask.shape[1]] = pmask.cpu()
        logp = {}
        for name, _ in versions:
            lp = torch.full((n, K), float("-inf"))
            with torch.no_grad():
                for sl, idx, pmask in packs:
                    logits = nets[name](b["obs"][sl].to(dev).float())[0].gather(1, idx).masked_fill(~pmask, float("-inf"))
                    lp[sl, :idx.shape[1]] = logits.log_softmax(-1).cpu()
            logp[name] = lp
        sets[src] = (logp, mask)
    res = {"games": args.games, "versions": versions, "sources": {}}
    for src, (logp, mask) in sets.items():
        zero = torch.zeros(mask.shape)
        pairs = {}
        for (a, sa), (c, sc) in itertools.combinations(versions, 2):
            pa, pc = logp[a], logp[c]
            p, q = pa.exp(), pc.exp()
            kl_ac = torch.where(mask, p * (pa - pc), zero).sum(1)  # KL(a‖c)
            kl_ca = torch.where(mask, q * (pc - pa), zero).sum(1)
            tv = 0.5 * torch.where(mask, (p - q).abs(), zero).sum(1)
            pairs[f"{a}|{c}"] = {"from": sa, "to": sc, "kl_fwd": float(kl_ac.mean()), "kl_rev": float(kl_ca.mean()),
                                 "kl_fwd_se": float(kl_ac.std() / len(kl_ac) ** 0.5),
                                 "kl_fwd_median": float(kl_ac.median()), "tv": float(tv.mean()),
                                 "top_agree": float((pa.argmax(1) == pc.argmax(1)).float().mean())}
        ent = {n: float(-(torch.where(mask, logp[n].exp() * logp[n], zero)).sum(1).mean()) for n, _ in versions}
        res["sources"][src] = {"n": int(mask.shape[0]), "pairs": pairs, "entropy": ent}
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    for src, r in res["sources"].items():
        print(f"== {src} の局面（{r['n']} 決定点）")
        for k, v in r["pairs"].items():
            print(f"{k:32s} {v['from']:>6d}->{v['to']:>6d}  KL {v['kl_fwd']:.5f}±{v['kl_fwd_se']:.5f} "
                  f"rev {v['kl_rev']:.5f} TV {v['tv']:.4f} top一致 {v['top_agree']:.3f}")


if __name__ == "__main__":
    main()
