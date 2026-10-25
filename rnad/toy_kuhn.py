"""クーンポーカーで R-NaD の変種の**偏り**を見る（tsuitate-bot の docs/rl-learning-speed.md）。

    python rnad/toy_kuhn.py --variant ref: --variant cap100:mucap=100 --variant pg:loss=pg --seeds 3 --steps 3000

学習なしの勾配の診断（gradsnr.py）は「分散がどれだけ減るか」は測れるが、「向きがどれだけずれるか（偏り）」は測れない
（本線では雑音のスケールが数千局あり、期待勾配が推定できない）。ここでは、厳密に搾取されやすさ（NashConv）を計算できる
小さいゲームで、**本物の `RNaDLearner`**（v-trace・NeuRD・正則化方策の差し替え）をそのまま回し、変種が混合均衡への収束を
遅らせたり、ずれた点で止まったりしないかを見る。

- ゲーム: クーンポーカー（J/Q/K、アンテ 1、先手は check/bet、…）。情報集合 12 個 × 行動 2。報酬は ±1・±2 を 1/2 倍
- ネット: 情報集合の one-hot → ロジット（表）と価値（表）。β = 16 なのでロジットは鋭くなれる（本線と同じしきい値）
- NashConv: 各プレイヤーの最善応答の価値の和（0 = ナッシュ均衡）。測るのは現在のネットとターゲット
- 変種の書き方は gradsnr.py と同じ（`名前:mucap=<1/μ の上限>,loss=<neurd|pg>,clip=<NeuRD のクリップ>`）

**MMD 学習器**（`mmd.py`、`train_rnad.py --algo mmd`）も同じ収集でそのまま回せる（docs/rl-learning-speed.md 10節）:

    python rnad/toy_kuhn.py --collect fast --games 512 --steps 6000 --lr 1e-2 --schedule 500 --target-avg 0.01 \\
        --variant ref: --variant M1:algo=mmd,filter=pps0.25 --variant M1n:algo=mmd,filter=none

  MMD の変種の項: `filter=none|pps<割合>|q<分位>|topk<割合>`（q は |A| の分位点以上、topk はちょうどその割合で同値は乱数）、
  `lam=<λ_A>`、`tempfix=<α>`（温度を固定）または `tempc=<coef>,tempd=<decay>,tempoff=<起点>`（既定 0.05・0.3・0）、
  `lrfix=<学習率>`（既定は上限 1e-2 を約200更新保ってから t^-1.1 で 5e-4 まで下げる）、`steps=<1更新の歩数>`（既定 5）、
  `magnet=uniform|mix<ε>|target|mixtarget<ε>`（既定 uniform。target は今のターゲット = 現在のネットの EMA を磁石にする
  「動く磁石」）、`target=<ターゲットの平均化係数>`（既定 --target-avg。target の磁石の動く速さも決める）
- `--expected`: **標本なしの期待値の更新**（価値は厳密、A はプレイヤーごとの GAE、更新は π=μ の PPO の期待勾配 ＋ 一様な磁石
  （選んだ標本だけ）をロジットで積分）で、行の絞り込み（全体の |A| の分位点以上）が固定点を均衡から外すかを見る。
  絞り込みなしは α-QRE に着く（α=0.05 で NashConv 0.038）。pps は期待値が絞り込みなしと同じなので、ここでは回さない
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mmd import MMDLearner  # noqa: E402
from mmd import make_config as make_mmd_config  # noqa: E402
from rnad import RNaDConfig, RNaDLearner, legal_policy  # noqa: E402

# 履歴 → (手番, 終局なら先手から見た報酬の符号付きの額を返す関数)。行動 0 = check/fold、1 = bet/call
HISTORIES = ["", "c", "b", "cb"]  # 手番のある履歴
N_INFO = 3 * len(HISTORIES)


def to_move(h):
    return len(h) % 2


def terminal_payoff(h, cards):
    """終局なら先手から見た損得（チップ）、終局でなければ None"""
    hi = 0 if cards[0] > cards[1] else 1
    win = 1 if hi == 0 else -1
    if h in ("cc",):
        return win * 1
    if h in ("bc", "cbc"):
        return win * 2
    if h == "bf":
        return 1
    if h == "cbf":
        return -1
    return None


def info_index(card, h):
    return card * len(HISTORIES) + HISTORIES.index(h)


def next_h(h, a):
    # 相手の bet の後の行動 0/1 は fold/call、それ以外は check/bet
    facing = h.endswith("b")
    return h + ("f" if a == 0 else "c") if facing else h + ("c" if a == 0 else "b")


class TabularNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {"planes": N_INFO, "channels": 1, "blocks": 0}
        self.logit = torch.nn.Parameter(torch.zeros(N_INFO, 2))
        self.value = torch.nn.Parameter(torch.zeros(N_INFO))

    def forward(self, x):
        return x @ self.logit, x @ self.value


@torch.no_grad()
def policy_table(net):
    eye = torch.eye(N_INFO)
    logits, _ = net(eye)
    return legal_policy(logits, torch.ones(N_INFO, 2, dtype=torch.bool)).numpy()


def nash_conv(pi):
    """各プレイヤーの最善応答の価値（チップ）の和。pi[情報集合] = [p0, p1]"""
    deals = [(a, b) for a in range(3) for b in range(3) if a != b]
    total = 0.0
    for br in (0, 1):
        # 最善応答側の情報集合ごとに、相手のカードを周辺化して行動の価値を比べる（後ろ向きに）
        value = {}

        def ev(h, cards):
            """br が最善応答・相手が pi のときの、br から見た期待値（チップ）"""
            pay = terminal_payoff(h, cards)
            if pay is not None:
                return pay if br == 0 else -pay
            p = to_move(h)
            if p != br:
                probs = pi[info_index(cards[p], h)]
                return sum(probs[a] * ev(next_h(h, a), cards) for a in (0, 1))
            return best(h, cards[br])[cards]

        def reach_opp(h, d):
            """相手（br でない側）の方策で履歴 h に着く確率（br 自身の行動は数えない）"""
            r = 1.0
            for i, ch in enumerate(h):
                p = i % 2
                if p == br:
                    continue
                prefix = h[:i]
                a = 1 if ch == "b" or (ch == "c" and prefix.endswith("b")) else 0
                r *= pi[info_index(d[p], prefix)][a]
            return r

        def best(h, my_card):
            key = (h, my_card)
            if key in value:
                return value[key]
            ds = [d for d in deals if d[br] == my_card]
            q = [sum(reach_opp(h, d) * ev(next_h(h, a), d) for d in ds) for a in (0, 1)]
            a_star = int(np.argmax(q))
            value[key] = {d: ev(next_h(h, a_star), d) for d in ds}
            return value[key]

        total += sum(ev("", d) for d in deals) / len(deals)
    return total


def collect_kuhn(net, games, rng):
    """`rnad.collect` と同じ形の batch（局ごとに手番を記録し、終局の報酬を [先手, 後手] で）"""
    T = 3
    valid = torch.zeros(T, games, dtype=torch.bool)
    player = torch.zeros(T, games, dtype=torch.long)
    reward = torch.zeros(T, games, 2)
    obs_l, act_l, mu_l, t_l, b_l = [], [], [], [], []
    pi = policy_table(net)
    for b in range(games):
        cards = tuple(rng.choice(3, 2, replace=False))
        h = ""
        t = 0
        while True:
            p = to_move(h)
            i = info_index(cards[p], h)
            a = int(rng.random() < pi[i][1])
            x = torch.zeros(N_INFO)
            x[i] = 1
            obs_l.append(x)
            act_l.append(a)
            mu_l.append(float(pi[i][a]))
            t_l.append(t)
            b_l.append(b)
            valid[t, b] = True
            player[t, b] = p
            h = next_h(h, a)
            pay = terminal_payoff(h, cards)
            if pay is not None:
                reward[t, b, 0] = pay / 2
                reward[t, b, 1] = -pay / 2
                break
            t += 1
    n = len(act_l)
    return {"obs": torch.stack(obs_l), "legal": torch.ones(n, 2, dtype=torch.bool),
            "action": torch.tensor(act_l), "mu_a": torch.tensor(mu_l, dtype=torch.float32).clamp(min=1e-30),
            "t": torch.tensor(t_l), "b": torch.tensor(b_l), "valid": valid, "player": player, "reward": reward}


@torch.no_grad()
def tables(net):
    """(方策 [12, 2], 合法手の対数確率 [12, 2], 価値 [12])"""
    logits, v = net(torch.eye(N_INFO))
    lp = logits.float().log_softmax(-1)
    return lp.exp().numpy(), lp.numpy(), v.float().numpy()


def collect_kuhn_fast(net, games, rng):
    """`collect_kuhn` と同じ分布の batch を numpy でまとめて作る（乱数の使い方は違う）。行は時刻の昇順・局の昇順
    （`rnad.collect` と同じ並び）。MMD 学習器が使う `v_actor`・`mu_logp`・`legal_cols`・`legal_counts`・`outcome` も付ける"""
    pi, lp, vt = tables(net)
    cards = np.argsort(rng.random((games, 3)), axis=1)[:, :2]
    c0, c1 = cards[:, 0], cards[:, 1]
    win0 = np.where(c0 > c1, 1, -1)
    u = rng.random((games, 3))
    i0 = c0 * len(HISTORIES) + HISTORIES.index("")
    a0 = (u[:, 0] < pi[i0, 1]).astype(np.int64)
    i1 = c1 * len(HISTORIES) + np.where(a0 == 1, HISTORIES.index("b"), HISTORIES.index("c"))
    a1 = (u[:, 1] < pi[i1, 1]).astype(np.int64)
    need2 = (a0 == 0) & (a1 == 1)
    i2 = c0 * len(HISTORIES) + HISTORIES.index("cb")
    a2 = (u[:, 2] < pi[i2, 1]).astype(np.int64)
    # 先手から見た損得: bf +1 / bc 2·勝敗 / cc 勝敗 / cbf −1 / cbc 2·勝敗
    pay = np.where(a0 == 1, np.where(a1 == 0, 1, 2 * win0),
                   np.where(a1 == 0, win0, np.where(a2 == 0, -1, 2 * win0)))
    t_end = np.where(need2, 2, 1)
    b_all = np.arange(games)
    rows_i = [i0, i1, i2[need2]]
    rows_a = [a0, a1, a2[need2]]
    rows_b = [b_all, b_all, b_all[need2]]
    info = np.concatenate(rows_i)
    act = np.concatenate(rows_a)
    b_idx = np.concatenate(rows_b)
    t_idx = np.concatenate([np.full(len(r), t) for t, r in enumerate(rows_b)])
    n = len(act)
    T = 3
    valid = torch.zeros(T, games, dtype=torch.bool)
    valid[torch.from_numpy(t_idx), torch.from_numpy(b_idx)] = True
    player = torch.zeros(T, games, dtype=torch.long)
    player[1] = 1
    reward = torch.zeros(T, games, 2)
    reward[torch.from_numpy(t_end), torch.from_numpy(b_all), 0] = torch.from_numpy(pay / 2).float()
    reward[torch.from_numpy(t_end), torch.from_numpy(b_all), 1] = torch.from_numpy(-pay / 2).float()
    obs = torch.zeros(n, N_INFO)
    obs[torch.arange(n), torch.from_numpy(info)] = 1
    outcome = np.stack([np.sign(pay), -np.sign(pay)], 1).astype(np.float32)
    return {"obs": obs, "legal": torch.ones(n, 2, dtype=torch.bool),
            "legal_cols": torch.tensor([0, 1], dtype=torch.int16).repeat(n),
            "legal_counts": torch.full((n,), 2, dtype=torch.int32),
            "action": torch.from_numpy(act), "mu_a": torch.from_numpy(pi[info, act]).float().clamp(min=1e-30),
            "mu_logp": torch.from_numpy(lp[info].reshape(-1)).float(), "v_actor": torch.from_numpy(vt[info]).float(),
            "t": torch.from_numpy(t_idx), "b": torch.from_numpy(b_idx), "valid": valid, "player": player,
            "reward": reward, "outcome": outcome}


MMD_KEYS = ("algo", "filter", "lam", "tempfix", "tempc", "tempd", "tempoff", "lrfix", "steps", "magnet", "target")


def parse_variant(spec):
    name, _, rest = spec.partition(":")
    v = {"name": name, "clip": 10_000.0, "mucap": 0.0, "loss": "neurd", "algo": "rnad"}
    for kv in filter(None, rest.split(",")):
        k, _, x = kv.partition("=")
        if k not in ("clip", "mucap", "loss") + MMD_KEYS:
            raise SystemExit(f"変種の項 {k} は clip / mucap / loss / {' / '.join(MMD_KEYS)} のどれか")
        v[k] = x if k in ("loss", "algo", "filter", "magnet") else float(x)
    if v["algo"] not in ("rnad", "mmd"):
        raise SystemExit(f"algo {v['algo']}（rnad / mmd）")
    if v["algo"] == "rnad" and any(k in v for k in MMD_KEYS[1:]):
        raise SystemExit(f"{spec}: MMD の項は algo=mmd のときだけ")
    return v


def mmd_config(v, args):
    """変種 → MMDConfig（docs/rl-learning-speed.md 10節 (b) の共通設定: 1更新 5 歩、学習率は上限 1e-2 を約200更新
    保ってから t^-1.1 で 5e-4 まで、磁石は一様、温度は 0.05/(t+1)^0.3）"""
    f = v.get("filter", "pps0.25")
    filt = ("none" if f == "none" else f"pps:{f[3:]},floor=0.05" if f.startswith("pps")
            else f"topk:{f[4:]}" if f.startswith("topk") else f"quantile:{f[1:]},min=0.01" if f.startswith("q")
            else None)
    if filt is None:
        raise SystemExit(f"filter {f}（none / pps<割合> / q<分位> / topk<割合>）")
    temp = ((v["tempfix"], 0.0, v["tempfix"], v["tempfix"]) if "tempfix" in v
            else (v.get("tempc", 0.05), v.get("tempd", 0.3), 0.1, 0.001))
    lr = ((v["lrfix"], 0.0, v["lrfix"], v["lrfix"]) if "lrfix" in v
          else (1e-2 * 201 ** 1.1, 1.1, 1e-2, 5e-4))
    mag = v.get("magnet", "uniform")
    mag = (mag if mag in ("uniform", "target") else f"mix:target,{mag[9:]}" if mag.startswith("mixtarget")
           else f"mix:init,{mag[3:]}")
    lam = v.get("lam", 0.8)
    return make_mmd_config("tsuitate", filter=filt, value_rows="auto", magnet=mag, gae_lambda=lam, td_lambda=0.8,
                           temp=temp, temp_t_off=v.get("tempoff", 0.0), lr=lr, steps=int(v.get("steps", 5)),
                           chunk=100_000, max_grad_norm=0.0, target_avg=v.get("target", args.target_avg))


def run(variant, seed, args):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if variant["algo"] == "mmd":
        learner = MMDLearner(TabularNet(), mmd_config(variant, args), torch.device("cpu"), seed=seed)
    else:
        cfg = RNaDConfig(eta=args.eta, beta=args.beta, lr=args.lr, schedule_sizes=(args.schedule,),
                         schedule_repeats=(1,), target_avg=args.target_avg, clip=variant["clip"],
                         inv_mu_cap=variant["mucap"], policy_loss=variant["loss"])
        learner = RNaDLearner(TabularNet(), cfg, torch.device("cpu"))
    fast = args.collect == "fast" or variant["algo"] == "mmd"
    curve = []
    kept = []
    for step in range(args.steps + 1):
        if step % args.every == 0:
            pi = policy_table(learner.net)
            rec = {"step": step, "nashconv": float(nash_conv(pi)),
                   "nashconv_target": float(nash_conv(policy_table(learner.target))),
                   "min_prob": float(pi.min()),
                   "entropy": float(-(pi * np.log(np.maximum(pi, 1e-300))).sum(1).mean())}
            if kept:
                # この区間で実際に残した行の割合（クーンは |A| に同じ値が多く、分位点で切ると 25% からずれうる）
                rec["keep_rate"] = float(np.mean(kept))
                kept = []
            if variant["algo"] == "mmd":
                rec["temperature"] = learner.temperature()
            curve.append(rec)
        if step < args.steps:
            batch = (collect_kuhn_fast if fast else collect_kuhn)(learner.net, args.games, rng)
            info = learner.update(batch)
            if "keep_rate" in info:
                kept.append(info["keep_rate"])
    return curve


# ---------------------------------------------------------------- 標本なしの期待値の更新（--expected）

DEALS = [(a, b) for a in range(3) for b in range(3) if a != b]


def _paths(pi):
    """全ての (配り, 行動列) の確率・各決定点 (手番, 情報集合, 行動)・先手から見た報酬（1/2 倍）"""
    out = []
    for d in DEALS:
        def rec(h, prob, decs):
            pay = terminal_payoff(h, d)
            if pay is not None:
                out.append((prob / len(DEALS), decs, pay / 2))
                return
            p = to_move(h)
            i = info_index(d[p], h)
            for a in (0, 1):
                if pi[i][a] > 0:
                    rec(next_h(h, a), prob * pi[i][a], decs + [(p, i, a)])
        rec("", 1.0, [])
    return out


def expected_grad(z, alpha, lam, q, thr_min=0.01):
    """ロジット z [12, 2] での期待値の更新の向き（π=μ の PPO の期待勾配 ＋ 一様な磁石。磁石も選んだ標本だけ）。
    価値は今の方策での厳密な値、A はプレイヤーごとの GAE(λ)。q > 0 なら全決定点（両者）の |A| の確率重み付き q 分位点
    （下限 thr_min）以上だけを選ぶ（Ataraxos の絞り込み）。(向き, π, しきい値)"""
    pi = np.exp(z - z.max(1, keepdims=True))
    pi /= pi.sum(1, keepdims=True)
    P = _paths(pi)
    num, den = np.zeros(N_INFO), np.zeros(N_INFO)
    for pr, decs, zz in P:
        for p, i, a in decs:
            num[i] += pr * (zz if p == 0 else -zz)
            den[i] += pr
    V = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
    S = []  # (確率, 情報集合, 行動, A)
    for pr, decs, zz in P:
        for k in (0, 1):
            mine = [(i, a) for p, i, a in decs if p == k]
            zk = zz if k == 0 else -zz
            nxt = None
            for j in range(len(mine) - 1, -1, -1):
                i, a = mine[j]
                delta = (zk if j == len(mine) - 1 else V[mine[j + 1][0]]) - V[i]
                nxt = delta if j == len(mine) - 1 else delta + lam * nxt
                S.append((pr, i, a, nxt))
    thr = 0.0
    if q > 0:
        absA = np.array([abs(x[3]) for x in S])
        w = np.array([x[0] for x in S])
        o = np.argsort(absA)
        cw = np.cumsum(w[o]) / w.sum()
        thr = max(float(absA[o][np.searchsorted(cw, q - 1e-12)]), thr_min)
    g = np.zeros((N_INFO, 2))
    psel = np.zeros(N_INFO)
    for pr, i, a, A in S:
        if q == 0 or abs(A) >= thr - 1e-12:
            psel[i] += pr
            g[i, a] += pr * A
    dz = g - g.sum(1, keepdims=True) * pi
    logp = np.log(np.maximum(pi, 1e-300))
    H = -(pi * logp).sum(1, keepdims=True)
    dz -= alpha * psel[:, None] * pi * (logp + H)
    return dz, pi, thr


def expected_run(alpha, lam, q, iters, lr):
    z = np.zeros((N_INFO, 2))
    avg = np.zeros((N_INFO, 2))
    n = 0
    for it in range(iters):
        dz, pi, thr = expected_grad(z, alpha, lam, q)
        z += lr * dz
        if it >= iters // 2:
            avg += pi
            n += 1
    return {"alpha": alpha, "lam": lam, "q": q, "iters": iters, "lr": lr, "nashconv_last": float(nash_conv(pi)),
            "nashconv_avg_late": float(nash_conv(avg / n)), "thr": thr}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--collect", choices=["loop", "fast"], default="loop",
                    help="R-NaD の収集（loop = 従来の1局ずつ / fast = numpy でまとめて。MMD は常に fast）")
    ap.add_argument("--expected", action="store_true", help="標本なしの期待値の更新（--alphas × --lams × --qs）")
    ap.add_argument("--alphas", default="0.05,0.02,0.01")
    ap.add_argument("--lams", default="0.8,0.5")
    ap.add_argument("--qs", default="0,0.75", help="絞り込みの分位（0 = なし）")
    ap.add_argument("--iters", type=int, default=60_000)
    ap.add_argument("--exp-lr", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eta", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=16.0)
    ap.add_argument("--schedule", type=int, default=200)
    ap.add_argument("--target-avg", type=float, default=0.02)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.expected:
        res = []
        for alpha in [float(x) for x in args.alphas.split(",")]:
            for lam in [float(x) for x in args.lams.split(",")]:
                for q in [float(x) for x in args.qs.split(",")]:
                    r = expected_run(alpha, lam, q, args.iters, args.exp_lr)
                    res.append(r)
                    print(f"α {alpha} λ {lam} 分位 {q}: NashConv 最後 {r['nashconv_last']:.4f} 後半の平均の方策 "
                          f"{r['nashconv_avg_late']:.4f}（しきい値 {r['thr']:.3f}）", flush=True)
        if args.out:
            pathlib.Path(args.out).write_text(json.dumps({"args": vars(args), "expected": res}, indent=1))
        return
    if not args.variant:
        raise SystemExit("--variant か --expected を指定する")
    results = {}
    for spec in args.variant:
        v = parse_variant(spec)
        curves = [run(v, s, args) for s in range(args.seeds)]
        results[v["name"]] = curves
        steps = [c["step"] for c in curves[0]]
        mean = [np.mean([cv[i]["nashconv"] for cv in curves]) for i in range(len(steps))]
        mean_t = [np.mean([cv[i]["nashconv_target"] for cv in curves]) for i in range(len(steps))]
        print(f"{v['name']:10s} NashConv（現在）" + " ".join(f"{m:.4f}" for m in mean), flush=True)
        print(f"{'':10s} NashConv（ターゲット）" + " ".join(f"{m:.4f}" for m in mean_t), flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({"args": vars(args), "results": results}, indent=1))


if __name__ == "__main__":
    main()
