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
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
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


def parse_variant(spec):
    name, _, rest = spec.partition(":")
    v = {"name": name, "clip": 10_000.0, "mucap": 0.0, "loss": "neurd"}
    for kv in filter(None, rest.split(",")):
        k, _, x = kv.partition("=")
        v[k] = x if k == "loss" else float(x)
    return v


def run(variant, seed, args):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = RNaDConfig(eta=args.eta, beta=args.beta, lr=args.lr, schedule_sizes=(args.schedule,), schedule_repeats=(1,),
                     target_avg=args.target_avg, clip=variant["clip"], inv_mu_cap=variant["mucap"],
                     policy_loss=variant["loss"])
    learner = RNaDLearner(TabularNet(), cfg, torch.device("cpu"))
    curve = []
    for step in range(args.steps + 1):
        if step % args.every == 0:
            curve.append({"step": step, "nashconv": float(nash_conv(policy_table(learner.net))),
                          "nashconv_target": float(nash_conv(policy_table(learner.target))),
                          "min_prob": float(policy_table(learner.net).min())})
        if step < args.steps:
            learner.update(collect_kuhn(learner.net, args.games, rng))
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True)
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
