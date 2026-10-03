"""リーグ学習の相手の選び方（PFSP: prioritized fictitious self-play。AlphaStar のリーグと同じ考え方）。

1回の更新の局の一部を、固定した相手（凍結版 `rl_vN`・`rl_policy:<重み>` など tsuitate-bot の
`strategy::make` が作れる名前）との対局に回す。相手ごとの勝率を学習中の対局から数え続け、
**負けやすい相手ほど多く当てる**（重み ∝ (1 − 勝率)^power）。苦手が克服されれば自然に別の相手へ移る。

- 勝率は相手ごとに直近 `window` 局の得点（勝 1・分 0.5・負 0）の平均を、事前の 1勝1敗で均したもの
  （まだ当たっていない相手は 0.5 から始まる）
- 局数は先後をそろえるため2局単位で配る（評価モードの VecEnv は偶数番の局で学習側が先手）。
  配った局は `VecEnv(opponents=[...])` の1つの env にまとめて回す（相手ごとの env だと推論が直列になる）
- 相手を固定した局は相手の手番が記録に現れないので、学習には学習側の手だけが効く
  （相手を固定した最善応答の学習。R-NaD の自己対局の収束保証の外）
"""

from collections import deque

import numpy as np


class League:
    def __init__(self, names, window=512, power=2.0):
        if not names:
            raise ValueError("リーグの相手が空")
        self.names = list(names)
        self.window = window
        self.power = power
        self.scores = {n: deque(maxlen=window) for n in self.names}

    def win_rate(self, name):
        s = self.scores[name]
        return (sum(s) + 1.0) / (len(s) + 2.0)

    def weights(self):
        w = np.array([(1.0 - self.win_rate(n)) ** self.power for n in self.names])
        # 全部に勝ち切っても配分が消えないよう、わずかに底上げする
        w = w + 1e-3
        return w / w.sum()

    def allocate(self, n_games, rng):
        """n_games 局（偶数）を相手ごとの局数へ配る。2局単位の多項分布"""
        if n_games % 2:
            raise ValueError(f"リーグの局数 {n_games} は偶数にする（先後をそろえるため）")
        pairs = rng.multinomial(n_games // 2, self.weights())
        return {n: int(k) * 2 for n, k in zip(self.names, pairs)}

    def record(self, finished):
        """評価モードの env の pop_finished() を取り込み、相手ごとの [勝, 負, 分] を返す
        （どの相手との局かは各局の `opponent`。リーグの外の相手の局は数えない）"""
        out = {}
        for f in finished:
            name = f["opponent"]
            if name not in self.scores:
                continue
            wld = out.setdefault(name, [0, 0, 0])
            if f["winner"] is None:
                score, wld[2] = 0.5, wld[2] + 1
            elif f["winner"] == f["learner"]:
                score, wld[0] = 1.0, wld[0] + 1
            else:
                score, wld[1] = 0.0, wld[1] + 1
            self.scores[name].append(score)
        return out

    def summary(self):
        ws = self.weights()
        return {n: {"wr": round(self.win_rate(n), 4), "weight": round(float(w), 4), "n": len(self.scores[n])}
                for n, w in zip(self.names, ws)}

    def state_dict(self):
        return {"names": self.names, "window": self.window, "power": self.power,
                "scores": {n: list(s) for n, s in self.scores.items()}}

    def load_state_dict(self, d):
        for n, s in d.get("scores", {}).items():
            if n in self.scores:
                self.scores[n].extend(s)
