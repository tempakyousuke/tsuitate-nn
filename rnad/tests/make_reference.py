"""OpenSpiel の R-NaD（JAX、削除直前の版 d1dcdf5）の損失を乱数入力で計算して npz に書く。
tsuitate-nn/rnad の PyTorch 実装の突き合わせ用（test_rnad.py のフィクスチャを作り直すとき）。

学習用の venv とは別の使い捨て venv で回す（JAX を学習側へ入れないため）:

    python3 -m venv /tmp/jaxenv
    /tmp/jaxenv/bin/pip install jax chex dm-tree numpy
    curl -sSfL -o /tmp/claude-rnad/rnad.py \\
      https://raw.githubusercontent.com/google-deepmind/open_spiel/d1dcdf5dc9c0a98a0ff0a6b476236d0cc91bf2d5/open_spiel/python/algorithms/rnad/rnad.py
    /tmp/jaxenv/bin/python rnad/tests/make_reference.py rnad/tests/openspiel_rnad_ref0.npz 0
"""
import sys
import typing
from typing import Any, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import tree_util as tree

src = open("/tmp/claude-rnad/rnad.py").read().split("\n")
# _legal_policy（262行目）〜 get_loss_nerd（589行目）だけを取り出す
body = "\n".join(src[261:589])
ns = dict(chex=chex, jax=jax, jnp=jnp, lax=lax, np=np, tree=tree,
          Any=Any, Sequence=Sequence, Tuple=Tuple)
exec(body, ns)

rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
T, B, A = 14, 5, 7
eta, beta, alpha = 0.2, 2.0, 0.37

valid = np.zeros((T, B), np.float32)
player = rng.integers(0, 2, size=(T, B)).astype(np.float32)
rewards = np.zeros((T, B, 2), np.float32)
lengths = rng.integers(3, T + 1, size=B)
for b, L in enumerate(lengths):
    valid[:L, b] = 1
    r = rng.choice([-1.0, 0.0, 1.0])
    rewards[L - 1, b] = [r, -r]
legal = (rng.random((T, B, A)) < 0.6).astype(np.float32)
legal[..., 0] = 1  # 少なくとも1つ
logit = rng.normal(size=(T, B, A)).astype(np.float32)
logit_prev = rng.normal(size=(T, B, A)).astype(np.float32)
logit_prev_ = rng.normal(size=(T, B, A)).astype(np.float32)
logit_mu = rng.normal(size=(T, B, A)).astype(np.float32)
v = rng.normal(size=(T, B, 1)).astype(np.float32) * 0.5
v_tgt = rng.normal(size=(T, B, 1)).astype(np.float32) * 0.5

_legal_policy = ns["_legal_policy"]
legal_log_policy = ns["legal_log_policy"]
mu = np.asarray(_legal_policy(jnp.asarray(logit_mu), jnp.asarray(legal)))
actions = np.zeros((T, B), np.int64)
for t in range(T):
    for b in range(B):
        p = mu[t, b].astype(np.float64)
        actions[t, b] = rng.choice(A, p=p / p.sum())
action_oh = np.eye(A, dtype=np.float32)[actions]


def loss_fn(logit, v):
    pi = _legal_policy(logit, legal)
    log_pi = legal_log_policy(logit, legal)
    log_pi_prev = legal_log_policy(jnp.asarray(logit_prev), legal)
    log_pi_prev_ = legal_log_policy(jnp.asarray(logit_prev_), legal)
    log_policy_reg = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)
    vt_list, hp_list, pt_list = [], [], []
    for p in range(2):
        vt, hp, pt = ns["v_trace"](
            jnp.asarray(v_tgt), jnp.asarray(valid), jnp.asarray(player), jnp.asarray(mu),
            pi, log_policy_reg, ns["_player_others"](jnp.asarray(player), jnp.asarray(valid), p),
            jnp.asarray(action_oh), jnp.asarray(rewards[:, :, p]), p,
            lambda_=1.0, c=1.0, rho=np.inf, eta=eta)
        vt_list.append(vt)
        hp_list.append(hp)
        pt_list.append(pt)
    loss_v = ns["get_loss_v"]([v] * 2, vt_list, hp_list)
    is_vector = jnp.expand_dims(jnp.ones_like(jnp.asarray(valid)), axis=-1)
    loss_nerd = ns["get_loss_nerd"]([logit] * 2, [pi] * 2, pt_list, jnp.asarray(valid),
                                    jnp.asarray(player), jnp.asarray(legal), [is_vector] * 2,
                                    clip=10_000, threshold=beta)
    return loss_v + loss_nerd, (loss_v, loss_nerd, vt_list, hp_list)


(total, (loss_v, loss_nerd, vt_list, hp_list)), (g_logit, g_v) = jax.value_and_grad(
    loss_fn, argnums=(0, 1), has_aux=True)(jnp.asarray(logit), jnp.asarray(v))

np.savez(sys.argv[1], T=T, B=B, A=A, eta=eta, beta=beta, alpha=alpha,
         valid=valid, player=player, rewards=rewards, legal=legal, logit=logit,
         logit_prev=logit_prev, logit_prev_=logit_prev_, mu=mu, actions=actions,
         v=v, v_tgt=v_tgt, loss_v=np.asarray(loss_v), loss_nerd=np.asarray(loss_nerd),
         total=np.asarray(total), g_logit=np.asarray(g_logit), g_v=np.asarray(g_v),
         v_target0=np.asarray(vt_list[0]), v_target1=np.asarray(vt_list[1]),
         has_played0=np.asarray(hp_list[0]), has_played1=np.asarray(hp_list[1]))
print("loss_v", float(loss_v), "loss_nerd", float(loss_nerd))
