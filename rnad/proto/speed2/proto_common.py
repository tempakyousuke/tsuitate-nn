"""proto/speed2 の試作が共有する置き場所と小道具（学習には使わない）。

- RNAD = tsuitate-nn/rnad（rnad.py などを読む）、MODELS = tsuitate-nn の隣の tsuitate-bot/models
- `make_learner`: 本線と同じ設定（η 0.2・β 16・lr 1e-4・Δm 10,000）で、現在・ターゲット = rl_deduce_v7、
  prev = rl_deduce_v6、prev_ = rl_deduce_v5 の学習器（step を与えて α を決める）
- `collect_batch`: 重みの自己対局で compact_legal の batch を作る（本線の収集と同じ関数）
"""

import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
RNAD = HERE.parent.parent
ROOT = RNAD.parent
MODELS = ROOT.parent / "tsuitate-bot" / "models"
DEFAULT_WEIGHTS = MODELS / "rl_deduce_v7.bin"
if str(RNAD) not in sys.path:
    sys.path.insert(0, str(RNAD))

import tsuitate_rl  # noqa: E402
from export_weights import read as read_weights  # noqa: E402
from model import NUM_PLANES  # noqa: E402
from rnad import RNaDConfig, RNaDLearner, collect  # noqa: E402


def env_kwargs(net):
    planes = net.config["planes"]
    return {"planes": planes} if planes != NUM_PLANES else {}


def make_learner(step, packed=True, chunk=1024, models=MODELS, device=torch.device("cpu")):
    """本線と同じ設定の学習器。online/target = v7、prev = v6、prev_ = v5、step_count = step"""
    models = pathlib.Path(models)
    cfg = RNaDConfig(eta=0.2, beta=16.0, clip=10_000.0, lr=1e-4, target_avg=0.001,
                     schedule_sizes=(10000,), schedule_repeats=(1,), chunk=chunk)
    L = RNaDLearner(read_weights(models / "rl_deduce_v7.bin"), cfg, device, packed=packed)
    L.prev.load_state_dict(read_weights(models / "rl_deduce_v6.bin").state_dict())
    L.prev_.load_state_dict(read_weights(models / "rl_deduce_v5.bin").state_dict())
    L.step_count = step
    return L


def collect_batch(games, seed=7, weights=DEFAULT_WEIGHTS, device=torch.device("cpu")):
    """重みのネットで games 局を自己対局した batch（compact_legal）"""
    net = read_weights(weights).to(device).eval()
    env = tsuitate_rl.VecEnv(games, seed=seed, auto_reset=False, **env_kwargs(net))
    torch.manual_seed(seed)
    return collect(env, net, device, compact_legal=True)


def load_or_collect(path, games, seed=7, weights=DEFAULT_WEIGHTS, device=torch.device("cpu")):
    """保存した batch（torch.save）があれば読み、無ければ games 局を device の順伝播で集める（batch 自体は CPU に置かれる）"""
    if path:
        return torch.load(path, map_location="cpu", weights_only=False)
    return collect_batch(games, seed, weights, device)
