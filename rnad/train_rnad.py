"""M2b: R-NaD の自己対局学習（tsuitate-bot の docs/rl-deepnash-design.md）。

    .venv/bin/python rnad/train_rnad.py --init out_rnad/bc/checkpoint.pt --out out_rnad/rnad_bc
    .venv/bin/python rnad/train_rnad.py --init none --out out_rnad/rnad_scratch   # ゼロから

1回の更新 = バッチ数ぶんの局を最後まで自己対局（tsuitate_rl.VecEnv、auto_reset=False）→
R-NaD の損失で1ステップ。学習の進み具合は**錨（既定は模倣学習のネット）との直接対戦の勝率**で
測る（凍結版 estimator は1手1秒以上かかり学習ループに入れられないため）。錨に勝ち越していけば
学習が進んでいる。区切りでターゲットネットの重みを Rust 用に書き出すので、CI の arena
（`rl_policy:release:...`）でも測れる。

防御特化（玉の周りの固め）は `--guard λ --league <本命> --league-frac 1`（rnad/README.md）。
速攻特化（決着の速さ）は `--speed λ`（自己対局＋リーグ。rnad/README.md）。

`--algo mmd` で学習器を Ataraxos 型（PPO の比クリップ ＋ 磁石への KL。rnad/mmd.py）に替える（既定は R-NaD のまま）。
R-NaD の checkpoint から続けるなら `--init <checkpoint.pt> --init-which both --init-step`（tsuitate-bot の
docs/rl-learning-speed.md 10節）。

記録: <out>/log.jsonl（更新ごと・評価ごとに1行）、<out>/checkpoint.pt、<out>/policy_<step>.bin
"""

import argparse
import copy
import hashlib
import json
import math
import pathlib
import time
from collections import Counter

import numpy as np
import torch

import tsuitate_rl
from export_weights import read as read_weights
from export_weights import write as write_weights
from guard import apply_win_bonus as apply_guard_win_bonus
from guard import summarize as summarize_guard
from speed import summarize as summarize_speed
from model import NUM_PLANES, PolicyValueNet
from league import League
import openings
from mmd import PRESETS as MMD_PRESETS
from mmd import MMDLearner
from mmd import make_config as make_mmd_config
from rnad import (PRECISIONS, MultiEnv, OpponentPool, PipelinedCollector, RNaDConfig, RNaDLearner, cfg_diff,
                  collect, play_match, set_precision)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_net(path: str, channels: int, blocks: int, which: str = "target", planes: int = NUM_PLANES) -> PolicyValueNet:
    """模倣学習（train_bc.py）の checkpoint か、R-NaD（train_rnad.py）の checkpoint から読む。
    R-NaD の checkpoint は `which`（既定はターゲットネット = 書き出す重みと同じ）を読む。
    `none` は乱数初期化（入力のプレーン数は `planes`）"""
    if path == "none":
        return PolicyValueNet(channels, blocks, planes)
    ck = torch.load(path, map_location="cpu")
    if "config" in ck:
        net = PolicyValueNet(ck["config"]["channels"], ck["config"]["blocks"], ck["config"]["planes"])
    elif "net" in ck:
        # ネットの形を持たない古い R-NaD の checkpoint（2026-09-28 以前）は引数の形で読む
        net = PolicyValueNet(channels, blocks)
    else:
        raise SystemExit(f"{path} は checkpoint として読めない")
    if "state_dict" not in ck and which not in ck:
        raise SystemExit(f"{path} に {which} が無い")
    net.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck[which])
    return net


def parse_lambda_schedule(spec: str, schedule: int):
    """ボーナスの λ の予定 "1.0:800,0.5:400,0" → (更新回数 step で使う λ を返す関数, 予定中の λ の最大)。
    区間の長さは正則化方策の差し替え間隔の倍数に限る（各区間は固定したゲームで収束させる前提）。
    最大値は報酬の縮め方（reward_scale）を予定全体で固定するのに使う"""
    if not spec:
        return (lambda step: 0.0), 0.0
    parts = []
    for item in spec.split(","):
        lam, _, n = item.partition(":")
        parts.append((float(lam), int(n) if n else None))
    for lam, n in parts[:-1]:
        if n is None or n % schedule:
            raise SystemExit(f"λ の区間 {lam}:{n} は --schedule {schedule} の倍数の長さにする")
    if parts[-1][1] is not None:
        raise SystemExit("λ の予定の最後は長さなし（以後ずっとその値）にする")

    def at(step):
        for lam, n in parts:
            if n is None or step < n:
                return lam
            step -= n
        return parts[-1][0]
    return at, max(lam for lam, _ in parts)


def save_checkpoint(learner, path: pathlib.Path, collect_cfg=None) -> None:
    """書きかけで止められても前の checkpoint が壊れないよう、別名に書いてから置き換える。
    `collect_cfg` は集め方の設定（序盤の強制など。学習器の設定 cfg とは別に、再開時に食い違いを止める）で、
    None なら書かない（既定の run の checkpoint は従来とビット単位で同じ）"""
    tmp = path.with_suffix(".tmp")
    state = learner.state_dict()
    torch.save(state if collect_cfg is None else {**state, "collect_cfg": collect_cfg}, tmp)
    tmp.replace(path)


def open_seed(seed, step):
    """序盤の強制の乱数列（更新ごと。同じ step から再開すれば同じ局に同じ定跡が付く。リーグの相手の配分
    `seed * 1_000_003 + step`・--pipeline の行動の乱数 `+ 17` とは別の列）"""
    return seed * 1_000_003 + 29 + step


# R-NaD だけの設定（--algo mmd で既定以外なら止める。黙って無視すると取り違える）
RNAD_ONLY = ("lr", "eta", "beta", "neurd_clip", "inv_mu_cap", "policy_loss", "c_vtrace", "c_ply", "schedule",
             "schedule_start", "adam_b1")


def _floats4(spec, name):
    if spec is None:
        return None
    vals = tuple(float(x) for x in spec.split(","))
    if len(vals) != 4:
        raise SystemExit(f"--{name} は 'coef,decay,ceil,floor' の4つ")
    return vals


def check_algo_args(ap, args):
    """--algo と、学習器ごとの設定の組み合わせを検査する。MMD なら MMDConfig を返す（R-NaD なら None）"""
    mmd_given = sorted(k for k, v in vars(args).items() if k.startswith("mmd_") and k != "mmd_preset" and v is not None)
    if args.algo == "rnad":
        if mmd_given or args.mmd_preset != "tsuitate" or args.init_which != "target" or args.init_step:
            raise SystemExit("--mmd-* / --init-which / --init-step は --algo mmd のときだけ使う")
        return None
    if not args.compact_legal:
        raise SystemExit("--algo mmd は --compact-legal と一緒に使う（収集方策の対数確率を合法手の添字の並びで持つ）")
    bad = [f for f, on in (("--fused", args.fused), ("--reuse", args.reuse != 1), ("--lr-sched", args.lr_sched),
                           ("--tail-avg", args.tail_avg), ("--league", args.league), ("--nyugyoku", args.nyugyoku),
                           ("--guard", args.guard), ("--speed", args.speed), ("--aux-truth", args.aux_truth > 0)) if on]
    bad += ["--" + k.replace("_", "-") for k in RNAD_ONLY if getattr(args, k) != ap.get_default(k)]
    if bad:
        raise SystemExit(f"--algo mmd では使えない: {' '.join(bad)}（学習率は --mmd-lr、1つの batch を何周するかは "
                         "--mmd-epochs。自己対局だけ・ボーナスなしで使う。docs/rl-learning-speed.md 10節）")
    try:
        return make_mmd_config(
            args.mmd_preset, filter=args.mmd_filter, value_rows=args.mmd_value_rows, filt_window=args.mmd_filt_window,
            pol_scale=args.mmd_pol_scale, clip=args.mmd_clip, kl=args.mmd_kl,
            decoupled=None if args.mmd_decoupled is None else args.mmd_decoupled == "on",
            prox_cap=args.mmd_prox_cap, vf=args.mmd_vf, gae_lambda=args.mmd_gae_lambda, td_lambda=args.mmd_td_lambda,
            lambda_unit=args.mmd_lambda_unit, magnet=args.mmd_magnet, magnet_avg=args.mmd_magnet_avg,
            temp=_floats4(args.mmd_temp, "mmd-temp"),
            temp_t_off=args.mmd_temp_t_off, lr=_floats4(args.mmd_lr, "mmd-lr"), lr_t_off=args.mmd_lr_t_off,
            t_scale=args.mmd_t_scale, steps=args.mmd_steps, chunk=args.mmd_chunk, epochs=args.mmd_epochs,
            max_grad_norm=args.mmd_max_grad_norm, adam_b1=args.mmd_adam_b1, value_warmup=args.mmd_value_warmup,
            target_avg=args.target_avg)
    except ValueError as e:
        raise SystemExit(f"MMD の設定: {e}") from e


def start_mmd_from_init(learner, args):
    """--init の checkpoint から MMD を始める（--init-which both なら net ← 現在のネット、ターゲットと磁石 ← ターゲット。
    --init-step なら step の番号を続ける）。Adam の状態・正則化方策・乱数状態は持ち込まない"""
    ick = torch.load(args.init, map_location="cpu")
    step = int(ick.get("step", 0)) if args.init_step else 0
    if args.init_step and "step" not in ick:
        raise SystemExit(f"{args.init} に step が無い（--init-step は R-NaD / MMD の checkpoint に使う）")
    if args.init_which == "both":
        if "net" not in ick or "target" not in ick:
            raise SystemExit(f"{args.init} に net と target が無い（--init-which both は R-NaD / MMD の checkpoint に使う）")
        net_sd, target_sd = ick["net"], ick["target"]
    else:
        net_sd = target_sd = learner.net.state_dict()
    sha = hashlib.sha256(pathlib.Path(args.init).read_bytes()).hexdigest()
    learner.start_from(net_sd, target_sd, step,
                       source={"path": str(args.init), "step": ick.get("step"), "which": args.init_which,
                               "algo": ick.get("algo", "rnad" if "prev" in ick else "bc"), "sha256": sha})
    print(f"MMD を {args.init}（{args.init_which}、step {ick.get('step')}）から始める。番号は step {step} から", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "out_rnad" / "bc" / "checkpoint.pt"),
                    help="初期化に使う checkpoint（none = 乱数初期化）")
    ap.add_argument("--anchor", default=str(ROOT / "out_rnad" / "bc" / "checkpoint.pt"),
                    help="進み具合を測る対戦相手（固定）")
    ap.add_argument("--out", default=str(ROOT / "out_rnad" / "rnad"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64, help="1回の更新で自己対局する局数")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-games", type=int, default=128)
    ap.add_argument("--export-every", type=int, default=100)
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="checkpoint を書く間隔（Spot で止められたらここから再開する）")
    ap.add_argument("--schedule", type=int, default=100, help="正則化方策を差し替える間隔（更新回数）")
    ap.add_argument("--schedule-start", type=int, default=0,
                    help="この更新回数を区切りにして、そこから --schedule ごとに差し替える（0 = 0 から数える）。"
                         "途中の区切りから間隔を変えて再開するときに使う（58,000 から Δm=10,000 なら 58000 → 区切りは 68,000, 78,000, …）")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eta", type=float, default=0.2)
    # NeuRD のしきい値（合法手の平均を引いたロジットが ±β を超える方向へは押さない）。
    # 参照の既定 2.0 は小さいゲーム用で、行動数の多い本ゲームの模倣ネットは最善手が +6.6
    # （中央値、99% 点で +14.9）にあるので、β=2 だと方策が平らへ潰される（2026-09-28 実測。
    # 300 更新で現在のネットが錨に 23% まで弱くなった）。模倣ネットの広がりより上に置く
    ap.add_argument("--beta", type=float, default=16.0)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux-truth", type=float, default=0.0,
                    help="真実の盤面を予測する補助ヘッドの損失の重み（0 = 使わない。報酬・方策の目的は変えない）")
    ap.add_argument("--override-config", action="store_true",
                    help="再開時に checkpoint と違う設定（学習率・β など）を使うことを許す。"
                         "指定しないと設定の食い違いは停止する")
    ap.add_argument("--nyugyoku", default="",
                    help="入玉ボーナスの λ の予定（例 '1.0:800,0.5:400,0.25:400,0'。空 = 使わない）。"
                         "区間は更新回数で、正則化方策の差し替え間隔の倍数")
    ap.add_argument("--nyugyoku-mode", choices=["enter", "graded"], default="enter",
                    help="enter = 敵陣への初進入で λ / graded = 前進 3・4・5・6 段の初到達ごとに λ/4")
    ap.add_argument("--nyugyoku-sum", choices=["zero", "own"], default="zero",
                    help="zero = 前進した側に +λ・相手に −λ（ゼロ和）/ own = 前進した側に +λ だけ"
                         "（非ゼロ和。R-NaD の収束保証は外れる）")
    ap.add_argument("--nyugyoku-cond", choices=["any", "win"], default="any",
                    help="any = 節目に着くたびに付ける（勝敗によらない）/ win = その局に勝ったときだけ付ける"
                         "（固め・速攻のボーナスとは併用できない）")
    ap.add_argument("--guard", default="",
                    help="玉の周りの固めのボーナスの λ の予定（書式は --nyugyoku と同じ。空 = 使わない）。"
                         "終局の報酬に λ × 対局を通した自玉の周りの占有率の平均を両者それぞれ足す"
                         "（非ゼロ和。tsuitate_rl.VecEnv の guard_bonus）。相手を固定する "
                         "--league ... --league-frac 1 と組み合わせる想定")
    ap.add_argument("--guard-radius", type=int, choices=[1, 2], default=1,
                    help="占有率を数える自玉からの距離（1 = 8近傍 / 2 = 24マス）")
    ap.add_argument("--guard-cond", choices=["any", "win"], default="any",
                    help="any = 勝敗によらず両者に付ける（tsuitate_rl の guard_bonus）/ win = 勝った局でだけ、"
                         "勝った側に付ける（env にはボーナスを渡さず、終局の記録から学習側で足す）")
    ap.add_argument("--speed", default="",
                    help="決着の速さのボーナスの λ の予定（書式は --nyugyoku と同じ。空 = 使わない）。"
                         "勝った側に λ × max(0, (H − 手数) / H) を足す（tsuitate_rl.VecEnv の speed_bonus。"
                         "手数は受理手の数、引き分けには付かない）")
    ap.add_argument("--speed-horizon", type=int, default=200,
                    help="速さの基準の手数 H（既定 200 = 手数上限。縮めると手数の差が効きやすくなる）")
    ap.add_argument("--speed-sum", choices=["own", "zero"], default="own",
                    help="own = 勝った側だけに足す（非ゼロ和）/ zero = 負けた側から同じ額を引く（ゼロ和。"
                         "負けそうな側に「手数を延ばせ」という動機ができる）")
    ap.add_argument("--open-book", default="",
                    help="序盤の強制の定跡ブック（tsuitate-bot の joseki.json の形式）。自己対局の局ごと・先後それぞれ"
                         "独立に確率 --open-p で、その側の最初の数手をこのラインから指させる（env の中で指すので学習器から"
                         "見ると環境の手。tsuitate-bot の docs/rl-opening-diversity.md）")
    ap.add_argument("--open-p", type=float, default=0.0,
                    help="側ごとに序盤を強制する確率（0 = 強制しない。両者とも強制される局は p²）")
    ap.add_argument("--open-len", default="",
                    help="強制する手数の範囲 'LO:HI'（局ごとに一様に引く。空 = ラインを切った後の最後まで）。"
                         "どれも駒がぶつかった時点（どちらかの駒取り・王手）で止める")
    ap.add_argument("--open-weight", choices=["category", "line"], default="category",
                    help="ラインの選び方（category = 系統ごとに同じ確率 / line = ラインごとに一様）")
    ap.add_argument("--league", default="",
                    help="リーグ学習の相手（カンマ区切り。tsuitate-bot の strategy::make の名前、例 "
                         "'rl_v15,rl_v16,rl_policy:/path/policy.bin'）。空 = 自己対局だけ")
    ap.add_argument("--league-frac", type=float, default=0.5,
                    help="1回の更新の局のうちリーグの相手に回す割合（残りは自己対局）")
    ap.add_argument("--league-window", type=int, default=512, help="相手ごとの勝率を数える直近の局数")
    ap.add_argument("--league-power", type=float, default=2.0,
                    help="相手の重み ∝ (1 − 勝率)^power（hard）/ (勝率 × (1 − 勝率))^power（even）。"
                         "大きいほど寄せ方が強い")
    ap.add_argument("--league-weighting", choices=["hard", "even"], default="hard",
                    help="hard = 負けやすい相手ほど多く当てる / even = 勝率が五分前後の相手ほど多く当てる")
    ap.add_argument("--league-device", choices=["rust", "gpu"], default="rust",
                    help="リーグの相手の手番をどこで指すか。rust = tsuitate_rl の評価モード（凍結版を CPU で"
                         "1手ずつ推論。相手10体・128局で収集 約8秒）/ gpu = 相手の重みを PyTorch へ読んで学習中の"
                         "ネットと同じ device でまとめて推論する（凍結版 rl_* と rl_policy:<パス> だけ。"
                         "相手の行動マスクは環境の今のマスク = 新マスクになる）")
    ap.add_argument("--frozen-models", default=str(ROOT.parent / "tsuitate-bot" / "models"),
                    help="--league-device gpu で凍結版の重み（<名前>.bin）を探すディレクトリ")
    # ---- 学習の速さの実験（docs/rl-learning-speed.md）。どれも既定は従来と同じ
    ap.add_argument("--packed", action="store_true",
                    help="方策の計算を合法手に詰めて行う（数式は同じで速い。rnad.pack_legal）")
    ap.add_argument("--target-avg", type=float, default=0.001,
                    help="ターゲットネットの平均化係数（参照 0.001）")
    ap.add_argument("--adam-b1", type=float, default=0.0, help="Adam の β1（参照 0）")
    ap.add_argument("--c-vtrace", type=float, default=1.0,
                    help="v-trace の c（参照 1.0）。1 未満は λ = c の λ-return（自己対局は集めた方策で更新するので"
                         " π/μ ≈ 1）。目標の分散を減らす代わりに、価値の当たらない区間の信号を削る")
    ap.add_argument("--neurd-clip", type=float, default=10_000.0,
                    help="NeuRD のアドバンテージのクリップ（参照 10,000 = 事実上無し）。まれにしか選ばれない手の"
                         " s = (結果 − v)/μ が勾配の分散のほとんどを占めるので、下げると分散が減る（偏りと引き換え）")
    ap.add_argument("--inv-mu-cap", type=float, default=0.0,
                    help="v-trace の s に掛かる 1/μ の上限（0 = 上限なし = 参照と同じ）。--neurd-clip と同じ狙いで、効く場所が違う")
    ap.add_argument("--policy-loss", choices=["neurd", "pg"], default="neurd",
                    help="方策の損失（neurd = 参照 / pg = 押し出しに各手の π を掛けたソフトマックス方策勾配。"
                         "まれな手の 1/μ による分散が消える）")
    ap.add_argument("--pipeline", action="store_true",
                    help="自己対局を別スレッドで集めて更新と重ねる（1更新遅れの方策で集める。rnad.PipelinedCollector）。"
                         "自己対局だけ・ボーナスなし・補助ヘッドなしの設定に限る")
    ap.add_argument("--reuse", type=int, default=1,
                    help="1つの batch で続けて更新する回数（2 以上で標本を使い回す。2回目以降は π/μ ≠ 1 を"
                         "v-trace が補正する。正則化方策の差し替え間隔は更新回数で数える）")
    ap.add_argument("--panel", default="",
                    help="固定パネル（speedlab.py panel の JSON。レートを固定した凍結版など）で学習の途中を測る。"
                         "結果は <out>/panel_eval.jsonl（speedlab.py report でそのまま学習曲線になる）。"
                         "手の乱数は局ごとの鍵で決めるので、どの run・どの書き出しも同じ局の組で測られる")
    ap.add_argument("--panel-every", type=int, default=500, help="パネルで測る間隔（更新回数。0 なら最初と最後だけ）")
    ap.add_argument("--panel-games", type=int, default=256, help="パネルの相手1体あたりの局数（偶数）")
    ap.add_argument("--panel-which", default="target", help="測るネット（target / current をカンマ区切り）")
    ap.add_argument("--reseed", action="store_true",
                    help="再開時に checkpoint の乱数状態を使わず --seed で乱数を初めから作り直す（同じ checkpoint から "
                         "seed だけ変えた複製の対照を回すとき。付けないと checkpoint の乱数状態が戻り、同じ自己対局が再生される）")
    ap.add_argument("--c-ply", default="",
                    help="終盤だけの λ（'時刻:λ'、例 '60:0.8'。collect の時刻 = 両者の試行の通し番号がこれ以上の手番で c = λ）")
    ap.add_argument("--lr-sched", default="",
                    help="段階（Δm）の中の学習率の予定。'cos:<最小の割合>' で段階の頭の lr から段階末の lr×割合へ余弦で下げ、"
                         "次の段階の頭で戻す")
    ap.add_argument("--tail-avg", action="store_true",
                    help="段階の後半（α = 1 の区間）の現在のネットの一様な平均を、段階の区切りで tail_<step>.bin に書く（学習には使わない）")
    ap.add_argument("--reset-adam", action="store_true",
                    help="再開時に Adam の状態（モーメントと step）を捨てる。更新の式を途中で変える A/B は両腕に付ける"
                         "（2次モーメントが古い勾配の大きさを b2 = 0.999 で何千更新も覚えているため）")
    ap.add_argument("--lr-warmup", type=int, default=0,
                    help="この実行の最初の N 更新で学習率を 0 から線形に上げる（--reset-adam の直後の2次モーメントは"
                         "数回分の勾配だけから推定されて揺れるため）。--algo mmd では MMD を始めた step から数え"
                         "（--reset-adam を付けた実行はその頭から）、--mmd-value-warmup があれば温めが終わった step から数え直す")
    ap.add_argument("--compact-legal", action="store_true",
                    help="batch の合法手のマスクを添字で持つ（768局で約 0.79GB → 約 11MB。更新の結果は同じ。"
                         "--pipeline で batch が2つ同時にあるときのメモリ対策）")
    ap.add_argument("--extra-ema", default="",
                    help="学習には使わない追加の指数移動平均（カンマ区切りの係数、例 '0.0003,0.003'）。書き出しの区切りで "
                         "ema<係数>_<step>.bin に書く（ターゲットの 0.001 以外の平均化で書き出しが強くなるかを見る。"
                         "状態は <out>/extra_ema.pt に保存して再開する）")
    ap.add_argument("--export-current", action="store_true",
                    help="書き出しの区切りで現在のネットも current_<step>.bin に書く（speedlab で測る）")
    ap.add_argument("--fast-collect", choices=["py", "rust"], default="",
                    help="自己対局の収集の CPU の仕事を減らす（batch はビット単位で同じ。rnad.collect の fast）。"
                         "py = Python だけの変更、rust = 生きている局だけを Rust で観測する（rl-env を作り直した "
                         "tsuitate_rl が要る）。--compact-legal と一緒に使い、相手をネットに指させるリーグ"
                         "（--league-device gpu）・入玉ボーナス・補助ヘッドとは使えない")
    ap.add_argument("--device-batch", action="store_true",
                    help="batch を更新の頭で1回だけデバイスへ送り、チャンクごとの転送と同期をなくす（結果は同じ。"
                         "--packed --compact-legal と一緒に使う。GPU のメモリを batch あたり約1GB 使う）")
    ap.add_argument("--fused", action="store_true",
                    help="1回目の現在のネットの順伝播を2回目へまとめる（決定点あたり順伝播1回分が減る。結果は同じ。"
                         "--league-device gpu とは使えない。勾配をチャンクごとに取っておくので GPU のメモリを約130MB 使う。"
                         "v-trace を CPU でチャンクごとに進めるので、--device-batch と併用してもチャンクごとに1回の同期は残る）")
    # ---- 学習器（docs/rl-learning-speed.md 10節）。既定は R-NaD のまま
    ap.add_argument("--algo", choices=["rnad", "mmd"], default="rnad",
                    help="rnad = R-NaD（従来）/ mmd = Ataraxos 型（PPO の比クリップ ＋ 近い方策と磁石への KL。rnad/mmd.py）。"
                         "mmd は自己対局だけ・ボーナスなし・--compact-legal が要る")
    ap.add_argument("--init-which", choices=["target", "net", "both"], default="target",
                    help="（--algo mmd）--init の checkpoint のどのネットから始めるか。target = ターゲット（従来と同じ）/ "
                         "net = 現在のネット / both = 現在のネットを学び、ターゲットと磁石はターゲットから（R-NaD の続きにする推奨）")
    ap.add_argument("--init-step", action="store_true",
                    help="（--algo mmd）--init の checkpoint の step から番号を続ける（書き出しの名前とパネルの横軸が本線とそろう）。"
                         "温度・学習率の予定と --lr-warmup はこの step から数える")
    mg = ap.add_argument_group("MMD（--algo mmd のときだけ。指定しなければ --mmd-preset の値）")
    mg.add_argument("--mmd-preset", choices=list(MMD_PRESETS), default="tsuitate",
                    help="tsuitate = 本線向けの既定（pps 0.5・動く磁石〔ターゲット〕・温度 0.05 固定・λ_A 0.5・1更新 4歩・学習率固定）/ "
                         "mixinit = 設計の初版（pps 0.25・始めの方策の固定の混合磁石・温度を下げる・λ_A 0.8・8歩。代理で崩れた形）/ "
                         "ataraxos = Ataraxos のコードの既定（|A| の 75%%点で切る・一様な磁石・λ_A 0.5・8歩・学習率も下げる）")
    mg.add_argument("--mmd-filter", default=None,
                    help="方策の行の選び方: pps:<割合>[,floor=<下限>] / quantile:<分位>[,min=<下限>] / topk:<割合> / none")
    mg.add_argument("--mmd-value-rows", default=None, help="価値の行: auto / all / filtered / uniform:<割合>")
    mg.add_argument("--mmd-filt-window", type=int, default=None, help="quantile の分位点に入れる過去の更新の数")
    mg.add_argument("--mmd-pol-scale", type=float, default=None, help="方策の項の倍率")
    mg.add_argument("--mmd-clip", type=float, default=None, help="比のクリップ ε")
    mg.add_argument("--mmd-kl", type=float, default=None, help="近い方策（pipeline なしなら収集方策）への逆KL の係数")
    mg.add_argument("--mmd-decoupled", choices=["on", "off"], default=None,
                    help="--pipeline でクリップと KL の中心を更新の頭の方策にする（decoupled PPO）")
    mg.add_argument("--mmd-prox-cap", type=float, default=None, help="decoupled の外側の重要度の上限")
    mg.add_argument("--mmd-vf", type=float, default=None, help="価値の損失の係数")
    mg.add_argument("--mmd-gae-lambda", type=float, default=None, help="アドバンテージ（GAE）の λ")
    mg.add_argument("--mmd-td-lambda", type=float, default=None, help="価値の目標（TD(λ)）の λ")
    mg.add_argument("--mmd-lambda-unit", choices=["attempt", "turn"], default=None,
                    help="λ をかける単位（attempt = 反則の試行も1回 / turn = 受理された手ごと）")
    mg.add_argument("--mmd-magnet", default=None, help="磁石: mix:init,<ε> / mix:target,<ε> / uniform / target")
    mg.add_argument("--mmd-magnet-avg", type=float, default=None,
                    help="動く磁石の EMA の係数（0 = 書き出しのターゲット〔--target-avg〕をそのまま磁石にする）")
    mg.add_argument("--mmd-temp", default=None, help="温度（磁石への KL の係数）の予定 'coef,decay,ceil,floor'")
    mg.add_argument("--mmd-temp-t-off", type=float, default=None, help="温度の予定の起点のずらし")
    mg.add_argument("--mmd-lr", default=None,
                    help="学習率の予定 'coef,decay,ceil,floor'（clip(coef/(t+1)^decay, floor, ceil)。固定なら '<lr>,0,<lr>,<lr>'）")
    mg.add_argument("--mmd-lr-t-off", type=float, default=None, help="学習率の予定の起点のずらし")
    mg.add_argument("--mmd-t-scale", type=float, default=None, help="予定の t の刻み（t = MMD の更新回数 × これ）")
    mg.add_argument("--mmd-steps", type=int, default=None, help="1更新の Adam の歩数（残した行を k 等分）")
    mg.add_argument("--mmd-chunk", type=int, default=None, help="逆伝播のチャンク（勾配を足してから1歩）")
    mg.add_argument("--mmd-epochs", type=int, default=None, help="残した行を何周するか")
    mg.add_argument("--mmd-max-grad-norm", type=float, default=None, help="勾配ノルムのクリップ（0 = なし）")
    mg.add_argument("--mmd-adam-b1", type=float, default=None, help="Adam の β1（R-NaD の --adam-b1 とは別）")
    mg.add_argument("--mmd-value-warmup", type=int, default=None,
                    help="最初の N 更新は価値の頭だけを全行で学ぶ（幹と方策は止める）")
    ap.add_argument("--threads", type=int, default=0, help="torch のスレッド数（0 = 既定）")
    ap.add_argument("--planes", type=int, default=NUM_PLANES,
                    help="--init none のときの入力のプレーン数（86 = v1 / 91 = v2 確定事実つき）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", choices=PRECISIONS, default="default",
                    help="ネットの計算精度（rnad.set_precision。default でも畳み込みは TF32 = tf32 と速さは同じ）")
    args = ap.parse_args()

    set_precision(args.precision)
    # 速くする選択肢（docs/rl-learning-speed.md「さらに速くする」）が使えない組み合わせを最初に止める
    if args.fast_collect and (not args.compact_legal or args.aux_truth > 0 or args.nyugyoku
                              or (args.league and args.league_device == "gpu")):
        raise SystemExit("--fast-collect は --compact-legal と一緒に使い、--league-device gpu・入玉ボーナス・"
                         "補助ヘッドとは使えない")
    if args.fast_collect == "rust" and not hasattr(tsuitate_rl.VecEnv, "observe_alive_f16_csr"):
        raise SystemExit("--fast-collect rust には VecEnv.observe_alive_f16_csr が要る（tsuitate-bot の rl-env を "
                         "maturin develop で作り直す）")
    if args.device_batch and not (args.compact_legal and (args.packed or args.algo == "mmd")):
        raise SystemExit("--device-batch は --packed --compact-legal と一緒に使う（--algo mmd は --compact-legal だけ）")
    mcfg = check_algo_args(ap, args)
    open_kw, open_cfg = {}, None
    if args.open_p or args.open_book or args.open_len or args.open_weight != "category":
        if not (0.0 < args.open_p <= 1.0) or not args.open_book:
            raise SystemExit("序盤の強制は --open-book と --open-p（0 < p ≤ 1）を一緒に指定する"
                             "（--open-len / --open-weight もこの2つと一緒に使う）")
        bad = [f for f, on in (("--league", args.league), ("--guard", args.guard), ("--speed", args.speed),
                               ("--nyugyoku", args.nyugyoku), ("--c-ply", args.c_ply),
                               ("--mmd-lambda-unit turn", mcfg is not None and mcfg.lambda_unit == "turn")) if on]
        if bad:
            # 強制した手で囲った形・延びた手数・玉の前進がボーナスに入る。turn は「同じ側が続けて指す」を
            # 反則とみなすので、相手の強制した手をはさんだ手番を取り違える。c_ply は collect の行の番号で数えるので、
            # 強制した手（行にならない）のぶん手数がずれる
            raise SystemExit(f"序盤の強制は {' '.join(bad)} と一緒に使えない（自己対局だけ・ボーナスなし）")
        if not hasattr(tsuitate_rl.VecEnv, "set_opening_seed"):
            raise SystemExit("tsuitate_rl が序盤の強制を知らない。tsuitate-bot の rl-env を maturin develop で作り直す")
        try:
            book = openings.load_book(args.open_book, args.open_weight)
            open_len = openings.parse_len(args.open_len)
            open_kw = openings.env_kwargs(book, args.open_p, open_len)
            tsuitate_rl.VecEnv(2, seed=0, auto_reset=False, **open_kw)  # ラインの検査（不正なら ValueError）
        except (ValueError, OSError) as e:
            raise SystemExit(f"序盤の強制の設定: {e}") from e
        open_cfg = {"book_sha256": book.sha256, "lines": len(book.lines), "p": args.open_p,
                    "len": list(open_len) if open_len else None, "weight": args.open_weight}
    if args.fused and args.league and args.league_device == "gpu":
        raise SystemExit("--fused は --league-device gpu とは使えない（相手の手番が batch の隙間になる）")
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.schedule_start:
        # 区切りの列 [0, start, start + schedule, …]（EntropySchedule は最後の大きさで繰り返す）
        if args.nyugyoku or args.guard or args.speed:
            raise SystemExit("--schedule-start とボーナスの λ の予定は併用できない（λ の区間を区切りに揃えられない）")
        sizes, repeats = (args.schedule_start, args.schedule), (1, 1)
    else:
        sizes, repeats = (args.schedule,), (1,)
    cfg = RNaDConfig(eta=args.eta, beta=args.beta, lr=args.lr, schedule_sizes=sizes,
                     schedule_repeats=repeats, aux_truth=args.aux_truth, target_avg=args.target_avg,
                     adam_b1=args.adam_b1, c_vtrace=args.c_vtrace, clip=args.neurd_clip,
                     inv_mu_cap=args.inv_mu_cap, policy_loss=args.policy_loss,
                     c_ply=((int(args.c_ply.split(":")[0]), float(args.c_ply.split(":")[1])) if args.c_ply else (0, 1.0)))
    lam_at, lam_max = parse_lambda_schedule(args.nyugyoku, args.schedule)
    guard_at, guard_max = parse_lambda_schedule(args.guard, args.schedule)
    speed_at, speed_max = parse_lambda_schedule(args.speed, args.schedule)
    # ボーナス込みの終局報酬は最大 1 + λ（入玉は節目の合計で λ、固めは占有率 ≤ 1 なので λ、
    # 速攻は速さの係数 ≤ 1 なので λ。zero でも負けた側は −1 − λ まで）。
    # 報酬と η をこの倍率で縮めて価値の頭（tanh）の範囲に収める（issue #4）。**予定全体で固定する**:
    # 区切りごとに変えると価値の目標の単位がそのたびに変わり、ゆっくり追従するターゲットネットの v
    # （target_avg=0.001、合わせ直すのに約1,000更新）が区間の大半で古い単位のままになる。
    # 固定でも報酬と η にそろって掛かるので、各区間の勝敗・ボーナス・正則化の比率は変わらない。
    # loss_v は 1/s²、v_target_mean・s_abs_mean は 1/s の単位になる（s の違う run は log の reward_scale で換算）
    reward_scale = 1.0 + lam_max + guard_max + speed_max
    if args.nyugyoku_cond == "win" and (args.guard or args.speed):
        raise SystemExit("--nyugyoku-cond win は勝ち負けを終局の報酬の符号で見るので、--guard / --speed と併用できない")
    if args.speed_horizon < 1:
        raise SystemExit("--speed-horizon は 1 以上")

    def env_kwargs(guard_lam, speed_lam):
        # 使わないボーナスの引数は渡さない（guard_bonus / speed_bonus の無い古い tsuitate_rl でも動くように）
        kw = {}
        if args.guard:
            # 勝ち限定は env ではボーナスを付けず、終局の記録から勝った側だけに足す（apply_guard_win_bonus）
            kw.update(guard_bonus=(guard_lam if args.guard_cond == "any" else 0.0), guard_radius=args.guard_radius)
        if args.speed:
            kw.update(speed_bonus=speed_lam, speed_horizon=args.speed_horizon, speed_sum=args.speed_sum)
        return kw
    if args.eval_games % 2:
        raise SystemExit("--eval-games は偶数にする（先後を同数にするため）")
    ck_path = out / "checkpoint.pt"
    ck = torch.load(ck_path, map_location=device) if ck_path.exists() else None
    if ck is not None and ck.get("algo", "rnad") != args.algo:
        raise SystemExit(f"{ck_path} は --algo {ck.get('algo', 'rnad')} の checkpoint（今は --algo {args.algo}）。"
                         "別の学習器として黙って再開しない")
    if ck is not None and "config" in ck:
        # 再開: ネットの形は checkpoint から（--init には依存しない）
        init_net = PolicyValueNet(ck["config"]["channels"], ck["config"]["blocks"], ck["config"]["planes"])
        if args.algo == "mmd" and (args.init_which != "target" or args.init_step):
            print("checkpoint から再開するので --init / --init-which / --init-step は使わない", flush=True)
    else:
        init_net = load_net(args.init, args.channels, args.blocks, planes=args.planes,
                            which=("net" if args.init_which in ("net", "both") else "target"))
    if args.algo == "mmd":
        learner = MMDLearner(init_net, mcfg, device, seed=args.seed, device_batch=args.device_batch,
                             stale_actor=args.pipeline)
        if ck is None and args.init != "none":
            start_mmd_from_init(learner, args)
    else:
        learner = RNaDLearner(init_net, cfg, device, packed=args.packed, device_batch=args.device_batch,
                              fused=args.fused)
    # 観測のプレーン数は学習するネットの形で決まる（v2 = 確定事実のプレーンつきの checkpoint は
    # widen_planes.py で作る）。錨・リーグの相手が v1 のネットでも、v2 の観測の先頭が v1 なので読める
    # （PolicyValueNet.trunk が切る）。v1 のときは引数を渡さない（planes の無い古い tsuitate_rl でも動くように）
    obs_kw = {"planes": init_net.config["planes"]} if init_net.config["planes"] != NUM_PLANES else {}
    anchor = load_net(args.anchor, args.channels, args.blocks, planes=init_net.config["planes"]).to(device).eval()
    eval_gen = torch.Generator(device=device)
    eval_gen.manual_seed(args.seed + 20_000)
    learner_cfg = mcfg if args.algo == "mmd" else cfg
    record = {"args": vars(args), "net": learner.net.config,
              **({"mmd": dict(mcfg.__dict__), "init_from": learner.init_from} if args.algo == "mmd"
                 else {"rnad": dict(cfg.__dict__)})}
    if open_cfg is not None:
        record["opening"] = open_cfg
    if ck is not None:
        diff = cfg_diff(ck["cfg"], learner_cfg)
        # 集め方の設定（序盤の強制）。再開で --open-* を付け忘れると黙って強制が止まるので、学習器の設定と同じく止める
        old_open = (ck.get("collect_cfg") or {}).get("opening")
        if old_open != open_cfg:
            diff["opening"] = (old_open, open_cfg)
        if diff and not args.override_config:
            raise SystemExit(f"checkpoint と設定が違う: {diff}（(保存, 今)）。"
                             "意図した変更なら --override-config を付ける")
        learner.load_state_dict(ck)
        if args.algo == "mmd":
            record["init_from"] = learner.init_from  # checkpoint から戻した値（record を作った時点ではまだ None）
        if args.reseed:
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            if args.algo == "mmd":
                learner.reseed(args.seed)
            print(f"乱数を --seed {args.seed} で作り直した（checkpoint の乱数状態は使わない）", flush=True)
        if diff:
            print(f"再開時に設定を上書き: {diff}")
        if set(diff) - {"opening"}:
            learner.apply_cfg(learner_cfg, **({"old": ck["cfg"]} if args.algo == "mmd" else {}))
        elif args.algo == "mmd":
            pass  # MMD の学習率は毎更新で予定から入れ直す
        else:
            # 学習率は optimizer の状態として checkpoint に入るので、--lr-warmup / --lr-sched で下げた値のまま止めた run を
            # そのフラグなしで再開すると、下がった学習率が黙って使われ続ける。設定の学習率へ戻す（フラグがあれば毎更新で上書き）
            for opt_ in (learner.opt, learner.aux_opt):
                if opt_ is not None:
                    for g in opt_.param_groups:
                        g["lr"] = cfg.lr
        print(f"再開: step {learner.step_count}")
        record["resumed_from_step"] = learner.step_count
        record["overridden"] = {k: list(v) for k, v in diff.items()}
        (out / f"config_resume_{learner.step_count}.json").write_text(json.dumps(record, indent=2))
    else:
        (out / "config.json").write_text(json.dumps(record, indent=2))

    if not 0.0 <= args.league_frac <= 1.0:
        raise SystemExit(f"--league-frac {args.league_frac} は 0〜1 にする")
    league = None
    n_league = 0
    if args.league:
        league = League([n.strip() for n in args.league.split(",") if n.strip()],
                        window=args.league_window, power=args.league_power,
                        weighting=args.league_weighting)
        if args.league_device == "gpu":
            # 相手の重みを PyTorch へ読む（凍結版は --frozen-models/<名前>.bin、rl_policy:<パス> はそのパス）
            opp_nets = {}
            for name in league.names:
                path = (name[len("rl_policy:"):] if name.startswith("rl_policy:")
                        else pathlib.Path(args.frozen_models) / f"{name}.bin")
                if not pathlib.Path(path).exists():
                    raise SystemExit(f"リーグの相手 {name} の重み {path} が無い（--league-device gpu は "
                                     "凍結版 rl_* と rl_policy:<パス> だけ）")
                opp_nets[name] = read_weights(path).to(device).eval()
            # 全相手を1回の推論で指させる（相手ごとに推論すると相手の数だけ直列になり速くならない）
            opp_nets = OpponentPool(opp_nets)
        else:
            # 相手を全部ここで一度作ってみる。tsuitate_rl が古いビルドだと新しい凍結版を知らず、
            # その相手が初めて配られた更新で落ちる（2026-10-03 に rl_v23 で実際に落ちた）
            for name in league.names:
                try:
                    tsuitate_rl.VecEnv(2, opponent=name, seed=0, auto_reset=False)
                except ValueError as e:
                    raise SystemExit(f"リーグの相手 {name} を作れない: {e}。tsuitate-bot を更新して "
                                     "tsuitate_rl を作り直す（maturin develop）") from e
        league_path = out / "league.json"
        if league_path.exists():
            league.load_state_dict(json.loads(league_path.read_text()))
        # 2局単位に丸める。round は偶数丸めなので、奇数の batch で frac=1 だと batch を超えうる
        n_league = min(int(round(args.batch * args.league_frac / 2)) * 2, args.batch // 2 * 2)
    n_self = args.batch - n_league
    # 自己対局の env はボーナスの λ を作るときに決めるので、λ が変わったら作り直す（下のループ）
    env_bonus = (guard_at(learner.step_count), speed_at(learner.step_count))
    env = (tsuitate_rl.VecEnv(n_self, seed=args.seed, auto_reset=False, **env_kwargs(*env_bonus), **obs_kw, **open_kw)
           if n_self else None)
    eval_env = tsuitate_rl.VecEnv(args.eval_games, seed=args.seed + 10_000, auto_reset=False, **obs_kw)
    log = open(out / "log.jsonl", "a")

    def save_all():
        save_checkpoint(learner, ck_path, None if open_cfg is None else {"opening": open_cfg})
        if extra_ema:
            tmp = out / "extra_ema.tmp"
            torch.save({"step": learner.step_count, "nets": {str(t): n.state_dict() for t, n in extra_ema.items()}}, tmp)
            tmp.replace(out / "extra_ema.pt")
        if args.tail_avg:
            tmp = out / "tail_avg.tmp"
            torch.save({"step": learner.step_count, **tail}, tmp)
            tmp.replace(out / "tail_avg.pt")
        if league is not None:
            (out / "league.json").write_text(json.dumps(league.state_dict()))

    def evaluate(step):
        # ターゲットは現在のネットにゆっくり追従するだけなので、両方を測る
        # （ターゲットだけだと現在のネットの劣化が隠れる。2026-09-28 の β=2 で実際に隠れた）
        for which, net in (("target", learner.target), ("current", learner.net)):
            t0 = time.time()
            w, l, d = play_match(eval_env, net, anchor, device, eval_gen)
            eval_env.pop_finished()
            n = w + l + d
            rate = (w + 0.5 * d) / n
            rec = {"step": step, "eval": which, "eval_vs_anchor": rate, "w": w, "l": l, "d": d,
                   "ci95": 1.96 * (rate * (1 - rate) / n) ** 0.5, "sec": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[eval] step {step} {which}: 錨に {w}勝{l}敗{d}分 = {rate:.3f}±{rec['ci95']:.3f}",
                  flush=True)

    if args.reuse < 1:
        raise SystemExit("--reuse は 1 以上")
    if args.reset_adam:
        learner.reset_optimizer_state()
        print(f"Adam の状態を捨てた（step {learner.step_count} から）", flush=True)
    # warmup の起点。MMD は MMD を始めた step（checkpoint に入る）から数えるので、同じコマンドで再開しても走り直さない。
    # ただし --reset-adam で Adam を捨てたら、今の step から数え直す（R-NaD と同じく、その実行の頭で温め直す）
    warm_from = learner.mmd_t0 if args.algo == "mmd" and not args.reset_adam else learner.step_count

    lr_min_frac = None
    if args.lr_sched:
        kind, _, frac_s = args.lr_sched.partition(":")
        if kind != "cos":
            raise SystemExit("--lr-sched は 'cos:<最小の割合>'")
        lr_min_frac = float(frac_s or 0.1)

    def set_warmup_lr():
        if args.algo == "mmd":
            # MMD は学習器が予定の学習率を毎更新で入れる。ここは warmup の倍率だけ。
            # 価値の温め（--mmd-value-warmup）の間は価値の頭だけが Adam の歩を踏み、幹と方策の頭は温めの後に初めて踏む
            # （Adam の状態が空のまま）ので、warmup は温めが終わった step から数え直す。価値の頭の学習率もそこで一度下がって
            # 上がり直す（価値の頭の Adam はもう温まっているので害は小さい）。温めが 0 なら MMD を始めた step から数えるだけ。
            # --reset-adam で温めの後から数え始めた実行は、温めの終わりをもう跨がないのでそのまま数える
            t = learner.step_count - warm_from
            policy_from = learner.mmd_t0 + learner.cfg.value_warmup  # 方策が動き始める step
            if learner.cfg.value_warmup > 0 and warm_from < policy_from <= learner.step_count:
                t = learner.step_count - policy_from
            learner.lr_scale = min(1.0, (t + 1) / args.lr_warmup) if args.lr_warmup > 0 else 1.0
            return
        lr = cfg.lr
        if lr_min_frac is not None:
            start, size = learner.schedule.position(learner.step_count)
            x = (learner.step_count - start) / size
            lr = cfg.lr * (lr_min_frac + (1 - lr_min_frac) * 0.5 * (1 + math.cos(math.pi * x)))
        if args.lr_warmup > 0:
            lr *= min(1.0, (learner.step_count - warm_from + 1) / args.lr_warmup)
        if args.lr_warmup > 0 or lr_min_frac is not None:
            for g in learner.opt.param_groups:
                g["lr"] = lr

    # 段階の後半の一様な平均（--tail-avg）。α = 1 の区間（段階の後半）の更新後のネットを足し、区切りで書いて捨てる
    tail = {"sum": None, "n": 0}
    if args.tail_avg and (out / "tail_avg.pt").exists():
        saved_tail = torch.load(out / "tail_avg.pt", map_location="cpu")
        if saved_tail.get("step") == learner.step_count:
            tail.update(sum=saved_tail["sum"], n=saved_tail["n"])

    def update_tail(info):
        if not args.tail_avg:
            return
        start, size = learner.schedule.position(learner.step_count - 1)
        if learner.step_count - 1 - start >= size // 2:
            with torch.no_grad():
                vec = torch.nn.utils.parameters_to_vector(learner.net.parameters()).detach().double().cpu()
                tail["sum"] = vec if tail["sum"] is None else tail["sum"] + vec
                tail["n"] += 1
        if info["reg_update"] and tail["n"]:
            want = size - size // 2
            if tail["n"] == want:
                net_ = copy.deepcopy(learner.net).eval()
                torch.nn.utils.vector_to_parameters((tail["sum"] / tail["n"]).float().to(device), net_.parameters())
                write_weights(net_, out / f"tail_{learner.step_count}.bin")
            else:
                # 段階の途中で再開して和が引き継げなかった（後半の一部しか入っていない）ので書かない
                print(f"[tail-avg] step {learner.step_count}: 後半 {want} 更新のうち {tail['n']} しか無いので書き出さない",
                      flush=True)
            tail["sum"], tail["n"] = None, 0
    pipe = None
    if args.pipeline:
        if league is not None or args.guard or args.speed or args.nyugyoku or cfg.aux_truth > 0:
            raise SystemExit("--pipeline は自己対局だけ・ボーナスなし・補助ヘッドなしの設定で使う")
        pipe = PipelinedCollector(
            lambda: tsuitate_rl.VecEnv(n_self, seed=args.seed, auto_reset=False, **obs_kw, **open_kw), device,
            seed=args.seed * 1_000_003 + 17 + learner.step_count, compact_legal=args.compact_legal,
            fast=args.fast_collect, actor_extras=args.algo == "mmd")
        # 序盤の強制の乱数列は「その batch を使う step」で決める（ループの前に頼む batch は今の step で使う）
        pipe_open_seed = open_seed(args.seed, learner.step_count) if open_kw else None
        pipe.submit(learner.net, opening_seed=pipe_open_seed)

    # 学習には使わない追加の指数移動平均（--extra-ema）。現在のネットの写しを係数ごとに持つ
    extra_ema = {}
    if args.extra_ema:
        ema_path = out / "extra_ema.pt"
        saved = torch.load(ema_path, map_location=device) if ema_path.exists() else {}
        # checkpoint と同じ時点に保存したものだけ使う（--extra-ema なしで回した区間をはさむと古い平均が残っている）
        saved = saved.get("nets", {}) if saved.get("step") == learner.step_count else {}
        for tau in [float(x) for x in args.extra_ema.split(",") if x.strip()]:
            # ターゲット（それまでの平均）から始める。現在のネットから始めると、学習の途中から付けたとき
            # 平均が今の雑音から始まり、忘れるのに約 3/係数 更新かかる
            net_ = copy.deepcopy(learner.target).eval()
            net_.requires_grad_(False)
            if str(tau) in saved:
                net_.load_state_dict(saved[str(tau)])
            extra_ema[tau] = net_

    def update_extra_ema():
        with torch.no_grad():
            for tau, net_ in extra_ema.items():
                for pe, p in zip(net_.parameters(), learner.net.parameters()):
                    pe.add_(tau * (p - pe))
                for be, b in zip(net_.buffers(), learner.net.buffers()):
                    be.copy_(b)

    panel = None
    if args.panel:
        import speedlab
        panel = json.loads(pathlib.Path(args.panel).read_text())
        if panel["planes"] != init_net.config["planes"]:
            raise SystemExit(f"パネルの planes {panel['planes']} が学習するネット {init_net.config['planes']} と違う")
        panel_members = [speedlab.Member(m, panel["planes"], device) for m in panel["members"]]
        if args.panel_games % 2:
            raise SystemExit("--panel-games は偶数（先後を同数にするため）")

    def panel_eval(step):
        """パネルの各相手と局ごとの結果を panel_eval.jsonl へ（speedlab.py eval と同じ形・同じ seed の決め方）"""
        nets = {"target": learner.target, "current": learner.net}
        with open(out / "panel_eval.jsonl", "a") as f:
            for which in [w.strip() for w in args.panel_which.split(",") if w.strip()]:
                t0 = time.time()
                res = []
                for m in panel_members:
                    o = speedlab.play_vs(nets[which], m, args.panel_games, speedlab.EVAL_SEED + speedlab.stable_seed(m.name),
                                         panel["planes"], per_game=True, device=device)
                    w, l, d = int((o > 0).sum()), int((o < 0).sum()), int((o == 0).sum())
                    res.append((m.name, w, l, d))
                    f.write(json.dumps({"step": step, "which": which, "opponent": m.name, "w": w, "l": l, "d": d,
                                        "games": args.panel_games, "outcomes": o.tolist()}) + "\n")
                f.flush()
                elo, se = speedlab.fit_fixed(res, panel["ratings"])
                print(f"[panel] step {step} {which}: Elo {elo:.0f}±{se:.0f}（"
                      + " ".join(f"{n} {w}-{l}-{d}" for n, w, l, d in res) + f"、{time.time() - t0:.0f}s）", flush=True)

    def crossed(step, every):
        """この反復（reuse 回の更新）で every の倍数をまたいだか（reuse=1 なら step % every == 0 と同じ）"""
        return every > 0 and step // every != (step - args.reuse) // every

    if panel is not None and not (out / "panel_eval.jsonl").exists():
        panel_eval(learner.step_count)  # 起点（再開でも最初の1回は測る）
    if learner.step_count == 0 and args.eval_every:
        evaluate(0)
    if ck is None and args.export_current:
        # 学習前の点（speedlab の曲線の起点）。--init-step で途中の番号から始めた run も、その番号で書く
        # （起点を同じ指し方〔speedlab eval --decode〕で測り直せるように）
        write_weights(learner.target, out / f"policy_{learner.step_count}.bin")
        write_weights(learner.net, out / f"current_{learner.step_count}.bin")
    while learner.step_count < args.steps:
        t0 = time.time()
        lam = lam_at(learner.step_count)
        guard_lam = guard_at(learner.step_count)
        speed_lam = speed_at(learner.step_count)
        if env is not None and (guard_lam, speed_lam) != env_bonus:
            env = tsuitate_rl.VecEnv(n_self, seed=args.seed, auto_reset=False,
                                     **env_kwargs(guard_lam, speed_lam), **obs_kw, **open_kw)
            env_bonus = (guard_lam, speed_lam)
        if pipe is not None:
            # 前の反復で頼んだ batch（1更新前のネットで集めたもの）を受け取り、今のネットで次を頼む
            batch, finished, busy = pipe.result()
            batch_open_seed = pipe_open_seed
            # 次の batch はこの反復の更新（reuse 回）の後に使うので、その step の乱数列で集める
            # （続けた run と再開した run で、同じ step の batch には同じ乱数列が付く）
            pipe_open_seed = open_seed(args.seed, learner.step_count + args.reuse) if open_kw else None
            pipe.submit(learner.net, opening_seed=pipe_open_seed)
            league_env = None
            league_finished = []
            t1 = time.time()
        else:
            envs = [env] if env is not None else []
            batch_open_seed = open_seed(args.seed, learner.step_count) if open_kw else None
            if open_kw and env is not None:
                env.set_opening_seed(batch_open_seed)  # collect の reset から効く
            league_env = None
            league_games = None  # --league-device gpu のときの局ごとの (相手, 学習側の色)
            if league is not None and n_league and args.league_device == "gpu":
                # 自己対局モードの env で、相手の手番だけ相手のネットに指させる（collect の opponents）。
                # 色は局ごとに交互（相手ごとの局数は偶数なので先後がそろう）
                rng = np.random.default_rng(args.seed * 1_000_003 + learner.step_count)
                names = [name for name, k in league.allocate(n_league, rng).items() for _ in range(k)]
                league_games = [(name, i % 2) for i, name in enumerate(names)]
                league_env = tsuitate_rl.VecEnv(n_league, seed=args.seed * 1_000_003 + learner.step_count,
                                                auto_reset=False, **env_kwargs(guard_lam, speed_lam), **obs_kw)
                envs.append(league_env)
            elif league is not None and n_league:
                # リーグの局は1つの env にまとめる（局番号 g の局は opponents[g % len] と指す。相手ごとに
                # env を分けると相手の推論が直列になる）。相手ごとの局数は偶数なので先後がそろう
                rng = np.random.default_rng(args.seed * 1_000_003 + learner.step_count)
                opponents = [name for name, k in league.allocate(n_league, rng).items() for _ in range(k)]
                league_env = tsuitate_rl.VecEnv(n_league, opponents=opponents,
                                                seed=args.seed * 1_000_003 + learner.step_count, auto_reset=False,
                                                **env_kwargs(guard_lam, speed_lam), **obs_kw)
                envs.append(league_env)
            batch = collect(MultiEnv(envs), learner.net, device,
                            nyugyoku_lambda=lam, nyugyoku_mode=args.nyugyoku_mode,
                            nyugyoku_zero_sum=args.nyugyoku_sum == "zero",
                            nyugyoku_win_only=args.nyugyoku_cond == "win",
                            opponents=(None if league_games is None else [None] * n_self + league_games),
                            opp_nets=(opp_nets if league_games is not None else None),
                            truth=cfg.aux_truth > 0, compact_legal=args.compact_legal, fast=args.fast_collect,
                            actor_extras=args.algo == "mmd")
            t1 = time.time()
            finished = env.pop_finished() if env is not None else []
            league_finished = league_env.pop_finished() if league_env is not None else []
            if league_games is not None:
                # 自己対局モードの env なので、どの相手・どちらの色の局かはこちらで付ける
                # （並びの位置 = 局番号の順。collect は最初に reset するので局番号は 0 からではない）
                league_finished = sorted(league_finished, key=lambda f: f["game_no"])
                assert len(league_finished) == len(league_games), (len(league_finished), len(league_games))
                for f, (name, color) in zip(league_finished, league_games):
                    f["opponent"], f["learner"] = name, color
        league_rec = league.record(league_finished) if league_env is not None else {}
        if args.guard and args.guard_cond == "win" and guard_lam:
            apply_guard_win_bonus(batch, [(0, finished), (n_self, league_finished)], guard_lam, args.guard_radius)
        for _ in range(args.reuse):
            set_warmup_lr()
            info = learner.update(batch, reward_scale=reward_scale)
            update_extra_ema()
            update_tail(info)
        t2 = time.time()
        step = learner.step_count
        rec = {"step": step, **info, "games": len(finished),
               "nyugyoku_lambda": lam, "nyugyoku_sum": args.nyugyoku_sum, "nyugyoku_cond": args.nyugyoku_cond, "guard_cond": args.guard_cond, "reward_scale": reward_scale,
               "collect_sec": t1 - t0, "update_sec": t2 - t1, "lr": learner.opt.param_groups[0]["lr"]}
        if pipe is not None:
            rec["collect_busy_sec"] = busy  # 別スレッドで集めるのにかかった時間（collect_sec は待った時間）
        if args.guard:
            # 玉の周りの固めの監視値（guard.summarize）。自己対局とリーグの局は分けて出す
            # （リーグの局は相手側の反則数 opp_fouls が見える = この戦法の狙いどおりか）
            rec["guard_lambda"] = guard_lam
            rec["guard_radius"] = args.guard_radius
            if finished:
                rec["guard_self"] = summarize_guard(finished, args.guard_radius)
            if league_finished:
                rec["guard_league"] = summarize_guard(league_finished, args.guard_radius)
        if args.speed:
            # 決着の速さの監視値（speed.summarize）。自己対局とリーグの局は分けて出す
            rec["speed_lambda"] = speed_lam
            rec["speed_horizon"] = args.speed_horizon
            rec["speed_sum"] = args.speed_sum
            if finished:
                rec["speed_self"] = summarize_speed(finished, args.speed_horizon)
            if league_finished:
                rec["speed_league"] = summarize_speed(league_finished, args.speed_horizon)
        # 手数・反則・終局理由・入玉・前進の集計は自己対局の局だけ（リーグの局は学習側の玉しか
        # 観測しないので混ぜると薄まる）。リーグを入れる前後でこれらの数字は単純には比べられない。
        # --league-frac 1 で自己対局が無いときは出さない（空の平均は NaN になり、厳密な JSON でなくなる）
        if finished:
            ent = batch["entered"][:n_self]
            res = batch["outcome"][:n_self][ent]
            rec.update({
                "plies": float(np.mean([f["plies"] for f in finished])),
                "fouls_per_side": float(np.mean([sum(f["fouls"]) / 2 for f in finished])),
                "reasons": dict(Counter(f["reason"] for f in finished)),
                "nyugyoku_rate": float(ent.mean()),
                "nyugyoku_w": int((res > 0).sum()), "nyugyoku_l": int((res < 0).sum()),
                "nyugyoku_d": int((res == 0).sum()),
                "king_advance_hist": np.bincount(batch["advance"][:n_self].ravel(), minlength=9).tolist(),
            })
        if open_kw and finished:
            # 序盤の強制の監視値（強制した側の割合・手数・やめた理由、最初の接触の手数、接触の前の反則）。
            # 強制した側の序盤の局面は学習の行に入らないので、pi_top_* やエントロピーは強制しない run と直接は比べられない
            rec["opening"] = {**openings.summarize(finished), "seed": batch_open_seed}
        if league is not None:
            # league_games = この更新の相手ごとの (勝, 負, 分)、league = 窓の勝率と次の配分の重み
            rec["league_games"] = league_rec
            rec["league"] = league.summary()
        log.write(json.dumps(rec) + "\n")
        log.flush()
        selfplay = (f"手数 {rec['plies']:.0f} 反則/側 {rec['fouls_per_side']:.2f} " if finished else "")
        nyugyoku = f"入玉 {rec['nyugyoku_rate']:.1%}" if finished else "入玉 -"
        print(f"step {step}: 標本 {info['samples']} {selfplay}"
              f"loss_v {info['loss_v']:.3f} nerd {info['loss_nerd']:.3f} "
              + (f"aux {info['loss_aux']:.3f} " if "loss_aux" in info else "") +
              f"α {info['alpha']:.2f}{' [正則化更新]' if info['reg_update'] else ''} "
              f"{nyugyoku}(λ {lam}) "
              f"({t1 - t0:.1f}s + {t2 - t1:.1f}s)", flush=True)
        if args.algo == "mmd" and info.get("minibatches"):
            print(f"  MMD: 温度 {info['temperature']:.4f} 行 {info['rows_kept']}/{info['samples']} "
                  f"KL(中心) {info['kl_prox_last']:.4f} 未使用の行の KL(μ) {info['kl_mu_unkept']:.4f} "
                  f"clip {info['clip_frac']:.2f} |g| p50 {info['grad_norm_p50']:.2f}（クリップ {info['clip_rate']:.0%}）"
                  f" エントロピー {info['entropy_mean']:.3f}", flush=True)
        if args.guard and crossed(step, 10):
            g = rec.get("guard_league") or rec.get("guard_self") or {}
            key = f"guard_r{args.guard_radius}"
            if g:
                print(f"  固め（λ {guard_lam}）: 占有率 {g[key]:.3f} 玉の近くへの打ち {g['near_drops']:.2f} "
                      f"検知 {g['near_drop_fouls']:.2f} 反則 {g['fouls']:.2f}"
                      + (f" / 相手 占有率 {g['opp_' + key]:.3f} 反則 {g['opp_fouls']:.2f}"
                         if "opp_fouls" in g else ""), flush=True)
        if args.speed and crossed(step, 10):
            parts = []
            for label, key in (("自己対局", "speed_self"), ("リーグ", "speed_league")):
                sp = rec.get(key) or {}
                if "win_plies" in sp:
                    parts.append(f"{label} 勝ち局 {sp['win_plies']:.0f}手 速さ {sp['win_speed']:.3f} "
                                 f"詰み {sp['win_mate']:.0%}")
            if parts:
                print(f"  速攻（λ {speed_lam}）: " + " / ".join(parts), flush=True)
        if league is not None and crossed(step, 10):
            print("  リーグ（勝率/重み）: " + "  ".join(
                f"{n} {s['wr']:.2f}/{s['weight']:.2f}" for n, s in rec["league"].items()), flush=True)
        if crossed(step, args.eval_every):
            evaluate(step)
        if panel is not None and (crossed(step, args.panel_every) or step >= args.steps):
            panel_eval(step)
        if crossed(step, args.checkpoint_every) or crossed(step, args.export_every):
            save_all()
        if crossed(step, args.export_every) or step >= args.steps:
            write_weights(learner.target, out / f"policy_{step}.bin")
            if args.export_current:
                write_weights(learner.net, out / f"current_{step}.bin")
            for tau, net_ in extra_ema.items():
                write_weights(net_, out / f"ema{tau:g}_{step}.bin")
    if pipe is not None:
        pipe.result()  # 最後に頼んだ batch は使わずに捨てる
        pipe.close()
    save_all()


if __name__ == "__main__":
    main()
