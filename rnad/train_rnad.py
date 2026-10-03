"""M2b: R-NaD の自己対局学習（tsuitate-bot の docs/rl-deepnash-design.md）。

    .venv/bin/python rnad/train_rnad.py --init out_rnad/bc/checkpoint.pt --out out_rnad/rnad_bc
    .venv/bin/python rnad/train_rnad.py --init none --out out_rnad/rnad_scratch   # ゼロから

1回の更新 = バッチ数ぶんの局を最後まで自己対局（tsuitate_rl.VecEnv、auto_reset=False）→
R-NaD の損失で1ステップ。学習の進み具合は**錨（既定は模倣学習のネット）との直接対戦の勝率**で
測る（凍結版 estimator は1手1秒以上かかり学習ループに入れられないため）。錨に勝ち越していけば
学習が進んでいる。区切りでターゲットネットの重みを Rust 用に書き出すので、CI の arena
（`rl_policy:release:...`）でも測れる。

防御特化（玉の周りの固め）は `--guard λ --league <本命> --league-frac 1`（rnad/README.md）。

記録: <out>/log.jsonl（更新ごと・評価ごとに1行）、<out>/checkpoint.pt、<out>/policy_<step>.bin
"""

import argparse
import json
import pathlib
import time
from collections import Counter

import numpy as np
import torch

import tsuitate_rl
from export_weights import write as write_weights
from guard import summarize as summarize_guard
from model import PolicyValueNet
from league import League
from rnad import (PRECISIONS, MultiEnv, RNaDConfig, RNaDLearner, cfg_diff, collect, play_match,
                  set_precision)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_net(path: str, channels: int, blocks: int, which: str = "target") -> PolicyValueNet:
    """模倣学習（train_bc.py）の checkpoint か、R-NaD（train_rnad.py）の checkpoint から読む。
    R-NaD の checkpoint は `which`（既定はターゲットネット = 書き出す重みと同じ）を読む"""
    if path == "none":
        return PolicyValueNet(channels, blocks)
    ck = torch.load(path, map_location="cpu")
    if "config" in ck:
        net = PolicyValueNet(ck["config"]["channels"], ck["config"]["blocks"], ck["config"]["planes"])
    elif "net" in ck:
        # ネットの形を持たない古い R-NaD の checkpoint（2026-09-28 以前）は引数の形で読む
        net = PolicyValueNet(channels, blocks)
    else:
        raise SystemExit(f"{path} は checkpoint として読めない")
    net.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck[which])
    return net


def parse_lambda_schedule(spec: str, schedule: int):
    """入玉ボーナスの λ の予定 "1.0:800,0.5:400,0" → 更新回数 step で使う λ を返す関数。
    区間の長さは正則化方策の差し替え間隔の倍数に限る（各区間は固定したゲームで収束させる前提）"""
    if not spec:
        return lambda step: 0.0
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
    return at


def save_checkpoint(learner, path: pathlib.Path) -> None:
    """書きかけで止められても前の checkpoint が壊れないよう、別名に書いてから置き換える"""
    tmp = path.with_suffix(".tmp")
    torch.save(learner.state_dict(), tmp)
    tmp.replace(path)


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
    ap.add_argument("--guard", default="",
                    help="玉の周りの固めのボーナスの λ の予定（書式は --nyugyoku と同じ。空 = 使わない）。"
                         "終局の報酬に λ × 対局を通した自玉の周りの占有率の平均を両者それぞれ足す"
                         "（非ゼロ和。tsuitate_rl.VecEnv の guard_bonus）。相手を固定する "
                         "--league ... --league-frac 1 と組み合わせる想定")
    ap.add_argument("--guard-radius", type=int, choices=[1, 2], default=1,
                    help="占有率を数える自玉からの距離（1 = 8近傍 / 2 = 24マス）")
    ap.add_argument("--league", default="",
                    help="リーグ学習の相手（カンマ区切り。tsuitate-bot の strategy::make の名前、例 "
                         "'rl_v15,rl_v16,rl_policy:/path/policy.bin'）。空 = 自己対局だけ")
    ap.add_argument("--league-frac", type=float, default=0.5,
                    help="1回の更新の局のうちリーグの相手に回す割合（残りは自己対局）")
    ap.add_argument("--league-window", type=int, default=512, help="相手ごとの勝率を数える直近の局数")
    ap.add_argument("--league-power", type=float, default=2.0,
                    help="相手の重み ∝ (1 − 勝率)^power。大きいほど苦手な相手へ寄せる")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", choices=PRECISIONS, default="default",
                    help="ネットの計算精度（rnad.set_precision。default でも畳み込みは TF32 = tf32 と速さは同じ）")
    args = ap.parse_args()

    set_precision(args.precision)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = RNaDConfig(eta=args.eta, beta=args.beta, lr=args.lr, schedule_sizes=(args.schedule,),
                     schedule_repeats=(1,))
    lam_at = parse_lambda_schedule(args.nyugyoku, args.schedule)
    guard_at = parse_lambda_schedule(args.guard, args.schedule)

    def env_kwargs(guard_lam):
        # --guard を使わないときは引数を渡さない（guard_bonus の無い古い tsuitate_rl でも動くように）
        return {"guard_bonus": guard_lam, "guard_radius": args.guard_radius} if args.guard else {}
    if args.eval_games % 2:
        raise SystemExit("--eval-games は偶数にする（先後を同数にするため）")
    ck_path = out / "checkpoint.pt"
    ck = torch.load(ck_path, map_location=device) if ck_path.exists() else None
    if ck is not None and "config" in ck:
        # 再開: ネットの形は checkpoint から（--init には依存しない）
        init_net = PolicyValueNet(ck["config"]["channels"], ck["config"]["blocks"], ck["config"]["planes"])
    else:
        init_net = load_net(args.init, args.channels, args.blocks)
    learner = RNaDLearner(init_net, cfg, device)
    anchor = load_net(args.anchor, args.channels, args.blocks).to(device).eval()
    eval_gen = torch.Generator(device=device)
    eval_gen.manual_seed(args.seed + 20_000)
    record = {"args": vars(args), "rnad": dict(cfg.__dict__), "net": learner.net.config}
    if ck is not None:
        diff = cfg_diff(ck["cfg"], cfg)
        if diff and not args.override_config:
            raise SystemExit(f"checkpoint と設定が違う: {diff}（(保存, 今)）。"
                             "意図した変更なら --override-config を付ける")
        learner.load_state_dict(ck)
        if diff:
            learner.apply_cfg(cfg)
            print(f"再開時に設定を上書き: {diff}")
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
                        window=args.league_window, power=args.league_power)
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
    # 自己対局の env は guard_bonus を作るときに決めるので、λ が変わったら作り直す（下のループ）
    env_guard = guard_at(learner.step_count)
    env = (tsuitate_rl.VecEnv(n_self, seed=args.seed, auto_reset=False, **env_kwargs(env_guard))
           if n_self else None)
    eval_env = tsuitate_rl.VecEnv(args.eval_games, seed=args.seed + 10_000, auto_reset=False)
    log = open(out / "log.jsonl", "a")

    def save_all():
        save_checkpoint(learner, ck_path)
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

    if learner.step_count == 0:
        evaluate(0)
    while learner.step_count < args.steps:
        t0 = time.time()
        lam = lam_at(learner.step_count)
        guard_lam = guard_at(learner.step_count)
        if env is not None and guard_lam != env_guard:
            env = tsuitate_rl.VecEnv(n_self, seed=args.seed, auto_reset=False, **env_kwargs(guard_lam))
            env_guard = guard_lam
        envs = [env] if env is not None else []
        league_env = None
        if league is not None and n_league:
            # リーグの局は1つの env にまとめる（局番号 g の局は opponents[g % len] と指す。相手ごとに
            # env を分けると相手の推論が直列になる）。相手ごとの局数は偶数なので先後がそろう
            rng = np.random.default_rng(args.seed * 1_000_003 + learner.step_count)
            opponents = [name for name, k in league.allocate(n_league, rng).items() for _ in range(k)]
            league_env = tsuitate_rl.VecEnv(n_league, opponents=opponents,
                                            seed=args.seed * 1_000_003 + learner.step_count, auto_reset=False,
                                            **env_kwargs(guard_lam))
            envs.append(league_env)
        batch = collect(MultiEnv(envs), learner.net, device,
                        nyugyoku_lambda=lam, nyugyoku_mode=args.nyugyoku_mode,
                        nyugyoku_zero_sum=args.nyugyoku_sum == "zero")
        t1 = time.time()
        finished = env.pop_finished() if env is not None else []
        league_finished = league_env.pop_finished() if league_env is not None else []
        league_rec = league.record(league_finished) if league_env is not None else {}
        # ボーナス込みの終局報酬は最大 1 + λ（入玉は節目の合計で λ、固めは占有率 ≤ 1 なので λ）。
        # 報酬と η をこの倍率で縮めて価値の頭（tanh）の範囲に収める（issue #4）
        reward_scale = 1.0 + lam + guard_lam
        info = learner.update(batch, reward_scale=reward_scale)
        t2 = time.time()
        step = learner.step_count
        rec = {"step": step, **info, "games": len(finished),
               "nyugyoku_lambda": lam, "nyugyoku_sum": args.nyugyoku_sum, "reward_scale": reward_scale,
               "collect_sec": t1 - t0, "update_sec": t2 - t1}
        if args.guard:
            # 玉の周りの固めの監視値（guard.summarize）。自己対局とリーグの局は分けて出す
            # （リーグの局は相手側の反則数 opp_fouls が見える = この戦法の狙いどおりか）
            rec["guard_lambda"] = guard_lam
            rec["guard_radius"] = args.guard_radius
            if finished:
                rec["guard_self"] = summarize_guard(finished, args.guard_radius)
            if league_finished:
                rec["guard_league"] = summarize_guard(league_finished, args.guard_radius)
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
              f"α {info['alpha']:.2f}{' [正則化更新]' if info['reg_update'] else ''} "
              f"{nyugyoku}(λ {lam}) "
              f"({t1 - t0:.1f}s + {t2 - t1:.1f}s)", flush=True)
        if args.guard and step % 10 == 0:
            g = rec.get("guard_league") or rec.get("guard_self") or {}
            key = f"guard_r{args.guard_radius}"
            if g:
                print(f"  固め（λ {guard_lam}）: 占有率 {g[key]:.3f} 玉の近くへの打ち {g['near_drops']:.2f} "
                      f"検知 {g['near_drop_fouls']:.2f} 反則 {g['fouls']:.2f}"
                      + (f" / 相手 占有率 {g['opp_' + key]:.3f} 反則 {g['opp_fouls']:.2f}"
                         if "opp_fouls" in g else ""), flush=True)
        if league is not None and step % 10 == 0:
            print("  リーグ（勝率/重み）: " + "  ".join(
                f"{n} {s['wr']:.2f}/{s['weight']:.2f}" for n, s in rec["league"].items()), flush=True)
        if step % args.eval_every == 0:
            evaluate(step)
        if step % args.checkpoint_every == 0 or step % args.export_every == 0:
            save_all()
        if step % args.export_every == 0 or step == args.steps:
            write_weights(learner.target, out / f"policy_{step}.bin")
    save_all()


if __name__ == "__main__":
    main()
