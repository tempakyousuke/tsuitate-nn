# rnad/ — DeepNash（R-NaD）路線の学習

設計は tsuitate-bot の `docs/rl-deepnash-design.md`。環境・観測エンコード・行動マスクは
tsuitate-bot 側（Rust）にあり、ここからは PyO3 の拡張 `tsuitate_rl` 経由で呼ぶ。

## 準備

```sh
# tsuitate_rl（tsuitate-bot/rl-env）をこの venv へ入れる。tsuitate-bot の src/rl を変えたら再実行
.venv/bin/python -m pip install maturin
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m ../tsuitate-bot/rl-env/Cargo.toml
```

## M2a: 行動クローニング（配管の検証）

```sh
.venv/bin/python rnad/fetch_records.py --runs 30          # 日次 Mine tsume の記録 → data/rl_records/
.venv/bin/python rnad/train_bc.py --out out_rnad/bc      # 学習（MPS で 1エポック約3分）
.venv/bin/python rnad/export_weights.py out_rnad/bc/checkpoint.pt out_rnad/bc/policy.bin
# tsuitate-bot 側で対戦
ARENA_MATCH_SEED=20260927 cargo run --release --bin arena -- 100 "rl_policy:<policy.bin の絶対パス>" heuristic
```

## M2b: R-NaD（自己対局）

```sh
.venv/bin/python rnad/tests/test_rnad.py                       # 参照実装（OpenSpiel の JAX 版）との一致
.venv/bin/python rnad/train_rnad.py --init out_rnad/bc/checkpoint.pt --out out_rnad/rnad_bc
```

- `rnad.py`: R-NaD 本体。OpenSpiel の rnad.py（削除直前の d1dcdf5）の移植で、**損失と勾配が参照と
  一致する**ことを `tests/` のフィクスチャ（参照実装で乱数入力から計算した npz）で検査する。
  行動数が 11,259 あるので、参照が作る [A] 次元の Q ベクトルは作らず、アドバンテージを
  ステップごとのスカラーへ分解して持つ（数学的に同じ）
- `train_rnad.py`: 1回の更新 = バッチ数ぶんの局を最後まで自己対局 → 1ステップ。進み具合は
  **錨（既定は模倣学習のネット）との直接対戦の勝率**で測る。`<out>/log.jsonl` に更新・評価ごとの記録、
  `policy_<step>.bin` はターゲットネットの Rust 用の重み（CI の arena で測れる）
- BatchNorm は常に eval モード（初期化したネットの統計に固定）

| ファイル | 中身 |
| --- | --- |
| `model.py` | 方策・価値ネット（ResNet。方策 [139,9,9]、価値 tanh）。**層を変えたら tsuitate-bot の `src/rl/policy_net.rs` と `export_weights.py` も直す** |
| `train_bc.py` | 記録を真似る教師あり学習。局単位で学習/検証を分ける |
| `export_weights.py` | Rust 推論用の重み（`TSRLPV01` 形式、BatchNorm は畳み込みへ畳み込む）。`--tiny --test-vector` で tsuitate-bot のテスト用フィクスチャを作る |
| `widen_planes.py` | checkpoint の入力プレーンを広げる（v1 → v2）・v3 の組を足す（`--opp-hand-planes` / `--opp-foul-planes`）。足した列は 0 で、元と同じ関数のまま。R-NaD・MMD の checkpoint を読める（下の「入力プレーン」） |
| `fetch_records.py` | GitHub Actions の対局記録を取る |
| `league.py` | リーグ学習の相手の選び方（負けやすい相手ほど多く当てる） |
| `guard.py` | 防御特化（玉の周りの固め）の監視値の集計 |
| `speedlab.py` | 学習の速さの比較: 固定パネル（`panel`）で run の書き出しを測り（`eval`、局ごとに鍵を決めた共通乱数）、腕ごとの学習曲線と対照とのペア差をまとめる（`report`）。締まりの効果を除いて比べるには `eval --decode greedy\|temp:<T>`（記録名 `<which>@greedy` など）、途中から始めた run の起点の書き出しは `export-ck`、起点を何度で指すと締まった腕にそろうかは `temp-match`（`--seed` を変えると別の局になる。パネルの評価の乱数は学習中のパネルの記録と対にするため変えない） |
| `log_windows.py` | 学習ログ（log.jsonl）の見張りの値（エントロピー・π>0.999 の割合・手数・反則・磁石への KL など）を step の窓ごとに平均して並べる。MMD の中止の門の形の指標はこの 100更新の窓の平均で見る（中央値は締まりを小さく見せる） |
| `openings.py` | 序盤の強制の定跡ブック（tsuitate-bot の `joseki.json` の形式）の読み込み・系統ごとの重み・env へ渡す引数・ログの監視値（`train_rnad.py --open-book`） |
| `opening_eval.py` | 序盤の多様さ（`entropy`: 初期局面の方策・最初の k 手の並びのエントロピー）と、序盤を強制した対局（`robust`: 相手 / 自分 / 両者の序盤を定跡に強制したときの得点・接触の前の反則を系統ごとに） |
| `gradsnr.py` | 学習なしの診断: 変種（クリップ・1/μ の上限・PG）ごとの局ごとの勾配の分散・期待勾配・雑音のスケール |
| `bench_update.py` | 1更新の壁時計（従来 / `--packed` / 連続 / `--pipeline`）と、packed の勾配の一致。`--collect-arms` / `--update-arms` で `--fast-collect` / `--device-batch` / `--fused` を同じ batch・同じ乱数のペアで測り、結果がビット単位で同じかも出す（`--step` で α を選ぶ・`--precision`） |
| `distill.py` | 方策ネットを小さいネットへ蒸留する（CPU の代理 P2 の出発点） |
| `toy_kuhn.py` | クーンポーカーで R-NaD の変種の偏りを厳密な NashConv で見る |
| `stage_log.py` | 学習ログ（log.jsonl）だけでの事前確認: α<1 / α=1 ごとの1更新の秒と律速、錨への勝率の段階内の形と段階の平らさ g（Δm を縮めるかの判断）、`--out-dir` で区切りでの停止の時間とループの外の雑費 |
| `stage_profile.py` | 1つの段階の中の書き出しの KL の曲線（段階の頭の書き出しからの KL と隣どうしの KL）と、段階内の収束時間 T_r のあてはめ |
| `kl_versions.py` | 凍結版どうしの方策の KL・全変動距離・最善手の一致率を共通の局面の上で総当たりに測る（Δm ごとの1段階の動き） |
| `proto/speed2/` | 「さらに速くする」の試作と検算（収集・更新のビット一致の高速化、段階の見張りの差分など。学習には使わない。回し方は中の README.md） |

### 結果（2026-09-27、64ch×6ブロック、4,800局・53万試行、3エポック）

- 検証局（240局・2.6万試行、棋譜の署名でまとめて分割）: 教師の手の的中率 **50.0%**
  （マスク内一様なら 1.5%）、反則だった試行に限ると 37.8%、交差エントロピー 1.92。
  価値の MSE 0.937（常に0と答えると 1.0 前後）
- arena（`ARENA_MATCH_SEED=20260927`、100局）: **vs heuristic 100勝0敗**、反則/局 0.99（heuristic 9.76）、
  思考 平均 20ms
- arena（同 seed、100局）: **vs estimator_v6 73勝27敗（73.0%±8.7）**、反則/局 5.81（v6 8.40）、
  思考 平均 21ms（v6 1,418ms）。**手元の実測で、マシンに他の負荷があった**（v6 は壁時計で思考を
  打ち切るので、負荷で弱く出た可能性がある）。正式な計測は CI の arena で取り直す

## 学習の速さの実験（tsuitate-bot の `docs/rl-learning-speed.md`）

どれも**既定は従来と同じ**（付けなければ本線の学習は変わらない）。測り方・結果・GPU での手順は tsuitate-bot の doc。

| フラグ | 中身 |
| --- | --- |
| `--packed` | 方策の計算を合法手に詰めて行う（数式は同じ。`tests/test_packed.py`）。重み 0 の正則化ネットの順伝播は常に省く |
| `--pipeline` | 自己対局を別スレッド・別 CUDA ストリームで集め、更新と重ねる（1更新遅れの方策。自己対局だけ・ボーナスなし・補助ヘッドなしの設定。`--league` / `--guard` / `--speed` / `--nyugyoku` / `--aux-truth` と併用すると止まる） |
| `--compact-legal` | batch の合法手のマスクを添字で持つ（768局で 0.79GB → 11MB。`--pipeline` のメモリ対策） |
| `--fast-collect py\|rust` | 自己対局の収集の CPU の仕事を減らす（**batch はビット単位で同じ**。`tests/test_fast_collect.py`）。py は Python だけ（生きている局の行を1回だけ抜く・観測を fp16 で送る・合法手の添字を1回だけ作る）、rust は加えて Rust 側で生きている局だけを観測する（`VecEnv.observe_alive_f16_csr`。tsuitate-bot の rl-env を `maturin develop` で作り直す）。`--compact-legal` と一緒に使い、`--league-device gpu`・入玉ボーナス・補助ヘッドとは使えない |
| `--device-batch` | batch を更新の頭で1回だけ GPU へ送り、チャンクごとの転送と同期をなくす（**結果はビット単位で同じ**。`tests/test_fast_update.py`）。`--packed --compact-legal` と一緒に使う。GPU のメモリを batch あたり約1GB 使う |
| `--fused` | 1回目（勾配なし）の現在のネットの順伝播を2回目へまとめる（決定点あたり順伝播1回分が減る。チャンクを後ろから回して v-trace をチャンクごとに進め、勾配はチャンクごとに取っておいて従来と同じ順に足すので**結果はビット単位で同じ**）。`--league-device gpu` とは使えない。GPU のメモリを約130MB 使う。v-trace は CPU でチャンクごとに進めるので、`--device-batch` と併用してもチャンクごとに1回の同期は残る |
| `--inv-mu-cap K` / `--neurd-clip C` | v-trace の s の 1/μ の上限 / NeuRD のクリップ（まれな手のスパイクを取る） |
| `--policy-loss pg` | NeuRD の押し出しに π を掛ける（ソフトマックス方策勾配） |
| `--c-vtrace` / `--c-ply 時刻:λ` | v-trace の c（λ-return）/ 終盤だけ λ |
| `--target-avg` / `--adam-b1` / `--reuse` | ターゲットの平均化係数 / Adam の β1 / 1つの batch で複数回更新 |
| `--lr-sched cos:<割合>` / `--lr-warmup N` | 段階の中の学習率の余弦 / 最初の N 更新の線形の立ち上げ |
| `--reset-adam` | 再開時に Adam の状態を捨てる（**途中から更新の式を変える A/B は両腕に付ける**。2次モーメントが古い勾配を何千更新も覚えている） |
| `--reseed` | 再開時に checkpoint の乱数状態を使わず `--seed` で作り直す（同じ checkpoint から seed だけ変えた複製の対照） |
| `--extra-ema 係数,...` / `--tail-avg` / `--export-current` | 学習に使わない追加の平均・段階の後半の一様な平均・現在のネットを書き出す |
| `--panel <panel.json>` | 学習ループの中で固定パネルを測り `panel_eval.jsonl` へ（`speedlab.py report` の入力と同じ形） |
| `--planes` / `--threads` | `--init none` の入力のプレーン数 / torch のスレッド数 |

## 入力プレーン（エンコーダ v3、tsuitate-bot の `docs/rl-deepnash-design.md`「エンコーダ v3」）

v2（91枚）の後ろに観測と自分側だけで決まる事実を2組足した 101枚。**組ごとに別のフラグ**で使うかを選べるので、
効果を組ごとに測れる（既定は付けない = v2 のまま）:

| フラグ | 組 | 中身 |
| --- | --- | --- |
| `--opp-hand-planes` | `opp_hand`（7枚） | 相手の持ち駒の上界（駒種別。取られた自駒の数で抑える） |
| `--opp-foul-planes` | `opp_fouls`（3枚） | 相手の反則の時系列（直前の手番 / 王手を受けていた手番 / 受けていない手番） |

- `train_rnad.py` に付けると、v2 の checkpoint（`out/checkpoint.pt` の再開でも `--init` でも）を読み込み時に v3 へ広げる。
  足した列は 0 なので元と同じ関数から始まる（Adam の 2次モーメントは使っている列の平均で入れ直す）
- 一度足した組は以後の再開でも同じフラグを付ける（付け忘れると止まる。組は外せない）。後から組を足すのはフラグを足すだけ
- 使わない組はネットが入力を 0 にしてから読む（`config["extras"]`）ので、stem のその列は 0 のまま動かず、書き出した重みでも 0。
  Rust 側は v3 の観測を全部書くだけで、組の選び方を知らなくてよい。錨・リーグ・パネルの v1 / v2 のネットは観測の先頭だけを読む
- 同じことを学習の外で行うなら `widen_planes.py <元> <出力> --opp-hand-planes`（v3 の checkpoint に組を足すときも同じ）
- 検査は `tests/test_extra_planes.py`（tsuitate_rl が入っていれば Rust 側の組の並びとも照合する）

学習ログには `lr`（実際の学習率）と、方策の固まり方の見張り `pi_top_mean` / `pi_top_gt_0999` / `entropy_mean` が出る。

## MMD 学習器（Ataraxos 型、`--algo mmd`）

`--algo mmd` で更新を R-NaD から **Ataraxos（Stratego）型**に替える（`mmd.py`。既定は R-NaD のまま）。
設計・根拠・計測と 3070 での A/B の手順は tsuitate-bot の `docs/rl-learning-speed.md` 10節。

- 損失は PPO の比クリップ ＋ 近い方策への逆KL ＋ 磁石への逆KL × 温度 ＋ 価値の MSE。**1/μ がどこにも無い**
  （R-NaD の勾配の雑音の 99.6〜99.9% は v-trace の s = (G − v)/μ）
- アドバンテージは手番側ごとの GAE（λ_A 0.5）、価値の目標は TD(λ 0.8)。どちらも収集時の価値（`collect(actor_extras=True)` の
  `v_actor`）で作る。反則で同じ側が続く手番も、`v_trace_scalars` を η=0・π=μ=1・c=λ で呼べばそのまま扱える
- 学習に使う行は既定で **|A| に比例した確率の間引き（pps、割合 0.5）**。重み s̄/p で戻すので期待値は全行と同じ
  （Ataraxos どおりの「|A| の 75%点以上だけ」はクーンで固定点を均衡から外すので、`--mmd-preset ataraxos` の腕に残す）
- 1回の収集を 4 歩の Adam に分ける（行を無作為に並べて4等分）。段階（Δm）は無く、**磁石は既定でターゲット（現在のネットの
  EMA = 動く磁石）、温度は 0.05 固定**。設計の初版の「固定の磁石 ＋ 温度を下げる」形（`--mmd-preset mixinit`）は、クーンで
  R-NaD の約8倍遅く、代理 P2 で自己対局が短い詰みの筋に固まった（tsuitate-bot の docs 10.4）。**この既定（温度 0.05・磁石 = 書き出しの
  EMA）は 3070 の A/B（本線の 108,000 から）で 400更新のうちに方策が締まって中止の門に触れ、棄却した**。current の伸びは自分の系列に集中した
  （tsuitate-bot の docs 10.7。再挑戦するなら温度 0.2・別のゆっくりした磁石 `--mmd-magnet-avg 0.0003` で1回だけ。いつ回すかの条件は 10.7）
- `--pipeline` ではクリップと KL の中心を更新の頭の方策にし、μ との比を外側の重みにする（decoupled PPO）

R-NaD の checkpoint から続ける（net ← 現在のネット、ターゲットと磁石 ← ターゲット、step の番号を続ける。Adam は新しく作る）:

```sh
.venv/bin/python -u rnad/train_rnad.py --algo mmd --init out_rnad/ab_base_108000.pt --init-which both --init-step \
    --out out_rnad/ab_mmd_s1 --seed <本線と同じ> --steps <終わりの step> --batch 768 --anchor <本線と同じ> \
    --pipeline --compact-legal --fast-collect rust --device-batch --precision tf32 --lr-warmup 100
```

| フラグ | 既定（`--mmd-preset tsuitate`） | `ataraxos` | 中身 |
| --- | --- | --- | --- |
| `--mmd-filter` | `pps:0.5,floor=0.05`（`mixinit` は 0.25） | `quantile:0.75,min=0.01` | 方策の行: `pps:<割合>[,floor=]` / `quantile:<分位>[,min=]` / `topk:<割合>`（ちょうどその割合、同値は乱数）/ `none` |
| `--mmd-value-rows` | `auto` | `filtered` | 価値の行: `auto`（pps は方策と同じ行・重み、quantile は `uniform:0.25`）/ `all` / `uniform:<割合>` / `filtered` |
| `--mmd-gae-lambda` / `--mmd-td-lambda` | 0.5 / 0.8（`mixinit` は 0.8 / 0.8） | 0.5 / 0.8 | A と価値の目標の λ |
| `--mmd-lambda-unit` | `attempt` | 同じ | λ を試行ごと（反則も1回）/ `turn`（受理された手ごと）/ `attempt+check`（自分が王手をかけた手では λ をかけない）/ `turn+check`（その両方）。+check は反則・王手を観測のスカラーから読む（`own_next_flags`）ので序盤の強制と使える |
| `--mmd-gae-lambda-unit` | なし（= `--mmd-lambda-unit`） | 同じ | GAE（アドバンテージ）の側だけの単位。価値の目標は変えずに、詰まされた責任を王手の手で減らさずに手前へ届ける A/B 用（tsuitate-bot の docs/rl-learning-speed.md 10.8）。ログに `lam_check_frac` / `lam_foul_frac` / `adv_mean_check` が出る |
| `--mmd-magnet` | `target`（`mixinit` は `mix:init,0.03`） | `uniform` | `mix:init,<ε>` / `mix:target,<ε>` / `uniform` / `target` |
| `--mmd-magnet-avg` | 0 | 0 | 動く磁石（`target` / `mix:target`）の EMA の係数。0 = 書き出しのターゲット（`--target-avg`）をそのまま磁石にする |
| `--mmd-temp` / `--mmd-temp-t-off` | `0.05,0,0.05,0.05`（固定）/ 0（`mixinit` は `0.05,0.3,0.1,0.001` / 2000） | `0.05,0.3,0.1,0.001` / 0 | 温度 clip(coef/(t+1)^decay, floor, ceil)、t = MMD の更新回数 + ずらし |
| `--mmd-lr` / `--mmd-lr-t-off` | `1e-4,0,1e-4,1e-4`（固定）/ 0 | `0.5,1.1,1e-4,5e-6` / 0 | 学習率の予定（同じ式）。`--lr-warmup` を掛ける |
| `--mmd-clip` / `--mmd-kl` / `--mmd-vf` / `--mmd-pol-scale` | 0.2 / 0.1 / 1.0 / 1.0 | 同じ | 比のクリップ / 近い方策への KL / 価値 / 方策の項の係数 |
| `--mmd-decoupled` / `--mmd-prox-cap` | on / 2.0 | off | pipeline の中心を更新の頭の方策にする / 外側の重みの上限 |
| `--mmd-steps` / `--mmd-chunk` / `--mmd-epochs` | 4（`mixinit` は 8）/ 2048 / 1 | 8 / 2048 / 1 | 1更新の Adam の歩数 / 逆伝播のチャンク / 何周するか |
| `--mmd-max-grad-norm` / `--mmd-adam-b1` | 0.267 / 0.9 | 同じ | 勾配ノルムのクリップ（0 = なし）/ Adam の β1 |
| `--mmd-value-warmup` / `--mmd-filt-window` / `--mmd-t-scale` | 0 / 1 / 1.0 | 同じ | 最初の N 更新は価値の頭だけ / quantile の窓 / 予定の t の刻み |
| `--init-which` / `--init-step` | `target` / なし | — | `--init` の checkpoint のどのネットから始めるか（`both` 推奨）/ step の番号を続ける |

- `--compact-legal` が要る。`--fast-collect`・`--pipeline`・`--device-batch`（`--packed` は無くてよい）・`--lr-warmup`・
  `--reset-adam`・`--reseed`・`--panel`・`--extra-ema`・`--export-current` は使える
- 止める組み合わせ: `--fused`・`--reuse` > 1・`--lr-sched`・`--tail-avg`・`--league`・ボーナス（`--nyugyoku` / `--guard` /
  `--speed`）・`--aux-truth`、R-NaD だけの設定（`--lr` `--eta` `--beta` `--neurd-clip` `--inv-mu-cap` `--policy-loss`
  `--c-vtrace` `--c-ply` `--schedule` `--schedule-start` `--adam-b1`）を既定以外にしたとき。R-NaD の `out` を MMD で
  （またはその逆で）再開しようとしたとき
- checkpoint は `algo: "mmd"` と `net` / `target` / `magnet_net` / `opt` / `step` / `mmd_t0` / 行の間引きの乱数を持つ。
  `--init` / `--anchor` / `distill.py` は R-NaD と同じに読める（`config` と `target` がある）。書き出しはターゲット
  （現在のネットの EMA）。**本線と同じ step 番号になるので、Release のタグには `mmd` を入れる**
- 再開: pipeline なしなら、続けて回した run とビット単位で同じ（`tests/test_mmd.py`）。pipeline では再開すると収集が
  `seed*1_000_003 + 17 + step` から始まるので、続けた run とは別の局になる（R-NaD と同じ）。`--lr-warmup` は MMD を
  始めた step から数えるので、同じコマンドで再開しても走り直さない。`--mmd-value-warmup` があれば、温めが終わった step（幹と方策の頭が
  初めて Adam の歩を踏む）から数え直す（2026-10-26 から。それまでは温めの間に使い切っていた）。`--reset-adam` を付けた実行は
  Adam が空になるので、その頭から数え直す（R-NaD と同じ。付けたまま再開すると毎回捨てるので、再開では外す）
- ログ（log.jsonl）: R-NaD と共通の `samples` `loss_v` `loss_nerd`（方策の項）`alpha`（1.0）`reg_update`（false）`lr`
  `collect_sec` `update_sec` に加え、`temperature` `mmd_t` `rows_kept` `keep_rate` `pps_c` `w_max`、信頼領域の
  `kl_prox_mean` / `kl_prox_last`（中心への KL）`kl_mu_unkept`（残さなかった行の更新後の KL(π‖μ)）`clip_frac`
  `ratio_dev_first` `prox_is_capped`、勾配の `grad_norm_p50/p90/max`（クリップ前）`clip_rate`、方策の `entropy_mean`
  `pi_top_mean` `pi_top_gt_0999`（収集方策の値）`magnet_kl`、価値の `calib_band`（|v| の帯ごとの mean(z·sign v)・mean|v|）
  `calib_slope_by_remain`（終局までの手番数の帯ごとの傾き）`resid_band`（V の帯ごとの mean(G_td − v)）
- `bench_update.py --mmd-arms 0.5,1 --calib net,target --loop-mmd <割合> --mmd-stale` で割合ごとの1更新の秒・勾配ノルムの
  分位・GPU のメモリ・価値の較正を測る（A/B の前に決めること。tsuitate-bot の `docs/rl-mmd-design.md` 6.0c。コマンドの全体は
  `docs/rl-learning-speed.md` 10.5）
- テスト: `tests/test_mmd.py`（λ-return と素直なループ・閉形式、extras の収集、Ataraxos の式の密な参照との損失と勾配、
  decoupled PPO、端から端の勾配、行の選び方の不偏性、予定、再開のビット一致、R-NaD の checkpoint からの始め方、
  止める組み合わせ、煙テスト）

## 序盤の強制（`--open-book`）

方策ネットの序盤は約40,000更新から飛車先の歩の速攻1本に固まり、ほかの手は確率 1e-18 まで落ちている（MMD の 138,000 では
1e-8 まで戻ったが、自己対局で指されることはなく序盤は 100% 同じ）。
自己対局の局ごと・**先後それぞれ独立に**確率 p で、その側の最初の数手を定跡のラインから指させ、どんな序盤から来ても
ぶつかった後をうまく指すことを学ばせる。強制した手は env の中で指すので学習器から見ると環境の手で、R-NaD・MMD とも
学習器は変えていない。設計と計測は tsuitate-bot の `docs/rl-opening-diversity.md`。

```sh
# tsuitate-bot の rl-env を入れ直してから（VecEnv の openings / opening_prob が要る）
.venv/bin/python rnad/train_rnad.py ... --open-book ~/Develop/tsuitate-bot/joseki.json --open-p 0.25
```

- `--open-p`: 側ごとの確率（両者とも強制される局は p²）。`--open-len LO:HI`: 強制する手数を局ごとに一様に引く
  （既定はラインを切った後の最後まで）。`--open-weight`: `category`（既定。系統ごとに同じ確率）/ `line`
- 止める条件: どちらかの駒取り・王手（両者とも止める）・その側の反則・予定の手数。定跡の最後の駒取りの手は指させない
  （初期配置の相手の駒に触れる手の手前で切る）
- 使えない組み合わせ: `--league`・`--guard`・`--speed`・`--nyugyoku`・`--c-ply`・`--mmd-lambda-unit turn`（`--mmd-gae-lambda-unit turn` も。`turn+check` は使える）。`--pipeline` は使える。
  `--open-len` / `--open-weight` は `--open-book` と `--open-p` と一緒に使う（片方だけなら止める）。起動時に tsuitate_rl が
  序盤の強制を知っているかを1局指して確かめる
- 設定は checkpoint（`collect_cfg`）と `config.json` に残り、再開で `--open-*` を付け忘れると止まる（`--override-config` で通す）。
  乱数は更新ごとの番号から作るので、再開しても同じ局に同じ定跡が付く
- ログの `opening`: 強制した側の割合・平均の手数・止めた理由・両者とも強制した局の割合・最初の接触の手数・接触の前の反則・
  序盤（各側の受理手 12 まで）の反則（`fouls_early_per_side`）・その batch の乱数列（`seed`。batch を使う step から作る）。
  強制した側の序盤の局面は学習の行に入らないので、`pi_top_*` やエントロピーは強制しない run と直接は比べられない
- 測る: `opening_eval.py entropy NAME=weights.bin ...`・`opening_eval.py robust --cand a.bin [--ref b.bin] --book joseki.json`
- テスト: `tests/test_openings.py`（ブック・env の挙動と p = 0 がブックなしと同じ局・collect の不変量・再開のビット一致・
  付け忘れの再開の停止・使えない組み合わせ）

## 防御特化（玉の周りの固め）モデル

「自玉の周りに駒を打ちまくって相手の打ち反則を誘い、自分の打ち反則で相手の駒を検知して取る」戦法の
練習相手を作る。本命をこの戦法に勝ち切れるように鍛えるためのもの。設計は tsuitate-bot の
`docs/rl-deepnash-design.md` の「防御特化（玉の周りの固め）モデル」。

```sh
# tsuitate-bot の main（PR #60 以降）の rl-env を入れ直してから
.venv/bin/python rnad/tests/test_guard.py                      # 報酬の配り方と集計の検査
.venv/bin/python rnad/train_rnad.py \
  --init <本命の checkpoint.pt> --anchor <本命の checkpoint.pt> --out out_rnad/guard_l05 \
  --league "rl_policy:<本命の policy.bin>" --league-frac 1 \
  --guard 0.5 --guard-radius 1 --steps <更新回数>
```

- **報酬** = 勝敗(±1) + λ × 対局を通した自玉の周りの占有率の平均（自玉から距離 r 以内の盤上のマスの
  うち、玉以外の自駒がいる割合。開始局面と受理手の直後ごとに測って平均）。加算は Rust 側の
  `VecEnv(guard_bonus=λ, guard_radius=r)` が終局の報酬に対して行い、両者それぞれに足す（**非ゼロ和**）
- **本命を固定して、その相手への最善応答として学習する**（`--league <本命> --league-frac 1`）のが想定。
  自己対局に入れると両側ともボーナス付きになり、R-NaD の収束保証が外れる
- `--guard` は `--nyugyoku` と同じ書式の λ の予定（`"1.0:400,0.5"` など）。入玉と違い、玉の周りに打つ
  手はランダム方策でも普通に出る（8近傍の占有率は平均 0.45）ので、固定の λ から始めてよい。
  自己対局の env は λ が変わったときに作り直す（`guard_bonus` は env を作るときに決まるため）
- 錨を本命にすると、`eval_vs_anchor` が「ボーナス抜きの勝敗で本命にどれだけ勝てるか」になる
- **監視**: `log.jsonl` の `guard_league`（相手固定の局）/ `guard_self`（自己対局の局）。
  10更新ごとに画面にも出す

  | キー | 中身 |
  | --- | --- |
  | `guard_r1` / `guard_r2` | 学習側の占有率の対局平均（`--guard-radius` の半径のもの） |
  | `near_drops` | 学習側の、自玉から距離2以内への打ち（受理）の1局あたりの回数 |
  | `near_drop_fouls` | 同じ範囲への打ちの反則のうち、打ったマスに相手の駒がいたもの（＝検知） |
  | `near_captures` | 同じ範囲での駒取り |
  | `fouls` | 学習側の反則数 |
  | `opp_guard_r*` / `opp_fouls` | 相手側の占有率と反則数（相手固定の局だけ）。**この戦法の狙いは相手の反則を誘うこと** |

- **確かめ方**: λ を 0（対照 = ボーナス無しの最善応答）/ 0.5 / 1.0 で振り、占有率・検知・相手の反則数・
  `eval_vs_anchor` を並べる。狙いどおりなら占有率と相手の反則数が上がり、本命への勝率は大きく
  落ちない。占有率だけ上がって勝率が崩れるなら λ が大きすぎる（ただ固めるだけのモデル）
- 計算の都合: 相手（`rl_policy:`）の手番は Rust 側で1手ずつ推論するので、自己対局より1局が重い

## 速攻特化（決着の速さ）モデル
tsuitate-bot の `docs/rl-deepnash-design.md` の速攻特化の案（tsuitate-bot PR #65 の `VecEnv(speed_bonus=...)`）。

```sh
# tsuitate-bot の main（PR #65 以降）の rl-env を入れ直してから
.venv/bin/python rnad/tests/test_speed.py                      # 報酬の配り方と集計の検査
.venv/bin/python rnad/train_rnad.py   --init <本線の checkpoint.pt> --anchor <本線の checkpoint.pt> --out out_rnad/speed   --speed 2.0 --speed-horizon 120 --league rl_v15,...,rl_v24 --league-frac 0.5 --steps <更新回数>
```

- **報酬** = 勝敗(±1) + 勝った側に λ × max(0, (H − 手数) / H)。手数は受理手の数で、引き分けには付かない。
  `--speed-sum own`（既定）は勝った側だけ（非ゼロ和）、`zero` は負けた側から同じ額を引く（負けそうな側に
  「手数を延ばせ」という動機ができるので既定にしない）
- `--speed` は `--nyugyoku` と同じ書式の λ の予定。`--speed-horizon` を縮めると手数の差が効きやすい
  （H=120 なら 50手で 0.58・100手で 0.17）
- 防御特化と違い、自己対局でも両側が速攻を目指すだけなので、**自己対局＋リーグ**で回せる
- **監視**: `log.jsonl` の `speed_self`（自己対局の局）/ `speed_league`（相手固定の局）。10更新ごとに画面にも出す

  | キー | 中身 |
  | --- | --- |
  | `win_plies` / `win_speed` | 学習側の勝ち局の平均手数と速さの係数の平均 |
  | `win_mate` | 学習側の勝ち局のうち詰みで終わった割合（残りは相手の反則負けなど） |
  | `loss_plies` | 学習側の負け局の平均手数（相手固定の局だけ） |
  | `wins` / `losses` / `draws` | 局数（自己対局は決着した局をすべて学習側の勝ちとして数える） |

### ボーナス込みの報酬の縮め方（入玉・防御・速攻特化に共通、issue #4）

- 価値の頭は tanh（−1〜1）なので、ボーナス込みの終局報酬（最大 1 + λ）と η を同じ倍率
  s = 1 + 入玉の λ の最大 + 固めの λ の最大 + 速攻の λ の最大 で割ってから学習する（`RNaDLearner.update(reward_scale=s)`）。
  s は **λ の予定全体で固定**（区切りごとに変えると価値の単位が変わり、ターゲットネットの v が追いつかない）。
  ボーナスなし（本線）は s = 1 で従来と同一
- **ログの比較**: `loss_v` は 1/s²、`v_target_mean`・`s_abs_mean` は 1/s の単位になる。s の違う run 同士は
  log.jsonl の `reward_scale` で換算してから比べる
- **既存の checkpoint からの再開**: s=1 で学んだ checkpoint を s>1 で続けると、Adam の2次モーメントが古い勾配の
  大きさのままなので、最初の約1,000更新は実効の学習率が最大で 1/s 程度まで下がる（新しく始めるなら関係ない）
