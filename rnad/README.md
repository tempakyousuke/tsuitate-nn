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
| `fetch_records.py` | GitHub Actions の対局記録を取る |
| `league.py` | リーグ学習の相手の選び方（負けやすい相手ほど多く当てる） |
| `guard.py` | 防御特化（玉の周りの固め）の監視値の集計 |

### 結果（2026-09-27、64ch×6ブロック、4,800局・53万試行、3エポック）

- 検証局（240局・2.6万試行、棋譜の署名でまとめて分割）: 教師の手の的中率 **50.0%**
  （マスク内一様なら 1.5%）、反則だった試行に限ると 37.8%、交差エントロピー 1.92。
  価値の MSE 0.937（常に0と答えると 1.0 前後）
- arena（`ARENA_MATCH_SEED=20260927`、100局）: **vs heuristic 100勝0敗**、反則/局 0.99（heuristic 9.76）、
  思考 平均 20ms
- arena（同 seed、100局）: **vs estimator_v6 73勝27敗（73.0%±8.7）**、反則/局 5.81（v6 8.40）、
  思考 平均 21ms（v6 1,418ms）。**手元の実測で、マシンに他の負荷があった**（v6 は壁時計で思考を
  打ち切るので、負荷で弱く出た可能性がある）。正式な計測は CI の arena で取り直す

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

### ボーナス込みの報酬の縮め方（入玉・防御特化に共通、issue #4）

- 価値の頭は tanh（−1〜1）なので、ボーナス込みの終局報酬（最大 1 + λ）と η を同じ倍率
  s = 1 + 入玉の λ の最大 + 固めの λ の最大 で割ってから学習する（`RNaDLearner.update(reward_scale=s)`）。
  s は **λ の予定全体で固定**（区切りごとに変えると価値の単位が変わり、ターゲットネットの v が追いつかない）。
  ボーナスなし（本線）は s = 1 で従来と同一
- **ログの比較**: `loss_v` は 1/s²、`v_target_mean`・`s_abs_mean` は 1/s の単位になる。s の違う run 同士は
  log.jsonl の `reward_scale` で換算してから比べる
- **既存の checkpoint からの再開**: s=1 で学んだ checkpoint を s>1 で続けると、Adam の2次モーメントが古い勾配の
  大きさのままなので、最初の約1,000更新は実効の学習率が最大で 1/s 程度まで下がる（新しく始めるなら関係ない）
