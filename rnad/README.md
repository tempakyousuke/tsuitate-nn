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

### 結果（2026-09-27、64ch×6ブロック、4,800局・53万試行、3エポック）

- 検証局（240局・2.6万試行、棋譜の署名でまとめて分割）: 教師の手の的中率 **50.0%**
  （マスク内一様なら 1.5%）、反則だった試行に限ると 37.8%、交差エントロピー 1.92。
  価値の MSE 0.937（常に0と答えると 1.0 前後）
- arena（`ARENA_MATCH_SEED=20260927`、100局）: **vs heuristic 100勝0敗**、反則/局 0.99（heuristic 9.76）、
  思考 平均 20ms
- arena（同 seed、100局）: **vs estimator_v6 73勝27敗（73.0%±8.7）**、反則/局 5.81（v6 8.40）、
  思考 平均 21ms（v6 1,418ms）。**手元の実測で、マシンに他の負荷があった**（v6 は壁時計で思考を
  打ち切るので、負荷で弱く出た可能性がある）。正式な計測は CI の arena で取り直す
