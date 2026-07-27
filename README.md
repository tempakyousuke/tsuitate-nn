# tsuitate-nn

[tsuitate-bot](https://github.com/tempakyousuke/tsuitate-bot) の局面valueネット学習パイプライン
（フェーズ1: 粒子上のvalueネット）。設計の背景は
`tsuitate-bot/docs/nn-value-phase1.md` を参照。

## リポジトリ境界

- **このリポジトリ**: Python/PyTorch の学習パイプライン。tsuitate-botが
  書き出した特徴量CSVを読み、ONNXモデルを書き出す。tsuitate-botには
  依存しない（データファイル経由の疎結合）
- **tsuitate-bot**: 特徴量抽出定義（`src/value_features.rs`）・学習データ
  書き出し（`bin/export_value_data`）・（将来の）推論統合。**特徴量の定義は
  あちらに一本化されている**ので、このリポジトリ側で特徴量を再定義しない
  こと（`model.py::INPUT_DIM`と`train.py`の次元チェックだけでズレを検出する）

## セットアップ

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

PyTorchの対応バージョンの都合上、**Python 3.12を使うこと**（3.14等の最新版は
まだホイールが無いことがある）。

## データ

`tsuitate-bot`側で自己対局データと学習データを生成・書き出す（相手を
estimator単体だけでなく凍結版とも混ぜて多様化するのが今の推奨）:

```
# tsuitate-bot 側で自己対局（ARENA_RECORD_DIRは常時有効）。相手を混ぜる例:
gh workflow run arena.yml --ref main -f games=1000 -f candidate=estimator \
  -f baselines="estimator estimator_v6 estimator_v7" -f shards=10
# 完了後、arena-records artifactをダウンロードして展開
cargo run --release --bin export_value_data -- <展開したjsonlすべて> > tsuitate-nn/data/value_data.csv
# pairwise補助loss用のデータも同じjsonl群から生成する（下記「学習」参照）
cargo run --release --bin export_pairwise_data -- <展開したjsonlすべて> > tsuitate-nn/data/pairwise.csv
```

## 学習

```
python train.py --data data/value_data.csv --pairwise-data data/pairwise.csv --out out/ \
  --pairwise-weight 20.0 --pairwise-margin 0.1
```

`--pairwise-data`は省略可（通常の勝敗回帰のみになる）が、**推奨は必須**:
最終勝敗ラベルだけでは「駒を危険にさらす手」と「攻めている・優勢な手」が
交絡し、局所的な交換損得の符号を学習できないcredit assignment問題がある
（詳細は`tsuitate-bot/docs/nn-value-phase1.md`）。`--pairwise-weight 20.0
--pairwise-margin 0.1`は既定の0.3/0.05よりかなり強めの値で、小さいグリッド
探索（weight={5,10,20,40}×margin={0.05,0.1,0.2}）の結果採用した値。

`out/model.onnx`（推論用）・`out/model.pt`（PyTorch state_dict）・
`out/norm_stats.json`（特徴量の正規化統計。**推論側は同じ正規化を適用する
こと**）を書き出す。

フェーズ1の特徴量は22次元（state16 + transition6）・データは3000局・
22万行程度で、**GPUは不要**（CPUで数秒〜数分）。将来もっと大きいモデル・
データセットで学習する場合は `scripts/gce/setup-train.sh` を使う
（tsuitate-botのSPSAチューニングVMと同じパターン:
tar転送→セットアップスクリプト→systemd常駐）。

学習は5seed以上で頑健性を必ず確認すること（`--seed`）。1本のseedだけの
「合格」は信用できない（データリークや過学習でなくても、val差が小さい
シナリオはseed次第で逆転しうる。過去に実際に誤判定した経緯あり）。

## オフライン検証

```
# tsuitate-bot 側で候補手ごとの特徴量を出す（.kifファイルを直接渡せる）
cargo run --release --bin eval_candidates -- scenarios/<名前>.kif <ply> <sente|gote> <USI手...>
python eval_scenario.py --candidates /tmp/candidates.csv --model out/
```

`eval_candidates`はstate特徴量を**着手前**（base、指す側視点）、
transition特徴量を**base→着手後**の遷移として出力する（学習データの
規約=常に「指す前・指す側視点」に合わせてある）。`eval_scenario.py`の
出力はそのまま「着手した側視点の勝率相当」で、値が高いほど良い
（反転不要）。

## 現状

フェーズ1（データ基盤＋オフライン学習）は2026-07-21時点で一区切り。
既知シナリオ2件（`scenarios/gold-check.kif`, `scenarios/kakudo.kif`）の
オフライン検証で5/5・4/5（5seed中）まで到達したが、厳密な成功条件
（両方5/5で安定）には一歩届かない。次の一手候補・経緯の全詳細は
`tsuitate-bot/docs/nn-value-phase1.md`参照。**bot本体への推論統合・
ONNX推論クレートの選定はまだ行っていない**。

## 信念ネット（NN段階②、2026-07-27）

`belief_model.py` / `train_belief.py` / `export_belief_weights.py` /
`analyze_belief.py` / `compare_probe.py`。

観測履歴だけからマスごとに「そこに相手の駒がいる確率」を予測する
pointwise MLP（35 → 64 → 32 → 占有ロジット + 駒種9クラスの補助ヘッド）。
特徴量の定義は tsuitate-bot の `src/belief_features.rs` に一本化してある。

```
# tsuitate-bot 側: 記録用アリーナ → artifact 回収 → 学習データ
cargo run --release --bin export_belief_data -- <jsonl...> > tsuitate-nn/data/belief.csv
# 学習（対局単位で train/val 分割・正規化統計も学習splitのみ）
python train_belief.py --data data/belief.csv --out out_belief_seed0/ --seed 0
# 現行の粒子フィルタとの突き合わせ（ゲート1）
cargo run --release --bin belief_probe -- <jsonl...> > probe.csv   # bot 側・重い
python analyze_belief.py --probe probe.csv --model out_belief_seed0/
# Rust へ重みを書き出す
python export_belief_weights.py --model-dir out_belief_seed0/ \
    --out ../tsuitate-bot/src/belief_nn.rs --note "..."
```

**結果**: オフライン較正は現行の粒子フィルタに決定的に勝つ
（アリーナ・ホールドアウトで対数損失 0.6227 → 0.3777）が、
**強さには変換できていない**（供給チャネルを2つ試して両方とも勝率中立）。
既定0のノブとして bot 側に入っている。全経緯は
`tsuitate-bot/docs/nn-stage2-belief-net.md`。
