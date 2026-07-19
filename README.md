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

`tsuitate-bot`側でデータを生成・書き出す:

```
# tsuitate-bot 側で自己対局（ARENA_RECORD_DIRは常時有効）
gh workflow run arena.yml --ref main -f games=600 -f candidate=estimator -f baselines=estimator -f shards=12
# 完了後、arena-records artifactをダウンロードして展開
cargo run --release --bin export_value_data -- <展開したjsonlすべて> > tsuitate-nn/data/value_data.csv
```

## 学習

```
python train.py --data data/value_data.csv --out out/
```

`out/model.onnx`（推論用）・`out/model.pt`（PyTorch state_dict）・
`out/norm_stats.json`（特徴量の正規化統計。**推論側は同じ正規化を適用する
こと**）を書き出す。

フェーズ1の特徴量は14次元・データは数万行程度で、**GPUは不要**（CPUで
数秒〜数分）。将来もっと大きいモデル・データセットで学習する場合は
`scripts/gce/setup-train.sh` を使う（tsuitate-botのSPSAチューニングVMと
同じパターン: tar転送→セットアップスクリプト→systemd常駐）。

## オフライン検証

```
# tsuitate-bot 側で候補手ごとの特徴量を出す
cargo run --release --bin eval_candidates -- <moves.json> <ply> <sente|gote> <USI手...>
```

の出力を学習済みモデルに食わせて、どの候補を高く評価するか確認する
（評価スクリプトは今後 `eval_scenario.py` として追加予定）。

## 現状

フェーズ1（データ基盤＋オフライン学習）進行中。bot本体への推論統合・
ONNX推論クレートの選定はまだ行っていない。
