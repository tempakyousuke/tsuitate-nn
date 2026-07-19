#!/bin/bash
# GCE学習VMのセットアップ（冪等）。VM上で実行する。
# tsuitate-botの scripts/gce/setup-tune.sh と同じパターン
# （tar転送→依存インストール→systemd常駐）だが、学習は有限ジョブなので
# 常駐はログ監視用の一度きり実行にしている（SPSAのような無限再開はしない）。
#
# 前提: /tmp/tsuitate-nn.tar.gz にコードとデータ（data/value_data.csv）が
# 転送済み
#
# 使い方（VM上で）:
#   bash setup-train.sh <train.pyへの追加引数...>
# 例:
#   bash setup-train.sh --epochs 2000 --data data/value_data.csv --out out/
#
# 注意: フェーズ1時点の小さいモデル・データ規模ではGPU/GCEは不要
# （ローカルCPUで数秒〜数分）。これは将来もっと大きいモデル・データセットで
# 学習する段階向けの雛型
set -euo pipefail

TRAIN_ARGS=("$@")

sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv > /dev/null

cd "$HOME"
rm -rf tsuitate-nn
mkdir tsuitate-nn
tar xzf /tmp/tsuitate-nn.tar.gz -C tsuitate-nn
cd tsuitate-nn

python3.12 -m venv .venv
source .venv/bin/activate
pip install --quiet -e .

echo "SETUP_DONE"
python train.py "${TRAIN_ARGS[@]}"
echo "TRAIN_DONE"
