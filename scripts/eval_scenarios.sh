#!/usr/bin/env bash
# valueネットのオフライン検証を4シナリオまとめて走らせる。
#
# 使い方: scripts/eval_scenarios.sh <モデルディレクトリ> [tsuitate-botのパス]
#   例: scripts/eval_scenarios.sh out_approach_seed0
#
# 各シナリオは「好手 > 悪手」になっていれば正解。判定は目視でできるよう
# 期待順序を各ブロックの見出しに出す（eval_scenario.py はスコア降順で並べる）。
#
# 固定回帰テスト兼アンカー:
#   gold-check … 33手目 5八4八金（悪手・利きが確定している敵駒の前へ無防備に出る）
#   kakudo     … 22手目 R*2d（悪手・角道の誤信で飛車を打つ）
# 接近特徴量（2026-07-25）で追加した観測対象:
#   tokin-bet  … 16手目 P*8h（好手・安い歩のプローブ）vs 8g8h（と金の大駒捕獲賭け）
#   watch-…    … 34手目 P*8h（悪手・脅威にならない歩打ち）vs 9h8h（と金を寄せる）
set -euo pipefail

MODEL="${1:?usage: eval_scenarios.sh <model-dir> [bot-path]}"
BOT="${2:-$HOME/Develop/tsuitate-bot}"
# 変数展開の直後に日本語が続く箇所は必ず ${...} で括る（bash が全角文字を
# 変数名の一部として読み、unbound variable になる環境がある）
PYTHON="${PYTHON:-$(dirname "$0")/../.venv/bin/python}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

run() {
  local name="$1" kif="$2" ply="$3" side="$4" expect="$5"
  shift 5
  echo "=== ${name}（期待: ${expect}）==="
  # cargo のビルド警告は検証結果を埋もれさせるので捨てる
  (cd "$BOT" && cargo run --release --quiet --bin eval_candidates -- \
    "scenarios/${kif}" "${ply}" "${side}" "$@" 2>/dev/null) > "${TMP}/${name}.csv"
  "$PYTHON" eval_scenario.py --candidates "${TMP}/${name}.csv" --model "${MODEL}"
  echo
}

run gold-check gold-check.kif 32 sente "5g5f > 5i4h" 5i4h 5g5f
run kakudo     kakudo.kif     21 gote  "P*2h > R*2d" 'R*2d' 'P*2h'
run tokin-bet  tokin-bet.kif  15 gote  "P*8h > 8g8h" 'P*8h' 8g8h
run watch      watch-estimator-vs-estimator-20260725-203544.kif 33 gote \
  "9h8h / 7i6i > P*8h" 'P*8h' 9h8h 7i6i
