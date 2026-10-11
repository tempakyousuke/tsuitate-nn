# main_from_speed3 の 15,000更新（= rl_v28）

GCE で続きを学習する（指し継ぎの実験など）ための学習状態。`out_rnad/` は git 管理外なので、ここに置いて `git add -f` した。

- `checkpoint_15000.pt`（sha256 `492bdf14f1c141bd396552eb609864be1abdfe0949e65ad9997c99ca8e8f152f`）: 学習状態一式
  （ネット・ターゲットネット・正則化ネット・Adam・step=15000・RNaD の設定）。`--out <dir>` の `<dir>/checkpoint.pt` として
  コピーすれば、そこから再開する（`train_rnad.py` は `<out>/checkpoint.pt` があれば再開し、ネットの形も checkpoint から取る）
- `policy_15000.bin`（sha256 `c2a4b8e901f0fe842703145f714556ae86228ebf3663eef6db969edb0094f514`）: 推論用に書き出した重み。
  tsuitate-bot の `models/rl_v28.bin` と同一
- `run_main_v28.sh`: ゲーム用 PC で 15,000 → 20,000 を回しているコマンド（錨・リーグ・ハイパーパラメータの参照用）

経緯: rl_speed_v3（速攻特化 11,000更新）のターゲットネットを `--init` で読み、本線の報酬（ボーナスなし）で 0 から 15,000更新。
自己対局半分＋凍結版リーグ（hard 配分、GPU）。10,000更新が rl_v27、15,000更新が rl_v28（tsuitate-bot PR #79 / #80）。

注意: 再開時に RNaD の設定（`--lr` / `--eta` / `--beta` / `--schedule` など）が checkpoint と違うと止まる
（意図した変更なら `--override-config`）。リーグの凍結版（rl_v28 まで）を作るには tsuitate-bot の main（660e2e9 以降）で
`tsuitate_rl` をビルドし直すこと。
