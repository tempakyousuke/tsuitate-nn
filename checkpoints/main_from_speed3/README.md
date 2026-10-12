# main_from_speed3 の 30,000更新（= rl_v29branch_v1）

GCE で rl_v29branch_v1 だけに勝つ搾取用の専門モデルを作るための学習状態。`out_rnad/` は git 管理外なので、ここに置いて `git add -f` した。

- `checkpoint_30000.pt`（sha256 `520e94c134fb971a8e2e828df3ef0985a3cb36d5563139a428ed3ef206780065`）: 学習状態一式
  （ネット・ターゲットネット・正則化ネット・Adam・step=30000・RNaD の設定）。`--out <dir>` の `<dir>/checkpoint.pt` として
  コピーすれば、そこから再開する（`train_rnad.py` は `<out>/checkpoint.pt` があれば再開し、ネットの形も checkpoint から取る）。
  新しい out で初期値にだけ使うなら `--init`（ターゲットネットだけを読む）
- `policy_30000.bin`（sha256 `e723100c496d56e3e740166c2830a9567e076f40a792e72dded5b6b228a39558`）: 推論用に書き出した重み。
  tsuitate-bot の `models/rl_v29branch_v1.bin` と同一
- `run_main_v29.sh`: 20,000 → 30,000 を回したコマンド（リーグ22体、錨 rl_v29）
- `run_main_v30.sh`: ゲーム用 PC で 30,000 → 35,000 を回しているコマンド（リーグに rl_v29branch_v1 を加えた23体、錨 rl_v29branch_v1）

経緯: rl_speed_v3（速攻特化 11,000更新）のターゲットネットを `--init` で読み、本線の報酬（ボーナスなし）で 0 から回した系列。
自己対局半分＋凍結版リーグ（hard 配分、GPU）。10,000 が rl_v27、15,000 が rl_v28、20,000 が rl_v29。
25,000・30,000 は rl_v29 に勝つが外の相手へのペア差が下がった（−2.1 / −4.0pt）ので本線にせず、30,000 を枝の系列
`rl_v29branch_v1` として凍結した（tsuitate-bot PR #83）。レーティング表では rl_v29 1684 / rl_v29branch_v1 1682 でほぼ同等。

注意: 再開時に RNaD の設定（`--lr` / `--eta` / `--beta` / `--schedule` など）が checkpoint と違うと止まる
（意図した変更なら `--override-config`）。リーグや相手に rl_v29branch_v1 を使うには tsuitate-bot の main（cac7905 以降）で
`tsuitate_rl` をビルドし直すこと。
