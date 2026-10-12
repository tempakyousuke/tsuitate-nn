#!/bin/bash
# 出し直した本線の続き（20,000 → 25,000、2026-10-12 ユーザー判断）。報酬はボーナスなし、自己対局半分＋凍結版22体（rl_v29 を追加）のリーグ（hard、GPU）。
# 錨は rl_v29（= checkpoint_20000.pt、評価だけ）。checkpoint.pt から再開する
#   bash run_main_v29.sh [目標の更新回数]
cd ~/Develop/tsuitate-nn
O=out_rnad/main_from_speed3
exec .venv/bin/python -u rnad/train_rnad.py --out $O --anchor $O/checkpoint_20000.pt --steps ${1:-25000} --batch 256   --eval-every 50 --eval-games 256 --export-every 250 --checkpoint-every 10   --schedule 200 --lr 0.0002 --eta 0.2 --beta 16.0 --precision tf32   --league rl_v15,rl_v16,rl_v17,rl_v18,rl_v19,rl_v20,rl_v21,rl_v22,rl_v23,rl_v24,rl_v25,rl_v26,rl_v27,rl_v28,rl_v29,rl_nyugyoku_v1,rl_nyugyoku_v2,rl_speed_v1,rl_speed_v2,rl_speed_v3,rl_speed_v4,rl_guard_v1   --league-frac 0.5 --league-device gpu >> $O/train.out 2>&1
