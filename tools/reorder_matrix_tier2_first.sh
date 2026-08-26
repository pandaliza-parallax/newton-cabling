#!/usr/bin/env bash
# Stop the running new-ckpt matrix chain and relaunch with tier2 FIRST, dp01 after.
# (Script file so the pkill patterns never appear in an interactive command line.)
set -u
pkill -f "eval_matrix_ac[t]" || true
sleep 1
pkill -f "record_sbot_scene_gs_cable.py --live-en[v]" || true
pkill -f "serve_act_polic[y]" || true
sleep 3
cd /home/pandaliza/parallax/newton-cabling
MODEL=act_c10_tier2 bash tools/eval_matrix_act.sh
MODEL=act_c10_dp01 bash tools/eval_matrix_act.sh
echo "=== BOTH NEW-CKPT MATRICES DONE ==="
