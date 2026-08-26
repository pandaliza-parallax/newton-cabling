#!/usr/bin/env bash
# Screening protocol: 12 conditions (6 train + 6 held-out), GPU-served ACT.
# Order: tier2 -> combined (flagship) -> dp01. Resume-safe (finished episodes skip).
set -u
cd /home/pandaliza/parallax/newton-cabling
export COND=/home/pandaliza/parallax/data/vla_train/evalmatrix_screen12.json
MODEL=act_c10_tier2 bash tools/eval_matrix_act.sh
MODEL=act_c10_combined bash tools/eval_matrix_act.sh
MODEL=act_c10_dp01 bash tools/eval_matrix_act.sh
echo "=== SCREENING QUEUE DONE ==="
