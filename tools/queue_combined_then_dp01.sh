#!/usr/bin/env bash
# Take over the eval queue: let the RUNNING tier2 matrix finish, then run the
# combined model (flagship: depaused base + tier2 recovery), then finish dp01.
# The old parent chain must be killed before this starts (leaves the tier2
# child matrix running untouched).
set -u
cd /home/pandaliza/parallax/newton-cabling
echo "=== waiting for the tier2 matrix to finish $(date +%H:%M) ==="
while pgrep -f "eval_matrix_ac[t]" > /dev/null; do sleep 60; done
MODEL=act_c10_combined bash tools/eval_matrix_act.sh
MODEL=act_c10_dp01 bash tools/eval_matrix_act.sh
echo "=== ALL NEW-CKPT MATRICES DONE ==="
