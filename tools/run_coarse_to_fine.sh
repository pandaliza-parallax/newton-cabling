#!/usr/bin/env bash
# Coarse-to-fine screening: every model runs the 6-condition first pass; models
# showing promise (a seat OR any bore entry: depth > 0 while lat < 4mm) graduate
# automatically to the 12-condition set. Full-20 finals stay a human decision.
# Resume-safe throughout (finished episodes skip).
set -u
cd /home/pandaliza/parallax/newton-cabling
MODELS="act_c10_combined act_c10_tier2 act_c10_dp01"

pkill -f "eval_matrix_ac[t]" || true; sleep 1
pkill -f "record_sbot_scene_gs_cable.py --live-en[v]" || true
pkill -f "serve_act_polic[y]" || true; sleep 3

for M in $MODELS; do
    COND=/home/pandaliza/parallax/data/vla_train/evalmatrix_screen6.json \
        MODEL=$M bash tools/eval_matrix_act.sh
done
echo "=== FIRST PASS (6) DONE ==="

GRADS=$(python3 - <<'EOF'
import glob, json, os
import numpy as np
conds = [c["name"] for c in json.load(open("/home/pandaliza/parallax/data/vla_train/evalmatrix_screen6.json"))]
for m in ["act_c10_combined", "act_c10_tier2", "act_c10_dp01"]:
    promising = False
    for n in conds:
        d = f"/home/pandaliza/parallax/data/vla_train/evalmatrix/{m}_{n}"
        try:
            r = json.load(open(os.path.join(d, "result.json")))
            if r["seated"]:
                promising = True; break
            tr = np.load(os.path.join(d, "trace.npz"))
            y, lat = tr["y_sock"] * 1000, tr["lat"] * 1000
            if len(y[lat < 4]) and y[lat < 4].max() > 0:
                promising = True; break
        except Exception:
            pass
    if promising:
        print(m)
EOF
)
echo "=== GRADUATES: ${GRADS:-none} ==="
for M in $GRADS; do
    COND=/home/pandaliza/parallax/data/vla_train/evalmatrix_screen12.json \
        MODEL=$M bash tools/eval_matrix_act.sh
done
echo "=== COARSE-TO-FINE DONE ==="
