#!/usr/bin/env bash
# Stage-B integrity gate for the tier-2 campaign. The 2026-08-20 disk-full window
# left ~35 failed/partial render episodes which resume logic would treat as done.
# This waits for the render driver, verifies every rendered episode's frame count
# against its Stage-A trajectory, deletes bad dumps, re-runs the render driver on
# the gaps, repeats until clean, then starts the C-F pipeline (gamma->difix->
# depause->convert->push).
set -u
REPO=/home/pandaliza/parallax/newton-cabling
SRC=/home/pandaliza/parallax/data/vla_train/tier2_recovery
GS=/home/pandaliza/parallax/data/vla_train/tier2_recovery_gs
cd "$REPO"

verify() {  # prints bad episode dirs (one per line), deletes them
    python3 - "$SRC" "$GS" <<'EOF'
import glob, os, shutil, sys
import numpy as np
src, gs = sys.argv[1], sys.argv[2]
bad = 0
for ep in sorted(glob.glob(os.path.join(src, "chunk_[0-9][0-9]", "ep_*"))):
    rel = os.path.relpath(ep, src)
    out = os.path.join(gs, rel)
    T = len(np.load(os.path.join(ep, "eef_traj.npy")))
    ok = os.path.isdir(out)
    if ok:
        for sub in ("image", "wrist_image"):
            d = os.path.join(out, sub)
            if not os.path.isdir(d) or len(os.listdir(d)) != T:
                ok = False
        for f in ("state.npy", "action.npy"):
            p = os.path.join(out, f)
            ok = ok and os.path.isfile(p) and len(np.load(p)) == T
    if os.path.isdir(out) and not ok:
        shutil.rmtree(out)
        print("BAD", rel)
        bad += 1
    elif not os.path.isdir(out):
        print("MISSING", rel)
        bad += 1
print(f"TOTAL_BAD {bad}")
EOF
}

for attempt in 1 2 3; do
    echo "=== waiting for render driver (attempt $attempt) $(date +%H:%M) ==="
    until grep -q "ALL CHUNKS RENDERED" /home/pandaliza/render_tier2.log 2>/dev/null; do sleep 120; done
    echo "=== verifying $(date +%H:%M) ==="
    OUT=$(verify)
    echo "$OUT" | tail -5
    N=$(echo "$OUT" | grep "^TOTAL_BAD" | awk '{print $2}')
    if [ "$N" = "0" ]; then
        echo "=== STAGE B VERIFIED CLEAN $(date +%H:%M) ==="
        setsid nohup bash "$REPO/tools/tier2_pipeline.sh" \
            > /home/pandaliza/tier2_pipeline.log 2>&1 < /dev/null &
        echo "=== pipeline launched ==="
        exit 0
    fi
    echo "=== $N bad episodes deleted; re-running render driver $(date +%H:%M) ==="
    grep -v "ALL CHUNKS RENDERED" /home/pandaliza/render_tier2.log > /home/pandaliza/render_tier2.log.tmp \
        && mv /home/pandaliza/render_tier2.log.tmp /home/pandaliza/render_tier2.log
    bash tools/render_tier2_campaign.sh >> /home/pandaliza/render_tier2.log 2>&1
done
echo "=== GAVE UP after 3 attempts — investigate ==="
exit 1
