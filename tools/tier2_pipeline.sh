#!/usr/bin/env bash
# Tier-2 recovery campaign, Stages C-F: waits for Stage B to finish, then
#   gamma (per-episode {1.0,1.2,1.4,1.8}) -> difix -> depause (0.2mm arc
#   resample, same as the drmix_1000 surgery) -> LeRobot batches -> HF push.
# Difix runs strictly AFTER all rendering (never concurrent with shadow passes).
set -u
REPO=/home/pandaliza/parallax/newton-cabling
ROOT=/home/pandaliza/parallax/data/vla_train
SRC=$ROOT/tier2_recovery            # Stage-A (split) tree: chunk_XX/ep_XXXX + phase.npy
GS=$ROOT/tier2_recovery_gs
FINAL=$ROOT/tier2_recovery_final
DIFIX=$ROOT/tier2_recovery_final_difix
DEP=$ROOT/tier2_recovery_depaused
RENDER_LOG=/home/pandaliza/render_tier2.log
cd "$REPO"

echo "=== waiting for Stage B $(date +%H:%M) ==="
until grep -q "ALL CHUNKS RENDERED" "$RENDER_LOG" 2>/dev/null; do sleep 120; done

echo "=== gamma $(date +%H:%M) ==="
.venv/bin/python tools/gamma_draw_chunks.py --src "$GS" --dst "$FINAL" || exit 1

echo "=== difix $(date +%H:%M) ==="
pkill -f "difix_serve[r]" || true; sleep 3   # free the GPU for the batch pipeline
for cd in "$FINAL"/chunk_*; do
    c=$(basename "$cd")
    ( cd /home/pandaliza/parallax/Difix3D && .venv/bin/python difix_datagen.py \
        --src "$cd" --dst "$DIFIX/$c" --target 999 ) || { echo "difix FAILED on $c"; exit 1; }
done

echo "=== depause $(date +%H:%M) ==="
python3 tools/depause_dataset.py --apply \
    --difix-root "$DIFIX" --raw-root "$SRC" --out "$DEP" || exit 1

echo "=== merge + convert + push $(date +%H:%M) ==="
python3 - <<'EOF'
import glob, os
root = os.path.expanduser("~/parallax/data/vla_train/tier2_recovery_depaused")
eps = sorted(glob.glob(os.path.join(root, "chunk_*", "ep_*")))
n = len(eps)
per = (n + 2) // 3      # 3 batches, each <= 250 for any n <= 750
assert per <= 250, f"{n} eps -> batch {per} > 250"
for bi, name in enumerate(["merged_a", "merged_b", "merged_c"]):
    batch = eps[bi * per:(bi + 1) * per]
    os.makedirs(os.path.join(root, name), exist_ok=True)
    for k, p in enumerate(batch):
        dst = os.path.join(root, name, f"ep_{k:04d}")
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(p), dst)
    print(name, len(batch))
EOF
if pgrep -f "datagen_to_lerobot" | grep -qv $$; then
    echo "FATAL: a converter is already running"; exit 1
fi
cd ~/parallax/openpi
for B in a b c; do
    REPO_ID=Parallax-Worlds/cable_tier2rec_${B}_depaused
    rm -rf ~/.cache/huggingface/lerobot/$REPO_ID
    echo "=== batch $B -> $REPO_ID $(date +%H:%M) ==="
    uv run python "$REPO/tools/datagen_to_lerobot.py" \
        --raw "$DEP/merged_$B" \
        --repo_id "$REPO_ID" \
        --truncate_after_seat -1 --push_to_hub || { echo "convert $B FAILED"; exit 1; }
done
echo "=== TIER2 PIPELINE DONE + PUSHED $(date +%H:%M) ==="
