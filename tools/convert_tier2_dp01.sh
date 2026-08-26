#!/usr/bin/env bash
# Convert + push the tier-2 recovery set re-depaused at the 0.1mm floor (dp01),
# for the consistent dp01+tier2 ACT mix. Same guard + <=250/process pattern.
set -u
if pgrep -f "datagen_to_lerobot" | grep -qv $$; then
    echo "FATAL: a converter is already running — refusing to start."
    exit 1
fi
set -e
SRC=/home/pandaliza/parallax/data/vla_train/tier2_recovery_dp01
cd ~/parallax/openpi
for B in a b c; do
    REPO_ID=Parallax-Worlds/cable_tier2rec_${B}_dp01
    rm -rf ~/.cache/huggingface/lerobot/$REPO_ID
    echo "=== batch $B -> $REPO_ID $(date +%H:%M) ==="
    uv run python ~/parallax/newton-cabling/tools/datagen_to_lerobot.py \
        --raw "$SRC/merged_$B" \
        --repo_id "$REPO_ID" \
        --truncate_after_seat -1 --push_to_hub
done
echo "=== TIER2 DP01 DONE + PUSHED $(date +%H:%M) ==="
