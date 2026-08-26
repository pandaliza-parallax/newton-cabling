#!/usr/bin/env bash
# Convert + push the DE-PAUSED drmix_1000 as four sequential 250-episode quarters.
# Same guard + sequential design as convert_halves.sh (OOM ceiling: never >250/process).
# Source tree: drmix_1000_depaused_f01 (arc-length resampled, actions recomputed —
# see tools/depause_dataset.py).
set -u
if pgrep -f "datagen_to_lerobot" | grep -qv $$; then
    echo "FATAL: a converter is already running — refusing to start."
    exit 1
fi
set -e
SRC=/home/pandaliza/parallax/data/vla_train/drmix_1000_depaused_f01
cd ~/parallax/openpi

for Q in ${QUARTERS:-a b c d}; do
    case $Q in
        a|b) REPO=Parallax-Worlds/cable_drmix_500_${Q}_dp01 ;;
        *)   REPO=Parallax-Worlds/cable_drmix_1000_${Q}_dp01 ;;
    esac
    rm -rf ~/.cache/huggingface/lerobot/$REPO
    echo "=== QUARTER $Q -> $REPO $(date +%H:%M) ==="
    uv run python ~/parallax/newton-cabling/tools/datagen_to_lerobot.py \
        --raw "$SRC/merged_$Q" \
        --repo_id "$REPO" \
        --truncate_after_seat -1 --push_to_hub
done
echo "=== ALL 4 DP01 QUARTERS DONE + PUSHED $(date +%H:%M) ==="
