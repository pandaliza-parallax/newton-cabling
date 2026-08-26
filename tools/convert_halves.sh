#!/usr/bin/env bash
# Convert + push the drmix 500-episode set as two sequential 250-episode halves.
# Sequential BY CONSTRUCTION (one script, one process at a time) with a guard against
# concurrent copies — two converters at ~24GB each OOM'd the 62GB box (2026-08-18).
set -u
if pgrep -f "datagen_to_lerobot" | grep -qv $$; then
    echo "FATAL: a datagen_to_lerobot process is already running — refusing to start a second."
    exit 1
fi
set -e
rm -rf ~/.cache/huggingface/lerobot/Parallax-Worlds/cable_drmix_500_a \
       ~/.cache/huggingface/lerobot/Parallax-Worlds/cable_drmix_500_b
cd ~/parallax/openpi

echo "=== HALF A (episodes 0-249) $(date +%H:%M) ==="
uv run python ~/parallax/newton-cabling/tools/datagen_to_lerobot.py \
    --raw /home/pandaliza/parallax/data/vla_train/drmix_1000_final_difix/merged500 \
    --repo_id Parallax-Worlds/cable_drmix_500_a \
    --limit 250 --truncate_after_seat -1 --push_to_hub

echo "=== HALF B (episodes 250-499) $(date +%H:%M) ==="
uv run python ~/parallax/newton-cabling/tools/datagen_to_lerobot.py \
    --raw /home/pandaliza/parallax/data/vla_train/drmix_1000_final_difix/merged500_b \
    --repo_id Parallax-Worlds/cable_drmix_500_b \
    --truncate_after_seat -1 --push_to_hub

echo "=== BOTH HALVES DONE + PUSHED $(date +%H:%M) ==="
