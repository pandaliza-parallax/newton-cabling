#!/usr/bin/env bash
# Convert + push the BACK 500 of drmix_1000 as two sequential 250-episode quarters.
# Same guard + sequential design as convert_halves.sh (OOM ceiling: never >250/process).
set -u
if pgrep -f "datagen_to_lerobot" | grep -qv $$; then
    echo "FATAL: a converter is already running — refusing to start."
    exit 1
fi
set -e
rm -rf ~/.cache/huggingface/lerobot/Parallax-Worlds/cable_drmix_1000_c \
       ~/.cache/huggingface/lerobot/Parallax-Worlds/cable_drmix_1000_d
cd ~/parallax/openpi

echo "=== QUARTER C (episodes 500-749) $(date +%H:%M) ==="
uv run python ~/parallax/newton-cabling/tools/datagen_to_lerobot.py \
    --raw /home/pandaliza/parallax/data/vla_train/drmix_1000_final_difix/merged1000_c \
    --repo_id Parallax-Worlds/cable_drmix_1000_c \
    --truncate_after_seat -1 --push_to_hub

echo "=== QUARTER D (episodes 750-999) $(date +%H:%M) ==="
uv run python ~/parallax/newton-cabling/tools/datagen_to_lerobot.py \
    --raw /home/pandaliza/parallax/data/vla_train/drmix_1000_final_difix/merged1000_d \
    --repo_id Parallax-Worlds/cable_drmix_1000_d \
    --truncate_after_seat -1 --push_to_hub

echo "=== QUARTERS C+D DONE + PUSHED $(date +%H:%M) ==="
