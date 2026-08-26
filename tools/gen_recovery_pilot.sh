#!/usr/bin/env bash
# Perturbation-recovery pilot (tier-2 campaign prototype), 2 x 25 episodes:
#   chunk_R  near-dock off-nominal starts: approach 15-22mm (pre-dock sits at
#            16-19mm) with the stage-4 8mm lateral disk -> every frame is a
#            corrective label near the diagnosed failure zone. No new code paths.
#   chunk_K  mid-flight kicks: ALIGN-phase lateral kicks (2-6mm over 4-8 frames),
#            marked in kick.npy; split with tools/split_kick_episodes.py before
#            Stage B so only the recovery (never the kick) becomes labels.
# Same servo plant + fixture + campaign COMMON DR as drmix_1000.
set -eu
REPO=/home/pandaliza/parallax/newton-cabling
OUT=/home/pandaliza/parallax/data/vla_train/recovery_pilot
GEN="$REPO/.venv/bin/python -m newton_cabling.scripted_controller.gen_trajectories"
COMMON="--servo-plant --jack-fixture --connector-usd cad_rj45.usd --grasp-roll-180
        --grasp-roll-deg 150 --jack-yaw 5 --cable-tilt -5 8 --grasp-roll-jitter 3
        --offset-z 3 --offset-uniform-disk --stage 4
        --envs 16 --rounds 5 --grip-from-head 50 --standoff-mm 5
        --push-correction 0.3 --action-noise 0.0 --max-save 25 --keep-existing"
cd "$REPO"
mkdir -p "$OUT"

echo "=== chunk_R: near-dock off-nominal starts $(date +%H:%M) ==="
$GEN $COMMON --out "$OUT/chunk_R" --seed 5000 \
    --approach-jitter 15 22 --steps 450 --max-keep 450

echo "=== chunk_K: ALIGN-phase kicks $(date +%H:%M) ==="
$GEN $COMMON --out "$OUT/chunk_K" --seed 5100 \
    --approach-jitter 25 35 --steps 600 --max-keep 600 \
    --kick-prob 0.02 --kick-max 2 --kick-mag-mm 2 6 --kick-frames 4 8

echo "=== splitting chunk_K at kick windows ==="
"$REPO/.venv/bin/python" "$REPO/tools/split_kick_episodes.py" \
    --in "$OUT/chunk_K" --out "$OUT/chunk_K_split" --min-frames 25
echo "=== RECOVERY PILOT DONE $(date +%H:%M) ==="
