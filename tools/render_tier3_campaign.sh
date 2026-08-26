#!/usr/bin/env bash
# Tier-2 recovery campaign, Stage B driver: renders tier3_recovery chunks as the
# gen driver completes them. Identical visuals/DR to render_dr_campaign.sh
# (bg-yaw + shadow per chunk, jack jitter, combined scan splat, light shadows),
# but the per-chunk episode count is computed from the tree (kick chunks are
# split into a variable number of segments).
set -u
REPO=/home/pandaliza/parallax/newton-cabling
SRC=${SRC:-/home/pandaliza/parallax/data/vla_train/tier3_recovery}
DST=${DST:-/home/pandaliza/parallax/data/vla_train/tier3_recovery_gs}
CHUNKS=${CHUNKS:-10}
CAMPAIGN_SEED=${CAMPAIGN_SEED:-999}
cd "$REPO"

keeper() { docker exec -d parallax_sim_fp bash -lc 'for i in $(seq 1 14400); do \
  chmod 666 /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
            /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer* 2>/dev/null; sleep 2; done'; }

chunk_ready() {  # per-chunk marker (written after gen AND split) or campaign done
    local cd
    cd=$(printf "chunk_%02d" "$1")
    [ -f "$SRC/$cd/_CHUNK_DONE" ] && return 0
    [ -f "$SRC/_GEN_DONE" ] && return 0
    return 1
}

for c in $(seq 0 $((CHUNKS - 1))); do
    cd=$(printf "chunk_%02d" "$c")
    until chunk_ready "$c"; do sleep 60; done
    EPS=$(ls -d "$SRC/$cd"/ep_* 2>/dev/null | wc -l)
    [ "$EPS" -gt 0 ] || { echo "[tier2-render] $cd EMPTY, skipping"; continue; }
    read BGYAW SSTR <<< "$(.venv/bin/python -c "
import numpy as np
r = np.random.default_rng($CAMPAIGN_SEED + $c)
print(int(r.uniform(0, 360)), round(float(r.uniform(0.5, 0.8)), 2))")"
    echo "[tier2-render] === $cd: $EPS eps, bg-yaw $BGYAW, shadow $SSTR ==="
    keeper
    bash tools/restart_gs_renderer.sh || { echo "[tier2-render] $cd renderer restart FAILED"; exit 1; }
    TRAJROOT="$SRC/$cd" \
    OUTROOT="$DST/$cd" \
    EEF_RPY="0 0 180" ALLOW_NONCANONICAL_EEF_RPY=1 \
    JACK_ALIGN_RPY="-90 0 90" CONN_RPY="-90 0 180" \
    FRONT_EYE="0.53 -0.681 0.984" \
    BG_YAW_DEG="$BGYAW" \
    JACK_JITTER_M=0.07 JACK_JITTER_SEED=$((CAMPAIGN_SEED + c)) \
    SHADOWS=1 SHADOW_STRENGTH="$SSTR" SHADOW_ACCUM=8 SHADOW_MASK_SCALE=0.4 \
    NEWTON_REF=0 AUTO_RESTART_RENDERER=0 \
    PLUG_HEAD=/home/pandaliza/parallax/newton-cabling/newton_cabling/assets/ethernet/headA_plug_cadframe_rigid50.ply \
    PLUG_TAIL=none \
    JACK_PLY=/home/pandaliza/parallax/newton-cabling/newton_cabling/assets/ethernet/jack_fixture_0813_01_registered.ply \
    bash tools/render_batch_v4.sh 0 "$EPS"
done
echo "[tier2-render] ALL CHUNKS RENDERED"
