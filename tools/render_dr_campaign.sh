#!/usr/bin/env bash
# Stage B campaign driver: render drmix chunks as generation completes them.
# Per chunk: draw bg-yaw + shadow strength (deterministic in chunk index), restart the
# renderer (bg pose is SETUP-cached) + refresh the SHM chmod keeper, render resume-safe.
# Difix is deliberately NOT run here (GPU discipline: never concurrent with shadows).
set -u
REPO=/home/pandaliza/parallax/newton-cabling
SRC=${SRC:-/home/pandaliza/parallax/data/vla_train/drmix_1000}
DST=${DST:-/home/pandaliza/parallax/data/vla_train/drmix_1000_gs}
CHUNKS=${CHUNKS:-20}
EPS=${EPS:-50}
CAMPAIGN_SEED=${CAMPAIGN_SEED:-777}
cd "$REPO"

keeper() { docker exec -d parallax_sim_fp bash -lc 'for i in $(seq 1 14400); do \
  chmod 666 /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
            /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer* 2>/dev/null; sleep 2; done'; }

chunk_ready() {  # generation of chunk $1 is complete when the driver moved past it
    local c=$1 nxt
    nxt=$(printf "chunk_%02d" $((c + 1)))
    grep -q "\"$nxt\"" "$SRC/_CHUNKS.json" 2>/dev/null && return 0
    [ -f "$SRC/_GEN_DONE" ] && return 0
    return 1
}

for c in $(seq 0 $((CHUNKS - 1))); do
    cd=$(printf "chunk_%02d" "$c")
    until chunk_ready "$c"; do sleep 60; done
    # deterministic per-chunk draws
    read BGYAW SSTR <<< "$(.venv/bin/python -c "
import numpy as np
r = np.random.default_rng($CAMPAIGN_SEED + $c)
print(int(r.uniform(0, 360)), round(float(r.uniform(0.5, 0.8)), 2))")"
    echo "[campaign] === $cd: bg-yaw $BGYAW, shadow $SSTR ==="
    keeper
    bash tools/restart_gs_renderer.sh || { echo "[campaign] $cd renderer restart FAILED"; exit 1; }
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
echo "[campaign] ALL CHUNKS RENDERED"
