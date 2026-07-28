#!/usr/bin/env bash
# Closed-loop VLA eval of the pi05 rj45_sbot_DIFIX policy on the held-out eval_traj_v2/ starts
# (seed 777, never trained). Uses the EXACT datagen_v2 scene + cameras (sources datagen_v2_config.sh)
# so the policy sees IN-DISTRIBUTION obs — this is the key difference from tools/eval_batch.sh, which
# used a different camera rig. Grasped-insertion start (--eval-grasped-start) matches v2 training.
#
# Prereqs (three processes on the one 5090 — pause any training, no VRAM for both):
#   1. parallax_sim renderer UP at the datagen_v2 splat config (19 splats incl. SIDE/MIRROR preview).
#   2. openpi policy server, in the OPENPI venv:
#        cd ~/parallax/openpi && uv run scripts/serve_policy.py policy:checkpoint \
#            --policy.config=pi05_rj45_sbot_v3_lora \
#            --policy.dir=checkpoints/pi05_rj45_sbot_v3/10000
#   3. This script, WITH SUDO (root-owned SHM), from the repo root:
#        sudo bash tools/eval_batch_v2.sh 0 60 [HOST:PORT]
set -u
START=${1:-0}
END=${2:-60}
SERVER=${3:-localhost:8000}
REPO=/home/pandaliza/parallax/newton-cabling
OUTROOT=${OUTROOT:-$REPO/evalrun_v3}   # override to keep runs side by side (each ep dir is rm -rf'd)
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim

# identical scene/robot/camera config as the training renders (single source of truth)
source "$REPO/tools/datagen_v2_config.sh"
RENDER_FLAGS=""
[ -n "${RENDER_GAMMA:-}" ] && RENDER_FLAGS="$RENDER_FLAGS --gamma $RENDER_GAMMA"
[ -n "${RENDER_GAIN:-}" ]  && RENDER_FLAGS="$RENDER_FLAGS --gain $RENDER_GAIN"
FRONT_FLAGS=""
[ -n "${FRONT_TARGET:-}" ] && FRONT_FLAGS="$FRONT_FLAGS --target $FRONT_TARGET"
WRIST_USD_FLAG=""
[ "${WRIST_CAM_FROM_USD:-0}" = "1" ] && WRIST_USD_FLAG="--wrist-cam-from-usd"
# OPTIONAL: Difix the obs before the policy sees it (matches the Difix'd frames it trained on).
#   start:  cd ~/parallax/Difix3D && .venv/bin/python difix_server.py --port 8021
#   use:    sudo DIFIX_SERVER=localhost:8021 bash tools/eval_batch_v2.sh 0 60
DIFIX_FLAG=""
[ -n "${DIFIX_SERVER:-}" ] && DIFIX_FLAG="--difix-server $DIFIX_SERVER"
# Trim the physics reset transient off the trajectory start, EXACTLY as datagen did -- otherwise the
# closed-loop starts on the raw frame-0 jerk pose (~3.5mm further out than training ever begins).
INSERT_ROT_FLAG=""
[ "${INSERT_HOLD_HOME_ROT:-0}" = "1" ] && INSERT_ROT_FLAG="--insert-hold-home-rot"

cd "$REPO"
mkdir -p "$OUTROOT"
seated=0; ran=0
for i in $(seq "$START" $((END - 1))); do
    ep=$(printf "ep_%04d" "$i")
    traj=eval_traj_v2/$ep/plug_traj.npy
    out=$OUTROOT/$ep
    [ -f "$traj" ] || { echo "[eval] $ep: no trajectory, skipping"; continue; }
    rm -rf "$out"
    echo "[eval] $ep ..."
    .venv/bin/python scripts/record_sbot_scene_gs.py \
        --plug-traj "$traj" \
        --jack-pos $JACK_POS \
        --base-pos $BASE_POS --base-yaw-deg $BASE_YAW_DEG \
        --arm-home-deg $ARM_HOME_DEG \
        --grasp-rpy $GRASP_RPY --grasp-protrude $GRASP_PROTRUDE --grasp-along-cord $GRASP_ALONG_CORD \
        --gripper-gs-dir /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_v3 \
        --wrist3-ply /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link_realpalm.ply \
        --table-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/table/splat_flat.ply \
        --bg-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/background/splat_open.ply \
        --wrist-cam --wrist-orbit $WRIST_ORBIT --wrist-side $WRIST_SIDE --wrist-up $WRIST_UP \
        --wrist-back $WRIST_BACK --wrist-aim-back $WRIST_AIM_BACK ${WRIST_USD_FLAG:-} --wrist-rigid \
        --dist-scale $DIST_SCALE --elev $ELEV --azim $AZIM $FRONT_FLAGS $RENDER_FLAGS \
        --camera-config configs/cameras.yaml \
        --trim-settle-mm $TRIM_SETTLE_MM --trim-settle-deg $TRIM_SETTLE_DEG \
        --start-min-outside-mm ${START_MIN_OUTSIDE_MM:-0} $INSERT_ROT_FLAG \
        --width ${WIDTH:-640} --height ${HEIGHT:-480} --dump-size $DUMP_SIZE \
        --policy-server "$SERVER" --eval-max-steps 300 --eval-grasped-start --grasp-tilt-rpy 0 45 0 ${DIFIX_FLAG:-} \
        --out "$out" > "$out.log" 2>&1
    ran=$((ran + 1))
    if grep -q '"seated": true' "$out/result.json" 2>/dev/null; then
        seated=$((seated + 1)); echo "[eval] $ep SEATED  ($(grep seat_step "$out/result.json"))"
    else
        echo "[eval] $ep failed  ($(grep final_y "$out/result.json" 2>/dev/null || echo 'no result'))"
    fi
done
echo "[eval] SEAT RATE: $seated/$ran"
echo "[eval] categorize failures:  .venv/bin/python tools/analyze_eval.py --root $OUTROOT"
