#!/usr/bin/env bash
# Closed-loop VLA eval of the pi05 rj45_sbot_Y45P45_difix policy on held-out starts, at the
# EXACT y45p45 datagen scene (same flags as tools/render_batch_v4.sh, single config source).
#
# Held-out starts: eef-mode trajectories generated with a NEVER-TRAINED seed, e.g.
#   .venv/bin/python newton_cabling/scripted_controller/gen_trajectories.py \
#       --out /home/pandaliza/parallax/data/vla_train/eval_traj_y45p45 \
#       --envs 16 --rounds 4 --steps 160 --max-keep 160 --no-cut-at-success \
#       --connector-usd scan_rj45.usd --grasp-roll-180 \
#       --max-save 64 --seed 777
#   (NO --grip-from-head: the y45p45 TRAINING set used the default 68mm grip.)
#
# Prereqs (three processes share the one 5090 -- no training in parallel):
#   1. GS renderer restarted at this splat set:   bash tools/restart_gs_renderer.sh
#   2. openpi policy server (OPENPI venv):
#        cd ~/parallax/openpi && uv run scripts/serve_policy.py policy:checkpoint \
#            --policy.config=pi05_rj45_sbot_y45p45_lora \
#            --policy.dir=checkpoints/pi05_rj45_sbot_y45p45_lora/hf/3000
#   3. (optional, matches training obs) Difix server:
#        cd ~/parallax/Difix3D && .venv/bin/python difix_server.py --port 8021
#   4. This script, WITH SUDO (root-owned SHM), from the repo root:
#        sudo ALLOW_NONCANONICAL_EEF_RPY=1 DIFIX_SERVER=localhost:8021 \
#            bash tools/eval_batch_y45p45.sh 0 60 [HOST:PORT]
set -u
START=${1:-0}
END=${2:-60}
SERVER=${3:-localhost:8000}
REPO=/home/pandaliza/parallax/newton-cabling
TRAJROOT=${TRAJROOT:-/home/pandaliza/parallax/data/vla_train/eval_traj_y45p45}
OUTROOT=${OUTROOT:-/home/pandaliza/parallax/data/vla_train/evalrun_y45p45}
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim

# ── scene geometry: IDENTICAL to render_batch_v4.sh (the y45p45 training renders) ──
PLUG_HEAD=${PLUG_HEAD:-$REPO/newton_cabling/assets/ethernet/headA_plug_roll210.ply}
PLUG_TAIL=${PLUG_TAIL:-none}
EEF_RPY=${EEF_RPY:-"0 0 180"}
JACK_ANCHOR=${JACK_ANCHOR:-"0 0 0.018"}
GRIP_THETA=${GRIP_THETA:-"-0.0152"}
JACK_ALIGN_RPY=${JACK_ALIGN_RPY:-"-90 0 0"}
CONN_RPY=${CONN_RPY:-"-90 0 0"}
FRONT_EYE=${FRONT_EYE:-"0.53 -0.681 0.984"}
GRIPPER_DIR=/root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_v3
WRIST3_PLY=/root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link_realpalm.ply
source "$REPO/tools/datagen_v2_config.sh"

RENDER_FLAGS=""
[ -n "${RENDER_GAMMA:-}" ] && RENDER_FLAGS="$RENDER_FLAGS --gamma $RENDER_GAMMA"
[ -n "${RENDER_GAIN:-}" ]  && RENDER_FLAGS="$RENDER_FLAGS --gain $RENDER_GAIN"
FRONT_FLAGS=""
[ -n "${FRONT_EYE:-}" ]    && FRONT_FLAGS="$FRONT_FLAGS --eye $FRONT_EYE"
[ -n "${FRONT_TARGET:-}" ] && FRONT_FLAGS="$FRONT_FLAGS --target $FRONT_TARGET"
WRIST_USD_FLAG=""
[ "${WRIST_CAM_FROM_USD:-0}" = "1" ] && WRIST_USD_FLAG="--wrist-cam-from-usd"
DIFIX_FLAG=""
[ -n "${DIFIX_SERVER:-}" ] && DIFIX_FLAG="--difix-server $DIFIX_SERVER"

# the EEF_RPY guard escape hatch is required for the rigid env's "0 0 180" seat frame
export ALLOW_NONCANONICAL_EEF_RPY=${ALLOW_NONCANONICAL_EEF_RPY:-1}

cd "$REPO"
mkdir -p "$OUTROOT"
seated=0; ran=0
for i in $(seq "$START" $((END - 1))); do
    ep=$(printf "ep_%04d" "$i")
    traj=$TRAJROOT/$ep/eef_traj.npy
    out=$OUTROOT/$ep
    [ -f "$traj" ] || { echo "[eval] $ep: no trajectory, skipping"; continue; }
    rm -rf "$out"
    echo "[eval] $ep ..."
    .venv/bin/python scripts/record_sbot_scene_gs_cable.py \
        --eef-traj "$traj" --eef-rpy $EEF_RPY \
        --jack-pos $JACK_POS --jack-align-rpy $JACK_ALIGN_RPY --conn-rpy $CONN_RPY \
        --jack-anchor $JACK_ANCHOR \
        --grip-theta $GRIP_THETA \
        --connector-ply $PLUG_HEAD --connector-tail-ply $PLUG_TAIL \
        --base-pos $BASE_POS --base-yaw-deg $BASE_YAW_DEG \
        --arm-home-deg $ARM_HOME_DEG \
        --grasp-rpy $GRASP_RPY --grasp-protrude $GRASP_PROTRUDE \
        --gripper-gs-dir $GRIPPER_DIR \
        --wrist3-ply $WRIST3_PLY \
        --camera-config $REPO/configs/cameras.yaml \
        --table-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/table/splat_flat.ply \
        --bg-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/background/splat_open.ply \
        --wrist-cam --wrist-orbit $WRIST_ORBIT --wrist-side $WRIST_SIDE --wrist-up $WRIST_UP \
        --wrist-back $WRIST_BACK --wrist-aim-back $WRIST_AIM_BACK ${WRIST_USD_FLAG:-} --wrist-rigid \
        --dist-scale $DIST_SCALE --elev $ELEV --azim $AZIM $FRONT_FLAGS $RENDER_FLAGS \
        --grasped-only \
        --width ${WIDTH:-640} --height ${HEIGHT:-480} --dump-size ${DUMP_SIZE:-512} \
        --policy-server "$SERVER" --eval-max-steps ${EVAL_MAX_STEPS:-300} \
        --eval-grasped-start ${DIFIX_FLAG:-} \
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
