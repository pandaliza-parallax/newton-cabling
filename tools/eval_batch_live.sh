#!/usr/bin/env bash
# LIVE-PHYSICS closed-loop VLA eval: the same policy and the same scene as
# tools/eval_batch_head55cad.sh, but the plug is a free VBD body held by finger friction against
# a real socket (--live-env) instead of a rigid latch on a kinematic replay.
#
# Held-out starts come from the env's own reset (curriculum stage + seed), not from a recorded
# trajectory -- so there is no eval_traj set to generate; SEED is the episode index.
#
# Prereqs are identical to eval_batch_head55cad.sh (GS renderer restarted at this splat set,
# policy server on :8000 under XLA_PYTHON_CLIENT_MEM_FRACTION=0.45, Difix on :8021), e.g.
#     DIFIX_SERVER=localhost:8021 bash tools/eval_batch_live.sh 0 20
set -u
START=${1:-0}
END=${2:-20}
SERVER=${3:-localhost:8000}
REPO=/home/pandaliza/parallax/newton-cabling
DATA=/home/pandaliza/parallax/data/vla_train
ETH=$REPO/newton_cabling/assets/ethernet
OUTROOT=${OUTROOT:-$DATA/evalrun_live_physics}
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim
export ALLOW_NONCANONICAL_EEF_RPY=${ALLOW_NONCANONICAL_EEF_RPY:-1}

LIVE_STAGE=${LIVE_STAGE:-4}
LIVE_TILT=${LIVE_TILT:-5}
GRIP_FROM_HEAD=${GRIP_FROM_HEAD:-55}
PLUG_HEAD=${PLUG_HEAD:-$ETH/headA_plug_cadframe_rigid.ply}
JACK_PLY=${JACK_PLY:-$ETH/jack_v2_registered.ply}
source "$REPO/tools/datagen_v2_config.sh"
DIFIX_FLAG=""
[ -n "${DIFIX_SERVER:-}" ] && DIFIX_FLAG="--difix-server $DIFIX_SERVER"

cd "$REPO"
mkdir -p "$OUTROOT"
seated=0; ran=0
for i in $(seq "$START" $((END - 1))); do
    ep=$(printf "ep_%04d" "$i")
    out=$OUTROOT/$ep
    rm -rf "$out"; mkdir -p "$out"
    echo "[live-eval] $ep (seed $i) ..."
    .venv/bin/python scripts/record_sbot_scene_gs_cable.py \
        --live-env --live-stage $LIVE_STAGE --live-tilt $LIVE_TILT --live-seed "$i" \
        --live-grip-from-head $GRIP_FROM_HEAD \
        --eef-rpy 0 0 180 \
        --jack-pos $JACK_POS --jack-align-rpy -90 0 0 --conn-rpy -90 0 0 \
        --jack-anchor 0 0 0.018 --jack-ply "$JACK_PLY" --grip-theta -0.0152 \
        --connector-ply "$PLUG_HEAD" --connector-tail-ply none \
        --base-pos $BASE_POS --base-yaw-deg $BASE_YAW_DEG --arm-home-deg $ARM_HOME_DEG \
        --grasp-rpy $GRASP_RPY --grasp-protrude $GRASP_PROTRUDE \
        --gripper-gs-dir /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_v3 \
        --wrist3-ply /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link_realpalm.ply \
        --camera-config $REPO/configs/cameras.yaml \
        --table-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/table/splat_flat.ply \
        --bg-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/background/splat_open.ply \
        --wrist-cam --wrist-orbit $WRIST_ORBIT --wrist-side $WRIST_SIDE --wrist-up $WRIST_UP \
        --wrist-back $WRIST_BACK --wrist-aim-back $WRIST_AIM_BACK --wrist-cam-from-usd --wrist-rigid \
        --dist-scale $DIST_SCALE --elev $ELEV --azim $AZIM --eye 0.53 -0.681 0.984 --target $FRONT_TARGET \
        --gamma 1.8 --grasped-only --width ${WIDTH:-640} --height ${HEIGHT:-480} --dump-size ${DUMP_SIZE:-512} \
        --policy-server "$SERVER" --eval-max-steps ${EVAL_MAX_STEPS:-200} ${DIFIX_FLAG:-} \
        --out "$out" > "$out.log" 2>&1
    ran=$((ran + 1))
    if grep -q '"seated": true' "$out/result.json" 2>/dev/null; then
        seated=$((seated + 1)); echo "[live-eval] $ep SEATED"
    else
        echo "[live-eval] $ep failed ($(grep final_y_sock "$out/result.json" 2>/dev/null | tr -d ' ' || echo 'no result'))"
    fi
done
echo "[live-eval] PHYSICS SEAT RATE: $seated/$ran"
