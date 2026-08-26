#!/usr/bin/env bash
# Closed-loop eval of the drmix policy: live physics on the CALIBRATED SERVO PLANT with the
# jack FIXTURE, campaign visual geometry (combined scan splat, rigid50 plug, -90 0 90 align),
# and DR-matched eval conditions (tilt -5..8, jack yaw +-5, roll base 150, grip + gamma cycling).
# Held-out seeds: 90000 + episode index (training: 40000s; eval traj set: 70000s).
#
# Prereqs: GS renderer restarted at this splat set, policy server :8000
# (pi05_drmix1000_lora), difix_server :8021.
#     DIFIX_SERVER=localhost:8021 bash tools/eval_batch_drmix.sh 0 100
set -u
START=${1:-0}
END=${2:-100}
SERVER=${3:-localhost:8000}
REPO=/home/pandaliza/parallax/newton-cabling
ETH=$REPO/newton_cabling/assets/ethernet
OUTROOT=${OUTROOT:-/home/pandaliza/parallax/data/vla_train/evalrun_drmix}
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim
export ALLOW_NONCANONICAL_EEF_RPY=1

source "$REPO/tools/datagen_v2_config.sh"
DIFIX_FLAG=""
[ -n "${DIFIX_SERVER:-}" ] && DIFIX_FLAG="--difix-server $DIFIX_SERVER"

cd "$REPO"
mkdir -p "$OUTROOT"
seated=0; ran=0
for i in $(seq "$START" $((END - 1))); do
    ep=$(printf "ep_%04d" "$i")
    out=$OUTROOT/$ep
    if grep -q '"seated"' "$out/result.json" 2>/dev/null; then
        echo "[drmix-eval] $ep already done, skipping"
        grep -q '"seated": true' "$out/result.json" && seated=$((seated + 1)); ran=$((ran + 1))
        continue
    fi
    rm -rf "$out"; mkdir -p "$out"
    grip=$((45 + 5 * (i % 3)))                             # 45 / 50 / 55 cycling
    case $((i % 4)) in 0) gamma=1.0;; 1) gamma=1.2;; 2) gamma=1.4;; 3) gamma=1.8;; esac
    echo "[drmix-eval] $ep (seed $((90000 + i)), grip $grip, gamma $gamma) ..."
    .venv/bin/python scripts/record_sbot_scene_gs_cable.py \
        --live-env --live-stage 4 --live-tilt -5 8 --live-seed $((90000 + i)) \
        --live-grip-from-head "$grip" \
        --live-fixture --live-servo --live-jack-yaw 5 --live-grasp-roll-deg 150 \
        --eef-rpy 0 0 180 \
        --jack-pos $JACK_POS --jack-align-rpy -90 0 90 --conn-rpy -90 0 180 \
        --jack-anchor 0 0 0.018 --jack-ply "$ETH/jack_fixture_0813_01_registered.ply" \
        --grip-theta -0.0152 \
        --connector-ply "$ETH/headA_plug_cadframe_rigid50.ply" --connector-tail-ply none \
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
        --gamma "$gamma" --grasped-only --width 640 --height 480 --dump-size 512 \
        --shadows --shadow-strength ${SHADOW_STRENGTH:-0.65} --shadow-accum 8 --shadow-mask-scale 0.4 \
        --policy-server "$SERVER" --eval-max-steps ${EVAL_MAX_STEPS:-450} ${DIFIX_FLAG:-} \
        --out "$out" > "$out.log" 2>&1
    ran=$((ran + 1))
    if grep -q '"seated": true' "$out/result.json" 2>/dev/null; then
        seated=$((seated + 1)); echo "[drmix-eval] $ep SEATED"
    else
        echo "[drmix-eval] $ep failed"
    fi
done
echo "[drmix-eval] SEAT RATE: $seated/$ran"
