#!/usr/bin/env bash
# BAKED-GRIP variant of render_batch_v2.sh — identical pipeline, but renders with
# record_sbot_scene_gs_bakedgrip.py: the gripper is baked into the original wrist_3 splat
# (flat/wrist_3_link.ply), no separate articulating finger splats, no jaw open/close. Sources the
# SAME tools/datagen_v2_config.sh (poses, cameras, gamma, resolution) so it's directly comparable.
#
# Output -> datagen_bakedgrip/ep_XXXX/ (image/, wrist_image/, state/action/phase.npy, meta.json).
# Episodes with a complete dump are SKIPPED (resume-safe).
#
# Run WITH SUDO (SHM is root-owned), renderer up:
#     sudo bash tools/render_batch_bakedgrip.sh 0 1     # smoke test ep_0000
#     sudo bash tools/render_batch_bakedgrip.sh 0 500   # full set
set -u
START=${1:-0}
END=${2:-500}
REPO=/home/pandaliza/parallax/newton-cabling
SCRIPT=record_sbot_scene_gs_bakedgrip.py
OUTROOT=$REPO/datagen_bakedgrip
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim

# LOCKED robot-pose + camera/render config — see tools/datagen_v2_config.sh (shared with v2).
source "$REPO/tools/datagen_v2_config.sh"
INSERT_ROT_FLAG=""
[ "${INSERT_HOLD_HOME_ROT:-0}" = "1" ] && INSERT_ROT_FLAG="--insert-hold-home-rot"
RENDER_FLAGS=""
[ -n "${RENDER_GAMMA:-}" ] && RENDER_FLAGS="$RENDER_FLAGS --gamma $RENDER_GAMMA"
[ -n "${RENDER_GAIN:-}" ]  && RENDER_FLAGS="$RENDER_FLAGS --gain $RENDER_GAIN"
FRONT_FLAGS=""
[ -n "${FRONT_EYE:-}" ]        && FRONT_FLAGS="$FRONT_FLAGS --eye $FRONT_EYE"
[ -n "${FRONT_TARGET:-}" ]     && FRONT_FLAGS="$FRONT_FLAGS --target $FRONT_TARGET"
[ -n "${FRONT_EYE_OFF:-}" ]    && FRONT_FLAGS="$FRONT_FLAGS --eye-offset $FRONT_EYE_OFF"
[ -n "${FRONT_TARGET_OFF:-}" ] && FRONT_FLAGS="$FRONT_FLAGS --target-offset $FRONT_TARGET_OFF"
SIDE_FLAGS=""
if [ "${SIDE_CAM:-0}" = "1" ]; then
    SIDE_FLAGS="--side-cam"
    [ -n "${SIDE_ELEV:-}" ]     && SIDE_FLAGS="$SIDE_FLAGS --side-elev $SIDE_ELEV"
    [ -n "${SIDE_Z:-}" ]        && SIDE_FLAGS="$SIDE_FLAGS --side-z $SIDE_Z"
    [ -n "${SIDE_TARGET_Z:-}" ] && SIDE_FLAGS="$SIDE_FLAGS --side-target-z $SIDE_TARGET_Z"
fi
[ "${MIRROR_CAM:-0}" = "1" ] && SIDE_FLAGS="$SIDE_FLAGS --mirror-cam"
WRIST_USD_FLAG=""
[ "${WRIST_CAM_FROM_USD:-0}" = "1" ] && WRIST_USD_FLAG="--wrist-cam-from-usd"

cd "$REPO"
mkdir -p "$OUTROOT"
done_n=0; skip_n=0; fail_n=0; consec_fail=0
for i in $(seq "$START" $((END - 1))); do
    ep=$(printf "ep_%04d" "$i")
    traj=seated_traj/$ep/plug_traj.npy
    out=$OUTROOT/$ep
    [ -f "$traj" ] || { echo "[batch] $ep: no trajectory, skipping"; continue; }
    if [ -f "$out/state.npy" ] && grep -q "\[dump\] FULL episode" "$out.log" 2>/dev/null; then
        skip_n=$((skip_n + 1)); continue
    fi
    echo "[batch] rendering $ep -> $out"
    .venv/bin/python "$SCRIPT" \
        --plug-traj "$traj" \
        --jack-pos $JACK_POS \
        --base-pos $BASE_POS --base-yaw-deg $BASE_YAW_DEG \
        --arm-home-deg $ARM_HOME_DEG \
        --grasp-rpy $GRASP_RPY --grasp-protrude $GRASP_PROTRUDE --grasp-along-cord $GRASP_ALONG_CORD \
        --table-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/table/splat_flat.ply \
        --bg-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/background/splat_open.ply \
        --wrist-cam --wrist-orbit $WRIST_ORBIT --wrist-side $WRIST_SIDE --wrist-up $WRIST_UP --wrist-back $WRIST_BACK --wrist-aim-back $WRIST_AIM_BACK ${WRIST_USD_FLAG:-} --wrist-rigid \
        --dist-scale $DIST_SCALE --elev $ELEV --azim $AZIM $FRONT_FLAGS $SIDE_FLAGS $RENDER_FLAGS \
        --grasped-only --trim-settle-mm $TRIM_SETTLE_MM --trim-settle-deg $TRIM_SETTLE_DEG --start-min-outside-mm ${START_MIN_OUTSIDE_MM:-0} $INSERT_ROT_FLAG \
        --width ${WIDTH:-852} --height ${HEIGHT:-640} \
        --dump "$out" --dump-size $DUMP_SIZE --no-preview --stop-after-seat $STOP_AFTER_SEAT \
        --out "$out/preview" > "$out.log" 2>&1
    if [ -f "$out/state.npy" ] && grep -q "\[dump\] FULL episode" "$out.log" 2>/dev/null; then
        done_n=$((done_n + 1)); consec_fail=0
        grep -h "\[profile\]" "$out.log" | sed "s/^/[batch] $ep /"
    else
        fail_n=$((fail_n + 1)); consec_fail=$((consec_fail + 1))
        echo "[batch] $ep FAILED (see $out.log)"
        if [ "$consec_fail" -ge 3 ]; then
            echo "[batch] ABORT: 3 consecutive failures (dead renderer / broken script?) — fix and rerun to resume"
            break
        fi
    fi
done
echo "[batch] finished: rendered $done_n, skipped $skip_n (already done), failed $fail_n"
echo "[batch] convert -> LeRobot (run in openpi venv):"
echo "  cd ~/parallax/openpi && uv run python ~/parallax/newton-cabling/tools/datagen_to_lerobot.py \\"
echo "      --raw ~/parallax/newton-cabling/datagen_bakedgrip --repo_id parallax/rj45_sbot_bakedgrip"
