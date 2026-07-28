#!/usr/bin/env bash
# datagen v2: batch-render seated PPO rollouts into openpi-format training data (FRONT + WRIST).
# DIFFERENCE FROM v1 (tools/render_batch.sh): each episode is the ALREADY-GRASPED INSERTION ONLY,
# and the plug trajectory's physics RESET TRANSIENT is trimmed off the start. This removes the
# grasp-handoff "jerk" (v1's ~13mm/12° one-step action spike, which was the source PPO trajectory's
# frame0->1 reset settle replayed as a rigid wrist motion). The IK rotation-target order is also
# corrected (wxyz->xyzw) in scripts/record_sbot_scene_gs.py. Everything else matches v1 exactly.
#
# Per episode -> datagen_v2/ep_XXXX/:
#   image/frame_*.png        FRONT cam, 224x224   (VARIABLE length: grasped insertion only)
#   wrist_image/frame_*.png  WRIST cam, 224x224
#   state.npy (T,10)         [eef_pos(3), eef_rot6d(6), gripper(1)]  absolute, base frame
#   action.npy (T,7)         [dpos(3), drotvec(3), gripper(1)]       delta to next frame (NO handoff spike)
#   phase.npy (T,)           all 2 = insertion (no hold/approach recorded)
#   meta.json
# Episodes with a complete dump are SKIPPED -> resume-safe after interruption.
#
# The renderer must be UP and loaded for this scene (restart it once before the batch; the splat
# set is identical across episodes so no restarts inside the loop).
#
# Run WITH SUDO (the SHM segments are root-owned), from the repo root:
#     sudo bash tools/render_batch_v2.sh 0 20      # ep_0000 .. ep_0019 (subset test)
#     sudo bash tools/render_batch_v2.sh 0 500     # the full set
set -u
START=${1:-0}
END=${2:-500}
REPO=/home/pandaliza/parallax/newton-cabling
OUTROOT=$REPO/datagen_v2
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim

# LOCKED robot-pose config (arm home joints, base, grasp, jack) — see tools/datagen_v2_config.sh.
# These match the real RO1 pendant + old datagen and are passed EXPLICITLY below so no code default
# can silently override the hand's initial position.
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
    .venv/bin/python scripts/record_sbot_scene_gs.py \
        --plug-traj "$traj" \
        --jack-pos $JACK_POS \
        --base-pos $BASE_POS --base-yaw-deg $BASE_YAW_DEG \
        --arm-home-deg $ARM_HOME_DEG \
        --grasp-rpy $GRASP_RPY --grasp-protrude $GRASP_PROTRUDE --grasp-along-cord $GRASP_ALONG_CORD \
        --gripper-gs-dir /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_proj \
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
        grep -h "\[profile\]" "$out.log" | sed "s/^/[batch] $ep /"   # per-stage time breakdown
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
echo "[batch] convert v2 -> LeRobot (run in openpi venv):"
echo "  cd ~/parallax/openpi && uv run python ~/parallax/newton-cabling/tools/datagen_to_lerobot.py \\"
echo "      --raw ~/parallax/newton-cabling/datagen_v2 --repo_id parallax/rj45_sbot_v2"
