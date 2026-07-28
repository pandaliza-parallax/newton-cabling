#!/usr/bin/env bash
# datagen v3: batch-render seated PPO rollouts into openpi-format training data (FRONT + WRIST).
#
# DIFFERENCE FROM v2 (tools/render_batch_v2_gripv3.sh):
#   a) 45-DEGREE GRASP  -- the gripper holds the plug tilted via --grasp-tilt-rpy (default "0 45 0").
#      NOTE: --grasp-rpy does NOT do this; it cancels out of the IK (it defines the EEF bookkeeping
#      frame only). --grasp-tilt-rpy tilts the HELD orientation while the position objective keeps
#      the hand at the calibrated grasp point, so --base-pos stays valid.
#   b) NEW WRIST CAMERA -- read live from /sbot/wrist_3_link/Camera in standardbot.usd via
#      --wrist-cam-from-usd. Nothing to set here; re-save the USD and it is picked up.
#   Also: gripper_cut_v3 splats + wrist_3_link_realpalm + calibrated cameras.yaml (as v2 gripv3 had).
#
# OUTPUT DIR IS datagen_v3 (NOT datagen_v2). The v2/gripv2/gripv3 scripts all wrote to the SAME
# datagen_v2 dir, so the data never recorded which gripper/config produced it -- that ambiguity cost
# real debugging time. This script also stamps OUTROOT/_CONFIG.txt with the exact provenance.
#
# Per episode -> datagen_v3/ep_XXXX/:
#   image/frame_*.png        FRONT cam        wrist_image/frame_*.png  WRIST cam
#   state.npy (T,10)         [eef_pos(3), eef_rot6d(6), gripper(1)]  absolute, base frame
#   action.npy (T,7)         [dpos(3), drotvec(3), gripper(1)]       delta to next frame
#   phase.npy (T,)           all 2 = insertion            meta.json
# Episodes with a complete dump are SKIPPED -> resume-safe after interruption.
#
# The renderer must be UP. The splat set is IDENTICAL across episodes (no restarts inside the loop),
# but RESTART IT ONCE before the batch -- it caches the splat set from the first client and silently
# ignores a changed one (it only compares COUNT):
#   docker exec parallax_sim_fp bash -lc "pkill -9 -f '[d]alus_sim_app'; sleep 3; \
#     rm -f /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
#           /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer*"
#   docker exec -d parallax_sim_fp bash -lc \
#     'cd /root/parallax/DalusSimCore && python3 dalus_sim_app.py > /tmp/render.log 2>&1'
#
# Run WITH SUDO (SHM is root-owned), from the repo root:
#     sudo bash tools/render_batch_v3.sh 0 5        # smoke test -- EYEBALL IT before the full set
#     sudo bash tools/render_batch_v3.sh 0 500      # the full set
#     sudo GRASP_TILT="0 30 0" bash tools/render_batch_v3.sh 0 500   # different tilt
set -u
START=${1:-0}
END=${2:-500}
REPO=/home/pandaliza/parallax/newton-cabling
OUTROOT=${OUTROOT:-$REPO/datagen_v3}
GRASP_TILT=${GRASP_TILT:-"0 45 0"}          # (a) the 45-degree grasp
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim

GRIPPER_DIR=/root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_v3
WRIST3_PLY=/root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link_realpalm.ply
SBOT_USD=/home/pandaliza/parallax/robo_maker/sbot/assets/standardbot.usd

# LOCKED robot-pose config (arm home joints, base, grasp, jack, cameras) -- single source of truth.
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
# NO side/mirror cameras. The config sets SIDE_CAM=1/MIRROR_CAM=1, but --mirror-cam is NOT
# preview-only: dump_cams() writes a mirror_image/ folder per episode (~14MB/ep = ~7GB over 500),
# and datagen_to_lerobot.py never reads it. Both cams also cost render time every frame. Only
# FRONT (image/) and WRIST (wrist_image/) are used for training, so we render exactly those.
SIDE_FLAGS=""
WRIST_USD_FLAG=""
[ "${WRIST_CAM_FROM_USD:-0}" = "1" ] && WRIST_USD_FLAG="--wrist-cam-from-usd"

cd "$REPO"
mkdir -p "$OUTROOT"
# --out is makedirs'd unconditionally even with --no-preview, so point every episode at ONE
# throwaway dir instead of creating an empty preview/ inside each episode.
PREVIEW_TMP=$(mktemp -d /tmp/render_v3_preview.XXXXXX)
trap 'rm -rf "$PREVIEW_TMP"' EXIT

# ── provenance stamp: so this dataset always records WHAT produced it ──────────────────────────
{
  echo "datagen_v3 generated $(date -Is)"
  echo "script      : tools/render_batch_v3.sh   range [$START,$END)"
  echo "grasp tilt  : $GRASP_TILT   (--grasp-tilt-rpy)"
  echo "gripper gs  : $GRIPPER_DIR"
  echo "wrist3 ply  : $WRIST3_PLY"
  echo "sbot usd    : $SBOT_USD  (mtime $(stat -c %y "$SBOT_USD" 2>/dev/null | cut -d'.' -f1))"
  echo "wrist cam   : from USD (--wrist-cam-from-usd=${WRIST_CAM_FROM_USD:-0})"
  echo "cameras.yaml: configs/cameras.yaml (d415 front / d405 wrist)"
  echo "base/jack   : BASE_POS=$BASE_POS  JACK_POS=$JACK_POS  ARM_HOME=$ARM_HOME_DEG"
  echo "render      : ${WIDTH:-640}x${HEIGHT:-480}  dump ${DUMP_SIZE}  gamma ${RENDER_GAMMA:-none}"
} > "$OUTROOT/_CONFIG.txt"
cat "$OUTROOT/_CONFIG.txt"

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
        --grasp-tilt-rpy $GRASP_TILT \
        --gripper-gs-dir $GRIPPER_DIR \
        --wrist3-ply $WRIST3_PLY \
        --camera-config /home/pandaliza/parallax/newton-cabling/configs/cameras.yaml \
        --table-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/table/splat_flat.ply \
        --bg-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/background/splat_open.ply \
        --wrist-cam --wrist-orbit $WRIST_ORBIT --wrist-side $WRIST_SIDE --wrist-up $WRIST_UP --wrist-back $WRIST_BACK --wrist-aim-back $WRIST_AIM_BACK ${WRIST_USD_FLAG:-} --wrist-rigid \
        --dist-scale $DIST_SCALE --elev $ELEV --azim $AZIM $FRONT_FLAGS $SIDE_FLAGS $RENDER_FLAGS \
        --grasped-only --trim-settle-mm $TRIM_SETTLE_MM --trim-settle-deg $TRIM_SETTLE_DEG --start-min-outside-mm ${START_MIN_OUTSIDE_MM:-0} $INSERT_ROT_FLAG \
        --width ${WIDTH:-640} --height ${HEIGHT:-480} \
        --dump "$out" --dump-size $DUMP_SIZE --no-preview --stop-after-seat $STOP_AFTER_SEAT \
        --out "$PREVIEW_TMP" > "$out.log" 2>&1
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
echo ""
echo "[batch] next steps:"
echo "  1. Difix the renders:"
echo "     cd ~/parallax/Difix3D && .venv/bin/python difix_datagen.py \\"
echo "         --src $OUTROOT --dst ${OUTROOT}_difix --target 500"
echo "  2. convert -> LeRobot (openpi venv):"
echo "     cd ~/parallax/openpi && uv run python ~/parallax/newton-cabling/tools/datagen_to_lerobot.py \\"
echo "         --raw ${OUTROOT}_difix --repo_id parallax/rj45_sbot_v3"
