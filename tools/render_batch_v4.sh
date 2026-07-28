#!/usr/bin/env bash
# datagen v4: batch-render CABLE PPO rollouts into openpi-format training data (FRONT + WRIST).
#
# This is the cable track -- gripper pose (policy) -> cable physics -> plug pose -- the INVERSE of
# v3 (tools/render_batch_v3.sh), which replayed a rigid plug trajectory and INFERRED the gripper
# from it via the cord-axis heuristic. Here the wrist trajectory is RECORDED (rl/gen_cable_traj.py)
# and the connector follows its own Newton-simulated pose. Nothing about the gripper is inferred.
#
# INPUT  : cable_traj/ep_XXXX/{eef_traj,conn_traj}.npy  (from rl/gen_cable_traj.py)
#          (T,7) [pos3, quat4 wxyz], FULLY seat-relative -- see gen_cable_traj.py.
# RENDER : scripts/record_sbot_scene_gs_cable.py --eef-traj (fork of the v3 renderer; v3 untouched).
#          --eef-rpy 180 0 0 (the renderer default) undoes cable_env's seat frame (180deg-about-x
#          flipped from world) so the arm reaches every frame (IK err ~0), the wrist clears the
#          table, and the dangling connector sits below the hand as in cable_env. Verify the
#          [calib] line: reach ~0.7m, min wrist z > 0.785.
#
# Per episode -> datagen_v4/ep_XXXX/:
#   image/frame_*.png        FRONT cam        wrist_image/frame_*.png  WRIST cam
#   state.npy (T,10)         [eef_pos(3), eef_rot6d(6), gripper(1)]  absolute, base frame
#   action.npy (T,7)         [dpos(3), drotvec(3), gripper(1)]       delta to next frame
#   phase.npy (T,)           all 2 = insertion            meta.json
# Episodes with a complete dump are SKIPPED -> resume-safe after interruption.
#
# NOTE: episodes are already cut at success+hold by gen_cable_traj.py (no --stop-after-seat here),
# and there is no reset transient to trim (no --trim-settle / --grasp-along-cord: those are the
# plug-inference path). The EEF-offset convention (--grasp-rpy/--grasp-protrude) is kept identical
# to v3 so the same run_vla.py state/action convention applies to the cable dataset.
#
# The renderer must be UP. Restart it ONCE before the batch (it caches the splat set by COUNT):
#   docker exec parallax_sim_fp bash -lc "pkill -9 -f '[d]alus_sim_app'; sleep 3; \
#     rm -f /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
#           /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer*"
#   docker exec -d parallax_sim_fp bash -lc \
#     'cd /root/parallax/DalusSimCore && python3 dalus_sim_app.py > /tmp/render.log 2>&1'
#
# Run WITH SUDO (SHM is root-owned), from the repo root:
#     sudo bash tools/render_batch_v4.sh 0 4        # smoke test -- EYEBALL IT before the full set
#     sudo bash tools/render_batch_v4.sh 0 500      # the full set
set -u
START=${1:-0}
END=${2:-500}
REPO=/home/pandaliza/parallax/newton-cabling
OUTROOT=${OUTROOT:-$REPO/datagen_v4}
TRAJROOT=${TRAJROOT:-$REPO/cable_traj}       # rl/gen_cable_traj.py output
EEF_RPY=${EEF_RPY:-"180 0 0"}                # seat-frame -> scene calibration (renderer default)
# Jack SPLAT facing. The jack mouth is the +z face of its obj frame (jack_anchor 0 0 0.030); with
# the code's intrinsic-XYZ euler the mouth-normal = R_align.[0,0,1]. v3's 90 0 -180 -> -y (correct
# for v3's -y approach; proven by v3 working). The cable plug approaches from +y (arm side), so the
# mouth must face +y: -90 0 0 -> [0,1,0]. NOTE yaw only SPINS the jack about the mouth axis (all
# 90 0 y face -y) -- it does NOT change the facing; ROLL is the lever. Splat-only (not jqw/seat_R/IK).
# If the RJ45 SLOT looks rotated 90deg after this, add a roll about the mouth axis (tweak here).
JACK_ALIGN_RPY=${JACK_ALIGN_RPY:-"-90 0 0"}
# Plug SPLAT orientation = v3's -90 0 0 (the renderer default; do NOT "fix" it).
# The plug splat's INSERTION axis is local **+z**, not +y: head spans z 6.9..18.5, tail_longer
# z -12.2..8.0, and conn_anchor z=17.2 is the mating face. With face_traj + eef_rpy 180 0 0 +
# jack_align -90 0 0, `-90 0 0` maps splat +z -> [0,-1,0] = straight into the jack mouth.
# `0 0 0` maps +z -> [0,0,-1] = the plug renders as a VERTICAL rod (a real bug that only became
# obvious once the longer tail made the long axis visible). Verify against the +z axis, not +y.
CONN_RPY=${CONN_RPY:-"-90 0 0"}
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim

# connector splats: HEAD (RJ45 plug) + TAIL (boot) = the two halves of cad_plug_registered, both
# ride the plug FACE pose. NOTE: adding the tail changes the splat COUNT 18->19, so the renderer
# MUST be restarted before this batch (it caches the set by count).
PLUG_HEAD=${PLUG_HEAD:-/home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/ethernet/cropped_plug_head.ply}
PLUG_TAIL=${PLUG_TAIL:-/home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/ethernet/cropped_plug_tail_longer.ply}
GRIPPER_DIR=/root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_v3
WRIST3_PLY=/root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link_realpalm.ply
SBOT_USD=/home/pandaliza/parallax/robo_maker/sbot/assets/standardbot.usd

# LOCKED robot-pose config (arm home joints, base, grasp, jack, cameras) -- single source of truth.
source "$REPO/tools/datagen_v2_config.sh"

# Derived renderer flags (built here, as in v3 -- NOT in the config). Plug-inference flags
# (INSERT_ROT_FLAG, trim/along-cord/stop-after-seat) are intentionally absent: this is the eef path.
RENDER_FLAGS=""
[ -n "${RENDER_GAMMA:-}" ] && RENDER_FLAGS="$RENDER_FLAGS --gamma $RENDER_GAMMA"
[ -n "${RENDER_GAIN:-}" ]  && RENDER_FLAGS="$RENDER_FLAGS --gain $RENDER_GAIN"
FRONT_FLAGS=""
[ -n "${FRONT_EYE:-}" ]        && FRONT_FLAGS="$FRONT_FLAGS --eye $FRONT_EYE"
[ -n "${FRONT_TARGET:-}" ]     && FRONT_FLAGS="$FRONT_FLAGS --target $FRONT_TARGET"
[ -n "${FRONT_EYE_OFF:-}" ]    && FRONT_FLAGS="$FRONT_FLAGS --eye-offset $FRONT_EYE_OFF"
[ -n "${FRONT_TARGET_OFF:-}" ] && FRONT_FLAGS="$FRONT_FLAGS --target-offset $FRONT_TARGET_OFF"
SIDE_FLAGS=""                                     # FRONT + WRIST only (datagen_to_lerobot ignores mirror/side)
WRIST_USD_FLAG=""
[ "${WRIST_CAM_FROM_USD:-0}" = "1" ] && WRIST_USD_FLAG="--wrist-cam-from-usd"

cd "$REPO"
# --out is makedirs'd even with --no-preview; point every episode at ONE throwaway dir.
PREVIEW_TMP=$(mktemp -d /tmp/render_v4_preview.XXXXXX)
trap 'rm -rf "$PREVIEW_TMP"' EXIT

# provenance stamp
mkdir -p "$OUTROOT"
{
  echo "datagen_v4 generated (cable PPO track)"
  echo "script      : tools/render_batch_v4.sh   range [$START,$END)"
  echo "traj source : $TRAJROOT/ep_XXXX/eef_traj.npy  (rl/gen_cable_traj.py)"
  echo "renderer    : scripts/record_sbot_scene_gs_cable.py --eef-traj  (eef-rpy $EEF_RPY)"
  echo "gripper gs  : $GRIPPER_DIR"
  echo "wrist3 ply  : $WRIST3_PLY"
  echo "sbot usd    : $SBOT_USD"
  echo "base/jack   : BASE_POS=$BASE_POS  JACK_POS=$JACK_POS  ARM_HOME=$ARM_HOME_DEG"
  echo "render      : ${WIDTH:-640}x${HEIGHT:-480}  dump $DUMP_SIZE  gamma ${RENDER_GAMMA:-off}"
} > "$OUTROOT/_CONFIG.txt"
cat "$OUTROOT/_CONFIG.txt"

done_n=0; skip_n=0; fail_n=0; consec_fail=0
for i in $(seq "$START" $((END - 1))); do
    ep=$(printf "ep_%04d" "$i")
    traj=$TRAJROOT/$ep/eef_traj.npy
    out=$OUTROOT/$ep
    [ -f "$traj" ] || { echo "[batch] $ep: no trajectory, skipping"; continue; }
    if [ -f "$out/state.npy" ] && grep -q "\[dump\] FULL episode" "$out.log" 2>/dev/null; then
        skip_n=$((skip_n + 1)); continue
    fi
    echo "[batch] rendering $ep -> $out"
    .venv/bin/python scripts/record_sbot_scene_gs_cable.py \
        --eef-traj "$traj" --eef-rpy $EEF_RPY \
        --jack-pos $JACK_POS --jack-align-rpy $JACK_ALIGN_RPY --conn-rpy $CONN_RPY \
        --connector-ply $PLUG_HEAD --connector-tail-ply $PLUG_TAIL \
        --base-pos $BASE_POS --base-yaw-deg $BASE_YAW_DEG \
        --arm-home-deg $ARM_HOME_DEG \
        --grasp-rpy $GRASP_RPY --grasp-protrude $GRASP_PROTRUDE \
        --gripper-gs-dir $GRIPPER_DIR \
        --wrist3-ply $WRIST3_PLY \
        --camera-config /home/pandaliza/parallax/newton-cabling/configs/cameras.yaml \
        --table-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/table/splat_flat.ply \
        --bg-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/background/splat_open.ply \
        --wrist-cam --wrist-orbit $WRIST_ORBIT --wrist-side $WRIST_SIDE --wrist-up $WRIST_UP --wrist-back $WRIST_BACK --wrist-aim-back $WRIST_AIM_BACK ${WRIST_USD_FLAG:-} --wrist-rigid \
        --dist-scale $DIST_SCALE --elev $ELEV --azim $AZIM $FRONT_FLAGS $SIDE_FLAGS $RENDER_FLAGS \
        --grasped-only \
        --width ${WIDTH:-640} --height ${HEIGHT:-480} \
        --dump "$out" --dump-size $DUMP_SIZE --no-preview \
        --out "$PREVIEW_TMP" > "$out.log" 2>&1
    if [ -f "$out/state.npy" ] && grep -q "\[dump\] FULL episode" "$out.log" 2>/dev/null; then
        done_n=$((done_n + 1)); consec_fail=0
        grep -h "\[profile\]" "$out.log" | sed "s/^/[batch] $ep /"
        # SIDE-BY-SIDE debug view: [GS FRONT | GS WRIST | Newton truth] per frame, from the SAME
        # saved traj GS just replayed (re-rolling the policy would differ -- VBD nondeterminism).
        # Frame indices map 1:1 because we render --grasped-only (no preamble). NEWTON_REF=0 to skip.
        if [ "${NEWTON_REF:-1}" = "1" ]; then
            mkdir -p "$OUTROOT/_compare"
            .venv/bin/python tools/compare_gs_newton.py --gs "$out" --traj "$TRAJROOT/$ep" \
                --out "$OUTROOT/_compare/$ep" --stride ${CMP_STRIDE:-8} >> "$out.log" 2>&1 \
                && echo "[batch] $ep compare -> $OUTROOT/_compare/${ep}_cmp_*.png"
        fi
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
echo "  2. convert -> LeRobot (openpi venv). --truncate_after_seat -1: episodes are ALREADY cut at"
echo "     success+hold by gen_cable_traj.py, and there is no seated_traj/plug_traj.npy to key off."
echo "     cd ~/parallax/openpi && uv run python ~/parallax/newton-cabling/tools/datagen_to_lerobot.py \\"
echo "         --raw ${OUTROOT}_difix --repo_id parallax/cable_sbot_v1 --truncate_after_seat -1"
