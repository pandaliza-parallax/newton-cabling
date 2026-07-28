#!/usr/bin/env bash
# Batch-render seated PPO rollouts into openpi-format training data (FRONT + WRIST cams).
# Config locked from command_data.txt (2026-07-06). FULL episode (12 hold + 45 approach with
# gripper open->close + 200 insertion = 257 frames). Per episode -> datagen/ep_XXXX/:
#   image/frame_*.png        FRONT cam, 224x224 (257 frames)
#   wrist_image/frame_*.png  WRIST cam, 224x224
#   state.npy (257,10)       [eef_pos(3), eef_rot6d(6), gripper(1)]  absolute, base frame
#   action.npy (257,7)       [dpos(3), drotvec(3), gripper(1)]       delta to next frame
#   phase.npy (257,)         0=hold 1=approach 2=insertion
#   meta.json, preview/      (stitched full-res FRONT|WRIST preview)
# Episodes with a complete dump are SKIPPED -> resume-safe after interruption.
#
# The renderer must be UP and loaded for this 19-splat scene (restart it once before the batch;
# the splat set is identical across episodes so no restarts are needed inside the loop).
#
# Run WITH SUDO (the SHM segments are root-owned), from the repo root:
#     sudo bash tools/render_batch.sh 0 20      # episodes ep_0000 .. ep_0019 (subset test)
#     sudo bash tools/render_batch.sh 0 500     # the full set
set -u
START=${1:-0}
END=${2:-500}
REPO=/home/pandaliza/parallax/newton-cabling
OUTROOT=$REPO/datagen
FRAMES_EXPECTED=257     # home_hold 12 + approach 45 + 200 replay
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim

cd "$REPO"
mkdir -p "$OUTROOT"
done_n=0; skip_n=0; fail_n=0; consec_fail=0
for i in $(seq "$START" $((END - 1))); do
    ep=$(printf "ep_%04d" "$i")
    traj=seated_traj/$ep/plug_traj.npy
    out=$OUTROOT/$ep
    [ -f "$traj" ] || { echo "[batch] $ep: no trajectory, skipping"; continue; }
    # complete = the dump finished ("[dump] FULL episode" printed after state/action/images are
    # written). Episodes are VARIABLE length now (--stop-after-seat), so no fixed frame count.
    if [ -f "$out/state.npy" ] && grep -q "\[dump\] FULL episode" "$out.log" 2>/dev/null; then
        skip_n=$((skip_n + 1)); continue
    fi
    echo "[batch] rendering $ep -> $out"
    .venv/bin/python record_sbot_scene_gs.py \
        --plug-traj "$traj" \
        --jack-pos 0.295 -0.876 0.835 \
        --base-pos 0.295 -0.45 0.87 --base-yaw-deg 0 \
        --grasp-rpy 0 -90 0 --grasp-protrude 0.025 --grasp-along-cord 0.027 \
        --gripper-gs-dir /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_proj \
        --table-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/table/splat_flat.ply \
        --bg-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/background/splat_open.ply \
        --wrist-cam --wrist-orbit 75 --wrist-side -0.25 --wrist-rigid \
        --dist-scale 1.2 --elev 5 --azim -45 \
        --dump "$out" --dump-size 224 --no-preview --stop-after-seat 20 \
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
