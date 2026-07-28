#!/usr/bin/env bash
# Closed-loop eval of a pi05 checkpoint on held-out plug starts (eval_traj/, seed 777 — never
# seen in training). Needs: (1) the parallax_sim renderer UP at the 19-splat scene config,
# (2) an openpi policy server, started in the OPENPI venv (pause training first — no VRAM for both):
#     cd ~/parallax/openpi && uv run scripts/serve_policy.py policy:checkpoint \
#         --policy.config=pi05_rj45_sbot_lora \
#         --policy.dir=checkpoints/pi05_rj45_sbot_lora/rj45_sbot_v1/<STEP>
# Run WITH SUDO from the repo root:
#     sudo bash tools/eval_batch.sh 0 20 [HOST:PORT]
set -u
START=${1:-0}
END=${2:-20}
SERVER=${3:-localhost:8000}
REPO=/home/pandaliza/parallax/newton-cabling
OUTROOT=$REPO/evalrun
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim

cd "$REPO"
mkdir -p "$OUTROOT"
seated=0; ran=0
for i in $(seq "$START" $((END - 1))); do
    ep=$(printf "ep_%04d" "$i")
    traj=eval_traj/$ep/plug_traj.npy
    out=$OUTROOT/$ep
    [ -f "$traj" ] || continue
    rm -rf "$out"    # clear stale frames/results from previous runs (episodes can be shorter now)
    echo "[eval] $ep ..."
    .venv/bin/python scripts/record_sbot_scene_gs.py \
        --plug-traj "$traj" \
        --jack-pos 0.295 -0.876 0.835 \
        --base-pos 0.295 -0.45 0.87 --base-yaw-deg 0 \
        --grasp-rpy 0 -90 0 --grasp-protrude 0.025 --grasp-along-cord 0.027 \
        --gripper-gs-dir /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_proj \
        --table-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/table/splat_flat.ply \
        --bg-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/background/splat_open.ply \
        --wrist-cam --wrist-orbit 75 --wrist-side -0.25 --wrist-rigid \
        --dist-scale 1.2 --elev 5 --azim -45 \
        --policy-server "$SERVER" --eval-max-steps 300 \
        --out "$out" > "$out.log" 2>&1
    ran=$((ran + 1))
    if grep -q '"seated": true' "$out/result.json" 2>/dev/null; then
        seated=$((seated + 1)); echo "[eval] $ep SEATED  ($(grep seat_step "$out/result.json"))"
    else
        echo "[eval] $ep failed  ($(grep final_y "$out/result.json" 2>/dev/null || echo 'no result'))"
    fi
done
echo "[eval] SEAT RATE: $seated/$ran  (PPO teacher reference: ~92-97%)"
