#!/usr/bin/env bash
# Closed-loop VLA eval of the pi05 rj45_sbot_HEAD55_JACK13_difix policy (trained off-box, step
# 3500) on held-out starts, at the EXACT datagen_head55cad_jackv2 scene.
#
# SCENE NOTE -- this is a TRANSFER eval by default. The checkpoint trained on
# datagen_head55_jack1p3all (traj cable_traj_head55_500 / scan_rj45.usd, jack splat
# cad_jack_registered_x1p3.ply, plug splat headA_plug_roll210.ply). The defaults below mirror
# datagen_head55cad_jackv2 instead (traj cad_rj45.usd, jack splat jack_v2_registered.ply, plug
# splat headA_plug_cadframe_rigid.ply) -- a different jack registration AND a different plug
# appearance. For the IN-DISTRIBUTION baseline, override:
#   TRAJROOT=$DATA/eval_traj_head55  OUTROOT=$DATA/evalrun_head55_jack13 \
#   JACK_PLY=$ETH/cad_jack_registered_x1p3.ply  PLUG_HEAD=$ETH/headA_plug_roll210.ply
# (and generate that trajectory set with --connector-usd scan_rj45.usd).
#
# Held-out starts: eef-mode trajectories at a NEVER-TRAINED seed (training used seed 0):
#   .venv/bin/python -m newton_cabling.scripted_controller.gen_trajectories \
#       --out /home/pandaliza/parallax/data/vla_train/eval_traj_head55cad \
#       --envs 16 --rounds 40 --steps 160 --max-keep 160 --no-cut-at-success \
#       --connector-usd cad_rj45.usd --grasp-roll-180 --grip-from-head 55 \
#       --max-save 60 --seed 777
#   (--grip-from-head 55 is REQUIRED: the head55 training set used a 55mm grip, not the 68mm default.)
#
# Prereqs (three processes share the one 5090 -- no training in parallel):
#   1. GS renderer restarted AT THIS SPLAT SET:   bash tools/restart_gs_renderer.sh
#      MANDATORY, not hygiene: the jack/plug ply swap keeps the splat COUNT at 19, and the
#      renderer caches by count -- without a restart it silently serves the previous scene's
#      splats and the eval looks plausible but is measuring the wrong jack.
#   2. openpi policy server (OPENPI venv):
#        cd ~/parallax/openpi && uv run scripts/serve_policy.py policy:checkpoint \
#            --policy.config=pi05_rj45_sbot_head55_jack13_lora \
#            --policy.dir=checkpoints/pi05_rj45_sbot_head55_jack13_difix/3500
#   3. Difix server (NOT optional -- the checkpoint trained on Difix'd frames):
#        cd ~/parallax/Difix3D && .venv/bin/python difix_server.py --port 8021
#   4. This script, WITH SUDO (root-owned SHM), from the repo root:
#        sudo ALLOW_NONCANONICAL_EEF_RPY=1 DIFIX_SERVER=localhost:8021 \
#            bash tools/eval_batch_head55cad.sh 0 60 [HOST:PORT]
#      (no-sudo path: run the chmod-666 keeper loop from tools/datagen_dr_jack.sh first)
set -u
START=${1:-0}
END=${2:-60}
SERVER=${3:-localhost:8000}
REPO=/home/pandaliza/parallax/newton-cabling
DATA=/home/pandaliza/parallax/data/vla_train
ETH=$REPO/newton_cabling/assets/ethernet
TRAJROOT=${TRAJROOT:-$DATA/eval_traj_head55cad}
OUTROOT=${OUTROOT:-$DATA/evalrun_head55cad_jackv2}
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim

# ── scene geometry: IDENTICAL to datagen_head55cad_jackv2/_CONFIG.txt ──
PLUG_HEAD=${PLUG_HEAD:-$ETH/headA_plug_cadframe_rigid.ply}
PLUG_TAIL=${PLUG_TAIL:-none}
JACK_PLY=${JACK_PLY:-$ETH/jack_v2_registered.ply}
EEF_RPY=${EEF_RPY:-"0 0 180"}
JACK_ANCHOR=${JACK_ANCHOR:-"0 0 0.018"}
GRIP_THETA=${GRIP_THETA:-"-0.0152"}
JACK_ALIGN_RPY=${JACK_ALIGN_RPY:-"-90 0 0"}
CONN_RPY=${CONN_RPY:-"-90 0 0"}
FRONT_EYE=${FRONT_EYE:-"0.53 -0.681 0.984"}
GRIPPER_DIR=/root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_v3
WRIST3_PLY=/root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link_realpalm.ply
source "$REPO/tools/datagen_v2_config.sh"   # arm home, base, grasp, cameras, gamma 1.8

# A missing ply would otherwise fall back to a stock splat / fail deep inside the renderer.
for f in "$PLUG_HEAD" "$JACK_PLY"; do
    [ -f "$f" ] || { echo "[eval] ERROR: splat not found: $f"; exit 2; }
done

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
{
  echo "eval of Parallax-Worlds/pi05_rj45_sbot_head55_jack13_difix @ 3500"
  echo "script    : tools/eval_batch_head55cad.sh   range [$START,$END)  server $SERVER"
  echo "traj      : $TRAJROOT   (held-out, seed 777)"
  echo "jack splat: $JACK_PLY"
  echo "plug splat: $PLUG_HEAD  tail=$PLUG_TAIL"
  echo "geometry  : EEF_RPY=\"$EEF_RPY\" JACK_ALIGN_RPY=\"$JACK_ALIGN_RPY\" JACK_ANCHOR=\"$JACK_ANCHOR\" CONN_RPY=\"$CONN_RPY\" GRIP_THETA=\"$GRIP_THETA\""
  echo "difix     : ${DIFIX_SERVER:-OFF (obs will NOT match training)}"
} > "$OUTROOT/_CONFIG.txt"
cat "$OUTROOT/_CONFIG.txt"

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
        --jack-anchor $JACK_ANCHOR --jack-ply "$JACK_PLY" \
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
