#!/usr/bin/env bash
# FAST wrist-camera check: renders a couple of frames at the grasp pose and dumps the WRIST view,
# so you can iterate on /sbot/wrist_3_link/Camera in standardbot.usd without waiting for a full
# 131-frame episode (~30s + 13s of preview PNG writes).
#
# Reads the camera live from the USD (--wrist-cam-from-usd), so just re-save the USD and re-run.
# Scene/camera config matches datagen_v2 so what you see is what training/eval would see.
#
# Run WITH SUDO (root-owned SHM), from the repo root:
#     sudo bash tools/check_wrist_cam.sh                 # default: out_wristcam
#     sudo bash tools/check_wrist_cam.sh out_mycheck     # custom out dir
#     sudo TILT="0 45 0" bash tools/check_wrist_cam.sh   # also apply a grasp tilt
#
# NOTE: if you change the SPLAT SET (gripper dir / wrist3 ply / bg), restart the renderer first:
#   docker exec parallax_sim_fp bash -lc "pkill -9 -f '[d]alus_sim_app'; sleep 3; \
#     rm -f /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
#           /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer*"
#   docker exec -d parallax_sim_fp bash -lc \
#     'cd /root/parallax/DalusSimCore && python3 dalus_sim_app.py > /tmp/render.log 2>&1'
set -u
OUT=${1:-out_wristcam}
REPO=/home/pandaliza/parallax/newton-cabling
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim
cd "$REPO"
source "$REPO/tools/datagen_v2_config.sh"

TILT_FLAG=""
[ -n "${TILT:-}" ] && TILT_FLAG="--grasp-tilt-rpy $TILT"

rm -rf "$OUT"
.venv/bin/python scripts/record_sbot_scene_gs.py \
    --plug-traj eval_traj_v2/ep_0009/plug_traj.npy \
    --jack-pos $JACK_POS \
    --base-pos $BASE_POS --base-yaw-deg $BASE_YAW_DEG \
    --arm-home-deg $ARM_HOME_DEG \
    --grasp-rpy $GRASP_RPY --grasp-protrude $GRASP_PROTRUDE --grasp-along-cord $GRASP_ALONG_CORD \
    ${TILT_FLAG:-} \
    --gripper-gs-dir /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_v3 \
    --wrist3-ply /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link_realpalm.ply \
    --table-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/table/splat_flat.ply \
    --bg-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/background/splat_open.ply \
    --camera-config configs/cameras.yaml --gamma 1.8 \
    --width ${WIDTH:-640} --height ${HEIGHT:-480} \
    --dist-scale $DIST_SCALE --elev $ELEV --azim $AZIM --target $FRONT_TARGET \
    --wrist-cam --wrist-cam-from-usd --wrist-rigid \
    --no-approach --grasped-only --stop-after-seat 0 \
    --out "$OUT" 2>&1 | grep -aE "\[wrist\]|GRASP TILT|\[scene\] wrist|IK reach|penetration|wrote|Error|Traceback"

# crop the WRIST half out of the FRONT|WRIST stitch for a clean look
FIRST=$(ls "$OUT"/frame_*.png 2>/dev/null | head -1)
if [ -n "$FIRST" ]; then
    python3 - "$FIRST" "$OUT/WRIST_VIEW.png" <<'PY'
import sys
from PIL import Image
im = Image.open(sys.argv[1]); w, h = im.size
im.crop((w // 2, 0, w, h)).save(sys.argv[2])
print(f"  wrist view -> {sys.argv[2]}  ({w//2}x{h})")
PY
else
    echo "  no frames written — check the log above"
fi
