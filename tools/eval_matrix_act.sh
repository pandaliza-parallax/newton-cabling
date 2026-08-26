#!/usr/bin/env bash
# Closed-loop eval matrix: act_c10_depaused + pi05-21000-depaused on 20 conditions
# (10 recorded TRAIN episodes + 10 held-out EVAL draws; evalmatrix_conditions.json),
# 450-step budget, video per episode. Waits for the tier-2 Stage-B verify gate so
# the renderer is free. ACT block runs beside batch difix (CPU policy); the pi05
# block pauses batch difix (GPU policy server needs the VRAM) and resumes it after.
set -u
REPO=/home/pandaliza/parallax/newton-cabling
COND=${COND:-/home/pandaliza/parallax/data/vla_train/evalmatrix_conditions.json}
OUTROOT=/home/pandaliza/parallax/data/vla_train/evalmatrix
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim
export ALLOW_NONCANONICAL_EEF_RPY=1
source "$REPO/tools/datagen_v2_config.sh"
cd "$REPO"
mkdir -p "$OUTROOT"

MODEL=${MODEL:?set MODEL to the checkpoint dir name under openpi/checkpoints}
# waits for nothing: caller ensures the renderer is free

echo "=== starting difix obs server $(date +%H:%M) ==="
pkill -f "difix_serve[r]" || true; sleep 2
( cd /home/pandaliza/parallax/Difix3D && setsid nohup .venv/bin/python difix_server.py --port 8021 \
    > /home/pandaliza/difix_server.log 2>&1 < /dev/null & )
sleep 20

keeper() { docker exec -d parallax_sim_fp bash -lc 'for i in $(seq 1 14400); do \
  chmod 666 /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
            /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer* 2>/dev/null; sleep 2; done'; }

run_block() {  # $1 = model tag (act_c10_depaused | pi05_21000)
    local MODEL=$1 CUR_BG=""
    local N=$(python3 -c "import json; print(len(json.load(open('$COND'))))")
    for i in $(seq 0 $((N - 1))); do
        eval "$(python3 - "$COND" "$i" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))[int(sys.argv[2])]
tilt = str(c["tilt"]) if c["tilt"] is not None else "-5 8"
print(f'NAME={c["name"]}; TILT="{tilt}"; ROLL={c["roll"]}; GRIP={c["grip"]}; '
      f'JPOS="{c["jack_pos"]}"; GAMMA={c["gamma"]}; BGY={c["bg_yaw"]}; '
      f'SHW={c["shadow"]}; SEED={c["seed"]}; SPLIT={c["split"]}; '
      f'JYAW={"0" if c["split"] == "train" else "5"}')
EOF
)"
        local OUT="$OUTROOT/${MODEL}_${NAME}"
        [ -f "$OUT/result.json" ] && { echo "=== skip $OUT (done)"; continue; }
        if [ "$BGY" != "$CUR_BG" ]; then
            echo "=== renderer restart for bg-yaw $BGY $(date +%H:%M) ==="
            keeper
            bash tools/restart_gs_renderer.sh || { echo "renderer restart FAILED"; exit 1; }
            CUR_BG="$BGY"
        fi
        rm -rf "$OUT"; mkdir -p "$OUT"
        echo "=== $MODEL $NAME (seed $SEED, grip $GRIP, γ$GAMMA, bg $BGY) $(date +%H:%M) ==="
        .venv/bin/python scripts/record_sbot_scene_gs_cable.py \
          --live-env --live-stage 4 --live-tilt $TILT --live-seed $SEED \
          --live-grip-from-head $GRIP --live-fixture --live-servo \
          --live-jack-yaw $JYAW --live-grasp-roll-deg $ROLL \
          --eef-rpy 0 0 180 \
          --jack-pos $JPOS --jack-align-rpy -90 0 90 --conn-rpy -90 0 180 \
          --jack-anchor 0 0 0.018 \
          --jack-ply $REPO/newton_cabling/assets/ethernet/jack_fixture_0813_01_registered.ply \
          --grip-theta -0.0152 \
          --connector-ply $REPO/newton_cabling/assets/ethernet/headA_plug_cadframe_rigid50.ply \
          --connector-tail-ply none \
          --base-pos $BASE_POS --base-yaw-deg $BASE_YAW_DEG --arm-home-deg $ARM_HOME_DEG \
          --grasp-rpy $GRASP_RPY --grasp-protrude $GRASP_PROTRUDE \
          --gripper-gs-dir /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_v3 \
          --wrist3-ply /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link_realpalm.ply \
          --camera-config $REPO/configs/cameras.yaml \
          --table-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/table/splat_flat.ply \
          --bg-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/background/splat_open.ply --bg-yaw-deg $BGY \
          --wrist-cam --wrist-orbit $WRIST_ORBIT --wrist-side $WRIST_SIDE --wrist-up $WRIST_UP \
          --wrist-back $WRIST_BACK --wrist-aim-back $WRIST_AIM_BACK --wrist-cam-from-usd --wrist-rigid \
          --dist-scale $DIST_SCALE --elev $ELEV --azim $AZIM --eye 0.53 -0.681 0.984 --target $FRONT_TARGET \
          --gamma $GAMMA --grasped-only --width 640 --height 480 --dump-size 512 \
          --shadows --shadow-strength $SHW --shadow-accum 8 --shadow-mask-scale 0.4 \
          --policy-server localhost:8000 --eval-max-steps 450 --difix-server localhost:8021 \
          --out "$OUT" > "$OUT.log" 2>&1
        ( cd "$OUT" && ffmpeg -y -loglevel error -framerate 30 -i frame_%04d.png \
            -vf "drawtext=text='$MODEL  $NAME':x=10:y=10:fontsize=22:fontcolor=yellow:box=1:boxcolor=black@0.6" \
            -c:v libx264 -pix_fmt yuv420p -crf 20 rollout.mp4 ) || true
        echo "=== $MODEL $NAME: $(grep -o '"seated": [a-z]*' $OUT/result.json 2>/dev/null || echo NO_RESULT)"
    done
}

echo "=== ACT block (CPU policy; batch difix keeps running) $(date +%H:%M) ==="
pkill -f "serve_act_polic[y]"; pkill -f "serve_polic[y]"; sleep 3
( setsid nohup /home/pandaliza/parallax/act_venv/bin/python $REPO/tools/serve_act_policy.py \
    --ckpt /home/pandaliza/parallax/openpi/checkpoints/$MODEL --port 8000 \
    > /home/pandaliza/act_server.log 2>&1 < /dev/null & )
until grep -q "listening" /home/pandaliza/act_server.log 2>/dev/null; do sleep 3; done
run_block "$MODEL"

pkill -f "serve_act_polic[y]"; sleep 2
python3 - "$OUTROOT" <<'EOF'
import glob, json, os, sys
rows = []
for rj in sorted(glob.glob(os.path.join(sys.argv[1], "*", "result.json"))):
    r = json.load(open(rj))
    name = os.path.basename(os.path.dirname(rj))
    rows.append((name, r.get("seated"), round(r.get("final_y_sock_mm", 0), 1)))
json.dump(rows, open(os.path.join(sys.argv[1], "summary.json"), "w"), indent=1)
for n, s, y in rows:
    print(f"{n:45s} seated={s} final_y={y}mm")
import collections
seat = collections.Counter(n.rsplit("_", 2)[0] for n, s, _ in rows if s)
print("SEATS BY MODEL PREFIX:", dict(seat))
EOF
echo "=== EVAL MATRIX DONE $(date +%H:%M) ==="
