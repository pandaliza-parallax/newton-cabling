#!/usr/bin/env bash
# Closed-loop rollout on TRAIN ep_0000 conditions for the two ACT baselines
# (act_c10: 10 actions/obs, act_c100: 100 actions/obs — chunk length set by the
# server, the client executes whole chunks). ACT is served on CPU (openpi-venv
# torch has no CUDA kernels for this GPU; 51M model, one forward per chunk).
# Mirrors eval_trainep_ckpts.sh conditions exactly for comparability.
# NOTE: caller must have paused the tier-2 render campaign (renderer conflict).
set -u
REPO=/home/pandaliza/parallax/newton-cabling
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim
export ALLOW_NONCANONICAL_EEF_RPY=1
source "$REPO/tools/datagen_v2_config.sh"
cd "$REPO"

echo "=== renderer restart + keeper $(date +%H:%M) ==="
docker exec -d parallax_sim_fp bash -lc 'for i in $(seq 1 14400); do \
  chmod 666 /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
            /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer* 2>/dev/null; sleep 2; done'
bash tools/restart_gs_renderer.sh || { echo "renderer restart FAILED"; exit 1; }

MODELS=${MODELS:-"act_c10 act_c100"}
for M in $MODELS; do
    echo "=== $M: starting CPU policy server $(date +%H:%M) ==="
    pkill -f "serve_act_polic[y]"; pkill -f "serve_polic[y]"; sleep 3
    ( cd /home/pandaliza/parallax/openpi && \
      setsid nohup uv run python $REPO/tools/serve_act_policy.py \
        --ckpt /home/pandaliza/parallax/openpi/checkpoints/$M --port 8000 \
        > /home/pandaliza/act_server.log 2>&1 < /dev/null & )
    until grep -q "listening" /home/pandaliza/act_server.log 2>/dev/null; do sleep 3; done

    OUT=/home/pandaliza/parallax/data/vla_train/evalrun_trainep0000_$M
    rm -rf "$OUT"; mkdir -p "$OUT"
    echo "=== $M: running episode $(date +%H:%M) ==="
    .venv/bin/python scripts/record_sbot_scene_gs_cable.py \
      --live-env --live-stage 4 --live-tilt 2.727 --live-seed 40000 \
      --live-grip-from-head 55 --live-fixture --live-servo \
      --live-jack-yaw 0 --live-grasp-roll-deg 151.508 \
      --eef-rpy 0 0 180 \
      --jack-pos 0.2361 -0.8233 0.835 --jack-align-rpy -90 0 90 --conn-rpy -90 0 180 \
      --jack-anchor 0 0 0.018 \
      --jack-ply /home/pandaliza/parallax/newton-cabling/newton_cabling/assets/ethernet/jack_fixture_0813_01_registered.ply \
      --grip-theta -0.0152 \
      --connector-ply /home/pandaliza/parallax/newton-cabling/newton_cabling/assets/ethernet/headA_plug_cadframe_rigid50.ply \
      --connector-tail-ply none \
      --base-pos $BASE_POS --base-yaw-deg $BASE_YAW_DEG --arm-home-deg $ARM_HOME_DEG \
      --grasp-rpy $GRASP_RPY --grasp-protrude $GRASP_PROTRUDE \
      --gripper-gs-dir /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_v3 \
      --wrist3-ply /root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link_realpalm.ply \
      --camera-config $REPO/configs/cameras.yaml \
      --table-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/table/splat_flat.ply \
      --bg-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/background/splat_open.ply --bg-yaw-deg 219 \
      --wrist-cam --wrist-orbit $WRIST_ORBIT --wrist-side $WRIST_SIDE --wrist-up $WRIST_UP \
      --wrist-back $WRIST_BACK --wrist-aim-back $WRIST_AIM_BACK --wrist-cam-from-usd --wrist-rigid \
      --dist-scale $DIST_SCALE --elev $ELEV --azim $AZIM --eye 0.53 -0.681 0.984 --target $FRONT_TARGET \
      --gamma 1.4 --grasped-only --width 640 --height 480 --dump-size 512 \
      --shadows --shadow-strength 0.61 --shadow-accum 8 --shadow-mask-scale 0.4 \
      --policy-server localhost:8000 --eval-max-steps ${EVAL_STEPS:-450} --difix-server localhost:8021 \
      --out "$OUT" > "$OUT.log" 2>&1
    ( cd "$OUT" && ffmpeg -y -loglevel error -framerate 30 -i frame_%04d.png \
        -vf "drawtext=text='ACT $M on TRAIN ep_0000 conditions':x=10:y=10:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.6" \
        -c:v libx264 -pix_fmt yuv420p -crf 20 rollout_trainep.mp4 )
    echo "=== $M: $(grep -o '"seated": [a-z]*' $OUT/result.json 2>/dev/null || echo 'NO RESULT') ==="
done
pkill -f "serve_act_polic[y]"
echo "=== ACT TRAINEP EVALS DONE ==="
