#!/usr/bin/env bash
# DEBUG-ONLY jack-splat previewer. Renders ONE episode with a swapped jack splat so you can
# eyeball the new 3-Aug scan (Ethernet_jack_2.0) before it goes anywhere near a datagen run.
#
# Nothing here writes to a production path: no OUTROOT of datagen_head*, no difix, no
# compare_gs_newton, no LeRobot conversion. tools/datagen_dr_jack.sh, tools/render_batch_v4.sh
# and scripts/record_sbot_scene_gs_cable.py are NOT touched -- this script only calls them.
#
# Every renderer flag below is copied verbatim from render_batch_v4.sh + datagen_v2_config.sh,
# so the ONLY difference between a debug frame and a production frame is --jack-ply (and the
# camera framing when ZOOM=1). If the jack looks wrong here, it is the splat, not the setup.
#
#   bash tools/render_jack_debug.sh                    # bake + render ep_0000, zoomed on the jack
#   ROLLS="0 90 180 270" bash tools/render_jack_debug.sh   # settle the roll ambiguity in one pass
#   AB=1 bash tools/render_jack_debug.sh               # also render the current x1p3 jack, same frame
#   ZOOM=0 bash tools/render_jack_debug.sh             # production front framing instead of a zoom
#   EP=ep_0042 JACK_PLY=/path/to.ply bash tools/render_jack_debug.sh
#
# THE RENDERER IS A SINGLETON. dalus_sim_app's SHM names are hardcoded (dal_buffer0/1,
# send_dal_buffer0/1 + their semaphores -- DalusSimCore/dalus_sim_core/ipc_receiver/config.py),
# so there is exactly one renderer and one client at a time. Worse, the app caches the splat
# set and a same-COUNT file swap does not invalidate it, so each jack ply needs its own
# restart. That kills any datagen batch mid-flight, which is why this script refuses to start
# while one is running (FORCE=1 to override).
set -u
REPO=/home/pandaliza/parallax/newton-cabling
ETH=$REPO/newton_cabling/assets/ethernet
EP=${EP:-ep_0000}
TRAJROOT=${TRAJROOT:-/home/pandaliza/parallax/data/vla_train/cable_traj_head55_500}
DBGROOT=${DBGROOT:-/home/pandaliza/parallax/data/vla_train/_jack_v2_debug}
ROLLS=${ROLLS:-}                      # empty = render JACK_PLY as-is; else bake one ply per roll
JACK_PLY=${JACK_PLY:-$ETH/jack_v2_registered.ply}
PROD_JACK=${PROD_JACK:-$ETH/cad_jack_registered_x1p3.ply}   # the A/B reference (current datagen)
ZOOM=${ZOOM:-1}
FRAME_RADIUS=${FRAME_RADIUS:-0.07}
AB=${AB:-0}

cd "$REPO"

if [ "${FORCE:-0}" != "1" ] && pgrep -f "[r]ender_batch_v4.sh" > /dev/null; then
  echo "[jackdbg] REFUSING TO RUN: a datagen batch is live --"
  pgrep -af "[r]ender_batch_v4.sh" | sed 's/^/[jackdbg]   /'
  echo "[jackdbg] Restarting the renderer for a splat swap would break it. Wait for it to"
  echo "[jackdbg] finish, or stop it (it is resume-safe: completed episodes are skipped on"
  echo "[jackdbg] rerun), then re-run. FORCE=1 to override."
  exit 3
fi

traj=$TRAJROOT/$EP/eef_traj.npy
[ -f "$traj" ] || { echo "[jackdbg] no trajectory at $traj"; exit 2; }

# LOCKED robot-pose config (arm home joints, base, grasp, jack, cameras) -- same source of
# truth render_batch_v4.sh uses.
source "$REPO/tools/datagen_v2_config.sh"

# render_batch_v4.sh geometry, verbatim. See that file's header for why each value is what
# it is -- none of them are eyeball parameters.
EEF_RPY=${EEF_RPY:-"0 0 180"}
JACK_ALIGN_RPY=${JACK_ALIGN_RPY:-"-90 0 0"}
CONN_RPY=${CONN_RPY:-"-90 0 0"}
JACK_ANCHOR=${JACK_ANCHOR:-"0 0 0.018"}
GRIP_THETA=${GRIP_THETA:-"-0.0152"}
PLUG_HEAD=${PLUG_HEAD:-$ETH/headA_plug_roll210.ply}
PLUG_TAIL=${PLUG_TAIL:-none}
GRIPPER_DIR=/root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_v3
WRIST3_PLY=/root/parallax/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link_realpalm.ply
export PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim

# ZOOM=1 frames the jack instead of the scene: --frame jack needs --eye/--target ABSENT
# (record_sbot_scene_gs_cable.py:625 -- an explicit --eye overrides the framing entirely).
if [ "$ZOOM" = "1" ]; then
  CAM_FLAGS="--frame jack --frame-radius $FRAME_RADIUS --dist-scale 0.9 --elev 12 --azim -50"
else
  CAM_FLAGS="--dist-scale $DIST_SCALE --elev $ELEV --azim $AZIM --eye ${FRONT_EYE:-0.53 -0.681 0.984} --target ${FRONT_TARGET:-0.29 -0.87 0.90}"
fi

restart_renderer() {
  docker exec parallax_sim_fp bash -lc "pkill -9 -f '[d]alus_sim_app'; sleep 3; \
    rm -f /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
          /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer*"
  docker exec -d parallax_sim_fp bash -lc \
    'cd /root/parallax/DalusSimCore && python3 dalus_sim_app.py > /tmp/render.log 2>&1'
  sleep 8
  # the app recreates its SHM mode-600 root; unprivileged clients need 666 (45min keeper)
  docker exec -d parallax_sim_fp bash -lc 'for i in $(seq 1 1350); do \
    chmod 666 /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
              /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer* 2>/dev/null; \
    sleep 2; done'
}

render_one() {   # $1 = tag, $2 = jack ply
  local tag=$1 ply=$2
  local out=$DBGROOT/${EP}_$tag
  echo "[jackdbg] === $tag -> $out"
  echo "[jackdbg]     jack ply: $ply"
  [ -f "$ply" ] || { echo "[jackdbg]     MISSING, skipped"; return 1; }
  mkdir -p "$out"
  restart_renderer
  .venv/bin/python scripts/record_sbot_scene_gs_cable.py \
      --eef-traj "$traj" --eef-rpy $EEF_RPY \
      --jack-pos $JACK_POS --jack-align-rpy $JACK_ALIGN_RPY --conn-rpy $CONN_RPY \
      --jack-anchor $JACK_ANCHOR --jack-ply "$ply" \
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
      --wrist-back $WRIST_BACK --wrist-aim-back $WRIST_AIM_BACK --wrist-cam-from-usd --wrist-rigid \
      $CAM_FLAGS --gamma ${RENDER_GAMMA:-1.8} \
      --grasped-only \
      --width ${WIDTH:-640} --height ${HEIGHT:-480} \
      --dump "$out" --dump-size $DUMP_SIZE --no-preview \
      --out "$out/_preview" > "$out.log" 2>&1
  if [ -f "$out/state.npy" ]; then
      echo "[jackdbg]     OK: $(ls "$out"/image/*.png 2>/dev/null | wc -l) front + $(ls "$out"/wrist_image/*.png 2>/dev/null | wc -l) wrist frames"
      grep -h "\[calib\]\|\[scene\] jack\|\[scene\] .* splats" "$out.log" | sed 's/^/[jackdbg]     /'
  else
      echo "[jackdbg]     FAILED -- see $out.log"; tail -5 "$out.log" | sed 's/^/[jackdbg]     /'
  fi
}

mkdir -p "$DBGROOT"
if [ -n "$ROLLS" ]; then
  for r in $ROLLS; do
    ply=$ETH/jack_v2_roll${r}.ply
    .venv/bin/python tools/bake_jack_v2.py --out "$ply" --roll-deg "$r" | sed 's/^/[jackdbg] bake /'
    render_one "v2roll$r" "$ply"
  done
else
  render_one "${TAG:-v2}" "$JACK_PLY"
fi
[ "$AB" = "1" ] && render_one "prod_x1p3" "$PROD_JACK"

echo "[jackdbg] done. renders under $DBGROOT/"
echo "[jackdbg] compare front frames:  eog $DBGROOT/${EP}_*/image/frame_0000.png"
echo "[jackdbg] compare wrist frames:  eog $DBGROOT/${EP}_*/wrist_image/frame_0000.png"
echo "[jackdbg] NOTE the renderer is left running on the LAST jack ply rendered -- restart it"
echo "[jackdbg] before resuming datagen (tools/datagen_dr_jack.sh does that itself per block)."
