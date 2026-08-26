#!/usr/bin/env bash
# 500-episode cable datagen with JACK-SCALE domain randomization (2026-07-31 grip series).
#
# Stage A  trajectories: newton_cabling.scripted_controller.gen_trajectories at
#          GRIP_FROM_HEAD (default 55mm), seed SEED, rounds until N_EPS are saved
#          (hold rate is ~25%: 16 envs/round -> ~4 saved/round).
# Stage B  GS render: splits [0, N_EPS) into one contiguous block per scale in SCALES
#          and runs tools/render_batch_v4.sh per block with JACK_PLY pointing at the
#          mouth-pivoted bake for that scale (1.0 = the stock cad_jack_registered.ply).
#          Blocks, not per-episode draws: the renderer caches splats by COUNT and
#          silently serves stale content on a same-count file swap, so every scale
#          change REQUIRES a renderer restart -- 4 restarts total this way. Downstream
#          training shuffles episodes, so block assignment == random assignment.
#          Per-episode scale labels land in $OUTROOT/_jack_scales.tsv.
#
# The renderer restart + SHM permission dance is the NO-SUDO recipe (docker group
# suffices): the app creates its SHM objects mode-600, so a chmod keeper loop runs in
# the container for the lifetime of each block.
#
# GPU note: shares the 5090 with any eval/policy servers -- do not run during evals.
# Resume: re-running is safe; gen is skipped when TRAJROOT already has >= N_EPS
# episodes, and render_batch_v4 skips episodes whose dump completed.
#
#     bash tools/datagen_dr_jack.sh                 # full run (~3-5h)
#     DRY_RUN=1 bash tools/datagen_dr_jack.sh       # print the plan, run nothing
#     SKIP_GEN=1 bash tools/datagen_dr_jack.sh      # trajectories already generated
set -u
REPO=/home/pandaliza/parallax/newton-cabling
DATA=/home/pandaliza/parallax/data/vla_train
N_EPS=${N_EPS:-500}
SEED=${SEED:-0}
GRIP_FROM_HEAD=${GRIP_FROM_HEAD:-55}
SCALES=${SCALES:-"1.0 1.1 1.2 1.3"}
TRAJROOT=${TRAJROOT:-$DATA/cable_traj_head${GRIP_FROM_HEAD}_500}
OUTROOT=${OUTROOT:-$DATA/datagen_head${GRIP_FROM_HEAD}_drjack}
ROUNDS=${ROUNDS:-$((N_EPS / 2))}       # ~4 saved/round; 2x margin, gen stops at --max-save
ETH=$REPO/newton_cabling/assets/ethernet
CONNECTOR_USD=${CONNECTOR_USD:-scan_rj45.usd}   # sim connector for gen_trajectories
STOCK_JACK=${JACK_BASE_PLY:-/home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/ethernet/cad_jack_registered.ply}
PLUG_HEAD=${PLUG_HEAD:-$ETH/headA_plug_roll210.ply}

run() { if [ "${DRY_RUN:-0}" = "1" ]; then echo "+ $*"; else "$@"; fi; }

cd "$REPO"

# ── Stage A: trajectories ─────────────────────────────────────────────────────────
have=$(ls -d "$TRAJROOT"/ep_* 2>/dev/null | wc -l)
if [ "${SKIP_GEN:-0}" = "1" ] || [ "$have" -ge "$N_EPS" ]; then
  echo "[dr] gen skipped ($have episodes already in $TRAJROOT)"
else
  echo "[dr] generating $N_EPS trajectories (grip $GRIP_FROM_HEAD mm, seed $SEED)"
  run .venv/bin/python -m newton_cabling.scripted_controller.gen_trajectories \
      --out "$TRAJROOT" --envs 16 --rounds "$ROUNDS" --steps 160 --max-keep 160 \
      --no-cut-at-success --connector-usd "$CONNECTOR_USD" --grasp-roll-180 \
      --max-save "$N_EPS" --seed "$SEED" --grip-from-head "$GRIP_FROM_HEAD" \
      --keep-existing
fi

# ── Stage B: render blocks, one jack scale each ───────────────────────────────────
restart_renderer() {
  run docker exec parallax_sim_fp bash -lc "pkill -9 -f '[d]alus_sim_app'; sleep 3; \
    rm -f /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
          /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer*"
  run docker exec -d parallax_sim_fp bash -lc \
    'cd /root/parallax/DalusSimCore && python3 dalus_sim_app.py > /tmp/render.log 2>&1'
  run sleep 8
  # 8h chmod keeper: the app recreates SHM mode-600 root; unprivileged clients need 666
  run docker exec -d parallax_sim_fp bash -lc 'for i in $(seq 1 14400); do \
    chmod 666 /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
              /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer* 2>/dev/null; \
    sleep 2; done'
}

mkdir -p "$OUTROOT"
nsc=$(echo $SCALES | wc -w)
i=0
: > "$OUTROOT/_jack_scales.tsv"
for s in $SCALES; do
  lo=$((i * N_EPS / nsc)); hi=$(((i + 1) * N_EPS / nsc)); i=$((i + 1))
  if [ "$s" = "1.0" ]; then
    ply=$STOCK_JACK
  else
    ply=$ETH/cad_jack_registered_x$(echo "$s" | tr '.' 'p').ply
    [ -f "$ply" ] || run .venv/bin/python tools/scale_jack_splat.py --scale "$s" --out "$ply"
  fi
  for e in $(seq "$lo" $((hi - 1))); do printf "ep_%04d\t%s\n" "$e" "$s"; done >> "$OUTROOT/_jack_scales.tsv"
  echo "[dr] block [$lo,$hi): jack x$s  ($ply)"
  restart_renderer
  run env ALLOW_NONCANONICAL_EEF_RPY=1 TRAJROOT="$TRAJROOT" OUTROOT="$OUTROOT" \
      EEF_RPY="0 0 180" PLUG_HEAD="$PLUG_HEAD" PLUG_TAIL=none \
      RENDER_GAMMA=1.8 FRONT_EYE="0.53 -0.681 0.984" JACK_PLY="$ply" \
      bash tools/render_batch_v4.sh "$lo" "$hi"
done

# ── Stage C: Difix (SKIP_DIFIX=1 to leave raw renders only) ───────────────────────
if [ "${SKIP_DIFIX:-0}" != "1" ]; then
  echo "[dr] difixing -> ${OUTROOT}_difix"
  # needs ~7GB free on the 5090: stop idle difix_server/eval processes first
  run bash -c "cd /home/pandaliza/parallax/Difix3D && .venv/bin/python difix_datagen.py \
      --src '$OUTROOT' --dst '${OUTROOT}_difix' --target $N_EPS" \
    || { echo "[dr] DIFIX FAILED -- rerun this stage with SKIP_GEN=1 after freeing GPU memory"; exit 1; }
fi

echo "[dr] done. renders in $OUTROOT, difixed in ${OUTROOT}_difix (scales: _jack_scales.tsv)"
echo "[dr] next (openpi venv): cd ~/parallax/openpi && uv run python \\"
echo "     ~/parallax/newton-cabling/tools/datagen_to_lerobot.py \\"
echo "     --raw ${OUTROOT}_difix --repo_id parallax/<name> --truncate_after_seat -1"
