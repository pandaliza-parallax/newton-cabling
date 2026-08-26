#!/usr/bin/env bash
# Incremental difix: process tier-2 chunks that are fully rendered AND fully
# gamma'd, into the pipeline's final_difix tree (its own difix stage then skips
# them). Stands down when the pipeline's difix stage starts.
set -u
SRC=/home/pandaliza/parallax/data/vla_train/tier2_recovery
FINAL=/home/pandaliza/parallax/data/vla_train/tier2_recovery_final
DIFIX=/home/pandaliza/parallax/data/vla_train/tier2_recovery_final_difix
while ! grep -q "=== difix" /home/pandaliza/tier2_pipeline.log 2>/dev/null; do
  for c in chunk_00 chunk_01 chunk_02 chunk_03 chunk_04 chunk_05 chunk_06 chunk_07; do
    ns=$(find $SRC/$c -maxdepth 1 -type d -name "ep_*" 2>/dev/null | wc -l)
    nf=$(find $FINAL/$c -maxdepth 1 -type d -name "ep_*" 2>/dev/null | wc -l)
    nd=$(find $DIFIX/$c -maxdepth 1 -type d -name "ep_*" 2>/dev/null | wc -l)
    if [ "$ns" -gt 0 ] && [ "$nf" -eq "$ns" ] && [ "$nd" -lt "$ns" ]; then
      echo "[inc-difix] $c: $nd/$ns done, running $(date +%H:%M)"
      ( cd /home/pandaliza/parallax/Difix3D && .venv/bin/python difix_datagen.py \
          --src $FINAL/$c --dst $DIFIX/$c --target 999 ) || echo "[inc-difix] $c FAILED"
    fi
  done
  sleep 600
done
echo "[inc-difix] pipeline difix stage started; stopping"
