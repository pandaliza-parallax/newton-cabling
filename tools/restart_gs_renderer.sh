#!/usr/bin/env bash
# Restart the DalusSimCore Gaussian-Splat renderer and VERIFY it actually restarted.
#
# WHY THIS EXISTS: the renderer caches the splat set from its first client and compares only
# the COUNT. Swapping one splat file for another of the same count is silently ignored --
# it keeps serving the OLD asset while every log line says the new one was requested. That
# produced pixel-identical renders across three different plug splats before it was caught
# by diffing the images. So: restart before EVERY splat change, and confirm by PID.
#
# Two traps this guards against:
#   * `pkill -f dalus_sim_app` matches the `docker exec` command string itself -> it kills
#     its own wrapper and the app survives. The [d] bracket avoids the self-match.
#   * stale POSIX semaphores/SHM survive a kill and make the next handshake hang or need
#     retries ("handshake OK (attempt 2)"), so unlink them explicitly.
#
#     bash tools/restart_gs_renderer.sh          # restart + verify (exit 1 if it did not)
#     bash tools/restart_gs_renderer.sh --status # report only, change nothing
#
# Exits non-zero if the PID did not change, so it is safe to chain:
#     bash tools/restart_gs_renderer.sh && sudo ... bash tools/render_batch_v4.sh 0 1
set -uo pipefail

CONTAINER=${CONTAINER:-parallax_sim_fp}
APP_DIR=${APP_DIR:-/root/parallax/DalusSimCore}
WAIT_S=${WAIT_S:-8}

pid_of() { docker exec "$CONTAINER" pgrep -f '[d]alus_sim_app' 2>/dev/null | head -1; }

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "[gs] ERROR: container '$CONTAINER' is not running"
    exit 1
fi

OLD=$(pid_of)
if [ "${1:-}" = "--status" ]; then
    echo "[gs] container $CONTAINER up; renderer pid=${OLD:-<none>}"
    [ -n "$OLD" ] && docker exec "$CONTAINER" bash -lc \
        "ps -o etime= -p $OLD | sed 's/^/[gs] uptime:/'"
    docker exec "$CONTAINER" bash -lc \
        "grep -c 'ignoring re-initialization' /tmp/render.log 2>/dev/null \
         | sed 's/^/[gs] ignored re-inits since start: /'"
    exit 0
fi

echo "[gs] restarting renderer in $CONTAINER (old pid=${OLD:-<none>})"
# [d] bracket: stops the pattern matching this very `docker exec` command line
docker exec "$CONTAINER" bash -lc "pkill -9 -f '[d]alus_sim_app'; sleep 3; \
  rm -f /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
        /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer*" || true
docker exec -d "$CONTAINER" bash -lc \
  "cd $APP_DIR && python3 dalus_sim_app.py > /tmp/render.log 2>&1"

for _ in $(seq 1 "$WAIT_S"); do
    sleep 1
    NEW=$(pid_of)
    [ -n "$NEW" ] && [ "$NEW" != "$OLD" ] && break
done
NEW=$(pid_of)

if [ -z "$NEW" ]; then
    echo "[gs] FAILED: no renderer process after ${WAIT_S}s -- check:"
    echo "     docker exec $CONTAINER tail -40 /tmp/render.log"
    exit 1
fi
if [ "$NEW" = "$OLD" ]; then
    echo "[gs] FAILED: pid unchanged ($OLD) -- the old process survived the kill."
    echo "     Any render now would silently reuse the CACHED splat set."
    exit 1
fi
echo "[gs] OK: restarted, pid $OLD -> $NEW (splat cache cleared)"
