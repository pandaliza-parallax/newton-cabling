# sbot GS rollout — RO1 arm inserting an RJ45 connector (Gaussian-Splat render)

Renders a **visual rollout** of the StandardBots RO1 arm carrying an ethernet plug and
inserting it into a jack on a table, composited through the `parallax_sim` Gaussian-Splat
renderer, with the arm driven by IK to follow a recorded plug trajectory.

Output is a stitched multi-camera `.mp4` + per-frame PNGs.

---

## Quick start (the working 3-camera command)

```bash
cd /home/pandaliza/parallax/newton-cabling
sudo PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim \
  .venv/bin/python scripts/record_sbot_scene_gs.py \
    --plug-traj seated_traj/ep_0000/plug_traj.npy \
    --mount --base-yaw-deg 180 --base-pos 0.295 -0.60 1.30 --grasp-rpy 0 -90 0 \
    --gripper-gs-dir /home/pandaliza/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_col \
    --bg-ply /home/pandaliza/parallax/gs-sim-vla/scene/assets/background/splat.ply \
    --frame table --azim -90 --elev 15 --side-cam --side-elev 6 --wrist-cam --wrist-orbit 90 \
    --out out_seated_scene_3cam
```

→ `out_seated_scene_3cam/sbot_scene.mp4` (3 cameras stitched) + `frame_*.png`.

**Requires `sudo`**: the `parallax_sim_fp` container runs as root, so its `/dev/shm` SHM
segments are root-owned; a non-root client gets `No permission to access this segment`.
sudo here is **not** passwordless — run it yourself.

### Why these flags
| flag | why |
|---|---|
| `--base-pos 0.295 -0.60 1.30 --mount` | arm on a pedestal ~30 cm behind the table, raised so the **structural** links clear the tabletop (forearm/upper-arm no longer punch through). |
| `--grasp-rpy 0 -90 0` | the real fix for table penetration: the hand approaches the plug **from above** → wrist_3 + fingers stay above the surface (only the plug tip touches). Keeps a horizontal insertion. IK stays 0.0 mm. |
| `--gripper-gs-dir .../gripper_col` | recolored gripper splats (so the AG-145 reads correctly, not rainbow). |
| `--frame table --azim -90 --elev 15` | main camera: front-on, low, framed on the table. |
| `--side-cam --side-elev 6` | static +x side camera. |
| `--wrist-cam --wrist-orbit 90` | eye-in-hand camera on `wrist_3`. |

---

## How it works

`scripts/record_sbot_scene_gs.py`:
1. loads `plug_traj.npy` `(T,7)` = per-step `[pos3, quat4 wxyz]` plug pose in the **socket frame**;
2. builds the RO1 (+ AG-145 gripper) in Newton, solves **IK** each frame so the grasp point tracks
   the mapped plug pose (`newton.ik`, position+rotation objectives on `wrist_3`);
3. maps every object's world pose and sends them + the camera(s) to the GS renderer over SHM;
4. saves the stitched frames/mp4.

**Connector pose each frame** = `jack_pos + R(jack_q)·traj[i][:3]`, quat
`jack_q ∘ traj[i][3:7] ∘ conn_align`. Frame 0 = start (~10 mm out), frame −1 = seated.

---

## Files that matter

**Entry point**
- `scripts/record_sbot_scene_gs.py` — the render/IK driver above.

**Render / sim library**
- `newton_cabling/render/gs_bridge.py` — `NewtonGSClient` (SHM to the renderer),
  `newton_pose`, `place_on_body`, `look_at_quat`, `make_intrinsics`, multi-camera
  (`extra_cameras` static + `render(dyn_cameras=)` per-frame, `compose_multicam`).
- `newton_cabling/sim/sbot.py` — `add_sbot`, `SBOT_HOME`, gripper coupling, PD gains.
- `newton_cabling/sim/safe_vbd.py` — VBD builder/solver; `newton_cabling/sim/recording.py` — optional `.rrd`.

**Trajectory source (`plug_traj.npy`)**
- `rl/gen_seated_traj.py` — **seated PPO** trajectories → `seated_traj/ep_*/plug_traj.npy`
  (pure physics, **no renderer/sudo**). Uses `rl/connector_env.py`, `rl/train_ppo.py`.
- `rl/record_policy.py --eval-vla` — the VLA producer → `v1_eval/ep_*/plug_traj.npy` (needs renderer).

**Splat assets**
- Scene: `gs-sim-vla/scene/assets/objects/{table/splat.ply, ethernet/cad_jack_registered.ply,
  ethernet/cad_plug_registered.ply}`, `.../background/splat.ply`.
- Robot: `parallax-demo-isaac-lab/assets/sbot_gs/flat/*.ply` (7 arm + 8 gripper),
  recolored gripper in `.../sbot_gs/gripper_col/`.
- Helpers: `tools/recolor_gripper.py`, `tools/strip_arm_sh.py`.

**Isolated debug viz** (no arm/IK): `scripts/viz_jack_connector.py` — jack + connector on the table only.

---

## Generating trajectories

```bash
# seated PPO rollouts (physics only, no renderer) -> seated_traj/ep_0000..0004
.venv/bin/python rl/gen_seated_traj.py --random-easy --asset cad_rj45   # loadable ckpts: the obs=12,act=6 cad_* runs
```
Notes:
- Loadable PPO checkpoints are the `obs=12, act=6` `cad_*` runs (e.g. `rl/runs/cad_res1p0_mu05_to5k`,
  `cad_dr_all3_2k`, ~94% seat). The older rj45 runs (`rl/runs/extend`, `ppo`, …) are **stale** (obs=9, won't load).
- `cad_rj45` matches the scene's `cad_plug_registered.ply` splat.
- All seated episodes end ~`y=+11.8 mm`, lateral ~0.3 mm.

---

## Operational gotchas

**Renderer restart is required whenever the splat COUNT/set changes** (e.g. toggling
`--mount` / `--bg-ply`, or switching between the ~4-splat `record_policy` eval scene and the
18-splat `record_sbot_scene_gs` scene). Symptom: handshake OK but the first render times out;
`tail /tmp/render.log` shows `Gaussian splat count changed (current=… incoming=…)`.

Clean restart recipe (kill renderer, unlink all 8 SHM objects, restart):
```bash
docker exec parallax_sim_fp bash -lc "pkill -9 -f '[d]alus_sim_app'; sleep 3; \
  rm -f /dev/shm/dal_buffer* /dev/shm/send_dal_buffer* \
        /dev/shm/sem.dal_sem_buffer* /dev/shm/sem.send_dal_sem_buffer*; \
  ls /dev/shm/ | grep -iE 'dal|send' || echo CLEAN"
docker exec -d parallax_sim_fp bash -lc 'cd /root/parallax/DalusSimCore && python3 dalus_sim_app.py > /tmp/render.log 2>&1'
```
- **Use the `[d]alus_sim_app` bracket trick** — a plain `pkill -f dalus_sim_app` matches (and kills)
  the cleaning shell itself before the `rm` runs.
- Handshake normally completes on **attempt 2**; the `WARN: RECEIVER SHM not setup yet` on attempt 1 is benign.
- Cameras can be added/changed **without** a restart (splat count unchanged; cameras go per-UPDATE).

**Table height**: the wood splat renders larger than its point bounds — the visible top is
~`z=1.45` (not the `ply_bounds` 0.785). Parts/reach targets should sit on the visible surface.
`--table-top-z` overrides it.

**Debug without a render**: `--dry-run` (+ `--diag` for per-link tabletop penetration after each
IK solve, or `--no-arm-ik` to park the arm at home and just follow the connector).

See `~/.claude/.../memory/sbot-scene-table-connector.md` and `gssim-renderer-shm-handshake.md`
for the full debugging history.
