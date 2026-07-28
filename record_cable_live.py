"""LIVE cable insertion: Newton physics + PPO policy driving the GS renderer in one process.

This is the third mode, alongside the two offline ones:

  rl/record_cable_env.py        policy -> rerun .rrd            (geometry only, no images)
  gen_cable_traj + batch_v4     policy -> .npy -> GS replay      (datagen; two passes)
  record_cable_live.py  (this)  policy -> Newton -> GS, live     (debug/demo; one pass)

Why it exists: the replay path has to re-derive every pose through a seat-frame transform
(--eef-rpy) and pick which recorded channel drives which splat -- a whole class of calibration
bugs. Here every splat rides its ACTUAL Newton body, so no world transform is needed at all.
The only calibrations left are splat->body-frame (--conn-rpy / --jack-rpy), which are the same
values the replay path uses because they describe the splat asset, not the scene.

    .venv/bin/python record_cable_live.py --checkpoint rl/runs/cable_v3/best_model.pt \
        --stage 4 --cable-tilt 5 --steps 130 --out live_out

Needs the DalusSimCore renderer UP (see tools/render_batch_v4.sh header) and sudo (SHM).
Splat count differs from the batch path -> RESTART the renderer when switching between them.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "rl"))
sys.path.insert(0, _HERE)
# DalusPySim (the parallax_sim renderer client). render_batch_v4.sh exports this as PYTHONPATH,
# but `sudo` strips the environment on a direct invocation -- and this script is ALWAYS run under
# sudo (SHM is root-owned). So add it here rather than relying on the caller's env.
_DALUS = os.environ.get("DALUS_PYSIM",
                        "/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim")
if os.path.isdir(_DALUS) and _DALUS not in sys.path:
    sys.path.insert(0, _DALUS)

from cable_env import CableInsertVecEnv  # noqa: E402
from train_ppo import ActorCritic  # noqa: E402
from newton_cabling.render.gs_bridge import (  # noqa: E402
    NewtonGSClient, euler_deg_to_quat_wxyz, look_at_quat, make_intrinsics, newton_pose,
    ply_centroid, quat_mul_wxyz, quat_rotate_wxyz,
)

DEV = "cuda:0"
HOST_PARALLAX = "/home/pandaliza/parallax"
CONTAINER_PARALLAX = "/root/parallax"
HOST_SBOT_GS = f"{HOST_PARALLAX}/parallax-demo-isaac-lab/assets/sbot_gs/flat"
HOST_GRIPPER_CUT = f"{HOST_PARALLAX}/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_v3"
HOST_WRIST3 = f"{HOST_PARALLAX}/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link_realpalm.ply"
HOST_GSVLA = f"{HOST_PARALLAX}/gs-sim-vla/scene/assets"

# same order/---names as record_sbot_scene_gs_cable.py; base_link is the fixed root (static).
LINK_SPLATS = (
    "base_link", "shoulder_link", "upper_arm_link", "forearm_link",
    "wrist_1_link", "wrist_2_link", "wrist_3_link",
    "finger1_knuckle_link", "finger1_inner_knuckle_link",
    "finger1_finger_link", "finger1_finger_tip_link",
    "finger2_knuckle_link", "finger2_inner_knuckle_link",
    "finger2_finger_link", "finger2_finger_tip_link",
)


def host_to_container(p):
    return str(p).replace(HOST_PARALLAX, CONTAINER_PARALLAX, 1)


def static_pose(world_pos, quat_wxyz, centroid):
    """Seat a splat's native centroid at world_pos (same helper as the replay renderer)."""
    pos = np.asarray(world_pos, float) - quat_rotate_wxyz(quat_wxyz, centroid)
    return pos.tolist(), list(quat_wxyz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="rl/runs/cable_v3/best_model.pt",
                    help="cable PPO checkpoint; omit for the scripted servo teacher")
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--cable-tilt", type=float, nargs="+", default=[5.0],
                    help="droop deg; ONE value = fixed, TWO = per-env uniform DR")
    ap.add_argument("--steps", type=int, default=130)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="dir to save rendered PNGs (default: none, viewer only)")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--show-viewer", action="store_true", help="live window (needs xhost +local:)")
    ap.add_argument("--dry-run", action="store_true", help="no renderer; validates poses only")
    ap.add_argument("--rrd", default=None,
                    help="also record the REAL Newton geometry to this .rrd (+ .rbl blueprint) — "
                         "actual meshes in the rerun viewer, driven by the SAME live physics as the "
                         "GS frames, so the two are synced step-for-step. Default: <out>/rollout.rrd")
    ap.add_argument("--no-rrd", action="store_true", help="skip the .rrd recording")
    ap.add_argument("--movie", default=None,
                    help="encode the saved GS frames into this .mp4 (needs ffmpeg). "
                         "Default: <out>/rollout.mp4 when --out is set")
    ap.add_argument("--no-movie", action="store_true", help="skip movie encoding")
    ap.add_argument("--fps", type=float, default=30.0, help="rrd/movie playback rate")
    # splats
    ap.add_argument("--connector-ply", default=f"{HOST_GSVLA}/objects/ethernet/cropped_plug_head.ply")
    ap.add_argument("--connector-tail-ply",
                    default=f"{HOST_GSVLA}/objects/ethernet/cropped_plug_tail_longer.ply",
                    help="boot/cable-exit splat; longer variant (20.2mm vs 11.4mm), same plug frame")
    ap.add_argument("--jack-ply", default=f"{HOST_GSVLA}/objects/ethernet/cad_jack_registered.ply")
    ap.add_argument("--gripper-gs-dir", default=HOST_GRIPPER_CUT)
    ap.add_argument("--wrist3-ply", default=HOST_WRIST3)
    ap.add_argument("--bg-ply", default=None, help="optional room splat")
    # splat->body calibrations: SAME values as the replay path (they describe the splat asset,
    # not the scene), which is exactly why the live path needs no world transform.
    ap.add_argument("--conn-rpy", type=float, nargs=3, default=[-90.0, 0.0, 0.0])
    ap.add_argument("--conn-anchor", type=float, nargs=3, default=[-0.0015, -0.0015, 0.0172])
    ap.add_argument("--jack-rpy", type=float, nargs=3, default=[90.0, 0.0, -180.0])
    ap.add_argument("--jack-anchor", type=float, nargs=3, default=[0.0, 0.0, 0.030])
    # camera: placed RELATIVE to the live jack, reproducing the v3 FRONT view geometry
    # (v3: jack [0.295,-0.876,0.835], eye [0.53,-1.071,0.984], target [0.29,-0.87,0.90]).
    ap.add_argument("--eye-off", type=float, nargs=3, default=[0.235, -0.195, 0.149])
    ap.add_argument("--target-off", type=float, nargs=3, default=[-0.005, 0.006, 0.065])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fov", type=float, default=55.0)
    ap.add_argument("--gamma", type=float, default=1.8)
    args = ap.parse_args()

    tilt = tuple(args.cable_tilt) if len(args.cable_tilt) > 1 else args.cable_tilt[0]
    env = CableInsertVecEnv(1, seed=args.seed, cable_tilt_deg=tilt)
    env.set_stage(args.stage)
    ac = None
    if args.checkpoint:
        ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
        ac.load_state_dict(torch.load(args.checkpoint, map_location=DEV))
        ac.eval()
        print(f"[live] policy {args.checkpoint}")
    else:
        print("[live] scripted servo teacher")

    labels = list(env.model.body_label)

    def _sfx(s):
        hits = [i for i, l in enumerate(labels) if l.endswith(s)]
        return hits[0] if hits else None

    link_idx = [_sfx(n) for n in LINK_SPLATS]
    missing = [n for n, i in zip(LINK_SPLATS, link_idx) if i is None]
    jb = int(env.jack_body[0])

    def _link_ply(n):
        if n == "wrist_3_link" and args.wrist3_ply:
            return args.wrist3_ply
        return f"{(args.gripper_gs_dir if 'finger' in n else HOST_SBOT_GS)}/{n}.ply"

    tail = args.connector_tail_ply if str(args.connector_tail_ply).lower() not in ("none", "") else None
    # splat[0] is treated as a STATIC background by the renderer -> put the jack first (it is
    # static within an episode); everything after it gets a live per-frame transform.
    ply_hosts = [args.jack_ply, args.connector_ply] + ([tail] if tail else []) \
        + [_link_ply(n) for n in LINK_SPLATS]
    obj_plys = [host_to_container(p) for p in ply_hosts]

    # RESET FIRST: jacks are parked off-scene (z=-5) until reset() places them for the episode,
    # so reading the jack pose (and anchoring the camera to it) before this gives garbage.
    obs = env.reset()
    bqn = env.state_0.body_q.numpy()
    jack_p = bqn[jb, :3].copy()
    conn_align = euler_deg_to_quat_wxyz(*args.conn_rpy)
    jack_align = euler_deg_to_quat_wxyz(*args.jack_rpy)
    conn_anchor = np.array(args.conn_anchor, float)
    jack_anchor = np.array(args.jack_anchor, float)

    eye = jack_p + np.array(args.eye_off, float)
    target = jack_p + np.array(args.target_off, float)
    cam_K = make_intrinsics(args.width, args.height, args.fov)
    cam_quat = look_at_quat(eye.tolist(), target.tolist())

    print(f"[live] {len(obj_plys)} splats: jack, connector{'+tail' if tail else ''} + "
          f"{len(LINK_SPLATS)} links   (base_link static: {link_idx[0] is None})")
    if missing:
        print(f"[live] NOTE: no body for {missing} -> held at their t=0 pose (base_link is the "
              f"fixed root, this is expected)")
    print(f"[live] jack @ {np.round(jack_p,3)}  eye {np.round(eye,3)} -> target {np.round(target,3)}")

    # The renderer treats splat[0] as a STATIC background -- it does NOT get a live per-frame
    # transform. NewtonGSClient only prepends bg_pose when a background exists, so with no
    # --bg-ply the JACK would land at index 0 and freeze at its registration pose (this is
    # exactly why the jack was invisible in the first live run). Synthesize an invisible
    # 1-gaussian background when none is given, so every real splat gets a live transform --
    # the same trick record_sbot_scene_gs_cable.py uses for --gripper-only.
    bg_host = args.bg_ply
    if not bg_host:
        bg_host = os.path.join(HOST_GSVLA, "objects", "_live_dummy_bg.ply")
        _fp = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
               "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
        _db = np.zeros((1, 14), "<f4")
        _db[0, 0:3] = [0.0, 0.0, 50.0]; _db[0, 3:6] = -6.0      # far away, tiny
        _db[0, 6] = 1.0; _db[0, 13] = -30.0                     # identity rot, ~zero opacity
        with open(bg_host, "wb") as f:
            f.write(("ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                     + "".join(f"property float {p}\n" for p in _fp) + "end_header\n").encode())
            f.write(_db.tobytes())
        print("[live] no --bg-ply: using an invisible dummy background so the jack "
              "(splat[0]) gets a LIVE transform instead of freezing")

    client = NewtonGSClient(
        ply_paths=obj_plys, cam_K=cam_K, cam_pos=eye.tolist(), cam_quat=cam_quat,
        bg_ply=host_to_container(bg_host),
        show_viewer=args.show_viewer, dry_run=args.dry_run,
    )

    # base_link is the fixed root -- no body_q row -- so pose it from cable_env's build transform
    # (env 0 base = (0, 0, 0.6), identity; see cable_env.py's add_sbot call).
    static0 = {n: ([0.0, 0.0, 0.6], [1.0, 0.0, 0.0, 0.0])
               for n, i in zip(LINK_SPLATS, link_idx) if i is None}

    def scene_poses(bq):
        fp, fq = env._face_pose(bq)
        fpw = fp[0]
        fqw = fq[0][[3, 0, 1, 2]]                                   # xyzw -> wxyz
        poses = [
            static_pose(bq[jb, :3], quat_mul_wxyz(list(bq[jb, 3:7][[3, 0, 1, 2]]),
                                                  list(jack_align)), jack_anchor),
            static_pose(fpw, quat_mul_wxyz(list(fqw), list(conn_align)), conn_anchor),
        ]
        if tail:
            poses.append(poses[1])                                  # boot rides the head pose
        for n, i in zip(LINK_SPLATS, link_idx):
            poses.append(static0[n] if i is None else newton_pose(bq, i))
        return poses

    def save_frame(rgb, t):
        """Write one GS frame as PNG (gamma-corrected)."""
        from PIL import Image  # noqa: PLC0415
        a = np.clip(np.asarray(rgb, np.float32), 0, 1)
        if args.gamma != 1.0:
            a = a ** (1.0 / args.gamma)
        view = a if a.ndim == 3 else a[0]
        Image.fromarray((view * 255).astype("uint8")).save(
            os.path.join(args.out, f"frame_{t:04d}.png"))

    # REAL Newton geometry -> .rrd (actual meshes in the rerun viewer). Same physics loop as the
    # GS frames, so the two recordings are synced step-for-step.
    rrd_path = None
    viewer = None
    if not args.no_rrd:
        rrd_path = args.rrd or (os.path.join(args.out, "rollout.rrd") if args.out else None)
        if rrd_path:
            os.makedirs(os.path.dirname(os.path.abspath(rrd_path)), exist_ok=True)
            sys.path.insert(0, _HERE)
            from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: PLC0415
            viewer = open_rrd_recorder(rrd_path)
            viewer.set_model(env.model)
            print(f"[live] recording Newton geometry -> {rrd_path}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
    # Is the jack actually inside the camera frustum? A jack that renders "missing" is usually
    # just out of view or behind the camera, not un-posed.
    _v = jack_p - eye
    _fwd = target - eye
    _cos = float(np.dot(_v, _fwd) / (np.linalg.norm(_v) * np.linalg.norm(_fwd) + 1e-9))
    print(f"[live] jack vs camera: dist {np.linalg.norm(_v):.3f}m, "
          f"off-axis {np.degrees(np.arccos(np.clip(_cos,-1,1))):.1f}deg "
          f"(fov {args.fov:.0f} -> half {args.fov/2:.0f}deg)"
          + ("  <== OUTSIDE the frustum, jack will not be visible" if np.degrees(np.arccos(np.clip(_cos,-1,1))) > args.fov / 2 else ""))
    seated_at = None                      # NOTE: env.reset() already ran (before the camera setup)
    import time as _t
    prof = {"physics": 0.0, "render": 0.0, "io": 0.0}
    for t in range(args.steps):
        t0 = _t.perf_counter()
        poses = scene_poses(env.state_0.body_q.numpy())
        t1 = _t.perf_counter()
        rgb = client.render(poses)
        t2 = _t.perf_counter()
        if args.out and t % args.save_every == 0 and not args.dry_run:
            save_frame(rgb, t)
        if viewer is not None:                       # real Newton meshes, same step
            viewer.begin_frame(t / args.fps)
            viewer.log_state(env.state_0)
            viewer.end_frame()
        t3 = _t.perf_counter()
        with torch.no_grad():
            a_ = env.servo_action() if ac is None else ac.mean_action(obs)
        obs, _, _, _, _ = env.step(a_)
        t4 = _t.perf_counter()
        prof["render"] += t2 - t1
        prof["io"] += t3 - t2
        prof["physics"] += (t1 - t0) + (t4 - t3)
        if seated_at is None and int(env.hold[0]) >= 20:
            seated_at = t
            print(f"[live] SEATED (hold 20) at step {t}")
        if t % 20 == 0:
            fp, fq = env._face_pose(env.state_0.body_q.numpy())
            print(f"[live] step {t:4d}  conn->seat {np.linalg.norm(fp[0]-env.seat_pos[0])*1000:6.1f}mm  "
                  f"hold {int(env.hold[0]):2d}", flush=True)
    n = max(args.steps, 1)
    print(f"[live] done. seated_at={seated_at}  "
          f"physics {prof['physics']/n*1000:.1f}ms/fr | render {prof['render']/n*1000:.1f}ms/fr | "
          f"io {prof['io']/n*1000:.1f}ms/fr  -> {n/sum(prof.values()):.1f} FPS")

    if rrd_path:
        rbl = os.path.splitext(rrd_path)[0] + ".rbl"
        auto_blueprint(rbl, env.model)
        print(f"[live] Newton geometry -> {rrd_path}\n"
              f"[live]   view:  uvx --from rerun-sdk rerun {rrd_path} {rbl}")

    if args.out and not args.dry_run and not args.no_movie:
        import glob  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        frames = sorted(glob.glob(os.path.join(args.out, "frame_*.png")))
        mp4 = args.movie or os.path.join(args.out, "rollout.mp4")
        if not frames:
            print("[live] no frames to encode")
        elif shutil.which("ffmpeg") is None:
            print(f"[live] {len(frames)} frames -> {args.out} (ffmpeg not found; skipping movie. "
                  f"Frames are numbered by SIM STEP, so with --save-every>1 they are not "
                  f"consecutive -- encode with: ffmpeg -framerate {args.fps:.0f} -pattern_type "
                  f"glob -i '{args.out}/frame_*.png' -pix_fmt yuv420p {mp4})")
        else:
            # glob pattern, not %04d: --save-every leaves gaps in the frame numbering
            cmd = ["ffmpeg", "-y", "-framerate", f"{args.fps:.0f}", "-pattern_type", "glob",
                   "-i", os.path.join(args.out, "frame_*.png"),
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", mp4]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"[live] {len(frames)} frames -> {mp4}")
            else:
                print(f"[live] ffmpeg failed (frames kept in {args.out}):\n"
                      f"{r.stderr.strip().splitlines()[-3:]}")


if __name__ == "__main__":
    main()
