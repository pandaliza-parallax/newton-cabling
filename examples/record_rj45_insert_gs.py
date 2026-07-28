# Headless RJ45 insertion (Newton) rendered through DalusPySim's Gaussian-Splat
# renderer, instead of (well, in addition to) rerun.
#
# Same scripted timeline as record_rj45_insert.py:
#   0.0-1.5s settle | 1.5-4.5s insert +Y | 4.5-6.0s hold | 6.0-8.0s pull-back | 8.0-9.0s release
# Each frame we step the Newton sim, read the plug body's world pose, and send
# (socket, plug) transforms + a camera to the renderer; the returned RGB is saved.
#
# The cable is intentionally NOT rendered through GS (a deformable can't be one rigid
# splat); it still exists in the physics and is visible in the .rrd. The plug splat
# (cord+plug) rides the plug body rigidly.
#
# Run (renderer container must be up, ipc: host):
#   docker compose -f /path/to/DalusSimCore/docker-compose.yml up -d        # the renderer
#   cd newton-cabling
#   uv pip install posix_ipc                                                # one-time
#   PYTHONPATH=/root/parallax/DataGenerator/sim_engine/DalusPySim \
#     .venv/bin/python record_rj45_insert_gs.py --out out_gs
#
# Paths passed to --ply-* must be valid INSIDE the renderer container (e.g. /root/...).
# Quick checks: --smoke renders one frame; --dry-run skips the renderer entirely
# (still runs Newton, returns black frames) to validate the sim+packaging wiring.

from __future__ import annotations

import argparse
import os

import newton
import newton._src.viewer.viewer_rerun as viewer_rerun_module
import newton.examples
import newton.usd
import warp as wp
from newton.examples.contacts.example_contacts_rj45_plug import Example
from newton.viewer import ViewerRerun
from pxr import Usd

from newton_cabling.render.gs_bridge import (
    NewtonGSClient,
    look_at_quat,
    make_intrinsics,
    newton_pose,
)

# As in record_rj45_insert.py: pretend to be a notebook so ViewerRerun keeps the
# .rrd file sink instead of replacing it with a live server.
viewer_rerun_module.is_jupyter_notebook = lambda: True

FPS = 60
DURATION_SECONDS = 9.0
INSERT_DEPTH = 0.035
PULLBACK_DEPTH = -0.05

# Container-visible defaults (host ~/parallax is bind-mounted to /root/parallax).
ETH = "/root/parallax/gs-sim-vla/scene/assets/objects/ethernet"
DEFAULT_MOUNT_PLY = f"{ETH}/splat-mount.ply"
DEFAULT_CORD_PLY = f"{ETH}/edited-splat-cord.ply"  # cropped to just the plug (cable removed)


def target_offset_y(t: float) -> float:
    if t < 1.5:
        return 0.0
    if t < 4.5:
        return INSERT_DEPTH * (t - 1.5) / 3.0
    if t < 6.0:
        return INSERT_DEPTH
    if t < 8.0:
        return INSERT_DEPTH + (PULLBACK_DEPTH - INSERT_DEPTH) * (t - 6.0) / 2.0
    return PULLBACK_DEPTH


def socket_world_pos() -> list[float]:
    """World translation of /World/Socket in the bundled rj45 asset (static mount)."""
    stage = Usd.Stage.Open(newton.examples.get_asset("rj45_plug.usd"))
    prim = stage.GetPrimAtPath("/World/Socket")
    p = wp.transform_get_translation(newton.usd.get_transform(prim, local=False))
    return [float(p[0]), float(p[1]), float(p[2])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply-mount", default=DEFAULT_MOUNT_PLY, help="socket/mount splat (container path)")
    ap.add_argument("--ply-cord", default=DEFAULT_CORD_PLY, help="plug/cord splat (container path)")
    ap.add_argument("--bg-ply", default=None, help="optional background splat (container path)")
    ap.add_argument("--out", default="out_gs", help="output dir for frames + gif")
    ap.add_argument("--rrd", default="rj45_insertion_gs.rrd")
    ap.add_argument("--width", type=int, default=852)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--fov-deg", type=float, default=45.0)
    ap.add_argument("--eye", type=float, nargs=3, default=None, help="camera eye xyz (world)")
    ap.add_argument("--target", type=float, nargs=3, default=None, help="camera target xyz (world)")
    ap.add_argument("--duration", type=float, default=DURATION_SECONDS)
    ap.add_argument("--show-viewer", action="store_true", help="also open the renderer's pygame viewer")
    ap.add_argument("--smoke", action="store_true", help="render a single frame and exit")
    ap.add_argument("--dry-run", action="store_true", help="skip the renderer/SHM (still runs Newton)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    viewer = ViewerRerun(record_to_rrd=args.rrd, keep_historical_data=True)
    example = Example(viewer, args=None)
    rest = example._rest_pos
    plug_body = example._plug_body

    socket_pos = socket_world_pos()
    target = list(args.target) if args.target is not None else socket_pos
    eye = list(args.eye) if args.eye is not None else [
        socket_pos[0] + 0.12, socket_pos[1] - 0.12, socket_pos[2] + 0.06
    ]
    cam_K = make_intrinsics(args.width, args.height, args.fov_deg)
    cam_quat = look_at_quat(eye, target, up=(0.0, 0.0, 1.0), convention="ros")
    print(f"[demo] socket={socket_pos}  cam eye={eye} target={target}")

    # Object order: [mount(socket, static), cord(plug, dynamic)]. render() poses
    # must follow the same order.
    client = NewtonGSClient(
        ply_paths=[args.ply_mount, args.ply_cord],
        cam_K=cam_K,
        cam_pos=eye,
        cam_quat=cam_quat,
        bg_ply=args.bg_ply,
        show_viewer=args.show_viewer,
        dry_run=args.dry_run,
    )

    socket_pose = (socket_pos, [1.0, 0.0, 0.0, 0.0])
    num_frames = 1 if args.smoke else int(args.duration * FPS)
    frames = []
    for frame in range(num_frames):
        t = example.sim_time
        tgt = wp.vec3(rest[0], rest[1] + target_offset_y(t), rest[2])
        example._pick_body.assign([-1])
        example._pick_target.assign([tgt])
        example.gizmo_tf = wp.transform(tgt, wp.quat_identity())

        if getattr(example, "graph", None):
            wp.capture_launch(example.graph)
        else:
            example.simulate()
        example.sim_time += example.frame_dt
        example.render()

        plug_pose = newton_pose(example.state_0.body_q.numpy(), plug_body)
        rgb = client.render([socket_pose, plug_pose])  # (H, W, 3) float[0,1]
        frames.append((rgb * 255.0).clip(0, 255).astype("uint8"))

        if frame % 60 == 0 or args.smoke:
            print(f"t={t:4.1f}s dy={target_offset_y(t):+.3f} plug_y={plug_pose[0][1]:+.4f}", flush=True)

    _save(frames, args.out, args.smoke)
    print(f"done: {len(frames)} frame(s) -> {args.out}  (physics rrd: {args.rrd})")


def _save(frames, out_dir, smoke) -> None:
    from PIL import Image

    if smoke or len(frames) == 1:
        Image.fromarray(frames[0]).save(os.path.join(out_dir, "frame_0000.png"))
        return
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(out_dir, f"frame_{i:04d}.png"))
    try:
        import imageio.v2 as imageio

        imageio.mimsave(os.path.join(out_dir, "insertion.gif"), frames[::2], fps=20)
    except Exception as e:
        print(f"[demo] gif skipped: {e}")


if __name__ == "__main__":
    main()
