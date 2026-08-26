"""Debug frame: the ethernet connector ONLY (no arm/table/gripper), Newton mesh vs GS splat.

Two static shapes (jack + plug), no simulation: places the REAL cad_rj45.usd meshes and the
production splats (cad_jack_registered.ply + headA_plug_ccreg.ply) at the SAME world poses
this repo's actual render pipeline computes for a given frame of a scripted-insertion
trajectory, using the exact calibration constants (jack/conn align+anchor, eef-rpy seat
transform) from tools/render_batch_v4.sh / scripts/record_sbot_scene_gs_cable.py. This checks
splat<->physics registration in isolation, without the rest of the scene in the way.

Output: LEFT = Gaussian-splat render, RIGHT = Newton mesh render (headless GL), stitched with
composite_lr and labelled.

Run (renderer container must be up):
    sudo PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim \
      .venv/bin/python tools/debug_cable_newton_vs_gs.py --frame 88 --out /tmp/cable_debug.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import newton
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from newton_cabling.connector import cad_rj45_connector
from newton_cabling.render.gs_bridge import (  # noqa: E402
    NewtonGSClient,
    euler_deg_to_quat_wxyz,
    look_at_quat,
    make_intrinsics,
    quat_mul_wxyz,
    quat_rotate_wxyz,
)
from newton_cabling.render.scene_gs_common import (  # noqa: E402
    eye_target_to_pitch_yaw,
    composite_lr,
    host_to_container,
    static_pose,
)
from newton_cabling.sim.scene import resolve_asset_path  # noqa: E402

# ── production calibration, copied verbatim from the ep_0000 cad_full render's _CONFIG.txt /
# tools/datagen_v2_config.sh (this is "the ethernet cable we are using" today) ──────────────
JACK_POS = [0.295, -0.876, 0.835]
JACK_RPY = [0.0, 0.0, 0.0]
JACK_ALIGN_RPY = [-90.0, 0.0, 0.0]
JACK_ANCHOR = [0.0, 0.0, 0.018]
CONN_RPY = [-90.0, 0.0, 0.0]
CONN_ANCHOR = [-0.0015, -0.0015, 0.0172]
EEF_RPY = [0.0, 0.0, 180.0]

HOST_ETH = "/home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/ethernet"
JACK_PLY = f"{HOST_ETH}/cad_jack_registered.ply"
PLUG_PLY = "/home/pandaliza/parallax/newton-cabling/newton_cabling/assets/ethernet/headA_plug_ccreg.ply"
TRAJ = "/home/pandaliza/parallax/data/vla_train/cable_traj_cad_full/ep_0000/face_traj.npy"
# The renderer treats splat[0] as a static background and composites the rest against it; with
# a single real object and no bg it has nothing left to composite ("max() arg is an empty
# sequence"). scripts/record_sbot_scene_gs_cable.py's --gripper-only path hits the same issue
# and fixes it with this exact invisible 1-point dummy background -- reuse it here too.
DUMMY_BG_PLY = "/home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/_gonly_bg.ply"


def _load_visual_mesh(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    usd_mesh = newton.usd.get_mesh(prim, load_normals=True)
    vertices = np.array(usd_mesh.vertices, dtype=np.float32)
    indices = np.array(usd_mesh.indices, dtype=np.int32)
    normals = np.array(usd_mesh.normals, dtype=np.float32) if usd_mesh.normals is not None else None
    return newton.Mesh(vertices, indices, normals=normals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", type=int, default=88, help="face_traj.npy frame index (seated ~80+)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--fov-deg", type=float, default=35.0)
    ap.add_argument("--dist", type=float, default=0.12, help="camera distance from the jack (m)")
    ap.add_argument("--out", default="/tmp/cable_debug.png")
    ap.add_argument("--dry-run", action="store_true", help="skip the GS renderer (Newton-only sanity check)")
    ap.add_argument("--plug-only", action="store_true", help="drop the jack from both panels (cable/plug only)")
    ap.add_argument("--jack-ply", default=JACK_PLY, help="jack splat (CAD-jack frame)")
    ap.add_argument("--plug-ply", default=PLUG_PLY, help="plug splat (renderer splat frame)")
    ap.add_argument("--traj", default=TRAJ, help="face_traj.npy to pose the connector from")
    ap.add_argument("--usd-asset", default="cad_rj45.usd",
                    help="connector USD for the Newton mesh side (e.g. splat_rj45.usd)")
    ap.add_argument("--flip-view", action="store_true", help="shorthand for --orbit-deg 180")
    ap.add_argument("--orbit-deg", type=float, default=0.0,
                    help="orbit the CAMERA this many degrees (about world Z, through the object) "
                         "from the default azimuth. Does not touch the connector's pose -- both "
                         "panels just get viewed from a different direction.")
    ap.add_argument("--roll-deg", type=float, default=0.0,
                    help="spin the CONNECTOR itself this many degrees about its own insertion "
                         "axis (local +Y, the plug/splat frame's canonical axis -- same convention "
                         "as bake_headA_splat.py's --roll-deg escape hatch). Applied identically "
                         "to the Newton mesh and the GS splat, on top of the real calibrated pose, "
                         "purely for this debug view -- does not change CONN_RPY/EEF_RPY or "
                         "anything the real render pipeline uses.")
    args = ap.parse_args()
    if args.flip_view:
        args.orbit_deg += 180.0

    # ── world poses, mirroring scripts/record_sbot_scene_gs_cable.py's CABLE/eef branch ──
    import newton.usd  # noqa: E402 (after CLI parsing so --help doesn't need CUDA)
    from pxr import Usd

    face = np.load(args.traj)
    s = face[min(args.frame, len(face) - 1)]

    jack_q = euler_deg_to_quat_wxyz(*JACK_RPY)                       # identity here
    jack_align = euler_deg_to_quat_wxyz(*JACK_ALIGN_RPY)
    jack_splat_q = quat_mul_wxyz(jack_q, jack_align)
    jack_splat_pose = static_pose(JACK_POS, jack_splat_q, JACK_ANCHOR)

    eef_align = euler_deg_to_quat_wxyz(*EEF_RPY)
    seat_R = quat_mul_wxyz(jack_q, eef_align)
    p_world = np.asarray(JACK_POS, float) + quat_rotate_wxyz(seat_R, s[:3])
    q_world = quat_mul_wxyz(seat_R, [float(x) for x in s[3:7]])
    if args.roll_deg:                                          # debug-only spin about local +Y
        q_world = quat_mul_wxyz(q_world, euler_deg_to_quat_wxyz(0.0, args.roll_deg, 0.0))
    conn_align = euler_deg_to_quat_wxyz(*CONN_RPY)
    conn_splat_q = quat_mul_wxyz(list(q_world), conn_align)
    conn_splat_pose = static_pose(p_world.tolist(), conn_splat_q, CONN_ANCHOR)

    print(f"[debug] frame={args.frame}  jack_world={JACK_POS}  plug_world={np.round(p_world, 4).tolist()}")

    # ── camera: tight on the plug alone, or the jack/plug midpoint if both are shown ───────
    if args.plug_only:
        center = p_world.copy()
    else:
        center = (np.asarray(JACK_POS, float) + p_world) / 2.0
    base_dir = np.array([0.10, -0.10, 0.05])
    th = math.radians(args.orbit_deg)
    ch, sh = math.cos(th), math.sin(th)
    eye_dir = np.array([
        ch * base_dir[0] - sh * base_dir[1],
        sh * base_dir[0] + ch * base_dir[1],
        base_dir[2],
    ])
    eye = center + eye_dir
    eye = center + (eye - center) / (np.linalg.norm(eye - center) + 1e-9) * args.dist
    cam_K = make_intrinsics(args.width, args.height, args.fov_deg)
    cam_quat = look_at_quat(eye.tolist(), center.tolist(), up=(0.0, 0.0, 1.0), convention="ros")

    # ── GS splat render: plug only, or jack + plug; no bg, no table, no arm either way ────
    gs_plys = [host_to_container(args.plug_ply)] if args.plug_only else [
        host_to_container(args.jack_ply), host_to_container(args.plug_ply)]
    gs_poses = [conn_splat_pose] if args.plug_only else [jack_splat_pose, conn_splat_pose]
    if not args.dry_run:
        client = NewtonGSClient(
            ply_paths=gs_plys,
            cam_K=cam_K, cam_pos=eye.tolist(), cam_quat=cam_quat,
            # ALWAYS give the renderer a dummy background: splat[0] is treated as a
            # STATIC bg (pose ignored), so without this the jack silently becomes the
            # background and never renders at its calibrated pose.
            bg_ply=host_to_container(DUMMY_BG_PLY),
        )
        gs_rgb = client.render(gs_poses)
        gs_u8 = (np.clip(gs_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        gs_u8 = np.zeros((args.height, args.width, 3), np.uint8)

    # ── Newton mesh render: the SAME connector mesh(es), as static world shapes ────────────
    spec = cad_rj45_connector(usd_asset_name=args.usd_asset)
    usd_path = resolve_asset_path(spec.usd_asset_name)
    stage = Usd.Stage.Open(usd_path)
    plug_mesh = _load_visual_mesh(stage, spec.plug_prim_path)

    builder = newton.ModelBuilder()

    def _wp_quat(q_wxyz):
        w, x, y, z = q_wxyz
        return wp.quat(x, y, z, w)

    if not args.plug_only:
        socket_mesh = _load_visual_mesh(stage, spec.socket_prim_path)
        builder.add_shape_mesh(
            -1, mesh=socket_mesh,
            xform=wp.transform(wp.vec3(*[float(v) for v in JACK_POS]), _wp_quat(jack_q)),
        )
    builder.add_shape_mesh(
        -1, mesh=plug_mesh,
        xform=wp.transform(wp.vec3(*[float(v) for v in p_world]), _wp_quat(q_world)),
    )
    model = builder.finalize()
    state = model.state()

    from newton.viewer import ViewerGL

    gl = ViewerGL(width=args.width, height=args.height, headless=True, vsync=False)
    gl.set_model(model)
    pitch, yaw = eye_target_to_pitch_yaw(eye.tolist(), center.tolist())
    gl.set_camera(wp.vec3(*[float(v) for v in eye]), pitch, yaw)
    gl.camera.fov = args.fov_deg
    gl.begin_frame(0.0)
    gl.log_state(state)
    gl.end_frame()
    newton_u8 = gl.get_frame().numpy()
    gl.close()

    frame = composite_lr(gs_u8, newton_u8, "GAUSSIAN SPLAT", "NEWTON")
    from PIL import Image

    Image.fromarray(frame).save(args.out)
    print(f"[debug] wrote {args.out}  ({frame.shape[1]}x{frame.shape[0]})")


if __name__ == "__main__":
    main()
