"""Minimal viz: render JUST the jack + connector splats (no arm / pedestal / IK), one frame.

Places the jack at --jack-pos/--jack-rpy and the connector at the chosen frame of a recorded
plug trajectory (socket-frame), mapped onto the jack EXACTLY as record_sbot_scene_gs.py does:
    conn_world_pos  = jack_pos + R(jack_q) . traj[i][:3]
    conn_world_quat = jack_q  o  traj[i][3:7]
So this isolates the jack<->connector alignment with none of the arm machinery.

Run (renderer container up + freshly restarted; SHM root-owned -> sudo):
    cd newton-cabling
    sudo PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim \
        .venv/bin/python viz_jack_connector.py --out viz_jc --frame 0
  --frame 0   the connector's INITIAL pose (~10mm off the jack)
  --frame -1  SEATED (connector in the jack)
  --dry-run   skip the renderer/SHM
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from newton_cabling.render.gs_bridge import (  # noqa: E402
    NewtonGSClient,
    euler_deg_to_quat_wxyz,
    look_at_quat,
    make_intrinsics,
    ply_centroid,
    quat_mul_wxyz,
    quat_rotate_wxyz,
)

HOST = "/home/pandaliza/parallax"
CONT = "/root/parallax"
G = f"{HOST}/gs-sim-vla/scene/assets"


def h2c(p: str) -> str:
    return p.replace(HOST, CONT)


def static_pose(world_pos, quat_wxyz, centroid):
    """Seat a splat's native centroid at ``world_pos`` with orientation ``quat_wxyz``."""
    pos = np.asarray(world_pos, float) - quat_rotate_wxyz(quat_wxyz, centroid)
    return pos.tolist(), list(quat_wxyz)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="viz_jc")
    ap.add_argument("--jack-ply", default=f"{G}/objects/ethernet/cad_jack_registered.ply")
    ap.add_argument("--connector-ply", default=f"{G}/objects/ethernet/cad_plug_registered.ply",
                    help="connector/plug splat (default = the EVAL's calibrated plug, matching ep_0000's frames)")
    ap.add_argument("--table-ply", default=f"{G}/objects/table/splat.ply")
    ap.add_argument("--table-pos", type=float, nargs=3, default=[0.50, -1.00, 0.091418],
                    help="table absolute placement (top ends up at z~0.785)")
    ap.add_argument("--no-table", dest="table", action="store_false", help="drop the table (objects only)")
    ap.add_argument("--traj", default="v1_eval/ep_0000/plug_traj.npy")
    ap.add_argument("--frame", type=int, default=0, help="trajectory frame to show (0=initial, -1=seated)")
    ap.add_argument("--jack-pos", type=float, nargs=3, default=[0.29, -0.872, 1.45],
                    help="jack world xyz (default: table-top CENTRE; z=1.45 = the EMPIRICAL render "
                         "surface — the table splat renders ~2x its raw bounds, measured via --zcol)")
    ap.add_argument("--jack-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="jack orient (deg)")
    ap.add_argument("--conn-rpy", type=float, nargs=3, default=[-90.0, 0.0, 0.0],
                    help="connector splat-frame rotation (deg); -90 0 0 = the eval's --plug-rot calibration")
    ap.add_argument("--conn-anchor", type=float, nargs=3, default=[-0.0015, -0.0015, 0.0172],
                    help="point on the connector splat (native frame) seated at the plug pose "
                         "(= the eval's --plug-anchor: the gold-contact mating face)")
    ap.add_argument("--bg-ply", default=None)
    ap.add_argument("--width", type=int, default=852)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--fov-deg", type=float, default=45.0)
    ap.add_argument("--cam-dir", type=float, nargs=3, default=[0.2, -0.9, 0.6],
                    help="eye direction from the jack<->connector midpoint (front + above the table)")
    ap.add_argument("--cam-dist", type=float, default=0.25,
                    help="eye distance (m) from the jack<->connector; ~0.25 shows the table surface around them")
    ap.add_argument("--zcol", type=float, nargs="+", default=None,
                    help="DIAGNOSTIC: render connector copies at these world-z heights at the jack xy "
                         "(side camera), to read off where the table-top surface actually is")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jack_c = ply_centroid(args.jack_ply)
    cord_c = np.array(args.conn_anchor, float)   # seat the plug's mating-face anchor (like the eval), not the centroid
    jq = euler_deg_to_quat_wxyz(*args.jack_rpy)
    jack_pos = np.array(args.jack_pos, float)

    traj = np.load(args.traj)
    s = traj[args.frame]
    cp = jack_pos + quat_rotate_wxyz(list(jq), s[:3])
    cq = quat_mul_wxyz(list(jq), [float(x) for x in s[3:7]])
    cq = quat_mul_wxyz(list(cq), euler_deg_to_quat_wxyz(*args.conn_rpy))     # connector splat-frame align
    print(f"[viz] traj frame {args.frame}/{len(traj)}: connector offset from jack = "
          f"{np.round((np.asarray(cp) - jack_pos) * 1000, 1)} mm; quat {np.round(cq, 3)}")

    jack_pose = static_pose(jack_pos, jq, jack_c)
    conn_pose = static_pose(cp, cq, cord_c)

    table_pose = (list(args.table_pos), [3.0, 0.0, 0.0, 0.0])           # absolute placement (render_config conv.)
    ident = [1.0, 0.0, 0.0, 0.0]
    if args.zcol:                                                       # diagnostic: connector column at known z
        obj_plys = ([h2c(args.table_ply)] if args.table else []) + [h2c(args.connector_ply)] * len(args.zcol)
        render_poses = ([table_pose] if args.table else []) + [
            static_pose([jack_pos[0], jack_pos[1], z], ident, cord_c) for z in args.zcol]
        center = np.array([jack_pos[0], jack_pos[1], float(np.mean(args.zcol))])
        print(f"[viz] z-column: connector at world z = {args.zcol}")
    else:
        obj_plys = ([h2c(args.table_ply)] if args.table else []) + [h2c(args.jack_ply), h2c(args.connector_ply)]
        render_poses = ([table_pose] if args.table else []) + [jack_pose, conn_pose]
        center = 0.5 * (jack_pos + np.asarray(cp))

    d = np.array(args.cam_dir, float)
    d /= np.linalg.norm(d) + 1e-9
    eye = center + d * args.cam_dist
    cam_K = make_intrinsics(args.width, args.height, args.fov_deg)
    cam_quat = look_at_quat(eye.tolist(), center.tolist(), up=(0.0, 0.0, 1.0), convention="ros")
    print(f"[viz] camera eye {np.round(eye, 3)} -> center {np.round(center, 3)}")
    client = NewtonGSClient(
        ply_paths=obj_plys, cam_K=cam_K, cam_pos=eye.tolist(), cam_quat=cam_quat,
        bg_ply=h2c(args.bg_ply) if args.bg_ply else None, dry_run=args.dry_run,
    )
    if args.dry_run:
        print("[viz] dry-run: poses computed, renderer skipped")
        return
    rgb = (client.render(render_poses) * 255.0).clip(0, 255).astype("uint8")
    from PIL import Image
    p = os.path.join(args.out, f"jc_frame{args.frame}.png")
    Image.fromarray(rgb).save(p)
    print(f"[viz] saved {p}")


if __name__ == "__main__":
    main()
