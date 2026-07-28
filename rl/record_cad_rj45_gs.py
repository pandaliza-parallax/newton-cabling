"""Render the REAL McMaster-CAD RJ45 insertion (`cad_rj45`) through the Gaussian-Splat
renderer — so the physics body and the rendered splat are the same part.

Unlike the top-level ``examples/record_rj45_insert_gs.py`` (which drives Newton's bundled *toy*
``rj45_plug.usd``), this drives the McMaster ``cad_rj45`` rig via ``ConnectorVecEnv`` with
the scripted base controller (zero residual — the path ``rl/smoke_asset.py`` shows seats
cad_rj45), and feeds the plug + socket poses to the splat renderer.

Splats (the real scanned part) are placed on the physics bodies by subtracting each
splat's native centroid (rough body-frame alignment — the exact splat↔body ICP is still
TODO; treat placement as approximate). Cable is not rendered (cad_rj45 has no cable).

Run (single clean parallax_sim renderer must be up — see examples/record_rj45_insert_gs.py header):
    cd newton-cabling
    sudo PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim \
      .venv/bin/python rl/record_cad_rj45_gs.py --smoke --out /tmp/gs_cad

--dry-run skips the renderer (still builds + steps the cad_rj45 sim on GPU).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # rl/ for connector_env
import newton  # noqa: E402
import newton.usd  # noqa: E402
import warp as wp  # noqa: E402
from connector_env import Z_LIFT, ConnectorVecEnv  # noqa: E402
from pxr import Usd  # noqa: E402

from newton_cabling.connector import cad_rj45_connector  # noqa: E402
from newton_cabling.render.gs_bridge import (  # noqa: E402
    NewtonGSClient,
    look_at_quat,
    make_intrinsics,
    newton_pose,
)
from newton_cabling.sim.scene import resolve_asset_path  # noqa: E402

DEV = "cuda:0"
HOST_ETH = "/home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/ethernet"
CONT_ETH = "/root/parallax/gs-sim-vla/scene/assets/objects/ethernet"


def ply_centroid(host_path: str) -> np.ndarray:
    """1-99th-pct centroid of a ply's gaussian means (to recentre the splat to a body)."""
    with open(host_path, "rb") as f:
        raw = f.read()
    end = raw.find(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:end].decode("ascii", "replace")
    n = next(int(ln.split()[-1]) for ln in hdr.splitlines() if ln.startswith("element vertex"))
    nprop = sum(1 for ln in hdr.splitlines() if ln.startswith("property"))
    xyz = np.frombuffer(raw[end : end + n * nprop * 4], dtype="<f4").reshape(n, nprop)[:, 0:3]
    lo, hi = np.percentile(xyz.astype(np.float64), 1, 0), np.percentile(xyz.astype(np.float64), 99, 0)
    return (lo + hi) / 2


def socket_world_pos() -> np.ndarray:
    """World pos of cad_rj45's static /World/Socket = prim translation + Z_LIFT (env 0)."""
    spec = cad_rj45_connector()
    stage = Usd.Stage.Open(resolve_asset_path(spec.usd_asset_name))
    prim = stage.GetPrimAtPath(spec.socket_prim_path)
    sb = wp.transform_get_translation(newton.usd.get_transform(prim, local=False))
    return np.array([sb[0], sb[1], sb[2]]) + Z_LIFT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply-mount", default=f"{HOST_ETH}/trellis-port.ply")  # TRELLIS port/jack
    ap.add_argument("--ply-plug", default=f"{HOST_ETH}/trellis-plug.ply")  # TRELLIS plug
    ap.add_argument("--bg-ply", default=None, help="optional room splat (host path)")
    ap.add_argument("--out", default="out_cad_gs")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--stage", type=int, default=0, help="curriculum stage (0=near-aligned)")
    ap.add_argument("--width", type=int, default=852)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--fov-deg", type=float, default=45.0)
    ap.add_argument("--dist-scale", type=float, default=2.5)
    ap.add_argument("--smoke", action="store_true", help="render one frame and exit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def h2c(p):
        return p.replace(HOST_ETH, CONT_ETH) if p else None

    # Recentre each splat to its body: subtract native centroid so the splat sits on the
    # physics-body origin (approximate; exact alignment is the deferred ICP).
    mount_rec = -ply_centroid(args.ply_mount)
    plug_rec = -ply_centroid(args.ply_plug)

    # Build the McMaster cad_rj45 rig + scripted base controller (proven by smoke_asset.py).
    env = ConnectorVecEnv(1, seed=0, random_easy=True, asset="cad_rj45")
    env.set_stage(args.stage)
    env.residual_scale = 0.0
    env.reset()
    plug_body = int(env.plug_idx.numpy()[0])
    socket_pos = socket_world_pos()

    # Camera frames the socket; plug inserts along +Y toward it.
    eye = (socket_pos + np.array([0.10, -0.12, 0.06]) * args.dist_scale).tolist()
    cam_K = make_intrinsics(args.width, args.height, args.fov_deg)
    cam_quat = look_at_quat(eye, socket_pos.tolist(), up=(0.0, 0.0, 1.0), convention="ros")
    print(f"[cad-gs] socket={np.round(socket_pos, 4)} plug_body={plug_body} eye={np.round(eye, 3)}")

    client = NewtonGSClient(
        ply_paths=[h2c(args.ply_mount), h2c(args.ply_plug)],
        cam_K=cam_K, cam_pos=eye, cam_quat=cam_quat,
        bg_ply=h2c(args.bg_ply), dry_run=args.dry_run,
    )

    zero = torch.zeros(1, env.act_dim, device=DEV)
    n_frames = 1 if args.smoke else args.steps
    frames = []
    for t in range(n_frames):
        if not args.smoke:
            env.step(zero)  # scripted insert; t=0 frame uses the reset pose
        bq = env.state_0.body_q.numpy()
        plug_p, plug_q = newton_pose(bq, plug_body)
        mount_pose = ((socket_pos + mount_rec).tolist(), [1.0, 0.0, 0.0, 0.0])
        plug_pose = ((np.array(plug_p) + plug_rec).tolist(), plug_q)
        rgb = client.render([mount_pose, plug_pose])
        frames.append((rgb * 255.0).clip(0, 255).astype("uint8"))
        if t % 20 == 0 or args.smoke:
            print(f"t={t:3d} plug_y={plug_p[1]:+.4f}", flush=True)

    from PIL import Image

    if args.smoke or len(frames) == 1:
        Image.fromarray(frames[0]).save(os.path.join(args.out, "frame_0000.png"))
    else:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(args.out, f"frame_{i:04d}.png"))
        try:
            import imageio.v2 as imageio

            imageio.mimsave(os.path.join(args.out, "insertion.gif"), frames[::2], fps=20)
        except Exception as e:  # noqa: BLE001
            print(f"[cad-gs] gif skipped: {e}")
    print(f"done: {len(frames)} frame(s) -> {args.out}")


if __name__ == "__main__":
    main()
