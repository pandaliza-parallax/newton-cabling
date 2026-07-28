#!/usr/bin/env python3
"""Standalone GS scene probe — compose room + table + the ethernet connector and
render directly (no Newton sim) to debug camera + placement fast. The camera
auto-frames the *connector* (focus splats) wherever it ends up, so the parts stay
centered while you tune their offset onto the table.

Scene (mirrors gs-sim-vla render_config.yml, parts swapped for our connector):
  background room splat  @ --bg-pos      (default origin)
  table splat            @ --table-pos   (default [0,0,0.091418], gs-sim-vla value)
  focus: mount + cord    @ --parts-offset (+ optional --recenter)

The focus splats are NOT origin-centred (mount centroid ~[0.066,-0.027,-0.008],
cord ~[0.162,-0.100,0.067], metres). --recenter drops each focus splat's centroid to
origin so --parts-offset becomes "where on the table to place the connector centre".

Needs the single parallax_sim renderer running (see examples/record_rj45_insert_gs.py header)
and the same sudo + PYTHONPATH. Example (connector on the table, in the room):

  sudo PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim \
    .venv/bin/python tools/gs_probe.py \
      --bg-ply    .../scene/assets/background/splat.ply \
      --table-ply .../scene/assets/objects/table/splat.ply \
      --recenter --parts-offset -0.205 0.125 0.83 \
      --out /tmp/gs_probe/scene.png

--dry-run prints the framing math and skips the renderer (no GPU/container needed).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from newton_cabling.render.gs_bridge import NewtonGSClient, look_at_quat, make_intrinsics

HOST_PARALLAX = "/home/pandaliza/parallax"
CONTAINER_PARALLAX = "/root/parallax"
HOST_ETH = f"{HOST_PARALLAX}/gs-sim-vla/scene/assets/objects/ethernet"


def host_to_container(p: str) -> str:
    """The renderer opens plys at /root/parallax/... (bind mount of host ~/parallax)."""
    return p.replace(HOST_PARALLAX, CONTAINER_PARALLAX)


def ply_bounds(host_path: str) -> tuple[np.ndarray, np.ndarray]:
    """1-99th-percentile xyz bounds of a ply's gaussian means (robust to strays).

    Reads the vertex stride from the header, so it handles SH-deg-0 (17 props, the
    ethernet splats) and SH-deg-3 (62 props, the table) alike.
    """
    with open(host_path, "rb") as f:
        raw = f.read()
    end = raw.find(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:end].decode("ascii", "replace")
    n = next(int(ln.split()[-1]) for ln in hdr.splitlines() if ln.startswith("element vertex"))
    nprop = sum(1 for ln in hdr.splitlines() if ln.startswith("property"))
    d = np.frombuffer(raw[end : end + n * nprop * 4], dtype="<f4").reshape(n, nprop)
    xyz = d[:, 0:3].astype(np.float64)
    return np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # focus (the connector — camera frames these)
    ap.add_argument("--ply", nargs="+",
                    default=[f"{HOST_ETH}/trellis-port.ply", f"{HOST_ETH}/trellis-plug.ply"],
                    help="host paths to the focus splats (camera frames their union)")
    ap.add_argument("--recenter", action="store_true",
                    help="drop the focus GROUP's union centre to origin (keeps mount/plug "
                         "relative arrangement) before --parts-offset")
    ap.add_argument("--parts-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="translation applied to the focus splats (set them on the table)")
    ap.add_argument("--plug-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="extra translation for the 2nd focus splat (the plug) — separate it "
                         "from the port (both splats register at the origin, so they overlap)")
    # static scene
    ap.add_argument("--bg-ply", default=None, help="room background splat (host path)")
    ap.add_argument("--bg-pos", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--table-ply", default=None, help="table splat (host path)")
    ap.add_argument("--table-pos", type=float, nargs=3, default=[0.0, 0.0, 0.091418],
                    help="table position (gs-sim-vla render_config value)")
    # camera
    ap.add_argument("--azim", type=float, default=-90.0, help="azimuth deg (0=+x, -90=from -y)")
    ap.add_argument("--elev", type=float, default=25.0, help="elevation deg above horizon")
    ap.add_argument("--dist-scale", type=float, default=2.2, help="eye dist = scale x diagonal")
    ap.add_argument("--convention", default="ros", choices=["ros", "opengl"])
    ap.add_argument("--fov-deg", type=float, default=45.0)
    ap.add_argument("--width", type=int, default=852)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--out", default="/tmp/gs_probe/frame.png")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    parts_off = np.asarray(args.parts_offset, float)

    # Group-recenter: one common offset = -(union centre), so mount and plug keep their
    # captured relative arrangement instead of collapsing onto each other. Then translate
    # the whole group by --parts-offset (e.g. onto the table).
    raw_bounds = [ply_bounds(p) for p in args.ply]
    if args.recenter:
        ulo = np.min([lo for lo, _ in raw_bounds], axis=0)
        uhi = np.max([hi for _, hi in raw_bounds], axis=0)
        common = -(ulo + uhi) / 2
    else:
        common = np.zeros(3)
    plug_off = np.asarray(args.plug_offset, float)
    focus_offsets, los, his = [], [], []
    for i, (lo, hi) in enumerate(raw_bounds):
        # index 1 (by [port, plug] convention) gets the extra plug offset
        off = common + parts_off + (plug_off if i == 1 else np.zeros(3))
        focus_offsets.append(off)
        los.append(lo + off)
        his.append(hi + off)

    allmin, allmax = np.min(los, axis=0), np.max(his, axis=0)
    center = (allmin + allmax) / 2
    diag = float(np.linalg.norm(allmax - allmin))
    a, e = np.deg2rad(args.azim), np.deg2rad(args.elev)
    direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    eye = center + direction * args.dist_scale * diag
    cam_K = make_intrinsics(args.width, args.height, args.fov_deg)
    cam_quat = look_at_quat(
        eye.tolist(), center.tolist(), up=(0, 0, 1), convention=args.convention
    )
    print(
        f"[probe] focus bbox {np.round(allmin, 4)}..{np.round(allmax, 4)} "
        f"center={np.round(center, 4)} diag={diag:.3f}"
    )
    print(f"[probe] eye={np.round(eye, 4)} azim={args.azim} elev={args.elev}")

    # Object order [table?, *focus]; background is the client's separate bg slot.
    static_plys, static_poses = [], []
    if args.table_ply:
        static_plys.append(host_to_container(args.table_ply))
        static_poses.append((list(args.table_pos), [1.0, 0.0, 0.0, 0.0]))
    obj_plys = static_plys + [host_to_container(p) for p in args.ply]
    obj_poses = static_poses + [
        (off.tolist(), [1.0, 0.0, 0.0, 0.0]) for off in focus_offsets
    ]

    client = NewtonGSClient(
        ply_paths=obj_plys, cam_K=cam_K, cam_pos=eye.tolist(), cam_quat=cam_quat,
        bg_ply=host_to_container(args.bg_ply) if args.bg_ply else None,
        bg_pose=(list(args.bg_pos), [1.0, 0.0, 0.0, 0.0]),
        dry_run=args.dry_run,
    )
    img = client.render(obj_poses)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    from PIL import Image

    Image.fromarray((img * 255.0).clip(0, 255).astype("uint8")).save(args.out)
    n = len(obj_plys) + (1 if args.bg_ply else 0)
    mode = "DRY-RUN black" if args.dry_run else "rendered"
    print(f"[probe] saved {args.out}  ({n} splats, {mode})")


if __name__ == "__main__":
    main()
