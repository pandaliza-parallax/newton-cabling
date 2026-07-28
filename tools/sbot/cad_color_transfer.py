"""CAD -> scan colour transfer: CAD-exact gripper splats coloured from the registered scan.

Instead of cutting the (inflated) scan, this SAMPLES each finger link's CAD surface (exact
geometry, even coverage, link-local -- same as synth_splats_from_meshes.py) and colours each
sample from the NEAREST Gaussian in the registered scan. Result: geometry hugs the CAD
perfectly (overlaps it, no inflation shell, no holes, articulates cleanly), with the scan's
real per-point f_dc colour instead of a flat grey.

Per-finger theta (finger1 @ -0.6, finger2 @ -0.3) matches the scan's asymmetric capture, so
each link's CAD is posed where that finger actually was when scanned before the colour lookup.

    python tools/sbot/cad_color_transfer.py --out .../sbot_gs/gripper_color
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from cut_splat_by_links import FINGER_LINKS, assemble_links, col, read_ply, write_flat_ply  # noqa: E402
from export_gripper_meshes import DEFAULT_USD  # noqa: E402
from synth_splats_from_meshes import link_local_mesh, synth_link  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="/home/pandaliza/parallax/sbot_assets/fingers/gripper_fingers_registered_full.ply")
    ap.add_argument("--usd", default=str(DEFAULT_USD))
    ap.add_argument("--out", default="/home/pandaliza/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_color")
    ap.add_argument("--theta1", type=float, default=-0.6, help="finger1 capture opening")
    ap.add_argument("--theta2", type=float, default=-0.3, help="finger2 capture opening")
    ap.add_argument("--spacing", type=float, default=0.0015, help="CAD sample spacing / gaussian size (m)")
    ap.add_argument("--opacity", type=float, default=4.0)
    ap.add_argument("--flatten", type=float, default=0.25)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.Open(a.usd)
    xc = UsdGeom.XformCache()
    props, data = read_ply(pathlib.Path(a.scan))
    scan_xyz = np.column_stack([col(props, data, k) for k in ("x", "y", "z")])
    scan_fdc = np.column_stack([col(props, data, f"f_dc_{i}") for i in range(3)])
    tree = cKDTree(scan_xyz)
    print(f"[transfer] scan {len(scan_xyz)} gaussians; sampling CAD @ {a.spacing*1000:.1f}mm")

    total = 0
    for group, theta in [(FINGER_LINKS[:4], a.theta1), (FINGER_LINKS[4:], a.theta2)]:
        asm = assemble_links(pathlib.Path(a.usd), theta, tuple(group))
        for name in group:
            lmesh = link_local_mesh(stage, name, xc)                    # CAD surface, link-local
            g = synth_link(lmesh, spacing=a.spacing, opacity=a.opacity, color=0.2, flatten=a.flatten)
            R, t = asm[name]["R"], asm[name]["t"]
            world = g["xyz"] @ R.T + t                                  # -> world at this finger's theta
            d, idx = tree.query(world)                                  # nearest scan gaussian
            fdc = scan_fdc[idx]                                         # transfer its colour
            short = name.replace("gripper_", "")
            write_flat_ply(out / f"{short}.ply", g["xyz"], g["scale"], g["rot"], fdc, g["opacity"])
            total += len(g["xyz"])
            print(f"  {short + '.ply':<28} {len(g['xyz']):5d} pts | colour-src dist "
                  f"median {np.median(d)*1000:4.1f}mm max {d.max()*1000:4.1f}mm")
    print(f"[transfer] {total} CAD-sampled gaussians, coloured from scan -> {out}")


if __name__ == "__main__":
    main()
