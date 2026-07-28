"""Per-link overlay of the CUT splat vs its CAD mesh, to debug alignment.

For each finger link: loads the cut <link>.ply (gripper_cut/) and the CAD gripper_<link>.obj
(gripper_meshes/), prints their centroids/bbox and the offset, and writes a color-coded
combined point cloud (cut = RED, CAD = GREEN) to <out>/<link>.ply for CloudCompare.

Both are in the link's LOCAL frame, so if the cut is correct they overlap. A consistent
non-zero offset => a frame bug (body-vs-link origin); a rotation => a quaternion-convention bug.

    python tools/sbot/debug_cut_vs_cad.py
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import trimesh

FINGERS = ["finger1_knuckle_link", "finger1_inner_knuckle_link", "finger1_finger_link",
           "finger1_finger_tip_link", "finger2_knuckle_link", "finger2_inner_knuckle_link",
           "finger2_finger_link", "finger2_finger_tip_link"]


def load_ply_xyz(path):
    raw = pathlib.Path(path).read_bytes()
    e = raw.find(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:e].decode("ascii", "replace")
    nm = [ln.split()[-1] for ln in hdr.splitlines() if ln.startswith("property")]
    n = next(int(ln.split()[-1]) for ln in hdr.splitlines() if ln.startswith("element vertex"))
    return np.frombuffer(raw[e:e + n * len(nm) * 4], dtype="<f4").reshape(n, len(nm))[:, :3].astype(np.float64)


def write_rgb_ply(path, pts, rgb):
    """pts: (N,3) float, rgb: (N,3) uint8."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="/home/pandaliza/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut")
    ap.add_argument("--cad", default="/home/pandaliza/parallax/robo_maker/sbot/assets/gripper_meshes")
    ap.add_argument("--out", default="/home/pandaliza/parallax/newton-cabling/debug_cut_vs_cad")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'link':28s} {'cut centroid':>24s} {'cad centroid':>24s} {'offset mm':>9s}  {'cut bbox mm':>18s} {'cad bbox mm':>18s}")
    for nm in FINGERS:
        cut = load_ply_xyz(f"{a.cut}/{nm}.ply")
        cad = trimesh.load(f"{a.cad}/gripper_{nm}.obj", process=False)
        cadv = np.asarray(cad.vertices, float)
        cc, dc = cut.mean(0), cadv.mean(0)
        off = np.linalg.norm(cc - dc) * 1000
        cbb = (cut.max(0) - cut.min(0)) * 1000
        dbb = (cadv.max(0) - cadv.min(0)) * 1000
        print(f"{nm:28s} {np.array2string(np.round(cc,3),separator=','):>24s} "
              f"{np.array2string(np.round(dc,3),separator=','):>24s} {off:9.1f}  "
              f"{np.array2string(np.round(cbb,0)):>18s} {np.array2string(np.round(dbb,0)):>18s}")
        pts = np.vstack([cut, cadv])
        rgb = np.vstack([np.tile([230, 40, 40], (len(cut), 1)),      # cut = red
                         np.tile([40, 210, 40], (len(cadv), 1))])    # cad = green
        write_rgb_ply(out / f"{nm}.ply", pts, rgb)
    print(f"\n[debug] wrote 8 per-link overlays (cut=RED, cad=GREEN) -> {out}/  (open in CloudCompare)")


if __name__ == "__main__":
    main()
