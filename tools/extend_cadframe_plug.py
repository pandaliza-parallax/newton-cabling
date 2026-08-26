"""Extend headA_plug_cadframe_rigid.ply with more cord, PRESERVING its registration.

The cadframe file is bake_headA_splat.py's default keep-40 output composed with a small
rigid registration (~29 deg roll + ~2 deg off-axis, CloudCompare-style) whose provenance
is lost. Rather than re-calibrate, this script re-derives that transform by ICP between
a fresh default keep-40 bake and the shipped file (converges to ~0.2 mm mean NN), then
bakes a longer-cord crop and applies the SAME transform -- xyz, quats, normals and the
45 deg-3 f_rest coeffs (per-band real-SH rotation, INRIA channel-major), so the result
drops into the pipeline under the SAME user-validated CONN_RPY="-90 0 180".

    .venv/bin/python tools/extend_cadframe_plug.py --keep-mm 50 \
        --out newton_cabling/assets/ethernet/headA_plug_cadframe_rigid50.ply
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bake_headA_splat import (  # noqa: E402
    mat_to_quat_wxyz, quat_mul_wxyz, read_ply, sh_rot_matrix, write_ply,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CADFRAME = os.path.join(HERE, "newton_cabling/assets/ethernet/headA_plug_cadframe_rigid.ply")


def bake(keep_mm, out):
    subprocess.run([sys.executable, os.path.join(HERE, "tools/bake_headA_splat.py"),
                    "--keep-mm", str(keep_mm), "--schema", "full", "--out", out],
                   check=True, capture_output=True)


def icp(src, dst, iters=20):
    """Rigid (R, t) with dst ~= src @ R.T + t."""
    tree = cKDTree(dst)
    R, t = np.eye(3), np.zeros(3)
    # coarse init: best z-roll about the conn anchor
    anchor = np.array([-0.0015, -0.0015, 0.0172])
    best = None
    for deg in range(-180, 181, 5):
        th = np.radians(deg); c, s = np.cos(th), np.sin(th)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        d = tree.query((src - anchor) @ Rz.T + anchor)[0].mean()
        if best is None or d < best[1]:
            best = (Rz, d)
    R = best[0]; t = anchor - R @ anchor
    for _ in range(iters):
        cur = src @ R.T + t
        q = dst[tree.query(cur)[1]]
        ca, cb = cur.mean(0), q.mean(0)
        H = (cur - ca).T @ (q - cb)
        U, _, Vt = np.linalg.svd(H)
        Rd = Vt.T @ U.T
        if np.linalg.det(Rd) < 0:
            Vt[-1] *= -1; Rd = Vt.T @ U.T
        R = Rd @ R
        t = Rd @ (t - ca) + cb
    return R, t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-mm", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        p40, p50 = os.path.join(td, "k40.ply"), os.path.join(td, "k50.ply")
        bake(40.0, p40)
        bake(args.keep_mm, p50)
        ref, _ = read_ply(CADFRAME)
        a40, _ = read_ply(p40)
        d50, props = read_ply(p50)

        xyz = lambda d: np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)  # noqa: E731
        R, t = icp(xyz(a40), xyz(ref))
        resid = cKDTree(xyz(ref)).query(xyz(a40) @ R.T + t)[0].mean()

        p = xyz(d50) @ R.T + t
        cols = {"x": p[:, 0], "y": p[:, 1], "z": p[:, 2]}
        nrm = np.stack([d50["nx"], d50["ny"], d50["nz"]], 1).astype(np.float64) @ R.T
        for k, nm in enumerate(("nx", "ny", "nz")):
            cols[nm] = nrm[:, k]
        qR = mat_to_quat_wxyz(R)
        q = np.stack([d50[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
        qn = quat_mul_wxyz(np.tile(qR, (len(q), 1)), q)
        for i in range(4):
            cols[f"rot_{i}"] = qn[:, i]
        for i in range(3):
            cols[f"scale_{i}"] = d50[f"scale_{i}"]
            cols[f"f_dc_{i}"] = d50[f"f_dc_{i}"]
        cols["opacity"] = d50["opacity"]
        rest = np.stack([d50[f"f_rest_{i}"] for i in range(45)], 1).astype(np.float64)
        blk = rest.reshape(len(rest), 3, 15)
        newblk, off = np.empty_like(blk), 0
        for l in range(1, 4):  # noqa: E741
            w = 2 * l + 1
            newblk[:, :, off:off + w] = blk[:, :, off:off + w] @ sh_rot_matrix(l, R).T
            off += w
        rest = newblk.reshape(len(rest), 45)
        for i in range(45):
            cols[f"f_rest_{i}"] = rest[:, i]

        order = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
                 + [f"f_rest_{i}" for i in range(45)]
                 + ["opacity", "scale_0", "scale_1", "scale_2",
                    "rot_0", "rot_1", "rot_2", "rot_3"])
        write_ply(args.out, cols, order)
        print(f"wrote {args.out}: {len(p)} splats (was 1272), "
              f"icp residual {resid*1000:.3f} mm, z range "
              f"[{p[:,2].min()*1000:.1f}, {p[:,2].max()*1000:.1f}] mm")


if __name__ == "__main__":
    main()
