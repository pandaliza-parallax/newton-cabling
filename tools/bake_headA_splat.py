"""Bake the headA SCAN splat into the render pipeline's plug-splat frame.

The physics already uses this scan: ``build_scan_rj45.py`` cropped headA, computed a
world->plug-frame transform (face at the tip/origin, insertion +Y, latch +Z) and saved it
as ``scan_frame_R.npy``. Reusing that SAME transform here is the whole point -- it puts the
splat exactly on the collision mesh, with no fresh registration and no drift between what
the physics simulates and what the camera sees.

Three frames are involved and they are NOT the same:
  * SCAN frame   (Ethernet_cable_headA/splat.ply): the parent scan's coordinates, metres.
  * PLUG frame   (scan_rj45.usd /World/Plug, i.e. what Newton simulates): mating face at
    the origin, insertion +Y, latch +Z, body spanning y in [-40mm, 0].
  * SPLAT frame  (cropped_plug_head.ply, what the renderer consumes): insertion **+Z**,
    mating face at ``--conn-anchor`` = (-1.5, -1.5, 17.2) mm. The v4 batch header is
    explicit that the splat axis is +Z, not +Y -- feeding it a +Y asset renders the plug
    as a vertical rod.

PLUG -> SPLAT is NOT a free choice, and in particular the roll is NOT an eyeball parameter
(an earlier version of this script claimed it was). The renderer draws the connector as

    world = cp + R(plug_q (x) conn_align) @ (v_splat - conn_anchor)

(``static_pose(cp, cq, conn_anchor)``, scene_gs_common.py:128) while Newton places a
PLUG-frame direction u at ``R(plug_q) @ u``. Equating the two gives u = R(conn_align) @ v,
so the bake is pinned by the calibration flag alone:

    plug -> splat  =  R(conn_align)^T  =  Rx(-90 deg)^T  =  Rx(+90 deg)

(``--conn-rpy -90 0 0`` with gs_bridge's intrinsic-XYZ euler => R = Rx(-90).) That maps
plug +Y (insertion) -> splat +Z and plug +Z (latch) -> splat **-Y**, which is exactly where
cropped_plug_head.ply puts its latch: in the rear head band (z 8..11mm) its -y extent grows
to -5.6mm while the front band (z 13..18mm) only reaches -3.8mm -- the latch's raised free
end. So --roll-deg 0 is correct; the flag is kept only as an escape hatch.

    p_splat = Rz(roll) @ Rx(90) @ R_scan @ (p_scan - tip) + conn_anchor

f_rest (45 coeffs, SH degree 3) is ROTATED, not dropped: bake_transform_sh.py's numerically
built real-SH rotation matrices are reused, and the renderer
(DalusSimCore/dalus_sim_core/gaussian_splat.py:112) reads f_rest as ``reshape(3, 15).T`` =
INRIA channel-major, which is the layout Nerfstudio wrote. ``--schema deg0`` emits the
14-property flat file instead (matches the other pipeline plug splats; the DC term is
isotropic so nothing is lost geometrically, only the view-dependent specular).

``--keep-mm 40`` crops to the RIGID body extent that Newton actually simulates
(build_scan_rj45.py's HEAD_LEN); the raw scan runs ~58mm back into the cord, and that extra
cord would render as a rigid straight stub while the sim's cable is deformable and unrendered.

    .venv/bin/python tools/bake_headA_splat.py \
        --out newton_cabling/assets/ethernet/headA_plug_registered.ply
"""

from __future__ import annotations

import argparse
import os

import numpy as np

SCAN_R = "newton_cabling/assets/ethernet/headA_scan_frame_R.npy"
HEADA = ("newton_cabling/assets/ethernet/assets/objects/"
         "Ethernet_cable_headA/splat.ply")
REF = ("/home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/"
       "ethernet/cropped_plug_head.ply")
# 14-property layout the renderer's existing plug splats use, in order
DEG0_PROPS = ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3",
              "scale_0", "scale_1", "scale_2", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]

# ── 3DGS real-SH basis (INRIA / DalusSimCore eval_sh), lifted from bake_transform_sh.py ──
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def sh_band(l, D):  # noqa: E741
    """Real-SH basis of band l evaluated at unit dirs D (K,3) -> (K, 2l+1)."""
    x, y, z = D[:, 0], D[:, 1], D[:, 2]
    if l == 1:
        return np.stack([-C1 * y, C1 * z, -C1 * x], 1)
    if l == 2:
        return np.stack([C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * z * z - x * x - y * y),
                         C2[3] * x * z, C2[4] * (x * x - y * y)], 1)
    if l == 3:
        return np.stack([C3[0] * y * (3 * x * x - y * y), C3[1] * x * y * z,
                         C3[2] * y * (4 * z * z - x * x - y * y),
                         C3[3] * z * (2 * z * z - 3 * x * x - 3 * y * y),
                         C3[4] * x * (4 * z * z - x * x - y * y),
                         C3[5] * z * (x * x - y * y), C3[6] * x * (x * x - 3 * y * y)], 1)
    raise ValueError(l)


def fib_sphere(k):
    """Deterministic ~even unit dirs (no RNG -> reproducible)."""
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], 1)


def sh_rot_matrix(l, R):  # noqa: E741
    """(2l+1)^2 matrix M with c' = M c for an object rotated by R:  f'(d) = f(R^-1 d)."""
    D = fib_sphere(max(4 * (2 * l + 1), 64))
    B = sh_band(l, D)
    return np.linalg.pinv(B) @ sh_band(l, D @ R)      # rows of D @ R are R^-1 d_i


# ── ply io ────────────────────────────────────────────────────────────────────────────
def read_ply(path):
    with open(path, "rb") as f:
        hdr = b""
        while b"end_header" not in hdr:
            hdr += f.readline()
        txt = hdr.decode("ascii", "ignore")
        n = int([ln for ln in txt.splitlines() if ln.startswith("element vertex")][0].split()[-1])
        props = [ln.split()[-1] for ln in txt.splitlines() if ln.startswith("property")]
        types = [ln.split()[1] for ln in txt.splitlines() if ln.startswith("property")]
        m = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}
        dt = np.dtype([(p, m[t]) for p, t in zip(props, types)])
        data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    return data, props


def write_ply(path, cols, order):
    n = len(cols["x"])
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    hdr += [f"property float {p}" for p in order] + ["end_header"]
    arr = np.empty(n, dtype=np.dtype([(p, "<f4") for p in order]))
    for p in order:
        arr[p] = np.asarray(cols[p]).astype(np.float32)
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode("ascii"))
        f.write(arr.tobytes())


def quat_mul_wxyz(a, b):
    """Hamilton product for (w,x,y,z) rows -- the 3DGS PLY rotation convention."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


def mat_to_quat_wxyz(M):
    t = np.trace(M)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        q = [(M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
    elif M[1, 1] > M[2, 2]:
        s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        q = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        q = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headA", default=HEADA)
    ap.add_argument("--scan-r", default=SCAN_R, help="scan_frame_R.npy from build_scan_rj45.py")
    ap.add_argument("--matrix", default=None,
                    help="4x4 CloudCompare 'Applied transformation matrix' (16 numbers, "
                         "whitespace/comma separated, or a .txt holding them) mapping the RAW "
                         "scan splat onto headA_plugframe.obj. Overrides --scan-r. A uniform "
                         "scale in the matrix IS applied -- to positions, to the gaussian "
                         "log-scales, and to the crop depth -- so the splat stays self-consistent.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--roll-deg", type=float, default=0.0,
                    help="extra roll about the insertion axis. Should be 0: the roll is PINNED by "
                         "--conn-rpy (see module docstring). Escape hatch only.")
    ap.add_argument("--anchor", type=float, nargs=3, default=[-0.0015, -0.0015, 0.0172],
                    help="splat-frame point the mating face must land on = the renderer's "
                         "--conn-anchor default. Keep in sync or pass a matching --conn-anchor.")
    ap.add_argument("--keep-mm", type=float, default=40.0,
                    help="crop to this depth (mm) behind the face = Newton's rigid plug body "
                         "(build_scan_rj45.py HEAD_LEN). 0 = keep the whole ~58mm scan.")
    ap.add_argument("--schema", choices=["full", "deg0"], default="full",
                    help="full = 62-prop, f_rest SH-rotated; deg0 = 14-prop flat (f_rest dropped)")
    args = ap.parse_args()

    if args.matrix:
        raw = args.matrix
        if os.path.exists(raw):
            with open(raw) as _fh:
                raw = _fh.read()
        v = np.array([float(x) for x in raw.replace(",", " ").split()], float)
        if v.size != 16:
            raise SystemExit(f"--matrix needs 16 numbers, got {v.size}")
        T = v.reshape(4, 4)
        A, tvec = T[:3, :3], T[:3, 3]
        scale = float(np.linalg.norm(A[0]))
        R = A / scale                           # rotation only; scale handled explicitly
        # CC's matrix maps p -> A@p + t. Express as (p - tip) @ R.T so the rest of the
        # pipeline is unchanged: tip is the scan-frame point that lands on the plug origin.
        tip = -(np.linalg.inv(A) @ tvec)
        print(f"[bake] CC matrix: scale {scale:.5f}  tip {np.round(tip, 5)}")
    else:
        RT = np.load(args.scan_r)
        R, tip = RT[:3], RT[3]                  # rows of R map scan-world -> plug frame
        scale = 1.0
    th = np.radians(args.roll_deg)
    c, s = np.cos(th), np.sin(th)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    Rx90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # plug +y->+z, +z->-y
    M = Rz @ Rx90 @ R                           # scan-world -> splat frame
    anchor = np.asarray(args.anchor, float)

    data, props = read_ply(args.headA)
    p = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    plug = scale * ((p - tip) @ R.T)            # PLUG frame (face y=0, insertion +y, latch +z)
    keep = np.ones(len(p), bool) if args.keep_mm <= 0 else plug[:, 1] >= -args.keep_mm / 1000.0
    out_xyz = scale * ((p - tip) @ M.T) + anchor

    cols = {"x": out_xyz[keep, 0], "y": out_xyz[keep, 1], "z": out_xyz[keep, 2]}
    qM = mat_to_quat_wxyz(M)
    q = np.stack([data[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    qn = quat_mul_wxyz(np.tile(qM, (len(q), 1)), q)[keep]
    for i in range(4):
        cols[f"rot_{i}"] = qn[:, i]
    for i in range(3):
        # 3DGS stores LOG scales: a uniform spatial scale s is + log(s)
        cols[f"scale_{i}"] = data[f"scale_{i}"][keep] + np.log(scale)
        cols[f"f_dc_{i}"] = data[f"f_dc_{i}"][keep]     # SH deg-0 is isotropic
    cols["opacity"] = data["opacity"][keep]

    order = list(DEG0_PROPS)
    n_rest = sum(1 for pr in props if pr.startswith("f_rest_"))
    if args.schema == "full":
        # normals (rotate; Nerfstudio usually writes zeros, but be correct anyway)
        if {"nx", "ny", "nz"} <= set(props):
            nrm = np.stack([data["nx"], data["ny"], data["nz"]], 1).astype(np.float64) @ M.T
            for k, nm in enumerate(("nx", "ny", "nz")):
                cols[nm] = nrm[keep, k]
        else:
            for nm in ("nx", "ny", "nz"):
                cols[nm] = np.zeros(int(keep.sum()))
        # f_rest: real-SH rotation, band by band, INRIA channel-major (3, per)
        assert n_rest == 45, f"expected 45 f_rest coeffs (deg 3), got {n_rest}"
        rest = np.stack([data[f"f_rest_{i}"] for i in range(n_rest)], 1).astype(np.float64)[keep]
        per = n_rest // 3
        deg = int(round((per + 1) ** 0.5)) - 1
        blk = rest.reshape(len(rest), 3, per)
        newblk, off = np.empty_like(blk), 0
        for l in range(1, deg + 1):  # noqa: E741
            w = 2 * l + 1
            newblk[:, :, off:off + w] = blk[:, :, off:off + w] @ sh_rot_matrix(l, M).T
            off += w
        rest = newblk.reshape(len(rest), n_rest)
        for i in range(n_rest):
            cols[f"f_rest_{i}"] = rest[:, i]
        order = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
                 + [f"f_rest_{i}" for i in range(n_rest)]
                 + ["opacity", "scale_0", "scale_1", "scale_2",
                    "rot_0", "rot_1", "rot_2", "rot_3"])
        print(f"[bake] rotated SH: degree {deg}, bands {list(range(1, deg + 1))}, "
              f"{n_rest} f_rest coeffs (channel-major, matches gaussian_splat.py reshape(3,15).T)")
    else:
        print(f"[bake] SH stripped to degree 0 ({n_rest} f_rest coeffs dropped)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_ply(args.out, cols, order)

    # ── verification ─────────────────────────────────────────────────────────────────
    kept = out_xyz[keep]
    lo, hi = kept.min(0) * 1000, kept.max(0) * 1000
    print(f"[bake] {int(keep.sum()):,}/{len(p):,} gaussians  roll {args.roll_deg:+.1f}deg  "
          f"keep {args.keep_mm:g}mm  schema {args.schema} ({len(order)} props) -> {args.out}")
    print(f"[bake] M (scan->splat) =\n{np.round(M, 5)}")
    print(f"[bake] anchor (face) mm = {np.round(anchor * 1000, 2)}   "
          f"insertion axis +Z   latch -> -Y")
    print(f"[bake] bbox mm  lo {np.round(lo, 2)}  hi {np.round(hi, 2)}  "
          f"span {np.round(hi - lo, 2)}")
    print(f"[bake] centroid mm {np.round(kept.mean(0) * 1000, 2)}")
    print(f"[bake] insertion(z) span mm [{lo[2]:.2f} .. {hi[2]:.2f}]  "
          f"face anchor z={anchor[2] * 1000:.1f}  -> {anchor[2] * 1000 - lo[2]:.1f}mm behind the face")
    if os.path.exists(REF):
        rd, _ = read_ply(REF)
        rp = np.stack([rd["x"], rd["y"], rd["z"]], 1)
        print(f"[bake] ref cropped_plug_head bbox mm  lo {np.round(rp.min(0) * 1000, 2)}  "
              f"hi {np.round(rp.max(0) * 1000, 2)}")


if __name__ == "__main__":
    main()
