"""Render ONE Gaussian-splat .ply through the parallax_sim GS renderer, auto-framed.

Shows a splat in its NATIVE frame (no arm/scene) -- e.g. the registered finger capture in its
captured configuration. Full-SH plys rainbow in this renderer, so f_rest is stripped to flat
(deg-0): you see the real per-Gaussian f_dc colour, matte. One static frame (or turntable).

Needs the parallax_sim renderer up at a 1-splat config (restart it first). Run under sudo:
    sudo PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim \
        .venv/bin/python tools/viz_splat.py \
        --ply /home/pandaliza/parallax/sbot_assets/fingers/gripper_fingers_registered_full.ply \
        --out viz_splat --white
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from newton_cabling.render.gs_bridge import NewtonGSClient, look_at_quat, make_intrinsics  # noqa: E402

FLAT = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]


def _load(path):
    raw = pathlib.Path(path).read_bytes()
    e = raw.find(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:e].decode("ascii", "replace")
    nm = [ln.split()[-1] for ln in hdr.splitlines() if ln.startswith("property")]
    n = next(int(ln.split()[-1]) for ln in hdr.splitlines() if ln.startswith("element vertex"))
    return nm, np.frombuffer(raw[e:e + n * len(nm) * 4], dtype="<f4").reshape(n, len(nm))


def _to_container(p):
    return p.replace("/home/pandaliza/parallax", "/root/parallax")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="splat ply to view (host path, must be bind-mounted)")
    ap.add_argument("--out", default="viz_splat")
    ap.add_argument("--dist", type=float, default=2.2, help="camera distance = dist * bbox diag")
    ap.add_argument("--dir", type=float, nargs=3, default=[0.6, -0.6, 0.4], help="camera direction from centre")
    ap.add_argument("--fov", type=float, default=45.0)
    ap.add_argument("--frames", type=int, default=1, help=">1 = turntable (orbit about +z)")
    ap.add_argument("--white", action="store_true", help="paint near-black bg white")
    ap.add_argument("--thr", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from PIL import Image

    nm, buf = _load(a.ply)
    xyz = buf[:, :3]
    centre = xyz.mean(0)
    diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    # strip to flat (deg-0) next to the source (bind-mounted); renderer reads f_dc, not f_rest
    flat = np.column_stack([buf[:, nm.index(k)] for k in FLAT]).astype("<f4")
    tmp = pathlib.Path(a.ply).with_name("_viz_flat.ply")
    proprows = "".join(f"property float {p}\n" for p in FLAT)
    hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(flat)}\n{proprows}end_header\n"
    tmp.write_bytes(hdr.encode("ascii") + flat.tobytes())
    # The renderer treats the FIRST splat as a static background (skipped for live transforms), so a
    # lone splat -> "max() arg is an empty sequence". Add an invisible 1-pt dummy bg so ours renders.
    dummy = pathlib.Path(a.ply).with_name("_viz_bg.ply")
    db = np.zeros((1, 14), "<f4")
    db[0, 0:3] = [0.0, 0.0, 50.0]; db[0, 3:6] = -6.0; db[0, 6] = 1.0; db[0, 13] = -30.0  # far, tiny, invisible
    dummy.write_bytes(f"ply\nformat binary_little_endian 1.0\nelement vertex 1\n{proprows}end_header\n".encode("ascii")
                      + db.tobytes())
    print(f"[viz] {len(flat)} gaussians  centre={np.round(centre, 3)}  diag={diag:.3f}  (SH stripped -> flat)")

    d = np.asarray(a.dir, float)
    d /= np.linalg.norm(d) + 1e-9
    for f in range(max(1, a.frames)):
        ang = 2 * np.pi * f / max(1, a.frames)
        c, s = np.cos(ang), np.sin(ang)
        dr = np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c, d[2]])   # orbit about +z
        eye = (centre + dr * a.dist * diag).tolist()
        client = NewtonGSClient([_to_container(str(tmp))], make_intrinsics(852, 640, a.fov),
                                eye, look_at_quat(eye, centre.tolist(), up=(0, 0, 1), convention="ros"),
                                bg_ply=_to_container(str(dummy)))
        rgb = client.render([([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])])   # splat at identity (native frame)
        u8 = (rgb * 255.0).clip(0, 255).astype("uint8")
        if a.white:
            u8[u8.max(axis=2) < a.thr] = 255
        Image.fromarray(u8).save(os.path.join(a.out, f"frame_{f:04d}.png"))
    print(f"[viz] wrote {max(1, a.frames)} frame(s) -> {a.out}/")


if __name__ == "__main__":
    main()
