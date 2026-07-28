#!/usr/bin/env python3
"""Strip spherical-harmonics (f_rest_*) from the sbot_gs arm splats -> flat SH-deg-0 plys.

Why: the parallax_sim renderer (DalusSimCore/gaussian_splat.py) reads f_rest as
``reshape(3,15).T`` -- INRIA channel-major layout. The sbot_gs arm splats store f_rest in
a different layout, so the (small) SH coefficients land on the wrong colour channels and
the arm renders as a view-dependent rainbow oil-slick. Dropping f_rest forces the
renderer's flat-colour path (``0.5 + SH_C0 * f_dc``), which is the correct matte albedo --
exactly right for a robot and immune to the layout mismatch.

Output keeps only x,y,z, scale_0-2, rot_0-3, f_dc_0-2, opacity (the props the flat path
reads, all by name) as binary_little_endian. Originals are left untouched; flats go to
``<sbot_gs>/flat/<name>.ply``. Same object COUNT, so restart the renderer afterwards
(it caches the splat set per SETUP).
"""

from __future__ import annotations

import pathlib

import numpy as np

SH_C0 = 0.28209479177387814
KEEP = ["x", "y", "z", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]


def read_ply(path: pathlib.Path):
    raw = path.read_bytes()
    end = raw.find(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:end].decode("ascii", "replace")
    names = [ln.split()[-1] for ln in hdr.splitlines() if ln.startswith("property")]
    n = next(int(ln.split()[-1]) for ln in hdr.splitlines() if ln.startswith("element vertex"))
    buf = np.frombuffer(raw[end:end + n * len(names) * 4], dtype="<f4").reshape(n, len(names))
    return {nm: buf[:, i] for i, nm in enumerate(names)}, n


def write_flat(path: pathlib.Path, cols: dict, n: int) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {k}\n" for k in KEEP)
        + "end_header\n"
    )
    data = np.stack([cols[k].astype("<f4") for k in KEEP], axis=1)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def main() -> None:
    src = pathlib.Path("/home/pandaliza/parallax/parallax-demo-isaac-lab/assets/sbot_gs")
    out = src / "flat"
    out.mkdir(exist_ok=True)
    links = ["base_link", "shoulder_link", "upper_arm_link", "forearm_link",
             "wrist_1_link", "wrist_2_link", "wrist_3_link"]
    for name in links:
        p = src / f"{name}.ply"
        cols, n = read_ply(p)
        for k in KEEP:
            if k not in cols:
                raise SystemExit(f"{p} missing required prop {k!r}; has {list(cols)[:8]}...")
        rgb = np.clip(0.5 + SH_C0 * np.stack([cols["f_dc_0"], cols["f_dc_1"], cols["f_dc_2"]], 1), 0, 1)
        had_sh = "f_rest_0" in cols
        write_flat(out / f"{name}.ply", cols, n)
        print(f"{name:16s} N={n:7d} had_sh={had_sh!s:5s} "
              f"flat RGB mean={(rgb.mean(0) * 255).astype(int)} "
              f"p10={(np.percentile(rgb, 10, 0) * 255).astype(int)} "
              f"p90={(np.percentile(rgb, 90, 0) * 255).astype(int)}")
    print(f"\nwrote {len(links)} flat plys -> {out}")


if __name__ == "__main__":
    main()
