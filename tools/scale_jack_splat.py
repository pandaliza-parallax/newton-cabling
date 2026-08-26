"""Bake a uniformly SCALED copy of the jack splat (naive similarity, mouth-pivoted).

A Gaussian splat scales in two coupled parts: centers move as v' = m + s*(v - m), and
every log-sigma gains ln(s) (scale_0..2 are LOG sigmas -- moving only the centers leaves
a sparser cloud of same-size blobs with gaps). Rotations, opacity and SH are unchanged
under a uniform scale.

The pivot m defaults to the jack MOUTH plane centre (0, 0, 0.030) in splat coordinates:
the mouth plane maps to itself, and since render_batch_v4's --jack-anchor pins a splat-
local point at --jack-pos unchanged, the scaled jack's PORT stays exactly where the
unscaled one renders -- only the housing grows around it. The Newton/physics socket
stays 1.0x, so the visual housing is deliberately s-times off from collision geometry.

    .venv/bin/python tools/scale_jack_splat.py --scale 1.1 \
        --out newton_cabling/assets/ethernet/cad_jack_registered_x1p1.ply
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bake_headA_splat import read_ply, write_ply  # noqa: E402

SRC = "/home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/ethernet/cad_jack_registered.ply"
MOUTH = (0.0, 0.0, 0.030)   # splat-local mouth-plane centre (v4 batch header geometry)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.1)
    ap.add_argument("--pivot", nargs=3, type=float, default=MOUTH,
                    help="splat-local pivot kept fixed (default: the mouth centre)")
    args = ap.parse_args()

    data, props = read_ply(args.src)
    cols = {p: np.asarray(data[p], np.float64).copy() for p in props}
    m = np.asarray(args.pivot)
    for ax, p in zip(range(3), ("x", "y", "z")):
        cols[p] = m[ax] + args.scale * (cols[p] - m[ax])
    dln = math.log(args.scale)
    for p in ("scale_0", "scale_1", "scale_2"):
        cols[p] += dln
    write_ply(args.out, cols, props)
    print(f"wrote {args.out}: {len(cols['x'])} splats, x{args.scale} about pivot {args.pivot}"
          f" (props: {props})")


if __name__ == "__main__":
    main()
