#!/usr/bin/env python3
"""Crop the baked-in gripper out of the wrist_3 splat (APPROXIMATE) -- NON-DESTRUCTIVELY.

The captured flat/wrist_3_link.ply bakes the whole gripper into it (~36k pts, extends to
x=-0.21 along the tool axis), which double-renders with the 8 synth articulating finger
splats (the "outer captured gripper + inner synth fingers" doubling). This keeps only the
wrist side -- points with local x >= --keep-x-above -- and writes a COPY to a new dir; the
original flat/ is left untouched. Point scripts/record_sbot_scene_gs.py at it with --wrist3-ply.

The wrist/gripper geometry is blended (no clean gap), so this is approximate: raise the
threshold toward 0 to strip more gripper (risk clipping the flange), lower it to keep more.
Splat FILE changes but the object COUNT doesn't -> restart the renderer after (it caches
splats per SETUP).

    python tools/crop_wrist3.py --keep-x-above -0.05
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np


def main() -> None:
    base = "/home/pandaliza/parallax/parallax-demo-isaac-lab/assets/sbot_gs"
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{base}/flat/wrist_3_link.ply")
    ap.add_argument("--out", default=f"{base}/arm_nogrip/wrist_3_link.ply")
    ap.add_argument("--keep-x-above", type=float, default=-0.05,
                    help="keep splat points with local x >= this; drops the -x gripper cluster")
    a = ap.parse_args()
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    raw = src.read_bytes()
    end = raw.find(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:end].decode("ascii", "replace")
    names = [ln.split()[-1] for ln in hdr.splitlines() if ln.startswith("property")]
    n = next(int(ln.split()[-1]) for ln in hdr.splitlines() if ln.startswith("element vertex"))
    buf = np.frombuffer(raw[end:end + n * len(names) * 4], dtype="<f4").reshape(n, len(names))

    kept = buf[buf[:, 0] >= a.keep_x_above]
    hdr2 = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n%send_header\n"
            % (len(kept), "".join(f"property float {p}\n" for p in names)))
    with open(out, "wb") as f:
        f.write(hdr2.encode("ascii"))
        f.write(np.ascontiguousarray(kept).astype("<f4").tobytes())
    print(f"[crop] {src.name}: {n} -> {len(kept)} pts (kept x>={a.keep_x_above}, "
          f"dropped {n - len(kept)} gripper pts)")
    print(f"[crop] wrote {out}  (original {src} untouched)")


if __name__ == "__main__":
    main()
