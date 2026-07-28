#!/usr/bin/env python3
"""Recolor the gripper (finger) splats to a realistic albedo -- NON-DESTRUCTIVELY.

Reads the flat DC-only finger plys and rewrites only f_dc so the displayed colour
(0.5 + SH_C0*f_dc) matches a target (a real AG-145 is matte near-black, not the current
flat 0.2 gray). Writes to a NEW dir; the working flat/ plys are left untouched. Point the
renderer at the output with scripts/record_sbot_scene_gs.py --gripper-gs-dir <out>.

    python tools/recolor_gripper.py --body 0.06 --tip 0.11
    # then: scripts/record_sbot_scene_gs.py --gripper-gs-dir .../sbot_gs/gripper_col ...
    # (restart the renderer first -- it caches splats per SETUP and won't reload same-count files)
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

SH_C0 = 0.28209479177387814
FINGERS = [
    "finger1_knuckle_link", "finger1_inner_knuckle_link", "finger1_finger_link", "finger1_finger_tip_link",
    "finger2_knuckle_link", "finger2_inner_knuckle_link", "finger2_finger_link", "finger2_finger_tip_link",
]


def load(p: pathlib.Path):
    raw = p.read_bytes()
    end = raw.find(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:end].decode("ascii", "replace")
    names = [ln.split()[-1] for ln in hdr.splitlines() if ln.startswith("property")]
    n = next(int(ln.split()[-1]) for ln in hdr.splitlines() if ln.startswith("element vertex"))
    buf = np.frombuffer(raw[end:end + n * len(names) * 4], dtype="<f4").reshape(n, len(names)).copy()
    return names, n, buf


def _rgb(v):
    v = list(v) * 3 if len(v) == 1 else list(v)
    return np.array(v[:3], float)


def main() -> None:
    base = "/home/pandaliza/parallax/parallax-demo-isaac-lab/assets/sbot_gs"
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{base}/flat", help="source flat finger plys")
    ap.add_argument("--out", default=f"{base}/gripper_col", help="output dir (new; originals untouched)")
    ap.add_argument("--body", type=float, nargs="+", default=[0.06], help="body albedo, gray or 'R G B' (0..1)")
    ap.add_argument("--tip", type=float, nargs="+", default=[0.11], help="fingertip albedo, gray or 'R G B'")
    a = ap.parse_args()
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    body_dc, tip_dc = (_rgb(a.body) - 0.5) / SH_C0, (_rgb(a.tip) - 0.5) / SH_C0
    for name in FINGERS:
        names, n, buf = load(src / f"{name}.ply")
        dc = tip_dc if "finger_tip" in name else body_dc
        for k in range(3):
            buf[:, names.index(f"f_dc_{k}")] = dc[k]
        hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n%send_header\n"
               % (n, "".join(f"property float {p}\n" for p in names)))
        with open(out / f"{name}.ply", "wb") as f:
            f.write(hdr.encode("ascii"))
            f.write(buf.astype("<f4").tobytes())
        print(f"[recolor] {name:28s} -> displayed {np.round(0.5 + SH_C0 * dc, 3)}")
    print(f"[recolor] wrote {len(FINGERS)} finger plys -> {out}  (originals in {src} untouched)")


if __name__ == "__main__":
    main()
