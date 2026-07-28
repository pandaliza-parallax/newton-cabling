"""Base-aware cut: like cut_splat_by_links.py, but ADDS the gripper PALM as a 9th
target so palm Gaussians are kept (routed to wrist_3) instead of dropped.

Difference from cut_splat_by_links.py:
  * targets = the 8 finger links (assembled from USD @ --theta, world frame) PLUS
    gripper_base_link, whose posed CAD is read from --base-obj (already in the SAME
    world frame as the registered splat -- e.g. gripper_base_registered.obj).
  * palm Gaussians (nearest the base) are written into WRIST_3's local frame, because
    gripper_base_link is RIGID to wrist_3 (it collapses into the wrist_3 body in Newton).
    Output file: gripper_base_link.ply -- merge it into the wrist_3 splat to render
    (it rides wrist_3 exactly, like add_palm_to_wrist3.py's synthetic palm).
  * fingers are written link-local, exactly as the original cutter.
  * --threshold is only for dropping true floaters (points far from ALL 9 targets);
    with a tight registration keep it loose (~15mm) so only reconstruction noise drops.

The splat is assumed ALREADY registered (in world frame); pass --matrix only if it isn't.

Run from the repo root (newton .venv):
    .venv/bin/python tools/sbot/cut_splat_by_links_base.py \
        --splat  /home/pandaliza/parallax/sbot_assets/base/standalone_gripper_3dgs_registered_tight.ply \
        --base-obj /home/pandaliza/parallax/sbot_assets/base/gripper_base_registered.obj \
        --theta -0.6 --threshold 0.015 \
        --out /home/pandaliza/parallax/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut_v3
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from cut_splat_by_links import (  # noqa: E402
    FINGER_LINKS, _mat_to_quat_wxyz, _quat_mul, assemble_links, assign_to_links,
    col, read_ply, write_flat_ply,
)
from export_gripper_meshes import DEFAULT_USD  # noqa: E402

WRIST = "wrist_3_link"
PALM = "gripper_base_link"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splat", type=pathlib.Path, required=True)
    ap.add_argument("--base-obj", type=pathlib.Path, required=True,
                    help="posed gripper_base_link CAD, in the SAME world frame as the splat")
    ap.add_argument("--matrix", type=pathlib.Path, default=None,
                    help="4x4 splat->world (only if --splat is NOT already registered)")
    ap.add_argument("--usd", type=pathlib.Path, default=DEFAULT_USD)
    ap.add_argument("--theta", type=float, default=-0.6)
    ap.add_argument("--threshold", type=float, default=0.015,
                    help="max dist (m) to nearest of the 9 targets; only true floaters drop")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    # ---- read splat
    props, data = read_ply(args.splat)
    xyz = np.column_stack([col(props, data, a) for a in ("x", "y", "z")])
    scale = np.column_stack([col(props, data, f"scale_{i}") for i in range(3)])
    rot = np.column_stack([col(props, data, f"rot_{i}") for i in range(4)])
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    f_dc = np.column_stack([col(props, data, f"f_dc_{i}") for i in range(3)])
    opacity = col(props, data, "opacity")

    if args.matrix is not None:
        M = np.loadtxt(args.matrix).reshape(4, 4)
        A, t = M[:3, :3], M[:3, 3]
        s = np.cbrt(abs(np.linalg.det(A)))
        U, _, Vt = np.linalg.svd(A / s)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        print(f"registration: scale={s:.5g}  |t|={np.linalg.norm(t):.4g}")
        xyz = s * (xyz @ R.T) + t
        scale = scale + np.log(s)
        rot = _quat_mul(_mat_to_quat_wxyz(R), rot)

    # ---- targets: 8 fingers (world @ theta) + wrist_3 frame (for palm routing) + posed base
    asm = assemble_links(args.usd, args.theta, FINGER_LINKS + (WRIST,))
    w3 = asm.pop(WRIST)                                   # routing frame for the palm
    links = asm                                          # 8 finger targets (with R,t)
    base_mesh = trimesh.load(str(args.base_obj), process=False)

    targets = dict(links)
    targets[PALM] = {"mesh": base_mesh}                  # 9th target (mesh only; routed to wrist_3)
    names = list(targets)
    link_idx, dist = assign_to_links(xyz, targets)
    keep = dist <= args.threshold
    print(f"{len(xyz)} gaussians; {keep.sum()} within {args.threshold * 1000:.0f}mm of a target "
          f"({(~keep).sum()} dropped as floaters)")

    args.out.mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(names):
        sel = keep & (link_idx == j)
        if not sel.any():
            print(f"  {name:<40} 0 gaussians"); continue
        if name == PALM:                                 # palm -> wrist_3 local frame
            R, t = w3["R"], w3["t"]
            short = "gripper_base_link"
        else:                                            # finger -> its own local frame
            R, t = links[name]["R"], links[name]["t"]
            short = name.replace("gripper_", "")
        local_xyz = (xyz[sel] - t) @ R
        local_rot = _quat_mul(_mat_to_quat_wxyz(R.T), rot[sel])
        write_flat_ply(args.out / f"{short}.ply", local_xyz, scale[sel], local_rot, f_dc[sel], opacity[sel])
        tag = "  (-> wrist_3 local)" if name == PALM else ""
        print(f"  {short + '.ply':<40} {sel.sum():>5} gaussians{tag}")


if __name__ == "__main__":
    main()
