"""Synthesize per-link gripper Gaussian splats by surface-sampling the CAD meshes.

The real TRELLIS finger splat is too inflated/asymmetric to cut into the 8 thin
finger sub-links (its Gaussians sit ~18 mm off the CAD surface). This sidesteps it:
for each gripper link we sample its exact CAD surface (from standardbot.usd, in the
link's local frame) and emit a dense cloud of small Gaussians that tile it. The
geometry is therefore exact and articulates correctly; the look is a flat CAD
colour rather than a photoreal texture.

Output matches the arm splats in sbot_gs/flat: the flat (SH deg-0) 14-property
layout, in each link's local frame, so the link's Newton body transform IS its
splat transform (1:1 pass-through, like scripts/record_sbot_gs.py). Defaults are tuned to
the arm splats -- ~1.5 mm isotropic Gaussians, solid opacity, the AG-145's dark
grey (colour 0.2, f_dc ~ -1.06).

Run from the repo root (newton .venv: pxr + trimesh):
    .venv/bin/python tools/sbot/synth_splats_from_meshes.py \
        --out ../parallax-demo-isaac-lab/assets/sbot_gs/flat
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import trimesh
from pxr import Usd, UsdGeom

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from cut_splat_by_links import LINK_SETS, write_flat_ply
from export_gripper_meshes import DEFAULT_USD, _matrix_to_link, _read_mesh

SH_C0 = 0.28209479177387814  # SH degree-0 basis: rendered colour = 0.5 + C0 * f_dc


def link_local_mesh(
    stage: Usd.Stage, name: str, xform_cache: UsdGeom.XformCache
) -> trimesh.Trimesh:
    link = stage.GetPrimAtPath(f"/sbot/{name}")
    mesh_prim = stage.GetPrimAtPath(f"/sbot/{name}/visuals")
    return _read_mesh(mesh_prim, _matrix_to_link(mesh_prim, link, xform_cache, world=False))


def synth_link(
    mesh: trimesh.Trimesh, *, spacing: float, opacity: float, color: float, flatten: float
) -> dict:
    """Sample ``mesh`` into a tiling Gaussian cloud (all arrays in mesh-local frame)."""
    n = max(500, round(mesh.area / (spacing * spacing)))
    points, face_idx = trimesh.sample.sample_surface(mesh, n)
    normals = mesh.face_normals[face_idx]

    # Disc-like Gaussians: two axes ~spacing in the surface tangent plane, the third
    # squashed by `flatten` along the surface normal -> a thin shell, fewer needed.
    log_tan = np.log(spacing)
    log_norm = np.log(spacing * flatten)
    scale = np.tile([log_tan, log_tan, log_norm], (n, 1))
    rot = _normals_to_quat(normals)  # local +z -> surface normal

    f_dc = np.full((n, 3), (color - 0.5) / SH_C0)
    opacities = np.full(n, opacity)
    return {"xyz": points, "scale": scale, "rot": rot, "f_dc": f_dc, "opacity": opacities}


def _normals_to_quat(normals: np.ndarray) -> np.ndarray:
    """Quaternions (w,x,y,z) rotating local +z onto each unit ``normal``."""
    z = np.array([0.0, 0.0, 1.0])
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    dot = n @ z
    axis = np.cross(np.tile(z, (len(n), 1)), n)
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
    out = np.zeros((len(n), 4))
    out[:, 0] = 1.0  # identity where normal == +z
    ok = axis_norm[:, 0] > 1e-8
    a = axis[ok] / axis_norm[ok]
    ang = np.arccos(np.clip(dot[ok], -1.0, 1.0))
    out[ok, 0] = np.cos(ang / 2)
    out[ok, 1:] = a * np.sin(ang / 2)[:, None]
    # antiparallel (normal == -z): 180 deg about any in-plane axis
    flip = (dot < -0.999999)
    out[flip] = [0.0, 1.0, 0.0, 0.0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usd", type=pathlib.Path, default=DEFAULT_USD)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("gripper_gs"))
    ap.add_argument(
        "--links", choices=tuple(LINK_SETS), default="fingers",
        help="which links to synthesize (default: the 8 finger links)",
    )
    ap.add_argument(
        "--spacing", type=float, default=0.0015, help="Gaussian size / sample spacing (m)"
    )
    ap.add_argument("--opacity", type=float, default=4.0, help="pre-sigmoid opacity (4.0 -> ~0.98)")
    ap.add_argument("--color", type=float, default=0.2, help="grey level 0..1 (AG-145 is dark)")
    ap.add_argument(
        "--flatten", type=float, default=0.25, help="normal-axis squash (disc Gaussians)"
    )
    args = ap.parse_args()

    stage = Usd.Stage.Open(str(args.usd))
    xform_cache = UsdGeom.XformCache()
    args.out.mkdir(parents=True, exist_ok=True)

    link_names = LINK_SETS[args.links]
    print(
        f"synthesizing {len(link_names)} link splats -> {args.out} "
        f"(spacing={args.spacing * 1000:.1f}mm)"
    )
    total = 0
    for name in link_names:
        mesh = link_local_mesh(stage, name, xform_cache)
        g = synth_link(
            mesh, spacing=args.spacing, opacity=args.opacity, color=args.color, flatten=args.flatten
        )
        short = name.replace("gripper_", "")
        write_flat_ply(
            args.out / f"{short}.ply", g["xyz"], g["scale"], g["rot"], g["f_dc"], g["opacity"]
        )
        total += len(g["xyz"])
        print(f"  {short + '.ply':<38} {len(g['xyz']):>6} gaussians  ({mesh.area * 1e4:.1f}cm2)")
    print(f"done: {total} gaussians across {len(link_names)} links")


if __name__ == "__main__":
    main()
