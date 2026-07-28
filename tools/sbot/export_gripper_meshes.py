"""Export each StandardBots gripper link's mesh from standardbot.usd to its own file.

The AG-145 gripper geometry is embedded inline in
``robo_maker/sbot/assets/standardbot.usd`` -- there are no loose per-link STL/OBJ
files anywhere (the AG-145 ROS package ships only 5 shared *part* meshes under
mismatched names, and the flat URDF points at per-link STLs that don't exist on
disk). So this USD is the only source for distinct, assembled per-link geometry,
e.g. to feed a per-link Gaussian-splat pipeline (the GS render currently has no
gripper splat).

Each link's mesh is exported in that link's *local* frame (origin at the link's
joint frame), which matches the link-local convention the arm splats already use
and the per-body transforms Newton applies on import. Coordinates are metres:
despite the stage's ``metersPerUnit = 0.01``, the points are authored in metres
(a knuckle is ~49 x 40 x 12 mm at raw values), which is also how ``add_usd`` reads
them -- so raw point values are written through unscaled.

Run from the repo root (uses the newton .venv, which has pxr + trimesh):
    .venv/bin/python tools/sbot/export_gripper_meshes.py
    .venv/bin/python tools/sbot/export_gripper_meshes.py --collision --format stl
    .venv/bin/python tools/sbot/export_gripper_meshes.py --world --out /tmp/sbot_gripper
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import trimesh
from pxr import Usd, UsdGeom

DEFAULT_USD = pathlib.Path(
    "/home/pandaliza/parallax/robo_maker/sbot/assets/standardbot.usd"
)
DEFAULT_OUT = pathlib.Path(
    "/home/pandaliza/parallax/robo_maker/sbot/assets/gripper_meshes"
)


def find_gripper_links(stage: Usd.Stage) -> list[Usd.Prim]:
    """Every ``gripper_*`` link Xform that carries a mesh, in stage order."""
    links = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Xform" or "gripper" not in prim.GetName():
            continue
        if any(c.GetTypeName() == "Mesh" for c in prim.GetChildren()):
            links.append(prim)
    return links


def _matrix_to_link(
    mesh_prim: Usd.Prim, link_prim: Usd.Prim, xform_cache: UsdGeom.XformCache, *, world: bool
) -> np.ndarray:
    """4x4 (row-vector) transform taking mesh-local points to the output frame.

    ``world=False`` (default) maps mesh-local -> link-local, so the export is
    centred on the link's joint frame; ``world=True`` maps to world space.
    """
    mesh_to_world = xform_cache.GetLocalToWorldTransform(mesh_prim)
    if world:
        return np.array(mesh_to_world, dtype=np.float64)
    link_to_world = xform_cache.GetLocalToWorldTransform(link_prim)
    # Row-vector convention: p_link = p_mesh * mesh_to_world * link_to_world^-1.
    return np.array(mesh_to_world * link_to_world.GetInverse(), dtype=np.float64)


def _read_mesh(mesh_prim: Usd.Prim, transform: np.ndarray) -> trimesh.Trimesh:
    mesh = UsdGeom.Mesh(mesh_prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if not np.all(counts == 3):
        # Fan-triangulate any non-triangular faces (the sbot meshes are all tris,
        # but stay robust if a re-export introduces quads/n-gons).
        tris, offset = [], 0
        for n in counts:
            for k in range(1, n - 1):
                tris.append((indices[offset], indices[offset + k], indices[offset + k + 1]))
            offset += n
        faces = np.asarray(tris, dtype=np.int64)
    else:
        faces = indices.reshape(-1, 3)
    homogeneous = np.c_[points, np.ones(len(points))]
    verts = (homogeneous @ transform)[:, :3]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def export(
    usd_path: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    sources: tuple[str, ...],
    fmt: str,
    world: bool,
) -> None:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"could not open USD: {usd_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    xform_cache = UsdGeom.XformCache()
    frame = "world" if world else "link-local"

    links = find_gripper_links(stage)
    print(f"{usd_path.name}: {len(links)} gripper links -> {out_dir} ({frame}, .{fmt})")
    written = 0
    for link in links:
        link_name = link.GetName()
        for source in sources:  # e.g. "visuals", "collisions"
            mesh_prim = stage.GetPrimAtPath(link.GetPath().AppendChild(source))
            if not mesh_prim or mesh_prim.GetTypeName() != "Mesh":
                continue
            tri = _read_mesh(mesh_prim, _matrix_to_link(mesh_prim, link, xform_cache, world=world))
            suffix = "" if source == "visuals" else f"_{source}"
            out_path = out_dir / f"{link_name}{suffix}.{fmt}"
            tri.export(out_path)
            extent = tri.extents * 1000.0  # mm
            print(
                f"  {out_path.name:<44} {len(tri.vertices):>5}v {len(tri.faces):>5}f "
                f"bbox=[{extent[0]:.1f} {extent[1]:.1f} {extent[2]:.1f}]mm"
            )
            written += 1
    print(f"wrote {written} mesh files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=pathlib.Path, default=DEFAULT_USD)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--format", default="obj", choices=("obj", "stl", "ply"))
    parser.add_argument(
        "--collision", action="store_true", help="also export each link's collision mesh"
    )
    parser.add_argument(
        "--world", action="store_true", help="export in world frame instead of link-local"
    )
    args = parser.parse_args()

    sources = ("visuals", "collisions") if args.collision else ("visuals",)
    export(args.usd, args.out, sources=sources, fmt=args.format, world=args.world)


if __name__ == "__main__":
    main()
