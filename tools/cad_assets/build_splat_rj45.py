"""Build splat_rj45.usd: a Newton connector rig whose PLUG is the REAL scanned cable+plug mesh,
sized to match the GS splat we actually render (headA_plug_ccreg.ply), instead of cad_rj45.usd's
idealized ~25mm plug body which has no cable and visibly mismatches the splat (see
tools/debug_cable_newton_vs_gs.py --plug-only: the splat shows ~68mm of real cable+plug, the CAD
mesh shows a bare 25mm block).

Provenance of the Plug geometry:
  * Source: newton_cabling/assets/ethernet/assets/objects/Ethernet_cable_headA/mesh.usd, the raw
    photogrammetry scan region already cropped down to "headA" (17264 verts) from the larger
    Ethernet_cable_corrected/mesh.usd scan (59116 verts).
  * headA_frame.json documents the mating-face tip (``tip_m``) and outward axis (``axis_out``) of
    this region, measured in the scan's own coordinates, plus the cylinder crop that produced it
    (60mm long, 16mm radius). Re-applying that same cylinder to headA/mesh.usd keeps ~100% of its
    vertices (it was already cropped to ~this exact region) -- so no further cropping is needed,
    this mainly re-derives axis_out/tip as an axis-aligned filter for provenance/reproducibility.
  * headA_raw_scanframe.obj and headA_plugframe.obj are the SAME mesh (identical vertex/face count,
    1:1 index correspondence) in two frames: the raw scan frame and the canonical PLUG frame
    (mating face at the origin, insertion +Y -- the convention newton_cabling.connector.
    cad_rj45_connector() and gs_bridge's --conn-rpy/--conn-anchor calibration both assume). The
    rigid transform (R, t) between them is recovered here via Kabsch/Procrustes over all 17264
    vertex pairs (residual: mean 0.7 micron, max 1.5 micron -- a clean rigid registration, not an
    approximation), then applied to the (identically-cropped) headA/mesh.usd to get the final Plug
    in canonical frame. Sanity checks: R @ axis_out == (0,1,0) and R @ tip + t == (0,0,0) exactly.

Socket + Latch are copied UNCHANGED from scan_rj45.usd (already a real scan, already in the
correct canonical frame for cad_rj45_connector()'s prim paths) -- only the Plug changes here.

Run:
    .venv/bin/python tools/cad_assets/build_splat_rj45.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

import newton
import newton.usd

REPO_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "newton_cabling", "assets")
HEADA_DIR = os.path.join(REPO_ASSETS, "ethernet", "assets", "objects", "Ethernet_cable_headA")
RAW_OBJ = os.path.join(REPO_ASSETS, "ethernet", "headA_raw_scanframe.obj")
PLUG_OBJ = os.path.join(REPO_ASSETS, "ethernet", "headA_plugframe.obj")
SCAN_USD = os.path.join(REPO_ASSETS, "scan_rj45.usd")
OUT = os.path.join(REPO_ASSETS, "splat_rj45.usd")


def _load_obj_verts(path: str) -> np.ndarray:
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.array(verts, dtype=np.float64)


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid (R, t) minimizing ||R@src + t - dst||; src/dst are (N,3), index-corresponding."""
    cs, cd = src.mean(0), dst.mean(0)
    A, B = src - cs, dst - cd
    U, _, Vt_ = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt_.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt_.T @ D @ U.T
    t = cd - R @ cs
    return R, t


def recover_transform() -> tuple[np.ndarray, np.ndarray]:
    raw = _load_obj_verts(RAW_OBJ)
    plug = _load_obj_verts(PLUG_OBJ)
    assert raw.shape == plug.shape, (raw.shape, plug.shape)
    R, t = _kabsch(raw, plug)
    resid = np.linalg.norm(((R @ raw.T).T + t) - plug, axis=1)
    print(f"[splat_rj45] raw->plug transform residual: mean {resid.mean()*1e6:.2f}um "
          f"max {resid.max()*1e6:.2f}um  (over {len(raw)} verts)")
    return R, t


def build_plug_mesh(R: np.ndarray, t: np.ndarray):
    with open(os.path.join(HEADA_DIR, "headA_frame.json")) as f:
        fr = json.load(f)
    tip = np.array(fr["tip_m"], dtype=np.float64)
    axis = np.array(fr["axis_out"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    length_m = fr["crop"]["length_m"]
    radius_m = fr["crop"]["radius_m"]

    # sanity: this R,t must send axis_out -> +Y and tip -> the origin (the canonical plug frame).
    ax_err = np.linalg.norm(R @ axis - np.array([0.0, 1.0, 0.0]))
    tip_err = np.linalg.norm(R @ tip + t)
    print(f"[splat_rj45] R@axis_out error: {ax_err:.2e}   R@tip+t error: {tip_err:.2e} m")
    assert ax_err < 1e-6 and tip_err < 1e-6, "transform does not match the documented plug frame"

    stage = Usd.Stage.Open(os.path.join(HEADA_DIR, "mesh.usd"))
    prim = stage.GetPrimAtPath("/root/mesh/mesh")
    m = newton.usd.get_mesh(prim, load_normals=True)
    V = np.array(m.vertices, dtype=np.float64)
    I = np.array(m.indices, dtype=np.int64).reshape(-1, 3)
    N = np.array(m.normals, dtype=np.float64) if m.normals is not None else None

    # axis_out points OUT of the tip (away from the cable), so the body is at negative `along`;
    # crop to headA_frame.json's documented cylinder (a no-op today -- headA/mesh.usd is already
    # ~exactly this region -- kept so the crop is reproducible if a wider parent scan is swapped in).
    rel = V - tip
    along = rel @ axis
    perp = np.linalg.norm(rel - np.outer(along, axis), axis=1)
    keep_vert = (along >= -length_m) & (along <= 1e-3) & (perp <= radius_m)
    face_keep = keep_vert[I].all(axis=1)
    kept_tris = I[face_keep]
    used = np.unique(kept_tris)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    new_tris = remap[kept_tris]
    new_V, new_N = V[used], (N[used] if N is not None else None)
    print(f"[splat_rj45] crop kept {len(used)}/{len(V)} verts, {len(new_tris)}/{len(I)} faces")

    plug_V = (R @ new_V.T).T + t
    plug_N = (R @ new_N.T).T if new_N is not None else None
    ext = (plug_V.max(0) - plug_V.min(0)) * 1000.0
    print(f"[splat_rj45] Plug bbox (mm): {np.round(plug_V.min(0)*1000,1)} .. "
          f"{np.round(plug_V.max(0)*1000,1)}  extent {np.round(ext,1)}")
    return plug_V, new_tris, plug_N


def _copy_mesh_prim(stage, dst_path: str, src_stage: Usd.Stage, src_path: str) -> None:
    prim = src_stage.GetPrimAtPath(src_path)
    m = newton.usd.get_mesh(prim, load_normals=True)
    V = np.array(m.vertices, dtype=np.float32)
    I = np.array(m.indices, dtype=np.int32)
    N = np.array(m.normals, dtype=np.float32) if m.normals is not None else None
    geom = UsdGeom.Mesh.Define(stage, dst_path)
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(I))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(I) // 3, 3, np.int32)))
    if N is not None:
        geom.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N))
    lo, hi = V.min(0), V.max(0)
    geom.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])


def main() -> None:
    R, t = recover_transform()
    plug_V, plug_I, plug_N = build_plug_mesh(R, t)

    if os.path.exists(OUT):
        os.remove(OUT)
    stage = Usd.Stage.CreateNew(OUT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    scan_stage = Usd.Stage.Open(SCAN_USD)
    _copy_mesh_prim(stage, "/World/Socket", scan_stage, "/World/Socket")
    _copy_mesh_prim(stage, "/World/Latch", scan_stage, "/World/Latch")

    geom = UsdGeom.Mesh.Define(stage, "/World/Plug")
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(plug_V.astype(np.float32)))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(plug_I.reshape(-1).astype(np.int32)))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(plug_I), 3, np.int32)))
    if plug_N is not None:
        geom.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(plug_N.astype(np.float32)))
    lo, hi = plug_V.min(0), plug_V.max(0)
    geom.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])

    stage.GetRootLayer().Save()
    print(f"[splat_rj45] wrote {OUT}")


if __name__ == "__main__":
    main()
