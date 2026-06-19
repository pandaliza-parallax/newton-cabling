"""Export plug+jack as RAW fine tessellations (NO voxel-remesh) from the canonical
McMaster STEP files, in the build_collision mating frame (insertion +Z, leading face
at z=0, jack mouth at MOUTH_Z), so build_cad_rj45_usd.py can consume them.

Self-contained: reads the two source STEPs directly from the gs-sim-vla ethernet
asset dir (the source of truth) and vendors the few build_collision helpers it needs
(cascadio tessellation, the X->Z reorient, the cavity locator). No dependency on the
build_collision.py module (which moved out of the ethernet dir).

The raw mesh is non-watertight but consistently wound; Newton's narrow-band SDF
tolerates it directly, preserving the real sub-mm clearance the voxel-remesh ate.

Run (uses the newton .venv which has cascadio + trimesh):
    .venv/bin/python tools/cad_assets/raw_export.py
"""

import os
import tempfile

import cascadio
import numpy as np
import trimesh

# Canonical source STEPs (the real McMaster parts), the single source of truth.
ETH = "/home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/ethernet"
PLUG_STEP = os.path.join(ETH, "9953K216_Category 5E Ethernet Cord.STEP")
JACK_STEP = os.path.join(ETH, "1422N17_Panel-Mount Data Adapter.STEP")

OUT = "/home/pandaliza/parallax/newton-cabling/tools/cad_assets/src"

# x-axis is the cord long axis; (x,y,z) -> (-z, y, x) sends +x (insertion) -> +z.
R_X_TO_Z = np.array(
    [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float
)

MOUTH_Z = 0.030  # jack cavity mouth height in the hole-mesh frame (m)


def load_step(path, lin=0.00025, ang=0.15):
    """Tessellate a STEP via cascadio (OpenCASCADE) -> a single concatenated trimesh."""
    glb = tempfile.mktemp(suffix=".glb")
    cascadio.step_to_glb(path, glb, lin, ang)
    s = trimesh.load(glb)
    return (
        trimesh.util.concatenate(list(s.geometry.values()))
        if isinstance(s, trimesh.Scene)
        else s
    )


def _cavity_center(jack):
    """Ray-cast the +Z port; return (cx, cy) of the full-width opening and the
    measured cavity depth (m), in the jack's current (centred) frame."""
    zmax = jack.bounds[1, 2]
    n = 81
    g = np.linspace(-8e-3, 8e-3, n)
    gx, gy = np.meshgrid(g, g)
    org = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, zmax + 5e-3)]
    dirs = np.tile([0, 0, -1.0], (len(org), 1))
    loc, ir, _ = jack.ray.intersects_location(org, dirs, multiple_hits=False)
    depth = np.full(gx.size, np.nan)
    depth[ir] = zmax - loc[:, 2]
    pts = org[:, :2][depth > 2e-3]  # rays that enter the cavity
    body = pts[np.abs(pts[:, 0]) > 3e-3]  # full-width region (exclude latch slot)
    cx = 0.0
    cy = 0.5 * (body[:, 1].min() + body[:, 1].max())
    return cx, cy, float(np.nanmedian(depth[depth > 2e-3]))


def export_plug(L=0.018):
    """PLUG: crop one connector head off the cord, reorient to +Z insertion, leading
    face at z=0, body centred in x,y (latch on +Y kept)."""
    cord = load_step(PLUG_STEP, lin=0.0005)
    x0 = cord.vertices[:, 0].min()
    head = cord.slice_plane(
        plane_origin=[x0 + L, 0, 0], plane_normal=[-1, 0, 0], cap=True
    )
    head.apply_transform(R_X_TO_Z)
    head.apply_translation([0, 0, -head.vertices[:, 2].min()])  # leading face -> z=0
    # CLEAN to the OUTER SHELL: the molded-cord tessellation is ~1300 components -- the
    # outer housing + boot, plus the 8 internal contact pins, 8 contact blades, the inner
    # carrier, and tessellation specks. Only the outer shell collides/renders; the internal
    # metalwork just makes the plug look like a lumpy mess (floating pins) in the viewer.
    # Keep the large outer components (>2% of verts), drop the internal pins + debris.
    comps = head.split(only_watertight=False)
    nv = len(head.vertices)
    outer = [c for c in comps if len(c.vertices) > 0.02 * nv]
    head = trimesh.util.concatenate(outer)
    print(f"  cleaned plug: kept {len(outer)}/{len(comps)} components (outer shell), "
          f"{len(head.vertices)} verts", flush=True)
    v = head.vertices
    ins = v[(v[:, 2] > 0.001) & (v[:, 2] < 0.011)]
    body = ins[np.abs(ins[:, 0]) > 0.003]
    cy = 0.5 * (body[:, 1].min() + body[:, 1].max())
    head.apply_translation([0, -cy, 0])
    head.merge_vertices()
    trimesh.repair.fix_normals(head)
    head.export(os.path.join(OUT, "plug_raw.obj"))
    bb = np.round((head.bounds[1] - head.bounds[0]) * 1000, 1).tolist()
    print(f"plug_raw: tris={len(head.faces)} bbox_mm={bb}", flush=True)


def export_jack():
    """JACK: centre, locate the cavity, crop to the receptacle block around it
    (drops panel flanges + far port, keeps the catch), reorient so mouth -> MOUTH_Z."""
    jack = load_step(JACK_STEP, lin=0.0005)
    jack.apply_translation(-jack.bounds.mean(axis=0))
    cx, cy, _ = _cavity_center(jack)
    zmax = jack.bounds[1, 2]
    for o, n in [
        ([cx + 0.0105, 0, 0], [-1, 0, 0]),
        ([cx - 0.0105, 0, 0], [1, 0, 0]),
        ([0, cy + 0.0105, 0], [0, -1, 0]),
        ([0, cy - 0.0105, 0], [0, 1, 0]),
        ([0, 0, zmax - 0.016], [0, 0, 1]),
    ]:
        jack = jack.slice_plane(plane_origin=o, plane_normal=n, cap=True)
    mouth_z = jack.bounds[1, 2]
    jack.apply_translation([-cx, -cy, MOUTH_Z - mouth_z])
    jack.merge_vertices()
    trimesh.repair.fix_normals(jack)
    jack.export(os.path.join(OUT, "jack_raw.obj"))
    bb = np.round((jack.bounds[1] - jack.bounds[0]) * 1000, 1).tolist()
    print(f"jack_raw: tris={len(jack.faces)} bbox_mm={bb}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    export_plug()
    export_jack()
    print("DONE", flush=True)
