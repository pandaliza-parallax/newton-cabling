"""Build a CLEAN, idealized cad_rj45.usd that LOOKS like the real McMaster parts
(1422N17 panel-mount jack + 9953K216 Cat5e RJ45 plug) without the messy multi-body
STEP tessellation.

Rationale (user choice): the raw STEP tessellations are 500-1300+ disjoint solids
(internal pins, contact blades, molded boots) and the collision-carve adds voxel
roughness, so they never render like the crisp McMaster CAD drawings. Instead we model
clean parametric primitives SIZED TO THE REAL PART DIMENSIONS, booleaned into watertight
solids. Crisp in rerun, light + robust for SDF collision, recognizable as RJ45.

Frame (Newton/rig): +Y = insertion, +Z = up, X = width. Lengths in metres.
  * Plug leading face at y=0, body back to y=-12mm, boot trailing (y<-12).
  * Jack mouth (flange face) at y=0, RJ45 cavity floor at y=+12mm (INSERT_DEPTH).
  * Latch is a SEPARATE cantilever body (/World/Latch) on a revolute hinge, like the
    original Newton example. LATCH_DOWN puts it on -Z with the jack keyway on -Z, matching
    the 1422N17 receptacle as drawn (contacts top, latch keyway bottom).

Prims authored: /World/Socket, /World/Plug, /World/Latch (same layout the rig + RL env
load). Run:  .venv/bin/python tools/cad_assets/build_cad_rj45_clean.py
"""

import os

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "newton_cabling", "assets", "cad_rj45.usd")

MM = 1.0e-3
INSERT_DEPTH = 0.012      # mouth -> cavity floor (matches the real ~12mm cavity)
LATCH_DOWN = True         # latch + jack keyway on -Z (matches the 1422N17 drawing)

# Real 8P8C plug + 1422N17 jack dimensions (mm), from the STEP + the standard.
PLUG_W = 11.68            # body width  (X)
PLUG_H = 8.0              # body height (Z)
PLUG_L = 12.0             # insertion length (Y), leading face -> body back
BOOT_L = 11.0             # molded strain-relief boot length behind the body
CLR = 0.35               # plug<->cavity clearance per side (snug, real RJ45 ~0.1-0.4)

FLANGE_W, FLANGE_H, FLANGE_T = 26.0, 31.0, 2.5   # square D-flange face + thickness
JACK_W, JACK_H, JACK_D = 19.0, 24.0, 16.0        # body behind the flange
MOUNT_DIA, MOUNT_OFF = 3.5, 12.0                  # 0.138in holes, diagonal offset


def box(sx, sy, sz, center=(0, 0, 0)):
    b = trimesh.creation.box(extents=(sx, sy, sz))
    b.apply_translation(center)
    return b


def cyl(radius, height, center, axis="y"):
    c = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    if axis == "y":
        c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    c.apply_translation(center)
    return c


def build_plug():
    """Clean RJ45 plug: a rectangular 8P8C body + a distinct, narrower rounded molded
    boot stepping down to a cable stub. Newton frame, metres (leading face y=0)."""
    s = MM
    # body: the rectangular plug housing (the part that inserts). Leading face at y=0.
    body = box(PLUG_W * s, PLUG_L * s, PLUG_H * s, center=(0, -PLUG_L * s / 2, 0))
    # molded boot: a rounded collar clearly NARROWER than the body (a real strain relief),
    # stepping down behind the body, then a thin cable stub. Built as two rounded boxes.
    collar = box(8.0 * s, 6.0 * s, 6.5 * s, center=(0, -(PLUG_L + 3.0) * s, -0.4 * s))
    cable = cyl(2.6 * s, 8.0 * s, center=(0, -(PLUG_L + 9.0) * s, -0.4 * s))
    plug = body.union(collar).union(cable)
    return plug


# Stepped keyway (the "what I want" cross-section): a symmetric 2-step pyramid below the
# body, pointing away from the body face. Each step (half-width, depth) in mm.
KEYWAY_STEPS = [(3.0, 1.2), (1.6, 1.2)]   # (x half-width, z depth) per step, wide->narrow


def build_latch():
    """The stepped KEYWAY is the articulated latch (/World/Latch). It's the symmetric
    2-step pyramid the user drew at the bottom-centre of the cross-section, extruded along
    the insertion length, hinged at the body face so the revolute joint gives it spring/flex.
    Returns (mesh, hinge_xyz), Newton frame, metres. Latch verts are absolute; the USD author
    offsets them by the hinge."""
    s = MM
    sgn = -1.0 if LATCH_DOWN else 1.0
    zface = sgn * PLUG_H * s / 2          # body face the keyway springs from
    hinge = np.array([0.0, -1.5 * s, zface])
    y0, y1 = -1.5 * s, -10.5 * s          # the keyway runs most of the body length
    ylen, ymid = abs(y1 - y0), (y0 + y1) / 2
    ov = 0.1 * s                          # overlap so consecutive steps union cleanly
    parts, z = [], zface
    for hx, dz in KEYWAY_STEPS:
        step = box(2 * hx * s, ylen, dz * s + ov, center=(0, ymid, z + sgn * dz * s / 2))
        parts.append(step)
        z = z + sgn * dz * s
    latch = parts[0]
    for p in parts[1:]:
        latch = latch.union(p)
    return latch, hinge


def build_jack():
    """Clean panel-mount jack: square D-flange + body, with a keyed RJ45 cavity carved
    from the mouth (y=0) to depth INSERT_DEPTH, + 2 mounting holes. Newton frame, metres."""
    s = MM
    sgn = -1.0 if LATCH_DOWN else 1.0
    # solid: flange plate (front) + body box (behind), unioned.
    flange = box(FLANGE_W * s, FLANGE_T * s, FLANGE_H * s, center=(0, FLANGE_T * s / 2, 0))
    bodyj = box(JACK_W * s, JACK_D * s, JACK_H * s, center=(0, JACK_D * s / 2, 0))
    solid = flange.union(bodyj)
    # cavity (subtract): main RJ45 rect + a latch keyway notch on the latch side, extruded
    # from in front of the mouth (y=-1mm) to the cavity floor (y=+INSERT_DEPTH).
    cav_w = (PLUG_W + 2 * CLR) * s
    cav_h = (PLUG_H + 2 * CLR) * s
    cy0, cy1 = -1.0 * s, INSERT_DEPTH
    rect = box(cav_w, (cy1 - cy0), cav_h, center=(0, (cy0 + cy1) / 2, 0))
    # stepped keyway: mirror the latch's KEYWAY_STEPS (+ clearance), extruded full depth, so
    # the stepped latch slides in. Steps accumulate from the body face down past the cavity.
    cavity = rect
    z = sgn * PLUG_H * s / 2
    for hx, dz in KEYWAY_STEPS:
        kw = box(2 * (hx * s + CLR * s), (cy1 - cy0), dz * s + 2 * CLR * s,
                 center=(0, (cy0 + cy1) / 2, z + sgn * dz * s / 2))
        cavity = cavity.union(kw)
        z = z + sgn * dz * s
    jack = solid.difference(cavity)
    # mounting holes through the flange (diagonal corners, like the drawing)
    for sx, sz in [(-1, 1), (1, -1)]:
        hole = cyl(MOUNT_DIA * s / 2, (FLANGE_T + 1) * s,
                   center=(sx * MOUNT_OFF * s, FLANGE_T * s / 2, sz * (FLANGE_H / 2 - 4.0) * s))
        jack = jack.difference(hole)
    return jack


def add_mesh_prim(stage, path, mesh, translate=None):
    geom = UsdGeom.Mesh.Define(stage, path)
    v = mesh.vertices.astype(np.float32)
    if translate is not None:
        t = np.asarray(translate, np.float32)
        v = v - t
        geom.AddTranslateOp().Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
    f = mesh.faces.astype(np.int32)
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    lo, hi = v.min(0), v.max(0)
    geom.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
    return geom


def report(name, m):
    bb = (m.bounds[1] - m.bounds[0]) * 1000
    print(f"  {name}: verts={len(m.vertices)} faces={len(m.faces)} watertight={m.is_watertight} "
          f"bbox_mm=[{bb[0]:.1f},{bb[1]:.1f},{bb[2]:.1f}]")


def main():
    print("building clean idealized cad_rj45 (sized to the real McMaster parts) ...")
    plug = build_plug()
    latch, hinge = build_latch()
    jack = build_jack()
    report("SOCKET(jack)", jack)
    report("PLUG", plug)
    report("LATCH", latch)
    print(f"  latch hinge (m): ({hinge[0]:.4f}, {hinge[1]:.4f}, {hinge[2]:.4f})  "
          f"axis=+x  side={'-Z (down)' if LATCH_DOWN else '+Z (up)'}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    stage = Usd.Stage.CreateNew(OUT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    add_mesh_prim(stage, "/World/Socket", jack)
    add_mesh_prim(stage, "/World/Plug", plug)
    add_mesh_prim(stage, "/World/Latch", latch, translate=hinge)
    stage.GetRootLayer().Save()
    print(f"\nwrote {OUT}")
    print(f"# LATCH hinge for LatchSpec: hinge_axis=(1,0,0), "
          f"hinge_offset_meters=({hinge[0]:.5f}, {hinge[1]:.5f}, {hinge[2]:.5f})")


if __name__ == "__main__":
    main()
