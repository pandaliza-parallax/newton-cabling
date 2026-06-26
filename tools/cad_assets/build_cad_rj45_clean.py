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
LATCH_DOWN = False # latch is a clip on +Z (top), like the demo + drawing

# Real 8P8C plug + 1422N17 jack dimensions (mm), from the STEP + the standard.
PLUG_W = 11.68            # body width  (X)
PLUG_H = 8.0              # body height (Z)
PLUG_L = 12.0             # insertion length (Y), leading face -> body back
BOOT_L = 11.0             # molded strain-relief boot length behind the body
CLR = 0.4                # plug<->cavity clearance per side (snug, > contact gap)
# Lead-in chamfer at the cavity mouth: flare the bore into a funnel (wide at the face,
# tapering to bore size CHAMFER_D deep) so a plug arriving a mm or two off-centre / a few
# degrees tilted slides down the slope and self-centres into the 0.4mm-clearance bore
# instead of ramming the flat flange face. Real RJ45 jacks have this; without it the base
# controller stalls at the mouth (gap=12mm) on the tilted curriculum stages.
CHAMFER_W = 2.5          # mm flare per side at the face
CHAMFER_D = 3.0          # mm taper depth (face -> bore size); lead-in angle ~40deg

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


# Demo-style RJ45 latch (matches example_contacts_rj45_plug + the McMaster drawing): a thin
# spring-clip on TOP (+Z), hinged at the front near the leading face, rising back to a raised
# press-tab, with a LOCALIZED catch shoulder that rides over a lip inside the jack and snaps
# back to lock. The plug BODY seats in the plain rectangular bore; only this small clip
# interacts with the lip, so it deflects/clicks WITHOUT jamming (unlike the old keyway bar).
# Sized to the real RJ45 latch (demo + McMaster drawing proportions): a SLANTED spring-clip
# resting angled up (front low at the body top, back press-tab raised ~3.4mm proud), with a
# catch shoulder on the underside. On insertion it flattens over the jack ledge then springs
# back up to lock (the demo "deflect on entry and latch").
LATCH_HINGE_Y = -1.0   # mm, hinge at the front (near the leading face), on the body top
LATCH_LEN = 13.0       # mm, clip length back toward the boot (longer, like the real latch)
LATCH_W = 6.0          # mm, clip width (x)
LATCH_T = 1.0          # mm, blade thickness
LATCH_RISE = 3.4       # mm, raised back press-tab proud of the body top (the slant)
CATCH_Y = -4.5         # mm (latch-relative) catch shoulder centre (underside)
CATCH_LEN = 2.2        # mm catch shoulder length
CATCH_DROP = 0.9       # mm the catch shoulder hangs below the slanted blade
# Jack catch ledge (protrudes UP from the bore, deep in): the catch shoulder rides over it.
LIP_Y = 2.0            # mm deep — where the catch shoulder actually travels (plug seats ~7mm,
#                        shoulder ends ~2.5mm in; a deeper ledge sits at the hinge + only jams)
LIP_RISE = 0.8         # mm the ledge protrudes up (sets the deflection; rigid-VBD: bigger = jams)
LIP_LEN = 1.6          # mm along insertion

ZTOP = PLUG_H / 2      # body top (mm); latch is on +Z


def build_latch():
    """Real-proportioned RJ45 spring-clip on TOP (+Z), SLANTED: a blade hinged at the front
    (low, body top) rising at an angle to a raised press-tab at the back, with a catch shoulder
    on the underside. Flattens over the jack ledge on insertion and snaps back up (the demo
    mechanism). Returns (mesh, hinge_xyz); verts absolute (USD author offsets by the hinge)."""
    s = MM
    ztop = ZTOP * s
    hinge = np.array([0.0, LATCH_HINGE_Y * s, ztop])
    wx, t = LATCH_W * s / 2, LATCH_T * s
    yf, yb = LATCH_HINGE_Y * s, (LATCH_HINGE_Y - LATCH_LEN) * s    # front, back
    zf, zb = ztop, ztop + LATCH_RISE * s                          # front low -> back high (slant)
    pts = []
    for sx in (-wx, wx):                                          # slanted slab (convex hull)
        pts += [[sx, yf, zf], [sx, yb, zb], [sx, yf, zf + t], [sx, yb, zb + t]]
    slab = trimesh.Trimesh(vertices=np.array(pts), process=False).convex_hull
    zc = ztop + LATCH_RISE * s * (LATCH_HINGE_Y - CATCH_Y) / LATCH_LEN  # blade underside z at the catch
    catch = box(2 * wx, CATCH_LEN * s, CATCH_DROP * s, center=(0, CATCH_Y * s, zc - CATCH_DROP * s / 2))
    latch = slab.union(catch)
    return latch, hinge


def build_jack():
    """Clean panel-mount jack: square D-flange + body, a plain RJ45 bore for the plug body +
    a TOP latch channel with a catch lip the clip rides over, + 2 mounting holes. Metres."""
    s = MM
    # solid: flange plate (front) + body box (behind), unioned.
    flange = box(FLANGE_W * s, FLANGE_T * s, FLANGE_H * s, center=(0, FLANGE_T * s / 2, 0))
    bodyj = box(JACK_W * s, JACK_D * s, JACK_H * s, center=(0, JACK_D * s / 2, 0))
    solid = flange.union(bodyj)
    cav_w = (PLUG_W + 2 * CLR) * s
    cav_h = (PLUG_H + 2 * CLR) * s
    cy0, cy1 = -1.0 * s, INSERT_DEPTH
    mid, depth = (cy0 + cy1) / 2, (cy1 - cy0)
    rect = box(cav_w, depth, cav_h, center=(0, mid, 0))                 # plain body bore
    # LEAD-IN CHAMFER: a funnel frustum at the mouth — wide (bore + CHAMFER_W per side) at
    # the face, tapering to bore size CHAMFER_D deep — so a slightly misaligned plug rides
    # the slope into the bore instead of jamming on the flat flange (see CHAMFER_* above).
    cw, cd = CHAMFER_W * s, CHAMFER_D * s
    fpts = []
    for sx in (-1, 1):
        for sz in (-1, 1):
            fpts.append([sx * (cav_w / 2 + cw), cy0, sz * (cav_h / 2 + cw)])   # wide @ face
            fpts.append([sx * cav_w / 2, cy0 + cd, sz * cav_h / 2])            # bore size, deep
    funnel = trimesh.Trimesh(vertices=np.array(fpts), process=False).convex_hull
    # latch channel on top: tall enough for the slanted clip AND its flatten/flip swing.
    chan_h = (LATCH_RISE + LATCH_T + CLR + 0.6) * s
    chan = box(2 * (LATCH_W / 2 + CLR) * s, depth, chan_h, center=(0, mid, cav_h / 2 + chan_h / 2))
    jack = solid.difference(rect.union(chan).union(funnel))
    # NOTE: a physics catch-ledge (LIP_*) was trialled here for a real latch "click" but
    # REMOVED — in rigid VBD a box ledge meets the box catch shoulder head-on (horizontal
    # contact normal), so it jams the plug rather than ramping the latch over: at the shoulder's
    # travel (~y=2mm) seating collapsed (plug stalled at 3mm); placed deep (y=6mm) it sat at the
    # hinge and only grazed (0.6deg) while still costing ~18pts of seating. A real click needs
    # the plastic to FLEX (soft-body), which rigid VBD doesn't model. For a latch visual use the
    # kinematic demo (rl/record_cad_latch.py). LIP_* kept as constants for any future soft latch.
    # mounting holes through the flange (diagonal corners, like the drawing)
    for sx, sz in [(-1, 1), (1, -1)]:
        hole = cyl(MOUNT_DIA * s / 2, (FLANGE_T + 1) * s,
                   center=(sx * MOUNT_OFF * s, FLANGE_T * s / 2, sz * (FLANGE_H / 2 - 4.0) * s))
        jack = jack.difference(hole)
    return jack


def add_mesh_prim(stage, path, mesh, translate=None):
    geom = UsdGeom.Mesh.Define(stage, path)
    # UNMERGE vertices: give every triangle its own 3 verts so the viewer (which computes
    # vertex normals from points+indices when none are authored) produces FLAT per-face
    # normals -> a matte CAD look instead of the glossy smooth-shaded gradient on the big
    # flat faces. Geometry/SDF is identical (same surface), only the shading changes.
    v = mesh.vertices[mesh.faces].reshape(-1, 3).astype(np.float32)
    f = np.arange(len(v), dtype=np.int32).reshape(-1, 3)
    if translate is not None:
        t = np.asarray(translate, np.float32)
        v = v - t
        geom.AddTranslateOp().Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
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
