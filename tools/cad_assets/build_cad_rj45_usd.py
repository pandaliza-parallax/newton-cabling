"""Build a Newton-ready merged USD (cad_rj45.usd) from the gs-sim-vla CAD meshes.

ADDITIVE: this does not touch the bundled `rj45_plug.usd` path. It reads the
watertight collision OBJs that gs-sim-vla's build_collision.py produced (real
McMaster parts: 9953K216 Cat5E plug, 1422N17 panel jack) and emits ONE usd with
prims /World/Socket, /World/Plug, /World/Latch so the existing connector rig +
RL env can load it exactly like the bundled asset.

Two reconciliations are baked here so the rest of the code is untouched:

1. AXIS: gs-sim-vla authored insertion along +Z (pocket opens +Z, plug leading
   face at z=0). The Newton rig assumes insertion along +Y. We rotate the user
   frame so the insertion direction (user -Z, leading-face-first) maps to Newton
   +Y, then translate the jack so its MOUTH plane sits at y=0 and the cavity
   floor at y=+0.012 (the measured 12 mm depth). The plug's leading face lands at
   y=0 (= the mouth reference the rig calls dy=0); seating moves it +12 mm.

2. LATCH: the user's plug is a single fused envelope (latch baked in), but the
   rig + RL kernels require a separate /World/Latch body. We author a tiny
   stand-in latch box that rides ~24 mm behind the plug origin (always outside
   the jack, which lives at y>=0), so it is physically negligible and every
   existing code path keeps working unchanged. RL success is depth/lateral/angle
   based, not latch-snap based, so this stand-in does not affect training.

Run (uses the newton .venv which has both trimesh + pxr):
    .venv/bin/python tools/cad_assets/build_cad_rj45_usd.py

Writes newton_cabling/assets/cad_rj45.usd and prints the geometry block to paste
into the RL asset profile.
"""

import os

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(HERE, "src")
OUT = os.environ.get("CAD_RJ45_OUT") or os.path.join(REPO, "newton_cabling", "assets", "cad_rj45.usd")

# Mating-frame constants measured by gs-sim-vla/build_collision.py (metres, user +Z frame).
MOUTH_Z = 0.030       # jack cavity mouth plane in the user hole frame
INSERT_DEPTH = 0.012  # mouth -> internal stop (cavity depth)

# user->Newton rotation: (x, y, z) -> (x, -z, y).
# Sends user +Z -> Newton -Y, so the user insertion direction (-Z, leading face
# first) maps to Newton +Y = the rig's "into the socket" direction. Right-handed.
R_USER_TO_NEWTON = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
)

TARGET_FACES = 400000  # decimate cap. Keep high: quadric decimation collapses the
#                        thin jack cavity walls and shrinks the mouth the plug enters.

# The two CAD parts were voxel-remeshed at the same 0.10mm pitch, so the plug outer
# envelope and the jack cavity snapped to IDENTICAL widths (0.00mm clearance in x) —
# a rigid SDF body can't enter a zero-clearance hole. Shrink the plug laterally about
# the cavity centre to restore a workable, realistic clearance (RJ45 is ~0.1-0.4mm;
# we use a touch more so the policy has room to find the hole). Recentred to the
# cavity so the clearance is symmetric.
PLUG_CLEARANCE_M = 0.00008  # minimal clearance -> largest plug the fit still guarantees inserts
#                             sim-friendly compromise (a bit looser than real, tight enough
#                             to keep the plug near full-size). Contact gap must be < this.


def load_obj(path):
    # process=True merges coincident verts so watertightness reflects real holes,
    # not the duplicate-vertex seams that marching-cubes OBJ export leaves behind.
    m = trimesh.load(path, process=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


def repair(m):
    """Clean topology + consistent winding for a good narrow-band SDF.

    Deliberately NO fill_holes: the jack cavity mouth is a legitimate boundary
    loop and fill_holes would cap it (sealing the hole the plug must enter).
    Newton's SDF is narrow-band, so an open pocket is fine — the bundled socket
    is open too.
    """
    m.merge_vertices()
    trimesh.repair.fix_winding(m)
    trimesh.repair.fix_normals(m)
    return m


def decimate(m, target):
    if len(m.faces) <= target:
        return repair(m)
    try:
        out = m.simplify_quadric_decimation(face_count=target)
        if len(out.faces) > 0:
            m = out
    except Exception as exc:  # fast_simplification not present, etc.
        print(f"  [decimate] skipped ({exc.__class__.__name__}: {exc})")
    return repair(m)


def latch_box(center, half):
    """Tiny axis-aligned box (8 verts, 12 tris) as the negligible stand-in latch."""
    c = np.asarray(center, float)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    verts = c + signs * half
    faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],   # -x, +x
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],   # -y, +y
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],   # -z, +z
    ], dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def add_mesh_prim(stage, path, mesh, translate=None):
    """Author a Mesh prim. If `translate` is given, set it as the prim's xform translate
    and store the verts RELATIVE to it -> the rig reads base_position=translate (used for
    the latch hinge) while the geometry stays in place. Newton reads both."""
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


def cavity_channel(jack, depth_min=0.006):
    """Map the cavity as a +Y ray-cast first-hit field and return (grid, open-mask,
    centre). 'open' = a ray that reaches at least `depth_min` into the cavity, i.e.
    the clear through-channel the plug must pass (excludes the mouth lips/chamfer
    that only go 1-4mm deep)."""
    N = 221
    g = np.linspace(-9e-3, 9e-3, N)
    gx, gz = np.meshgrid(g, g)
    org = np.c_[gx.ravel(), np.full(gx.size, -6e-3), gz.ravel()]
    loc, ir, _ = jack.ray.intersects_location(
        org, np.tile([0, 1.0, 0], (gx.size, 1)), multiple_hits=False)
    fy = np.full(gx.size, np.nan)
    fy[ir] = loc[:, 1]
    openm = (fy > depth_min).reshape(N, N)   # [z, x]
    op = openm.ravel()
    cx = float(org[op][:, 0].mean())
    cz = float(org[op][:, 2].mean())
    floor = float(np.nanmedian(fy[op]))
    return dict(g=g, openm=openm, cx=cx, cz=cz, floor=floor, jack=jack)


def inscribe_rect(ch, phx, phz):
    """Largest factor t such that the centred rectangle (cx±t*phx, cz±t*phz) is fully
    inside the open channel — i.e. the biggest plug-shaped rectangle that actually fits
    the rounded/lipped opening (the plug is rectangular, the hole is not)."""
    g, openm, cx, cz = ch["g"], ch["openm"], ch["cx"], ch["cz"]
    def fits(t):
        xs = cx + np.linspace(-t * phx, t * phx, 31)
        zs = cz + np.linspace(-t * phz, t * phz, 31)
        ix = np.clip(np.round((xs - g[0]) / (g[-1] - g[0]) * (len(g) - 1)).astype(int), 0, len(g) - 1)
        iz = np.clip(np.round((zs - g[0]) / (g[-1] - g[0]) * (len(g) - 1)).astype(int), 0, len(g) - 1)
        return openm[np.ix_(iz, ix)].all()
    t = 0.0
    for cand in np.linspace(1.5, 0.2, 80):
        if fits(cand):
            t = cand
            break
    return t


def cavity_depth_field(jack):
    """Per-(x,z) cavity depth: how far a +Y ray penetrates before hitting solid.
    Returns (grid g, depth field [z,x] in metres, NaN where the ray misses)."""
    N = 221
    g = np.linspace(-9e-3, 9e-3, N)
    gx, gz = np.meshgrid(g, g)
    org = np.c_[gx.ravel(), np.full(gx.size, -6e-3), gz.ravel()]
    loc, ir, _ = jack.ray.intersects_location(
        org, np.tile([0, 1.0, 0], (gx.size, 1)), multiple_hits=False)
    depth = np.full(gx.size, np.nan)
    depth[ir] = loc[:, 1]            # world-y of first solid = cavity floor at this (x,z)
    return g, depth.reshape(N, N)


# Follow the original Newton example (example_contacts_rj45_plug / rj45_plug.usd):
# a REAL separate articulated latch (cantilever on a revolute hinge) attached to the
# connector, NOT a fused single body or a dummy stand-in. STANDIN_LATCH=False splits
# the real latch tab off into /World/Latch.
#
# SOCKET = the carved jack (build_jack_carved.py): the real 1422N17 is a feed-through
# coupler whose internal contact block leaves only a ~7x5mm channel, so a uniform-scaled
# rigid plug would have to shrink to ~60% to pass it (see ASSETS.md "Key facts"). Carving
# the block out restores a full-size open bore — the full-size real plug (11.7x8mm) seats,
# matching the original example's clean-bore + protruding-latch layout. The real internal
# catch-click is a further-fidelity option: rebuild jack_carved with the catch preserved
# (build_jack_carved.py has the `in_catch` mask) and tune the latch spring in rerun.
FORCE_FIT_SCALE = 0.97  # full-size: carved bore admits the full plug; latch rides on top
STANDIN_LATCH = False   # real latch split into /World/Latch (revolute), like the example


def fit_plug_to_cavity(plug, ch, clearance, seat_depth=0.012, nbins=10):
    if FORCE_FIT_SCALE is not None:
        v = plug.vertices; ins = v[(v[:, 1] > -seat_depth) & (v[:, 1] < 5e-4)]
        pcx = 0.5 * (ins[:, 0].min() + ins[:, 0].max()); pcz = 0.5 * (ins[:, 2].min() + ins[:, 2].max())
        print(f"  fit FORCED scale={FORCE_FIT_SCALE} (full-size test)")
        return dict(s=FORCE_FIT_SCALE, cx=ch["cx"], cz=ch["cz"], pcx=pcx, pcz=pcz)
    """Depth-matched, shape-aware lateral fit. When the plug seats (origin at
    +seat_depth) its material at local-y `ly` sits at world depth (seat_depth+ly), so
    each plug slice only has to clear the cavity cross-section AT THE DEPTH IT REACHES:
    the trailing flare maps to the wider mouth, the tip to the tight floor. Clearance
    is purely LATERAL (the tip is *supposed* to touch the floor). Largest uniform
    scale s (about the plug centroid, recentred on the cavity) that keeps every plug
    vertex `clearance` inside the cavity walls at its own depth."""
    from scipy.ndimage import distance_transform_edt
    g, floor = cavity_depth_field(ch["jack"])
    dx = g[1] - g[0]
    floor0 = np.nan_to_num(floor, nan=-1.0)
    # per-depth lateral clearance field: edt_bins[b] = distance to wall in the region
    # that is open at least to depth d_b. Deeper bins -> tighter region.
    dbins = np.linspace(0.001, seat_depth, nbins)
    edt_bins = np.stack([distance_transform_edt(floor0 >= d) * dx for d in dbins])
    cx, cz = ch["cx"], ch["cz"]

    v = plug.vertices
    inside = v[(v[:, 1] > -seat_depth) & (v[:, 1] < 5e-4)]   # the part that enters the cavity
    fx, fz, fy = inside[:, 0], inside[:, 2], inside[:, 1]
    pcx, pcz = 0.5 * (fx.min() + fx.max()), 0.5 * (fz.min() + fz.max())
    reach = seat_depth + fy                                  # world depth each vertex reaches
    bidx = np.clip(((reach - dbins[0]) / (dbins[-1] - dbins[0]) * (nbins - 1)).astype(int), 0, nbins - 1)

    def gi(a):
        return np.clip(np.round((a - g[0]) / (g[-1] - g[0]) * (len(g) - 1)).astype(int), 0, len(g) - 1)

    def fits(s):
        ix, iz = gi(cx + (fx - pcx) * s), gi(cz + (fz - pcz) * s)
        return bool((edt_bins[bidx, iz, ix] >= clearance).all())

    s = 0.2
    for cand in np.linspace(1.0, 0.2, 120):
        if fits(cand):
            s = cand
            break
    print(f"  fit (depth-matched lateral): scale={s:.3f}  body now "
          f"x={(fx.max()-fx.min())*s*1000:.1f} z={(fz.max()-fz.min())*s*1000:.1f}mm "
          f"(real 11.7x8 ; clearance {clearance*1000:.2f}/side)")
    return dict(s=s, cx=cx, cz=cz, pcx=pcx, pcz=pcz)


def apply_fit(mesh, fit):
    """Apply a fit transform (lateral recentre + uniform x,z scale) to a mesh, in place.
    Used on BOTH the plug body and the latch tab so they stay attached."""
    v = mesh.vertices.copy()
    v[:, 0] = fit["cx"] + (v[:, 0] - fit["pcx"]) * fit["s"]
    v[:, 2] = fit["cz"] + (v[:, 2] - fit["pcz"]) * fit["s"]
    mesh.vertices = v
    return mesh


def split_latch(plug, z0=0.0050, z1=0.0148, ylo=0.0022, yhi=0.0075, xhalf=0.0033):
    """Split the plug (USER frame: insertion +z, latch on +y) into connector BODY +
    latch cantilever by FACE SELECTION (no boolean) -- faces whose centroid is in the
    latch box go to the latch, the rest to the body. This works on non-watertight RAW
    tessellations (boolean needs watertight); the open cuts are fine because Newton's
    build_sdf handles non-watertight meshes (consistent winding).

    Box tuned to the real 9953K216 latch (mapped from plug_raw.obj): the spring tab
    rides above the ~1.94mm body roofline, its catching WINGS span z[6,9]mm at x±3.05,
    the press tip rises to ~6.5mm at z~14, and the wide material at z>15 is the cable
    BOOT (kept on the body, excluded by z1). Capturing the wings is what lets the
    de-latched body clear the jack's catch shoulders so the depth-matched fit keeps it
    near full size (instead of shrinking the whole body to dodge the catch)."""
    c = plug.triangles_center
    inbox = ((np.abs(c[:, 0]) <= xhalf) & (c[:, 1] >= ylo) & (c[:, 1] <= yhi)
             & (c[:, 2] >= z0) & (c[:, 2] <= z1))
    latch = plug.submesh([np.where(inbox)[0]], append=True)
    body = plug.submesh([np.where(~inbox)[0]], append=True)
    return body, latch


def report(name, m):
    bb = (m.bounds[1] - m.bounds[0]) * 1000.0
    print(f"  {name}: verts={len(m.vertices)} faces={len(m.faces)} watertight={m.is_watertight} "
          f"bbox_mm=[{bb[0]:.1f},{bb[1]:.1f},{bb[2]:.1f}] "
          f"y(mm)=[{m.bounds[0,1]*1000:.1f},{m.bounds[1,1]*1000:.1f}]")


def _pick(*names):
    for n in names:
        p = os.path.join(SRC, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(names)


def main():
    print("loading source OBJs ...")
    # SOCKET = the carved jack (block carved out -> full-size open bore so the rigid
    # full plug seats; build_jack_carved.py). The real jack (jack_raw.obj) keeps the
    # coupler's internal contact block and only admits a ~7mm plug, so it's the fallback.
    # PLUG = the raw STEP tessellation (real clearance, no voxel); raw_export.py
    # regenerates *_raw.obj from the source STEP. The latch (incl. wings) is split off
    # into a separate articulated /World/Latch (see split_latch + STANDIN_LATCH above).
    jack_src = _pick("jack_carved_meters.obj", "jack_raw.obj", "jack_step.obj", "jack_meters.obj")
    plug_src = _pick("plug_raw.obj", "plug_step.obj", "plug_meters.obj")
    print(f"  jack: {os.path.basename(jack_src)}   plug: {os.path.basename(plug_src)}")
    jack = load_obj(jack_src)
    plug = load_obj(plug_src)
    print(f"  raw jack faces={len(jack.faces)}  plug faces={len(plug.faces)}")

    jack = decimate(jack, TARGET_FACES)

    if STANDIN_LATCH:
        # boolean split needs watertight; raw tessellations aren't. Use the whole plug as
        # the body + a tiny stand-in latch (no boolean) to test the raw mesh in Newton.
        body = plug
        latch = latch_box(center=(0.0, -0.016, 0.0), half=0.001)
        print(f"STANDIN latch (no boolean); body = whole plug ({len(body.faces)} tris)")
    else:
        # split the plug into connector BODY + real latch TAB (USER frame, before rotation)
        body, latch = split_latch(plug)
        body = decimate(body, TARGET_FACES)
        print(f"split plug -> body ({len(body.faces)} tris) + latch tab ({len(latch.faces)} tris)")

    # rotate user -> Newton frame (jack, body, latch together)
    for m in (jack, body, latch):
        m.apply_transform(R_USER_TO_NEWTON)
    # jack: put the MOUTH plane (was user z=MOUTH_Z -> Newton y=-MOUTH_Z) at y=0,
    # so the cavity floor lands at y=+INSERT_DEPTH (deeper = +Y).
    jack.apply_translation([0.0, MOUTH_Z, 0.0])
    # body/latch leading face is already at y=0 (was user z=0) -> the rig's dy=0 mouth ref.

    # ORIENTATION: roll the WHOLE assembly (jack + body + latch together) 180 about the
    # insertion axis (Y) so the latch/contacts sit on the side that matches the real
    # 1422N17 jack's keying as mounted (the plug read "upside down" otherwise). Flipping
    # all three together preserves the plug<->jack mate; rotating about Y keeps the mouth at
    # y=0 and only flips x,z signs. Set FLIP_180Y=False for the un-rolled (latch on +Z) build.
    FLIP_180Y = True
    if FLIP_180Y:
        R180Y = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
        for m in (jack, body, latch):
            m.apply_transform(R180Y)

    print("after axis fix + jack mouth->y=0 :")
    report("SOCKET(jack)", jack)
    report("BODY (raw)", body)

    # restore a workable clearance: depth-matched fit of the BODY to the cavity channel,
    # then apply the SAME transform to the latch tab so it stays attached.
    ch = cavity_channel(jack, depth_min=0.003)
    print(f"  cavity channel: centre x={ch['cx']*1000:.2f} z={ch['cz']*1000:.2f}  "
          f"floor y={ch['floor']*1000:.1f}mm")
    fit = fit_plug_to_cavity(body, ch, PLUG_CLEARANCE_M)
    apply_fit(body, fit)
    apply_fit(latch, fit)
    report("BODY (fitted)", body)
    report("LATCH tab (fitted)", latch)

    # plug body lateral half-extents in the cross-section just inside the mouth
    pv = body.vertices
    near = pv[(pv[:, 1] > 0.0) & (pv[:, 1] < INSERT_DEPTH)]
    half_x = float(np.abs(near[:, 0]).max()) if len(near) else 0.0
    half_z = float(np.abs(near[:, 2]).max()) if len(near) else 0.0

    # latch hinge = the tab's root: front (toward leading face = +y max) and the edge
    # that joins the body (the z nearest the plug-body centroid). The tab flexes about
    # x (free wing swings in z). Latch is on the jack's catch side (bottom, -z).
    lv = latch.vertices
    body_cz = float(body.vertices[:, 2].mean())
    hinge_y = float(lv[:, 1].max())
    hinge_z = float(lv[:, 2].max()) if body_cz > lv[:, 2].mean() else float(lv[:, 2].min())
    hinge = (0.0, hinge_y, hinge_z)
    print(f"  latch tab newton: y=[{lv[:,1].min()*1000:.1f},{lv[:,1].max()*1000:.1f}] "
          f"z=[{lv[:,2].min()*1000:.1f},{lv[:,2].max()*1000:.1f}]  hinge=({hinge[1]*1000:.1f},{hinge[2]*1000:.1f})mm axis=+x")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)
    stage = Usd.Stage.CreateNew(OUT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    add_mesh_prim(stage, "/World/Socket", jack)
    add_mesh_prim(stage, "/World/Plug", body)
    # latch prim translate = hinge -> rig pivots at the tab root (proper flex), no gap.
    add_mesh_prim(stage, "/World/Latch", latch, translate=hinge)
    stage.GetRootLayer().Save()
    print(f"\nwrote {OUT}")
    print(f"# LATCH hinge for cad_rj45_connector LatchSpec: hinge_axis=(1,0,0), "
          f"hinge_offset_meters=({hinge[0]:.5f}, {hinge[1]:.5f}, {hinge[2]:.5f})")

    print("\n# ===== cad_rj45 RL geometry profile (paste into connector_env ASSET_PROFILES) =====")
    print(f"#   plug body half-extent: x={half_x*1000:.2f}mm  z={half_z*1000:.2f}mm")
    print(f"#   cavity floor (seat) at y=+{INSERT_DEPTH*1000:.0f}mm past the mouth")
    print("'cad_rj45': dict(")
    print("    spec='cad_rj45',")
    print("    plug_y=(0.0, 0.0, 0.0),          # plug leading face is already at the mouth")
    print(f"    seat_aim_dy={INSERT_DEPTH:.4f},          # +{INSERT_DEPTH*1000:.0f}mm to the cavity floor")
    print("    box=0.05,")
    print("    re_lat=0.004, re_approach_min=0.010, re_approach_max=0.030,")
    print("    seat_depth_tol=0.005, seat_offset=0.003,")
    print("),")


if __name__ == "__main__":
    main()
