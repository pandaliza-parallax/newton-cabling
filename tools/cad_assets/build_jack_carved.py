"""Re-carve the real jack so it admits a FULL-SIZE RJ45 plug.

The 1422N17 is a feed-through coupler: its real cavity has an internal contact
block + lipped mouth, leaving only a ~7x5mm clear channel — too small for its own
11.7x8mm plug in rigid sim. This keeps the real jack HOUSING (from the STEP-derived
jack_meters.obj) but voxel-carves a clean, full-size rounded-rectangular pocket
(sized to the real plug + clearance) at the real port location, removing the
internal block. Result: faithful exterior, full-size open cavity.

    .venv/bin/python tools/cad_assets/build_jack_carved.py
    -> writes tools/cad_assets/src/jack_carved_meters.obj (used by build_cad_rj45_usd.py)
"""
import os

import numpy as np
import trimesh
from trimesh.voxel import VoxelGrid

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
JACK = os.path.join(SRC, "jack_meters.obj")
# Carve to the REAL raw plug envelope (not the voxel-inflated plug_meters): plug_meters
# is ~0.1mm fatter per side AND its cross-section bakes in the latch, so a pocket sized to
# it leaves the de-latched body loose (the body slides free while only the sprung latch
# touches the wall). plug_raw is the same mesh the USD build uses -> a snug, matched bore.
PLUG = os.path.join(SRC, "plug_raw.obj")
OUT = os.path.join(SRC, "jack_carved_meters.obj")

# user frame: insertion +Z, mouth at jack max-z, lateral plane = (x, y); catch on -Y
PITCH = 0.20e-3
CLEARANCE = 0.0002    # snug pocket: real RJ45 clearance is ~0.1mm, so the plug INTERACTS
#                       with the bore on insertion instead of sliding through a loose hole.
DEPTH = 0.012         # pocket depth from the mouth (matches the real 12mm cavity)
CORNER_R = 0.0015     # rounded corners (match the plug + keep the SDF smooth)
# PRESERVE the real latch catch: a recess at bottom-center (x in [-2.8,2.8], y down to
# -7.3) flanked by shoulder lips at y~-4.2. Do NOT carve this region, so the latch
# (which rides on the bottom) flexes under the lip and springs into the recess to lock.
CATCH_XHALF = 0.0035   # protect x in [-3.5, 3.5] (the recess + a bit of the inner lips)
CATCH_YTOP = -0.0035   # protect y below this (the catch slot + shoulders)
CATCH_DEPTH = 0.008    # the catch lives in the front ~8mm (near the mouth)


def cavity_center(jack):
    mz = jack.bounds[1, 2]
    N = 121
    g = np.linspace(-9e-3, 9e-3, N)
    gx, gy = np.meshgrid(g, g)
    org = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, mz + 5e-3)]
    loc, ir, _ = jack.ray.intersects_location(
        org, np.tile([0, 0, -1.0], (gx.size, 1)), multiple_hits=False)
    d = np.full(gx.size, np.nan)
    d[ir] = mz - loc[:, 2]
    ent = d > 2e-3
    return 0.0, float(np.median(org[ent][:, 1])), mz   # cx (symmetric), cy, mouth_z


def main():
    import sys
    sys.path.insert(0, HERE)
    from build_cad_rj45_usd import split_latch  # same split the USD build uses

    jack = trimesh.load(JACK, process=True)
    plug = trimesh.load(PLUG, process=True)
    cx, cy, mz = cavity_center(jack)

    # STEPPED cavity (so the de-latched BODY interacts AND the latch has a slot): a real
    # RJ45 bore is wide+short for the housing then a narrow top slot for the latch. A
    # single rounded-rect sized to the whole plug envelope leaves the body loose by the
    # latch's height (the body rattles low while only the sprung latch touches the wall).
    body, latch = split_latch(plug)  # user frame: lateral (x,y), latch on +y
    bi = body.vertices[(body.vertices[:, 2] > 0) & (body.vertices[:, 2] < DEPTH)]
    li = latch.vertices[(latch.vertices[:, 2] > 0) & (latch.vertices[:, 2] < DEPTH)]
    # body BULK: the wide rectangular housing (|x| up to its max) below where it necks to
    # the latch shoulder. STEP_Y (plug frame) = where the cross-section narrows to ~half.
    STEP_Y = 0.0021
    bulk = bi[bi[:, 1] <= STEP_Y]
    bhx = np.abs(bulk[:, 0]).max() + CLEARANCE          # body pocket half-width (snug)
    by_lo = bulk[:, 1].min() - CLEARANCE                # body pocket floor (plug frame)
    lhx = np.abs(li[:, 0]).max() + CLEARANCE + 3.0e-4   # latch slot half-width + flex room
    ly_hi = li[:, 1].max() + CLEARANCE                  # latch slot ceiling (plug frame)
    # centre the plug envelope on the jack cavity (the USD fit recentres the plug to this).
    shift = cy - 0.5 * (by_lo + ly_hi)
    by_lo += shift; STEP_Yc = STEP_Y + shift; ly_hi += shift
    print(f"jack mouth z={mz*1000:.1f}mm cavity y={cy*1000:.2f}mm")
    print(f"BODY pocket: x±{bhx*1000:.2f} y[{by_lo*1000:.2f},{STEP_Yc*1000:.2f}]mm | "
          f"LATCH slot: x±{lhx*1000:.2f} y[{STEP_Yc*1000:.2f},{ly_hi*1000:.2f}]mm  depth {DEPTH*1000:.0f}mm")

    vox = jack.voxelized(PITCH).fill()
    occ = vox.matrix.copy()
    tf = vox.transform
    nx, ny, nz = occ.shape
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)], 1)
    world = (tf @ idx.T).T[:, :3]
    wx, wy, wz = world[:, 0], world[:, 1], world[:, 2]

    def rrect(hx, hy_lo, hy_hi):  # rounded-x rectangle band in y
        ddx = np.maximum(np.abs(wx - cx) - (hx - CORNER_R), 0.0)
        return (np.abs(wx - cx) <= hx) & (wy >= hy_lo) & (wy <= hy_hi) & (ddx <= CORNER_R + 1e-9)

    body_pocket = rrect(bhx, by_lo, STEP_Yc)
    latch_slot = rrect(lhx, STEP_Yc, ly_hi)
    in_z = (wz <= mz + 1e-4) & (wz >= mz - DEPTH)
    carve = ((body_pocket | latch_slot) & in_z).reshape(occ.shape)  # stepped pocket
    carved = occ & ~carve
    print(f"carved {carve.sum()} voxels ({carve.sum()/occ.sum()*100:.0f}% of solid)")

    mc = VoxelGrid(carved).marching_cubes
    mc.apply_transform(tf)
    mc.merge_vertices()
    trimesh.repair.fix_winding(mc)
    trimesh.repair.fix_normals(mc)
    mc.export(OUT)
    bb = (mc.bounds[1] - mc.bounds[0]) * 1000
    print(f"wrote {OUT}  verts={len(mc.vertices)} faces={len(mc.faces)} bbox_mm={np.round(bb,1).tolist()}")


if __name__ == "__main__":
    main()
