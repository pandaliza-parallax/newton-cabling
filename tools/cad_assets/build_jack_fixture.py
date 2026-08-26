"""Build a physical 3D-print FULL-CAPTURE FIXTURE for the 1422N17 panel-mount RJ45
jack (McMaster "Panel-Mount Data Adapter") as a SINGLE connected body.

Convention: SOCKET UP. The robot inserts the plug straight down (+Z) into the jack.

The jack drops in from the top and is located to a rigid, repeatable pose by:
  * a tapered BODY POCKET carved to the jack's real outer envelope (grips X/Y/yaw
    and tilt along the full body length), and
  * a FLANGE RECESS above it that seats the flange (the +Z-down insertion stop) and
    laterally captures it; the socket collar is left proud so the plug is accessible.
A rear feed-through hole keeps the coupler's back open; 4 counterbored holes bolt the
fixture to a bench.  The jack itself bolts down through its flange's own two screw
holes into Ø2.5 pilot holes drilled in the ledge (self-tap M3) -- positive +Z
retention.  (Without the screws the fit is still a gravity drop-in: a single rigid
print cannot trap the flange from above without a snap feature.)

Why the pocket is a voxel outer-envelope: the jack has an axial feed-through tunnel,
so subtracting the raw shell would leave a post inside the rear socket.  We voxelize,
fill each z-slice's holes (closes the tunnel + cavity), dilate for clearance and
marching-cubes it -> one solid, a guaranteed superset of the real part.

Reads the STEP directly (cascadio -> trimesh); booleans via manifold3d.

Run (newton .venv has cascadio + trimesh + manifold3d + scipy):
    .venv/bin/python tools/cad_assets/build_jack_fixture.py
"""
import os
import tempfile

import cascadio
import numpy as np
import trimesh
from scipy.ndimage import binary_dilation, binary_fill_holes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JACK = ("/home/pandaliza/parallax/gs-sim-vla/scene/assets/objects/"
        "ethernet/1422N17_Panel-Mount Data Adapter.STEP")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.makedirs(OUT, exist_ok=True)
ENGINE = "manifold"

# ---- tunables (mm) --------------------------------------------------------
CLEAR      = 0.40   # body-pocket clearance (FDM slip fit); = PITCH*DIL_ITERS
FLANGE_CL  = 0.40   # flange-recess clearance per side
WALL       = 6.0    # sleeve wall thickness around the flange
JACK_MULT  = 4.0    # fixture outer envelope = JACK_MULT * jack bbox, per axis
                    # (sets base footprint + base thickness; pocket/fit unscaled)
MOUNT_D    = 5.5    # bolt-down clearance holes (M5)
MOUNT_CB   = 10.0   # counterbore dia for an M5 socket head
MOUNT_GRIP = 8.0    # material left under the screw head (counterbores go the rest
                    # of the way down, so short standard M5 screws still reach)
REAR_X     = 16.0   # rear feed-through hole (fits a mating plug + cable)
REAR_Y     = 14.0
JACK_SCREWS = True  # bolt the jack down: pilot holes in the ledge under the two
PILOT_D    = 2.5    #   flange screw holes (auto-located from the STEP); Ø2.5
PILOT_DEPTH = 12.0  #   self-tap pilot for M3, this deep below the ledge
PITCH      = 0.20   # voxel pitch for the outer-envelope pocket (mm)
DIL_ITERS  = 2      # dilation steps -> pocket clearance = PITCH*DIL_ITERS


def box(sx, sy, sz, cz):
    b = trimesh.creation.box((sx, sy, sz))
    b.apply_translation([0, 0, cz])
    return b


def cyl(d, h, cx, cy, cz, n=48):
    c = trimesh.creation.cylinder(radius=d / 2.0, height=h, sections=n)
    c.apply_translation([cx, cy, cz])
    return c


def load_jack_mm():
    glb = tempfile.mktemp(suffix=".glb")
    cascadio.step_to_glb(JACK, glb, 0.00025, 0.15)
    s = trimesh.load(glb)
    m = (trimesh.util.concatenate(list(s.geometry.values()))
         if isinstance(s, trimesh.Scene) else s)
    m.apply_scale(1000.0)                       # metres -> mm
    V = m.vertices
    V[:, 0] -= 0.5 * (V[:, 0].min() + V[:, 0].max())
    V[:, 1] -= 0.5 * (V[:, 1].min() + V[:, 1].max())
    V[:, 2] -= V[:, 2].min()
    m.vertices = V
    return m


def measure(m):
    """Locate the flange slab (max footprint) + collar + body extents."""
    V = m.vertices
    zc = np.linspace(0, V[:, 2].max(), 500)
    hx = np.zeros_like(zc); hy = np.zeros_like(zc)
    for i, z in enumerate(zc):
        sl = V[np.abs(V[:, 2] - z) < 0.15]
        if len(sl) >= 3:
            hx[i] = np.ptp(sl[:, 0]); hy[i] = np.ptp(sl[:, 1])
    fm = (hx > 0.9 * hx.max()) & (hy > 0.9 * hy.max())
    fz = zc[fm]
    g = dict(flange_bot=float(fz.min()), flange_top=float(fz.max()),
             flange_x=float(hx.max()), flange_y=float(hy.max()),
             top=float(V[:, 2].max()))
    col = V[V[:, 2] > g["flange_top"] + 0.3]
    g["collar_x"] = float(np.ptp(col[:, 0])); g["collar_y"] = float(np.ptp(col[:, 1]))
    bod = V[V[:, 2] < g["flange_bot"] - 0.3]
    g["body_x"] = float(np.ptp(bod[:, 0])); g["body_y"] = float(np.ptp(bod[:, 1]))
    return g


def flange_screw_holes(m, g):
    """Locate the jack flange's screw holes: small (~Ø3.5) closed loops in a
    cross-section through the flange slab, in world XY."""
    zmid = 0.5 * (g["flange_bot"] + g["flange_top"])
    sec = m.section(plane_origin=[0, 0, zmid], plane_normal=[0, 0, 1])
    holes = []
    for loop in sec.discrete:
        span = loop.max(axis=0) - loop.min(axis=0)
        if 2.0 < span[0] < 5.0 and 2.0 < span[1] < 5.0:
            c = loop.mean(axis=0)
            holes.append((float(c[0]), float(c[1])))
    return holes


def pocket_solid(m, z1, pitch=PITCH, iters=DIL_ITERS):
    """The jack's OUTER envelope grown by ~pitch*iters, as ONE solid, cropped to
    z<z1.  Voxelize -> fill each z-slice's holes (closes the feed-through tunnel and
    the socket cavity so no post is left inside them) -> dilate for clearance ->
    marching cubes.  Guaranteed connected + a superset of the real part.

    The jack is cropped to the BODY (z<z1) *before* the envelope: otherwise the
    flange gets dilated too and its growth spills below z1, eating the seat ledge."""
    m = m.slice_plane([0, 0, z1], [0, 0, -1], cap=True)
    vg = m.voxelized(pitch).fill()
    mat = np.asarray(vg.matrix, dtype=bool)
    for k in range(mat.shape[2]):                      # k is the world-z axis
        mat[:, :, k] = binary_fill_holes(mat[:, :, k])
    mat = binary_dilation(mat, iterations=iters)
    pk = trimesh.voxel.VoxelGrid(mat, transform=vg.transform).marching_cubes
    pk.apply_transform(vg.transform)
    trimesh.repair.fix_normals(pk); trimesh.repair.fix_winding(pk)
    return pk.slice_plane([0, 0, z1], [0, 0, -1], cap=True)   # keep body region only


def build_fixture(m, g):
    fb, ft = g["flange_bot"], g["flange_top"]
    OUT_X = g["flange_x"] + 2 * (FLANGE_CL + WALL)
    OUT_Y = g["flange_y"] + 2 * (FLANGE_CL + WALL)

    # outer envelope = JACK_MULT * the jack's bbox, per axis.  X/Y grow the base
    # plate (ears); Z grows the base plate DOWN into a pedestal (the jack stays at
    # z=0-up so the pocket carve is untouched).
    jx, jy, jz = m.bounds[1] - m.bounds[0]
    BX, BY = JACK_MULT * jx, JACK_MULT * jy
    base_t = JACK_MULT * jz - ft
    assert BX > OUT_X + 2 * MOUNT_CB and BY > OUT_Y + 2 * MOUNT_CB, \
        "JACK_MULT too small: no room for the bolt ears"
    assert base_t >= MOUNT_GRIP + 2, "JACK_MULT too small: base thinner than the grip"
    mx = OUT_X / 2 + (BX - OUT_X) / 4                   # bolt centres, mid-ear
    my = OUT_Y / 2 + (BY - OUT_Y) / 4

    # sleeve up to the flange top + base pedestal below z=0  -> union = one blank
    solid = trimesh.boolean.union(
        [box(OUT_X, OUT_Y, ft, ft / 2), box(BX, BY, base_t, -base_t / 2)],
        engine=ENGINE)

    # carve: tapered body pocket (0..fb) then the flange recess.  The recess floor
    # lands EXACTLY at fb -- that annulus (pocket mouth -> recess wall) is the ledge
    # the flange seats on, so it must not be cut any lower.
    solid = solid.difference(pocket_solid(m, fb), engine=ENGINE)
    rh = (ft - fb) + 1                                  # poke 1mm above the top face
    solid = solid.difference(
        box(g["flange_x"] + 2 * FLANGE_CL, g["flange_y"] + 2 * FLANGE_CL,
            rh, fb + rh / 2), engine=ENGINE)
    # rear feed-through slot through the base pedestal
    solid = solid.difference(box(REAR_X, REAR_Y, base_t + 2, -base_t / 2 + 0.5),
                             engine=ENGINE)
    # bolt-down holes (through) + TOP counterbores (print support-free base-down).
    # The counterbore reaches down to MOUNT_GRIP above the bottom, so the screw
    # only threads through MOUNT_GRIP mm of plastic regardless of base thickness.
    cbh = base_t - MOUNT_GRIP
    for sx in (-1, 1):
        for sy in (-1, 1):
            solid = solid.difference(cyl(MOUNT_D, base_t + 2, sx * mx, sy * my,
                                         -base_t / 2), engine=ENGINE)
            solid = solid.difference(cyl(MOUNT_CB, cbh + 0.5, sx * mx, sy * my,
                                         -cbh / 2 + 0.25), engine=ENGINE)
    # jack bolt-down: pilot holes in the ledge under the flange's screw holes
    # (the jack body steps inward below the flange, so the material there is solid)
    screw_xy = flange_screw_holes(m, g) if JACK_SCREWS else []
    for hx, hy in screw_xy:
        solid = solid.difference(
            cyl(PILOT_D, PILOT_DEPTH + 1, hx, hy, fb - PILOT_DEPTH / 2 + 0.5),
            engine=ENGINE)
    return solid, dict(OUT_X=OUT_X, OUT_Y=OUT_Y, BX=BX, BY=BY, mx=mx, my=my,
                       base_t=base_t, cbh=cbh, screw_xy=screw_xy)


def render(fix, jack, fn):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for a, (i, j, nm) in zip(axes, [(0, 1, "XY (top-down)"),
                                    (0, 2, "XZ (section)"),
                                    (1, 2, "YZ (section)")]):
        if nm.startswith("XY"):
            a.scatter(fix.vertices[:, i], fix.vertices[:, j], s=0.3, alpha=0.12, c="0.5")
            a.scatter(jack.vertices[:, i], jack.vertices[:, j], s=0.3, alpha=0.12,
                      c="tab:green")
        else:
            nrm = [0, 1, 0] if nm.startswith("XZ") else [1, 0, 0]
            for mesh, c in [(fix, "0.3"), (jack, "tab:green")]:
                sec = mesh.section(plane_origin=[0, 0, 0], plane_normal=nrm)
                if sec is None:
                    continue
                for ent in sec.entities:
                    pl = sec.vertices[ent.points]
                    a.plot(pl[:, i], pl[:, 2], c=c, lw=0.8)
        a.set_aspect("equal"); a.set_title(nm)
        a.set_xlabel("XYZ"[i] + " mm"); a.set_ylabel("XYZ"[j] + " mm"); a.grid(alpha=0.3)
    for c, l in [("0.3", "fixture"), ("tab:green", "jack")]:
        axes[1].plot([], [], c=c, label=l)
    axes[1].legend(fontsize=8)
    fig.suptitle("1422N17 jack single-body full-capture fixture (grey) + jack (green)")
    fig.tight_layout(); fig.savefig(fn, dpi=95); plt.close()


def stats(name, m):
    b = m.bounds
    print(f"  {name}: {np.round(b[1]-b[0],2)} mm  bodies={m.body_count} "
          f"watertight={m.is_watertight} vol={m.volume/1000:.1f} cm^3 faces={len(m.faces)}")


def main():
    print("loading + measuring jack ...")
    jack = load_jack_mm()
    g = measure(jack)
    print("  flange z {flange_bot:.2f}..{flange_top:.2f}  outer {flange_x:.1f}x"
          "{flange_y:.1f}".format(**g))
    print("  collar {collar_x:.1f}x{collar_y:.1f}  body {body_x:.1f}x{body_y:.1f}"
          "  top z={top:.1f}".format(**g))
    print("building fixture ...")
    fix, bi = build_fixture(jack, g)
    fix = trimesh.boolean.union([fix], engine=ENGINE)     # clean manifold export
    fix.merge_vertices(); trimesh.repair.fix_normals(fix)
    print("output:")
    print("  base {BX:.1f}x{BY:.1f}x{base_t:.1f}  bolt centres +/-{mx:.1f}/+/-{my:.1f}"
          "  counterbore depth {cbh:.1f}".format(**bi))
    print("  jack screw pilots at", [(round(x, 1), round(y, 1))
                                     for x, y in bi["screw_xy"]])
    stats("fixture", fix)
    fix.export(os.path.join(OUT, "jack_fixture.stl"))
    fix.export(os.path.join(OUT, "jack_fixture.obj"))
    render(fix, jack, os.path.join(OUT, "jack_fixture_preview.png"))
    print(f"wrote jack_fixture STL+OBJ + preview to {OUT}")


if __name__ == "__main__":
    main()
