"""Bake the 3D-print jack fixture into the Newton SOCKET frame (jack_fixture_rj45.usd).

The physical bench fixture (build_jack_fixture.py: fixtures/jack_fixture.stl, mm,
SOCKET-UP frame — jack base at z=0, mouth at z=30, bbox-centred XY) is brought into
the exact frame the sim's /World/Socket mesh lives in (metres, mouth plane y=0,
cavity deeper = +Y), so the env can add it as a second shape on the kinematic jack
body with an IDENTITY local transform and it lands flush around the socket.

Registration is derived, not eyeballed. fixtures/jack_seated.stl is the SAME 1422N17
STEP tessellation placed in the fixture frame, and the USD socket is that same part
(carved + decimated) in the socket frame; both frames share the STEP's orientation up
to the known build_cad_rj45_usd chain, so the rotation is exact:

    R_total = R180Y . R_USER_TO_NEWTON . (mm->m)   ==  (x,y,z)_fix -> (-x, -z, -y)/1000

and only the translation is solved, from the jack itself:
  * y: the fixture-frame mouth plane (jack_seated max z) -> y = 0
  * x,z: bbox centres of R_total(jack_seated) matched to the USD socket's bbox
    centres (outer shells are the same solid; the carve only opens the interior)

Residuals (per-axis extent mismatch between the registered jack and the USD socket)
are printed and must stay sub-mm; a section-overlay preview PNG is written next to
the STLs for a visual check.

Run (newton .venv):
    .venv/bin/python tools/cad_assets/build_jack_fixture_usd.py
Writes newton_cabling/assets/jack_fixture_rj45.usd + fixtures/jack_fixture_usd_preview.png
"""

import os

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, Vt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIXTURES = os.path.join(HERE, "fixtures")
SOCKET_USD = os.path.join(REPO, "newton_cabling", "assets", "cad_rj45.usd")
OUT = os.path.join(REPO, "newton_cabling", "assets", "jack_fixture_rj45.usd")
PREVIEW = os.path.join(FIXTURES, "jack_fixture_usd_preview.png")

# fixture(mm) -> Newton socket frame(m): R180Y . R_USER_TO_NEWTON (build_cad_rj45_usd)
# composed = (x, y, z) -> (-x, -z, -y), then mm -> m.
R_TOTAL = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]) / 1000.0


def load_stl(name):
    m = trimesh.load(os.path.join(FIXTURES, name), process=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


def load_usd_socket():
    stage = Usd.Stage.Open(SOCKET_USD)
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Socket"))
    v = np.array(geom.GetPointsAttr().Get(), dtype=np.float64)
    f = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def section_xy(mesh, axis, value, color, ax, ix, iy, lw=0.8):
    """Plot the section of `mesh` at plane axis=value onto ax using coords (ix, iy)."""
    normal = np.zeros(3)
    normal[axis] = 1.0
    origin = np.zeros(3)
    origin[axis] = value
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None:
        return
    for ent in sec.entities:
        pl = sec.vertices[ent.points]
        ax.plot(pl[:, ix] * 1000, pl[:, iy] * 1000, c=color, lw=lw)


def main():
    print("loading meshes ...")
    seated = load_stl("jack_seated.stl")     # the jack, fixture frame (mm)
    fixture = load_stl("jack_fixture.stl")   # the fixture body, fixture frame (mm)
    socket = load_usd_socket()               # the SAME jack, socket frame (m)

    # rotate both into socket orientation (still un-translated)
    seated_r = seated.copy()
    seated_r.vertices = seated_r.vertices @ R_TOTAL.T
    # translation: mouth plane (fixture z max -> rotated y min side) to y=0, then
    # bbox-centre match in x/z against the USD socket (same outer shell).
    t = np.zeros(3)
    t[1] = -seated_r.bounds[0, 1]                      # mouth -> y = 0
    sb, jb = socket.bounds, seated_r.bounds
    t[0] = 0.5 * (sb[0, 0] + sb[1, 0]) - 0.5 * (jb[0, 0] + jb[1, 0])
    t[2] = 0.5 * (sb[0, 2] + sb[1, 2]) - 0.5 * (jb[0, 2] + jb[1, 2])
    seated_r.vertices = seated_r.vertices + t

    # x/z: same outer shell (both bboxes are set by the flange) -> must match tightly.
    # y: the USD socket is the jack CROPPED to its front 16mm receptacle block
    # (raw_export crops at zmax-0.016), so its y-range must be a strict subset of the
    # full jack's, with both mouths on y=0.
    ext_jack = (seated_r.bounds[1] - seated_r.bounds[0]) * 1000
    ext_sock = (socket.bounds[1] - socket.bounds[0]) * 1000
    print(f"  registered jack extents  {np.round(ext_jack, 2)} mm")
    print(f"  USD socket extents       {np.round(ext_sock, 2)} mm (y-cropped block)")
    lat = np.abs(ext_jack - ext_sock)[[0, 2]]
    print(f"  lateral extent mismatch  x {lat[0]:.3f}  z {lat[1]:.3f} mm")
    print(f"  jack y-range after reg   [{seated_r.bounds[0,1]*1000:.2f}, "
          f"{seated_r.bounds[1,1]*1000:.2f}] mm (socket "
          f"[{socket.bounds[0,1]*1000:.2f}, {socket.bounds[1,1]*1000:.2f}])")
    if lat.max() > 0.5:
        raise SystemExit("registration FAILED: lateral outer shells disagree by >0.5mm")
    if abs(socket.bounds[0, 1] - seated_r.bounds[0, 1]) > 5e-4:
        raise SystemExit("registration FAILED: mouth planes disagree by >0.5mm")
    if socket.bounds[1, 1] > seated_r.bounds[1, 1] + 5e-4:
        raise SystemExit("registration FAILED: socket deeper than the full jack")

    fixture_r = fixture.copy()
    fixture_r.vertices = fixture_r.vertices @ R_TOTAL.T + t
    fb = fixture_r.bounds * 1000
    print(f"  fixture in socket frame: x[{fb[0,0]:.1f},{fb[1,0]:.1f}] "
          f"y[{fb[0,1]:.1f},{fb[1,1]:.1f}] z[{fb[0,2]:.1f},{fb[1,2]:.1f}] mm "
          f"({len(fixture_r.faces)} faces)")

    # preview: socket (green) + registered jack (blue) + fixture (grey) sections
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    panels = [
        (2, 0.0, 0, 1, "x-y section at z=0 (top-down: mouth left, +y deeper)"),
        (0, 0.0, 1, 2, "y-z section at x=0 (side view)"),
        (1, 0.004, 0, 2, "x-z section at y=+4mm (through the cavity)"),
    ]
    for ax, (axis, value, ix, iy, title) in zip(axes, panels):
        section_xy(fixture_r, axis, value, "0.4", ax, ix, iy)
        section_xy(socket, axis, value, "tab:green", ax, ix, iy, lw=1.2)
        section_xy(seated_r, axis, value, "tab:blue", ax, ix, iy, lw=0.6)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("xyz"[ix] + " mm")
        ax.set_ylabel("xyz"[iy] + " mm")
        ax.grid(alpha=0.3)
    for c, l in [("0.4", "fixture"), ("tab:green", "USD socket"), ("tab:blue", "jack (registered)")]:
        axes[0].plot([], [], c=c, label=l)
    axes[0].legend(fontsize=8)
    fig.suptitle("jack_fixture_rj45.usd registration (Newton socket frame)")
    fig.tight_layout()
    fig.savefig(PREVIEW, dpi=95)
    plt.close()

    if os.path.exists(OUT):
        os.remove(OUT)
    stage = Usd.Stage.CreateNew(OUT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    geom = UsdGeom.Mesh.Define(stage, "/World/Fixture")
    v = fixture_r.vertices.astype(np.float32)
    f = fixture_r.faces.astype(np.int32)
    geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    lo, hi = v.min(0), v.max(0)
    geom.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
    stage.GetRootLayer().Save()
    print(f"wrote {OUT}")
    print(f"wrote {PREVIEW}")


if __name__ == "__main__":
    main()
