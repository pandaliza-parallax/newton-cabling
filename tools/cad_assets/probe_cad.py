"""Probe the cad_rj45 merged USD: does the plug cross-section actually fit the jack
cavity, and how deep does the cavity go (in the Newton +Y frame)?"""
import os

import numpy as np
import trimesh
from pxr import Usd, UsdGeom

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
USD = os.path.join(REPO, "newton_cabling", "assets", "cad_rj45.usd")


def load(prim_path):
    s = Usd.Stage.Open(USD)
    m = UsdGeom.Mesh(s.GetPrimAtPath(prim_path))
    v = np.array(m.GetPointsAttr().Get())
    f = np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


jack = load("/World/Socket")
plug = load("/World/Plug")
print(f"jack y-span mm: [{jack.bounds[0,1]*1000:.1f}, {jack.bounds[1,1]*1000:.1f}]")
print(f"plug y-span mm: [{plug.bounds[0,1]*1000:.1f}, {plug.bounds[1,1]*1000:.1f}]")

# ── jack cavity: ray-cast +Y from below the mouth on an x,z grid; first hit y for
#    each ray tells us the cavity floor; rays that miss (no hit until deep) are the
#    open cavity footprint.
N = 121
g = np.linspace(-9e-3, 9e-3, N)
gx, gz = np.meshgrid(g, g)
org = np.c_[gx.ravel(), np.full(gx.size, -5e-3), gz.ravel()]
dirs = np.tile([0, 1.0, 0], (len(org), 1))
loc, ir, _ = jack.ray.intersects_location(org, dirs, multiple_hits=False)
firsty = np.full(gx.size, np.nan)
firsty[ir] = loc[:, 1]
# cavity footprint = rays whose first solid hit is deep (past the mouth plane y=0)
entered = firsty > 1e-3
fp = org[entered][:, [0, 2]]
if len(fp):
    print(f"\ncavity opening footprint (x,z) mm: "
          f"x=[{fp[:,0].min()*1000:.1f},{fp[:,0].max()*1000:.1f}] "
          f"z=[{fp[:,1].min()*1000:.1f},{fp[:,1].max()*1000:.1f}]")
    print(f"cavity floor (first hit y) mm: median {np.nanmedian(firsty[entered])*1000:.1f}  "
          f"max {np.nanmax(firsty[entered])*1000:.1f}")
else:
    print("\nNO cavity rays entered — plug face is hitting a closed wall at the mouth!")
    print(f"  nearest first-hit y for center ray: {firsty[gx.size//2]*1000:.2f} mm")

# ── plug cross-section just inside the leading (+Y) face
pv = plug.vertices
for ymm in (-1, -3, -6, -10):
    sl = pv[np.abs(pv[:, 1] - ymm * 1e-3) < 0.5e-3]
    if len(sl):
        print(f"plug slice y={ymm:>3}mm: x=[{sl[:,0].min()*1000:.1f},{sl[:,0].max()*1000:.1f}] "
              f"z=[{sl[:,2].min()*1000:.1f},{sl[:,2].max()*1000:.1f}]")

# ── mouth-opening map: classify each +Y ray near the mouth plane.
#    'W' = solid right at the mouth (rim/wall, blocks a plug here)
#    '.' = open into the cavity   ' ' = outside the jack
#    overlay '#' = plug tip footprint (y in [-2,0]mm) so we see if it fits the hole.
ptip = pv[pv[:, 1] > -2e-3]
px0, px1 = ptip[:, 0].min(), ptip[:, 0].max()
pz0, pz1 = ptip[:, 2].min(), ptip[:, 2].max()
print("\nmouth opening (rows=z -9..9mm, cols=x -9..9mm); '#'=plug tip, 'W'=rim, '.'=open")
M = 25
gg = np.linspace(-9e-3, 9e-3, M)
fy = firsty.reshape(N, N)  # indexed [z, x]
for zi, zc in enumerate(gg):
    row = ""
    for xc in gg:
        ix = int(round((xc + 9e-3) / 18e-3 * (N - 1)))
        iz = int(round((zc + 9e-3) / 18e-3 * (N - 1)))
        y = fy[iz, ix]
        inplug = (px0 <= xc <= px1) and (pz0 <= zc <= pz1)
        if np.isnan(y):
            cell = " "
        elif y < 1e-3:
            cell = "W"
        else:
            cell = "."
        row += "#" if inplug else cell
    print("  " + row)
