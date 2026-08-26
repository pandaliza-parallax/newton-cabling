#!/usr/bin/env python3
"""Replay a saved Stage-A episode (ep_XXXX/*_traj.npy, seat-relative) into a .rrd.

Exact replay of what gen_trajectories saved -- no re-simulation, so no
nondeterminism. Everything is in the SEAT frame: origin = seated face pose,
+y = insertion axis (deeper), jack mouth at y = -SEAT_AIM_DY.

    .venv/bin/python tools/traj_to_rrd.py \
        --ep /home/pandaliza/parallax/data/vla_train/roll3_fixture_sample/ep_0000 \
        [--out ep_0000.rrd]

View:  uvx --from rerun-sdk rerun <out>.rrd
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import rerun as rr

PHASES = ["SETTLE", "ALIGN", "PUSH", "HOLD", "RETREAT"]
PHASE_COLOR = [(150, 150, 150), (66, 135, 245), (245, 166, 35), (80, 200, 120), (220, 70, 70)]


def _rot(q_wxyz: np.ndarray, v) -> np.ndarray:
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v, np.float64)


def axes(entity: str, pose7: np.ndarray, scale: float) -> None:
    p, (w, x, y, z) = pose7[:3], pose7[3:7]
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    rr.log(entity, rr.Arrows3D(
        origins=[p, p, p], vectors=(R * scale).T,
        colors=[(230, 80, 80), (80, 220, 80), (80, 120, 240)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ep", required=True, help="ep_XXXX dir with *_traj.npy + meta.json")
    ap.add_argument("--out", default=None, help="output .rrd (default: <ep>.rrd next to the dir)")
    ap.add_argument("--cads", action="store_true",
                    help="log the fixture + seated-jack CAD meshes (STLs, CAD mm -> seat frame) "
                         "and a CAD-proportioned plug body riding the face trajectory")
    args = ap.parse_args()

    ep = args.ep.rstrip("/")
    out = args.out or ep + ".rrd"
    ld = lambda n: np.load(os.path.join(ep, n))
    eef, face, conn = ld("eef_traj.npy"), ld("face_traj.npy"), ld("conn_traj.npy")
    rods, tips, phase = ld("rods_traj.npy"), ld("tips_traj.npy"), ld("phase.npy")
    meta = json.load(open(os.path.join(ep, "meta.json")))
    T = len(eef)

    rr.init("traj-replay", spawn=False)
    rr.save(out)

    # static scene: seat frame, jack mouth plane, full face path
    rr.log("seat", rr.Arrows3D(origins=np.zeros((3, 3)), vectors=np.eye(3) * 0.02,
                               colors=[(230, 80, 80), (80, 220, 80), (80, 120, 240)]), static=True)
    jack_p = np.asarray(meta.get("jack_pos_seatrel", [0, -0.012, 0]))
    rr.log("jack_mouth", rr.Points3D([jack_p], radii=0.004, colors=[(240, 240, 90)],
                                     labels=["jack mouth"]), static=True)
    rr.log("face_path", rr.LineStrips3D([face[:, :3]], colors=[(200, 200, 200)]), static=True)
    rr.log("meta", rr.TextDocument(json.dumps(meta, indent=1)), static=True)

    if args.cads:
        import trimesh
        # CAD jack frame (mm, mouth plane z=30, +z = mouth outward, +y = down) -> seat frame
        # (origin = seated face = 12mm PAST the mouth, +y = deeper, +z = up):
        #   z_cad -> -y_seat   (mouth normal points back toward the plug start)
        #   y_cad -> -z_seat   (CAD +y is down)   =>   x_cad -> -x_seat (right-handed)
        R_cs = np.array([[-1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, -1.0, 0.0]])
        mouth_cad = np.array([0.0, 0.0, 0.030])
        seat_off = np.array([0.0, -0.012, 0.0])   # mouth sits at y = -SEAT_AIM_DY
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name, path, color in (
                ("cad/fixture", "tools/cad_assets/fixtures/jack_fixture.stl", (110, 110, 120)),
                ("cad/jack", "tools/cad_assets/fixtures/jack_seated.stl", (40, 40, 45))):
            m = trimesh.load(os.path.join(repo, path))
            v = (np.asarray(m.vertices) * 0.001 - mouth_cad) @ R_cs.T + seat_off
            rr.log(name, rr.Mesh3D(vertex_positions=v, triangle_indices=np.asarray(m.faces),
                                   albedo_factor=color), static=True)
        # the REAL parametric plug from cad_rj45.usd (/World/Plug, authored in the FACE
        # frame): logged once, driven per frame by a Transform3D on the same entity
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(os.path.join(repo, "newton_cabling/assets/cad_rj45.usd"))
        geom = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Plug"))
        pv = np.asarray(geom.GetPointsAttr().Get(), np.float64)
        cnt = np.asarray(geom.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(geom.GetFaceVertexIndicesAttr().Get())
        tris, k = [], 0
        for c in cnt:                                     # fan-triangulate any n-gons
            for i in range(1, c - 1):
                tris.append((idx[k], idx[k + i], idx[k + i + 1]))
            k += c
        rr.log("cad/plug", rr.Mesh3D(vertex_positions=pv, triangle_indices=np.asarray(tris),
                                     albedo_factor=(215, 215, 220)), static=True)

    for t in range(T):
        rr.set_time("frame", sequence=t)
        ph = int(phase[t])
        axes("eef", eef[t], 0.015)
        axes("face", face[t], 0.008)
        rr.log("face/pt", rr.Points3D([face[t, :3]], radii=0.0025, colors=[PHASE_COLOR[ph]]))
        rr.log("conn", rr.Points3D([conn[t, :3]], radii=0.003, colors=[(230, 120, 60)]))
        rr.log("rods", rr.Points3D(rods[t], radii=0.002, colors=[(160, 160, 220)]))
        rr.log("tips", rr.Points3D(tips[t], radii=0.002, colors=[(120, 220, 220)]))
        rr.log("cable", rr.LineStrips3D(
            [np.vstack([eef[t, :3], rods[t].reshape(-1, 3), conn[t, :3]])],
            colors=[(220, 90, 90)]))
        rr.log("phase", rr.TextLog(f"{PHASES[ph]}", color=PHASE_COLOR[ph]))
        if args.cads:
            # the CAD plug entity follows the FACE pose (mesh is authored in the face frame)
            fq = face[t, 3:7]
            rr.log("cad/plug", rr.Transform3D(
                translation=face[t, :3],
                quaternion=rr.Quaternion(xyzw=[fq[1], fq[2], fq[3], fq[0]])))

    print(f"[rrd] {T} frames ({meta.get('frames')}) -> {out}")
    print(f"[rrd] view: uvx --from rerun-sdk rerun {out}")


if __name__ == "__main__":
    main()
