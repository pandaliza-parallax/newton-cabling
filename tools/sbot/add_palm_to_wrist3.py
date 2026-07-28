"""Merge a dense CAD-sampled gripper PALM (gripper_base_link) into the cropped wrist_3 splat.

The scan's palm front face is hollow (self-occluded by the fingers during capture), so the
plane-cropped wrist_3 bake renders see-through between the flange and the fingers. The palm
is RIGID to wrist_3 (gripper_base_link collapses into the wrist_3 body in Newton), so the fix
is asset-side: sample the gripper_base_link CAD surface (dense, even), colour each sample from
the nearest scan gaussian (like cad_color_transfer.py), transform into wrist_3's local frame,
and APPEND to arm_nogrip/wrist_3_link.ply.

Re-runnable: rows appended by this tool are tagged by exact opacity value and stripped first.

    python tools/sbot/add_palm_to_wrist3.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import newton
import numpy as np
import warp as wp
from pxr import Usd, UsdGeom
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from cut_splat_by_links import col, read_ply  # noqa: E402
from export_gripper_meshes import DEFAULT_USD  # noqa: E402
from synth_splats_from_meshes import link_local_mesh, synth_link  # noqa: E402

from newton_cabling.sim.sbot import add_sbot, set_gripper  # noqa: E402

PALM = "gripper_base_link"
WRIST = "wrist_3_link"
TAG_OPACITY = 3.9375  # exact float marks palm rows appended by this tool (re-run safe)


def usd_world(stage, xc, name):
    m = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(f"/sbot/{name}")), float).T
    return m[:3, :3], m[:3, 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="/home/pandaliza/parallax/sbot_assets/fingers/gripper_fingers_registered_full.ply")
    ap.add_argument("--usd", default=str(DEFAULT_USD))
    ap.add_argument("--wrist3", default="/home/pandaliza/parallax/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link.ply")
    ap.add_argument("--theta", type=float, default=-0.6, help="scan capture opening (colour lookup pose)")
    ap.add_argument("--spacing", type=float, default=0.002)
    ap.add_argument("--flatten", type=float, default=0.25)
    a = ap.parse_args()

    stage = Usd.Stage.Open(a.usd)
    xc = UsdGeom.XformCache()

    # palm CAD surface samples, in gripper_base_link's local frame
    lmesh = link_local_mesh(stage, PALM, xc)
    g = synth_link(lmesh, spacing=a.spacing, opacity=TAG_OPACITY, color=0.2, flatten=a.flatten)
    print(f"[palm] sampled {len(g['xyz'])} gaussians from {PALM} CAD @ {a.spacing * 1000:.1f}mm")

    # rigid offset wrist_3 -> palm from the USD rest pose (constant: fixed joint)
    Rw, tw = usd_world(stage, xc, WRIST)
    Rp, tp = usd_world(stage, xc, PALM)
    Rrel = Rw.T @ Rp
    trel = Rw.T @ (tp - tw)
    palm_w3 = g["xyz"] @ Rrel.T + trel                     # palm samples in wrist_3 local frame
    rot_w3 = (Rotation.from_matrix(Rrel) * Rotation.from_quat(g["rot"][:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]

    # colour from the registered scan: pose the palm in the scan's (FK) world frame at capture theta
    b = newton.ModelBuilder()
    h = add_sbot(b, wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()))
    set_gripper(b, h, a.theta)
    m = b.finalize(); s = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s)
    bq = s.body_q.numpy(); labels = list(m.body_label)
    wi = next(i for i, l in enumerate(labels) if l.endswith(WRIST))
    Rfk = Rotation.from_quat(bq[wi][3:7]).as_matrix()
    world = (palm_w3 @ Rfk.T) + bq[wi][:3]
    props, data = read_ply(pathlib.Path(a.scan))
    sxyz = np.column_stack([col(props, data, k) for k in ("x", "y", "z")])
    sfdc = np.column_stack([col(props, data, f"f_dc_{i}") for i in range(3)])
    d, idx = cKDTree(sxyz).query(world)
    fdc = sfdc[idx]

    # merge into the cropped wrist_3 ply (strip any previous palm rows first)
    wprops, wdata = read_ply(pathlib.Path(a.wrist3))
    assert len(wprops) == 14, f"expected flat 14-prop ply, got {len(wprops)}"
    old = wdata[np.abs(wdata[:, 13] - TAG_OPACITY) > 1e-6]

    # far colour sources (scan never saw the palm face) pull pale finger-pad colours -> cap at
    # 10mm and fall back to the BAKED palm's own median colour (real captured palm = black).
    near = d <= 0.010
    palm_bake = old[(old[:, 0] >= -0.12) & (old[:, 0] <= -0.05)]
    fallback = np.median(palm_bake[:, 10:13], axis=0)
    fdc[~near] = fallback
    print(f"[palm] colour: {near.sum()} from scan (<=10mm), {(~near).sum()} baked-palm median "
          f"{np.round(fallback, 2)} (src dist median {np.median(d) * 1000:.1f}mm)")
    rows = np.zeros((len(palm_w3), 14))
    rows[:, 0:3] = palm_w3
    rows[:, 3:6] = g["scale"]
    rows[:, 6:10] = rot_w3
    rows[:, 10:13] = fdc
    rows[:, 13] = TAG_OPACITY
    out = np.vstack([old, rows]).astype("<f4")
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n%send_header\n"
           % (len(out), "".join(f"property float {p}\n" for p in wprops)))
    pathlib.Path(a.wrist3).write_bytes(hdr.encode("ascii") + out.tobytes())
    print(f"[palm] {len(old)} wrist pts + {len(rows)} palm pts -> {a.wrist3}")


if __name__ == "__main__":
    main()
