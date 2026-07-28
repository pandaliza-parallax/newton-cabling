"""StandardBots RO1 arm + the ethernet/table scene, rendered through DalusPySim's
Gaussian-Splat renderer (the "combined scene").

Extends record_sbot_gs.py: the seven arm-link splats are driven by the Newton sbot sim
(as before), and three scene splats are added --

  * table  -- static, placed *absolutely* (render_config.yml convention, top at z~0.886)
  * jack   -- static ethernet port, placed on the table top (origin-centred CAD splat)
  * connector wire -- rides the gripper (wrist_3_link) so the arm visibly carries it

This is a render-only composition: the jack/connector are NOT in the Newton physics (no
contact / grasp). The connector follows wrist_3's pose rigidly via gs_bridge.place_on_body
(recentred by its native centroid, plus a tunable grasp offset/rotation so it seats in the
jaws). The AG-145 gripper now renders too -- its 8 finger links are dynamic bodies with
synthesized CAD splats, so the jaws open/close with the arm.

Everything is in metres, Z-up -- arm links ~0.1-0.7 m, jack ~2 cm, table ~1.2 m -- so no
scaling. Placement (table_pos / jack_pos / grasp_offset / camera) is approximate by design:
validate the wiring with --dry-run, then tune the offsets by eyeballing rendered frames,
exactly like tools/gs_probe.py.

Run (renderer container up, ipc: host; SHM root-owned -> sudo):
    cd newton-cabling
    sudo PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim \
        .venv/bin/python record_sbot_scene_gs.py --out out_sbot_scene
    --smoke    one static home-pose frame (tune placement fast)
    --dry-run  skip renderer/SHM (still runs Newton, black frames)
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

import newton
import numpy as np
import warp as wp

newton.use_coord_layout_targets = True

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from newton.solvers import SolverVBD  # noqa: E402

from newton_cabling.render.gs_bridge import (  # noqa: E402
    NewtonGSClient,
    euler_deg_to_quat_wxyz,
    look_at_quat,
    make_intrinsics,
    newton_pose,
    place_on_body,
    ply_centroid,
    quat_mul_wxyz,
    quat_rotate_wxyz,
)
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: E402
from newton_cabling.sim.safe_vbd import finalize_for_vbd, new_vbd_builder  # noqa: E402
from newton_cabling.sim.sbot import (  # noqa: E402
    GRIPPER_THETA_CLOSED,
    GRIPPER_THETA_OPEN,
    SBOT_HOME,
    add_sbot,
    set_arm_home,
    set_gripper,
    set_pd_gains,
)

FPS = 30
DURATION_SECONDS = 5.0
SIM_SUBSTEPS = 8
TABLE_TOP_Z = 0.886  # render_config.yml: table top in the Office_1/table frame

# Arm links in Isaac's registration order; base_link is the fixed root (no body_q row).
# The 8 AG-145 finger links follow -- dynamic bodies (synthesized splats from
# tools/sbot/synth_splats_from_meshes.py) that articulate with the gripper.
LINK_SPLATS: tuple[str, ...] = (
    "base_link", "shoulder_link", "upper_arm_link", "forearm_link",
    "wrist_1_link", "wrist_2_link", "wrist_3_link",
    "finger1_knuckle_link", "finger1_inner_knuckle_link",
    "finger1_finger_link", "finger1_finger_tip_link",
    "finger2_knuckle_link", "finger2_inner_knuckle_link",
    "finger2_finger_link", "finger2_finger_tip_link",
)
GRIPPER_LINK = "wrist_3_link"  # the connector rides this body

HOST_PARALLAX = "/home/pandaliza/parallax"
CONTAINER_PARALLAX = "/root/parallax"
# flat/ = SH stripped to deg-0 (tools/strip_arm_sh.py): the renderer mis-reads the arm
# splats' f_rest layout -> rainbow; the flat path (0.5+SH_C0*f_dc) gives correct matte colour.
HOST_SBOT_GS = f"{HOST_PARALLAX}/parallax-demo-isaac-lab/assets/sbot_gs/flat"
# Integrated gripper (default): the SH-cut per-finger splats (tools/sbot/cut_splat_by_links.py,
# real captured colour, per-finger theta) + the gripper-cropped wrist_3 (tools/crop_wrist3.py)
# so the baked-in gripper in the wrist_3 capture doesn't double the cut fingers.
HOST_GRIPPER_CUT = f"{HOST_PARALLAX}/parallax-demo-isaac-lab/assets/sbot_gs/gripper_cut"
HOST_WRIST3_CROP = f"{HOST_PARALLAX}/parallax-demo-isaac-lab/assets/sbot_gs/arm_nogrip/wrist_3_link.ply"
HOST_GSVLA = f"{HOST_PARALLAX}/gs-sim-vla/scene/assets"


def host_to_container(p: str) -> str:
    return p.replace(HOST_PARALLAX, CONTAINER_PARALLAX)


def xform_pose(xf: wp.transform):
    p = wp.transform_get_translation(xf)
    q = wp.transform_get_rotation(xf)  # [x, y, z, w]
    return [float(p[0]), float(p[1]), float(p[2])], [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def ply_bounds(host_path: str, plo: float = 1.0, phi: float = 99.0):
    """(min, max) xyz of a ply's gaussian means at the given percentiles."""
    raw = open(host_path, "rb").read()
    end = raw.find(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:end].decode("ascii", "replace")
    n = next(int(l.split()[-1]) for l in hdr.splitlines() if l.startswith("element vertex"))
    npp = sum(1 for l in hdr.splitlines() if l.startswith("property"))
    xyz = np.frombuffer(raw[end:end + n * npp * 4], dtype="<f4").reshape(n, npp)[:, :3].astype(np.float64)
    return np.percentile(xyz, plo, 0), np.percentile(xyz, phi, 0)


def resolve_body(model: newton.Model, name: str) -> int:
    labels = list(model.body_label)
    exact = [i for i, lbl in enumerate(labels) if lbl == name]
    if len(exact) == 1:
        return exact[0]
    suffix = [i for i, lbl in enumerate(labels) if lbl.endswith(name)]
    if len(suffix) == 1:
        return suffix[0]
    raise RuntimeError(f"could not resolve body {name!r}; labels: {labels}")


def static_pose(world_pos, quat_wxyz, centroid):
    """World pose for a static splat: seat its native centroid at ``world_pos``."""
    pos = np.asarray(world_pos, float) - quat_rotate_wxyz(quat_wxyz, centroid)
    return pos.tolist(), list(quat_wxyz)


def make_pedestal_ply(path, sx, sy, h, *, color=0.2, spacing=0.01, opacity=6.0, flatten=0.25):
    """Synthesize a solid matte box-column Gaussian splat (flat SH deg-0, 14-prop layout,
    same as the arm splats). Box centred at the origin (spans ±h/2 in z), so seating its
    centroid at [x, y, h/2] makes it stand floor(z=0)->top(z=h). Returns the gaussian count."""
    import trimesh  # noqa: PLC0415
    SH_C0 = 0.28209479177387814
    mesh = trimesh.creation.box(extents=[sx, sy, h])
    n = max(2000, round(mesh.area / (spacing * spacing)))
    pts, fidx = trimesh.sample.sample_surface(mesh, n)
    n = len(pts)
    nrm = mesh.face_normals[fidx]
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    z = np.array([0.0, 0.0, 1.0])
    dot = nrm @ z
    axis = np.cross(np.tile(z, (n, 1)), nrm)
    an = np.linalg.norm(axis, axis=1, keepdims=True)
    rot = np.zeros((n, 4)); rot[:, 0] = 1.0                  # local +z -> face normal
    ok = an[:, 0] > 1e-8
    ang = np.arccos(np.clip(dot[ok], -1.0, 1.0))
    rot[ok, 0] = np.cos(ang / 2.0)
    rot[ok, 1:] = (axis[ok] / an[ok]) * np.sin(ang / 2.0)[:, None]
    rot[dot < -0.999999] = [0.0, 1.0, 0.0, 0.0]
    log_t, log_n = np.log(spacing), np.log(spacing * flatten)
    scale = np.tile([log_t, log_t, log_n], (n, 1))
    f_dc = np.full((n, 3), (color - 0.5) / SH_C0)
    op = np.full((n, 1), opacity)
    arr = np.concatenate([pts, scale, rot, f_dc, op], axis=1).astype("<f4")   # 3+3+4+3+1 = 14
    props = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
             "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n%send_header\n"
           % (n, "".join(f"property float {p}\n" for p in props)))
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii")); f.write(arr.tobytes())
    return n


def quat_conj_wxyz(q):
    """Conjugate (= inverse for a unit quaternion) of a scalar-first quat."""
    return [q[0], -q[1], -q[2], -q[3]]


def ride_pose(body_pos, body_quat_wxyz, *, align, centroid, offset):
    """World pose for a splat riding a body: recentre by centroid, align, + grasp offset."""
    pos, q = place_on_body(body_pos, body_quat_wxyz, align_quat_wxyz=align, centroid=centroid)
    pos = (np.asarray(pos, float) + quat_rotate_wxyz(body_quat_wxyz, offset)).tolist()
    return pos, q


def eye_target_to_pitch_yaw(eye, target):
    """Newton Camera (Z-up) pitch/yaw from an eye->target direction (mirrors Camera)."""
    d = np.asarray(target, float) - np.asarray(eye, float)
    d /= np.linalg.norm(d) + 1e-9
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, d[2]))))
    yaw = math.degrees(math.atan2(d[1], d[0]))
    return pitch, yaw


def composite_lr(left_u8, right_u8, left_label="NEWTON", right_label="GAUSSIAN SPLAT"):
    """Stitch two (H,W,3) uint8 frames side by side with labels."""
    from PIL import Image, ImageDraw

    h = max(left_u8.shape[0], right_u8.shape[0])
    canvas = np.zeros((h, left_u8.shape[1] + right_u8.shape[1], 3), np.uint8)
    canvas[: left_u8.shape[0], : left_u8.shape[1]] = left_u8
    canvas[: right_u8.shape[0], left_u8.shape[1]:] = right_u8
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.text((8, 8), left_label, fill=(255, 255, 255))
    d.text((left_u8.shape[1] + 8, 8), right_label, fill=(255, 255, 255))
    return np.asarray(im)


def compose_multicam(rgb_out, labels=("FRONT", "SIDE +x", "WRIST")):
    """client.render() output -> one uint8 frame. (H,W,3) float passes through; (C,H,W,3)
    multi-cam is stitched left-to-right with per-camera labels (front | side | wrist ...)."""
    arr = np.asarray(rgb_out)
    if arr.ndim == 3:
        return (arr * 255.0).clip(0, 255).astype("uint8")
    u8 = (arr * 255.0).clip(0, 255).astype("uint8")
    if u8.shape[0] == 1:
        return u8[0]
    lbl = lambda i: labels[i] if i < len(labels) else f"cam{i}"  # noqa: E731
    frame = composite_lr(u8[0], u8[1], lbl(0), lbl(1))
    for k in range(2, u8.shape[0]):
        frame = composite_lr(frame, u8[k], "", lbl(k))
    return frame


def arm_target(t: float, duration: float, *, still: bool) -> list[float]:
    q = list(SBOT_HOME)
    if not still:
        phase = math.sin(2.0 * math.pi * t / max(duration, 1e-6))
        # q[1] += 0.22 * phase  # shoulder pitch: reach forward (+y), kept short of the table
        # q[4] += 0.3 * phase  # wrist_2 flourish
    return q


def gripper_theta(t: float, duration: float, *, still: bool) -> float:
    """Open -> closed -> open over the clip (one cosine cycle); ``still`` holds open."""
    if still:
        return GRIPPER_THETA_OPEN
    s = 0.5 - 0.5 * math.cos(2.0 * math.pi * t / max(duration, 1e-6))  # 0 -> 1 -> 0
    return GRIPPER_THETA_OPEN + (GRIPPER_THETA_CLOSED - GRIPPER_THETA_OPEN) * s


# ── EEF training-data helpers (openpi-format dump) ──────────────────────────────────
def quat_to_rot6d(q):
    """Scalar-first unit quat [w,x,y,z] -> 6D rotation rep (first two columns of R, flat)."""
    w, x, y, z = (float(v) for v in q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R[:, :2].T.reshape(-1)                      # [col0(3), col1(3)]


def quat_to_rotvec(q):
    """Scalar-first unit quat -> axis-angle 3-vector (rotvec); angle wrapped to [-pi, pi]."""
    w = float(max(-1.0, min(1.0, q[0])))
    ang = 2.0 * math.acos(w)
    if ang > math.pi:
        ang -= 2.0 * math.pi
    s = math.sqrt(max(1e-12, 1.0 - w * w))
    if s < 1e-8:
        return np.zeros(3)
    return (np.array([float(q[1]), float(q[2]), float(q[3])]) / s) * ang


def eef_world_to_base(cp, cq, base_pos, base_quat):
    """(pos, quat_wxyz) in world -> the robot base frame (inv(base) ∘ pose)."""
    binv = [base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]]
    p = quat_rotate_wxyz(binv, np.asarray(cp, float) - np.asarray(base_pos, float))
    q = quat_mul_wxyz(binv, [float(v) for v in cq])
    return np.asarray(p, float), q


def dump_cams(rgb_out, cam_labels, dump_dir, i, size=None):
    """Save the un-stitched FRONT->image, WRIST->wrist_image, MIRROR->mirror_image views for frame i (opt. resized NxN)."""
    from PIL import Image  # noqa: PLC0415
    arr = np.asarray(rgb_out)
    views = {"FRONT": arr} if arr.ndim == 3 else {lbl: arr[k] for k, lbl in enumerate(cam_labels)}
    for lbl, sub in (("FRONT", "image"), ("WRIST", "wrist_image"), ("MIRROR", "mirror_image")):
        if lbl in views:
            d = os.path.join(dump_dir, sub)
            os.makedirs(d, exist_ok=True)
            im = Image.fromarray((views[lbl] * 255.0).clip(0, 255).astype("uint8"))
            if size is not None:
                im = im.resize((size, size), Image.LANCZOS)
            im.save(os.path.join(d, f"frame_{i:04d}.png"))


def dump_episode(dump_dir, eef_world, base_pos, base_yaw_deg, gripper_vals, fps, phases=None):
    """Write state (T,10) + action (T,7) + meta from the EEF world-pose sequence.
    state  = [eef_pos(3), eef_rot6d(6), gripper(1)]   absolute, robot base frame
    action = [dpos(3), drotvec(3), gripper(1)]        base-frame delta to the next frame.
    gripper_vals: scalar or per-frame sequence (0=open .. 1=closed); the action's gripper
    channel is the NEXT frame's value (the command that produces it)."""
    import json  # noqa: PLC0415
    th = math.radians(base_yaw_deg) / 2.0
    base_quat = [math.cos(th), 0.0, 0.0, math.sin(th)]                 # base yaw about +z (wxyz)
    base = [eef_world_to_base(cp, cq, base_pos, base_quat) for (cp, cq) in eef_world]
    g = (np.full(len(base), float(gripper_vals)) if np.isscalar(gripper_vals)
         else np.asarray(gripper_vals, np.float32))
    states, actions = [], []
    for i, (p, q) in enumerate(base):
        states.append(np.concatenate([p, quat_to_rot6d(q), [g[i]]]).astype(np.float32))
        if i + 1 < len(base):
            p1, q1 = base[i + 1]
            qrel = quat_mul_wxyz([q[0], -q[1], -q[2], -q[3]], q1)      # rel rotation, frame i -> i+1
            actions.append(np.concatenate([p1 - p, quat_to_rotvec(qrel), [g[i + 1]]]).astype(np.float32))
        else:
            actions.append(np.concatenate([np.zeros(6), [g[i]]]).astype(np.float32))
    states, actions = np.stack(states), np.stack(actions)
    np.save(os.path.join(dump_dir, "state.npy"), states)
    np.save(os.path.join(dump_dir, "action.npy"), actions)
    if phases is not None:
        np.save(os.path.join(dump_dir, "phase.npy"), np.asarray(phases, np.int8))
    with open(os.path.join(dump_dir, "meta.json"), "w") as f:
        json.dump({"T": int(len(states)), "fps": int(fps),
                   "state_layout": "[eef_pos(3), eef_rot6d(6), gripper(1)] absolute, robot base frame",
                   "action_layout": "[dpos(3), drotvec_axisangle(3), gripper(1)] base-frame delta to next frame",
                   "eef": "hand grasp point (wrist_3 * grasp_offset); coincides with the plug once grasped",
                   "gripper": "0=open .. 1=closed (opens/closes during approach)",
                   "phase_layout": "0=home hold, 1=approach, 2=insertion (phase.npy)",
                   "images": {"image": "base_0_rgb (FRONT cam)", "wrist_image": "left_wrist_0_rgb (WRIST cam)"}},
                  f, indent=2)
    return states.shape, actions.shape
    return states.shape, actions.shape


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out_sbot_scene")
    ap.add_argument("--rrd", default=None, help="also record physics to this .rrd")
    # scene assets (host paths; rewritten to container paths automatically)
    ap.add_argument("--table-ply", default=f"{HOST_GSVLA}/objects/table/splat.ply")
    ap.add_argument("--jack-ply", default=f"{HOST_GSVLA}/objects/ethernet/cad_jack_registered.ply")
    ap.add_argument("--connector-ply", default=f"{HOST_GSVLA}/objects/ethernet/cad_plug_registered.ply",
                    help="connector/plug splat (default = the eval's calibrated plug, matches ep_0000)")
    ap.add_argument("--gripper-gs-dir", default=HOST_GRIPPER_CUT,
                    help="dir for the 8 gripper FINGER splats. DEFAULT is the SH-cut fingers (gripper_cut). "
                         "Pass HOST_SBOT_GS's flat dir for the old synth gripper. FILES differ from a prior "
                         "run -> restart the renderer after switching (it caches splats per SETUP).")
    ap.add_argument("--wrist3-ply", default=HOST_WRIST3_CROP,
                    help="wrist_3 splat file. DEFAULT is the gripper-CROPPED wrist_3 (arm_nogrip/), needed so "
                         "the wrist_3 capture's baked-in gripper doesn't double the cut fingers. Pass the "
                         "flat/wrist_3_link.ply to use the original. Restart the renderer after switching.")
    ap.add_argument("--gripper-only", action="store_true",
                    help="DEBUG: render ONLY the 8 gripper finger splats (no arm/table/jack/connector/bg/"
                         "pedestal), still driven by the sim. Restart the renderer (splat count changes).")
    ap.add_argument("--bg-white", action="store_true",
                    help="post-process: paint near-black (uncovered) background pixels white. Best with "
                         "--gripper-only (no bg splat). Note: keys on darkness, so faint splat edges may whiten.")
    ap.add_argument("--bg-white-thr", type=int, default=8,
                    help="--bg-white: max pixel value (0-255) treated as background")
    ap.add_argument("--compare-uncut", action="store_true",
                    help="--gripper-only: also render a SECOND, static copy of the gripper held at the "
                         "capture pose (theta=-0.6), offset to the side = the uncut splat next to the "
                         "articulating one. Splat count doubles -> restart the renderer.")
    ap.add_argument("--uncut-gap", type=float, default=0.14,
                    help="--compare-uncut: sideways gap (m) between the articulating and uncut copies")
    ap.add_argument("--conn-rpy", type=float, nargs=3, default=[-90.0, 0.0, 0.0],
                    help="connector splat-frame rotation (deg); -90 0 0 = eval's --plug-rot calibration")
    ap.add_argument("--conn-anchor", type=float, nargs=3, default=[-0.0015, -0.0015, 0.0172],
                    help="connector splat point seated at the plug pose (= eval's --plug-anchor)")
    ap.add_argument("--table-top-z", type=float, default=None,
                    help="override the table-top surface z (default: ply_bounds top ~0.785, which is "
                         "correct in the full scene with the default 3/4 camera). Jack/parts sit here.")
    ap.add_argument("--bg-ply", default=None,
                    help=f"background room splat (host path), e.g. {HOST_GSVLA}/background/splat.ply")
    ap.add_argument("--bg-pos", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="background placement (render_config uses origin)")
    # placement (tune by eyeballing frames)
    # Table moved forward (+y) so the floor-mounted arm doesn't pass through it (its
    # home pose threads through the table top otherwise); the arm works in front of /
    # over the near edge. The collision box auto-matches the table splat. Tune to taste.
    ap.add_argument("--table-pos", type=float, nargs=3, default=[0.50, -1.00, 0.091418])
    ap.add_argument("--jack-pos", type=float, nargs=3, default=None,
                    help="jack world xyz (default: centre of the table top)")
    ap.add_argument("--no-table-collision", dest="table_collision", action="store_false",
                    help="disable the static table collision box (render-only; arm clips through)")
    ap.add_argument("--table-box-thick", type=float, default=0.10,
                    help="thickness (m) of the static table-top collision slab")
    ap.add_argument("--jack-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="PHYSICAL jack body orientation (deg); also rotates the plug trajectory "
                         "(jqw). Default identity = matches the eval's socket-body frame.")
    ap.add_argument("--jack-align-rpy", type=float, nargs=3, default=[90.0, 0.0, -180.0],
                    help="jack SPLAT obj->newton calibration (deg) = eval's --mount-rot; applied to "
                         "the jack splat only, on top of --jack-rpy. Default (90,0,-180).")
    ap.add_argument("--jack-anchor", type=float, nargs=3, default=[0.0, 0.0, 0.030],
                    help="jack obj-frame point (cavity MOUTH) pinned to jack_pos = eval's "
                         "--mount-anchor; NOT the geometric centroid (which mis-seats the cavity).")
    ap.add_argument("--grasp-offset", type=float, nargs=3, default=None,
                    help="connector offset in wrist_3 frame (default: auto = midpoint of the fingertips, "
                         "so the connector sits between the jaws)")
    ap.add_argument("--grasp-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="connector orient relative to wrist_3 (deg)")
    ap.add_argument("--grasp-along-cord", type=float, default=0.0,
                    help="slide the grip point ALONG the plug/cord axis (m). Positive = the hand "
                         "shifts toward the CABLE end (jaws pinch the end of the cord, head fully "
                         "out); negative = toward the head. Direction is derived from --grasp-rpy, "
                         "so it stays on the cord for any grasp orientation (unlike --grasp-protrude, "
                         "which moves along the TOOL axis).")
    ap.add_argument("--grasp-protrude", type=float, default=0.05,
                    help="push the connector this far (m) past the fingertip-link origins along the "
                         "tool axis. Default 0.05 = jaws pinch the plug's REAR (cable-exit) end just "
                         "inside the physical tips (tips end at 0.228 along the axis, mating face at "
                         "0.192+protrude, body 22mm): head fully visible, hand holds the cable end. "
                         "0.036 = plug fully enclosed; >0.07 = plug floats clear of the jaws.")
    ap.add_argument("--grip", choices=["closed", "open", "cycle"], default="closed",
                    help="gripper jaws: held closed (default), held open, or cycled open/closed")
    # policy-rollout replay: drive the connector (and arm via IK) along a recorded plug trajectory
    ap.add_argument("--plug-traj", default=None,
                    help="(T,7) .npy of [pos3,quat4(wxyz)] plug poses in the SOCKET frame, from "
                         "rl/record_policy.py --eval-vla (ep_*/plug_traj.npy). Connector follows it; "
                         "arm tracks via IK so the gripper carries it. KINEMATIC replay (no new physics).")
    ap.add_argument("--no-arm-ik", action="store_true",
                    help="--plug-traj: move ONLY the connector along the trajectory (arm stays home); "
                         "use to validate the socket->jack frame mapping before enabling IK")
    # home -> grasp -> insertion sequence (prepended to the IK replay so the arm doesn't snap
    # straight to the grasp pose at frame 0)
    ap.add_argument("--home-hold", type=int, default=12,
                    help="frames to hold the fixed home pose (gripper open) before approaching")
    ap.add_argument("--approach-frames", type=int, default=45,
                    help="frames to move from home to the grasp pose (eased); gripper closes over the "
                         "last third. 0 or --no-approach = snap straight into the insertion (old behavior)")
    ap.add_argument("--no-approach", action="store_true",
                    help="skip the home->grasp phase; start the insertion at the grasp pose (old behavior)")
    ap.add_argument("--grasped-only", action="store_true",
                    help="datagen v2: record ONLY the already-grasped insertion. The arm is positioned "
                         "at the grasp pose with the jaws closed (approach solved but NOT dumped), then "
                         "the insertion is recorded. Episode = grasped insertion, no home-hold/approach.")
    ap.add_argument("--trim-settle-mm", type=float, default=0.0,
                    help="datagen v2: drop leading --plug-traj frames while per-step |dpos| exceeds this "
                         "(mm), removing the physics RESET TRANSIENT at the trajectory start (the source "
                         "of the grasp-handoff jerk). 3.0 recommended; 0 = off (v1 behavior).")
    ap.add_argument("--trim-settle-deg", type=float, default=3.0,
                    help="datagen v2: rotation companion to --trim-settle-mm. The plug's ORIENTATION "
                         "settles a few frames after its position, so the trim also drops leading frames "
                         "while per-step |drot| exceeds this (deg). Only active when --trim-settle-mm > 0.")
    ap.add_argument("--insert-hold-home-rot", action="store_true",
                    help="datagen v2: during the insertion the gripper HOLDS its home orientation and only "
                         "tracks the cable POSITION (the flexible cord absorbs the plug's reorientation), "
                         "so the arm stays near home the whole time. Default (off) = the gripper also rides "
                         "the cable's RELATIVE rotation, which can drift the arm far on wiggly rollouts.")
    # camera
    ap.add_argument("--width", type=int, default=852)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--fov-deg", type=float, default=45.0)
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="global brightness on the rendered RGB: >1 lifts shadows (e.g. the plug "
                         "interior); applied before --gain. 1.0 = off")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="global brightness multiplier on the rendered RGB (applied after --gamma, "
                         "clipped to 1.0). 1.0 = off")
    ap.add_argument("--eye", type=float, nargs=3, default=None)
    ap.add_argument("--eye-offset", type=float, nargs=3, default=None,
                    help="dx dy dz (m) ADDED to the front-cam eye AFTER placement (relative nudge from "
                         "wherever ELEV/AZIM/DIST_SCALE or --eye put it)")
    ap.add_argument("--target", type=float, nargs=3, default=None)
    ap.add_argument("--target-offset", type=float, nargs=3, default=None,
                    help="dx dy dz (m) ADDED to the front-cam aim point (relative nudge; e.g. lower dz "
                         "to tilt down)")
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--dist-scale", type=float, default=1.5)
    ap.add_argument("--frame", choices=["scene", "table", "jack"], default="scene",
                    help="what the camera frames: whole scene, the table top, or zoomed on the jack")
    ap.add_argument("--frame-radius", type=float, default=0.18,
                    help="half-size (m) of the box framed for --frame jack (smaller = tighter zoom)")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--duration", type=float, default=DURATION_SECONDS)
    ap.add_argument("--base-yaw-deg", type=float, default=180.0,
                    help="rotate the whole robot about Z (180 = arm faces -y, toward the camera, "
                         "so the table sits in front of it)")
    ap.add_argument("--base-pos", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="arm base world xyz (move the robot up to the table so the jack at the table "
                         "CENTRE is within reach; floor z=0)")
    ap.add_argument("--mount", action="store_true",
                    help="render a matte box-column pedestal from the floor (z=0) up to the arm base "
                         "(--base-pos z), directly under the base; use with a raised/behind --base-pos")
    ap.add_argument("--mount-size", type=float, default=0.15, help="pedestal footprint (m, square)")
    ap.add_argument("--mount-color", type=float, default=0.18, help="pedestal grey level 0..1 (dark)")
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--side-by-side", action="store_true",
                    help="render Newton's mesh view (headless GL) at the same camera and stitch it "
                         "left of the GS frame -> one composite PNG/mp4")
    ap.add_argument("--gl-fov", type=float, default=None,
                    help="FOV for the Newton GL view (default = --fov-deg; tune if framing differs)")
    ap.add_argument("--side-cam", action="store_true",
                    help="also render a 2nd camera looking along +x (perpendicular to the front view), "
                         "beside the table; frames are stitched front|side into one image. Same splat "
                         "set -> no renderer restart.")
    ap.add_argument("--mirror-cam", action="store_true",
                    help="extra camera DIAGONALLY OPPOSITE the front cam: the front eye reflected THROUGH "
                         "the aim point (gripper), looking back at it -> sees the far side. Stitched (preview).")
    ap.add_argument("--side-elev", type=float, default=None,
                    help="--side-cam elevation in deg above level (default: same as --elev)")
    ap.add_argument("--side-z", type=float, default=None,
                    help="--side-cam: override the eye's world Z (m) to raise/lower the camera "
                         "(default: height from --side-elev)")
    ap.add_argument("--side-target-z", type=float, default=None,
                    help="--side-cam: world Z (m) of the aim point; set below --side-z so the camera "
                         "tilts DOWN (default: aims at the scene centre)")
    ap.add_argument("--wrist-cam", action="store_true",
                    help="also render a DYNAMIC eye-in-hand camera on wrist_3, aimed along the tool "
                         "axis at the grasp point (plug+jack). Stitched after front/side.")
    ap.add_argument("--wrist-fov", type=float, default=75.0, help="--wrist-cam FOV deg (wide)")
    ap.add_argument("--wrist-back", type=float, default=0.10,
                    help="--wrist-cam: metres from the connector back toward the wrist along the tool axis")
    ap.add_argument("--wrist-side", type=float, default=0.12,
                    help="--wrist-cam: metres to offset the eye SIDEWAYS (perp to the tool axis) so the "
                         "black gripper doesn't fill the frame; 0 = straight down the tool axis")
    ap.add_argument("--wrist-orbit", type=float, default=0.0,
                    help="--wrist-cam: orbit the eye this many deg AROUND the tool axis on the "
                         "perpendicular circle (0 = horizontal side; 90 = a quarter-turn around, "
                         "perpendicular to the initial view). Still aims at the connector.")
    ap.add_argument("--wrist-up", type=float, default=0.05,
                    help="--wrist-cam: metres to lift the eye along world +z")
    ap.add_argument("--wrist-aim-back", type=float, default=0.0,
                    help="--wrist-cam: metres to shift the LOOK-AT point back along the tool axis toward "
                         "the wrist, tilting the view up toward the gripper fingers (0 = aim at the plug tip)")
    ap.add_argument("--wrist-rigid", action="store_true",
                    help="--wrist-cam: bolt the camera RIGIDLY to wrist_3 (first frame's pose baked "
                         "into the wrist frame; gripper stays frozen in view like a real eye-in-hand). "
                         "Default: steadicam behavior — re-aims at the grasp point with a level "
                         "horizon each frame, so the gripper drifts in frame as the wrist reorients.")
    ap.add_argument("--show-viewer", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="render a single static home-pose frame")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--diag", action="store_true",
                    help="--plug-traj: report per-link table penetration (link origin below tbl_top "
                         "within the table footprint) after IK each frame; no render needed (use with "
                         "--dry-run). TOOL links (fingers/wrist_3) at the surface are expected.")
    ap.add_argument("--arm-home-deg", type=float, nargs=6,
                    default=[4.0, -19.5, -113.0, 43.5, -268.9, -178.0],
                    help="arm home joint angles j0..j5 in DEGREES (overrides SBOT_HOME, which is "
                         "radians). Default = the RO1 pendant elbow-up pose. With IK on this is the "
                         "initial pose / IK seed branch; with --no-arm-ik it is the fixed arm pose.")
    ap.add_argument("--dump", default=None,
                    help="also write an openpi-format episode here: per-frame image/ + wrist_image/ PNGs, "
                         "state.npy (T,10)=[eef_pos(3),eef_rot6d(6),gripper(1)] base frame, action.npy (T,7)="
                         "[dpos(3),drotvec(3),gripper(1)] base-frame delta. EEF = the plug tip (conn_world).")
    ap.add_argument("--dump-size", type=int, default=None,
                    help="--dump: resize saved images to NxN (e.g. 224 = openpi model res, ~14x less disk). "
                         "Default None = full render resolution.")
    ap.add_argument("--no-preview", action="store_true",
                    help="skip the stitched full-res preview entirely (no compose, no --out PNGs). "
                         "Profiling showed the preview writes are ~78%% of episode wall time; use "
                         "for data-gen batches where only the --dump output matters.")
    ap.add_argument("--policy-server", default=None,
                    help="HOST:PORT of an openpi serve_policy.py websocket. CLOSED-LOOP EVAL: the "
                         "policy drives the EEF (10-D state in, 7-D delta-action chunks out) instead "
                         "of replaying the trajectory; --plug-traj only provides the plug START pose. "
                         "Success = plug reaches seat depth (socket y >= 11mm). Needs arm IK.")
    ap.add_argument("--eval-max-steps", type=int, default=300,
                    help="--policy-server: step budget before declaring failure")
    ap.add_argument("--stop-after-seat", type=int, default=None,
                    help="--plug-traj: stop rendering N frames after the plug SEATS (traj y >= 11mm) "
                         "instead of playing the full recorded tail (median seat ~frame 52/200 -> "
                         "~150 dead frames). Halves data-gen render time; episodes become "
                         "variable-length.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── robot (mirrors record_sbot_gs.py; visual-only) ──────────────────────────────
    builder = new_vbd_builder(gravity=-9.81)
    th = math.radians(args.base_yaw_deg) / 2.0
    base_xform = wp.transform(wp.vec3(*args.base_pos), wp.quat(0.0, 0.0, math.sin(th), math.cos(th)))
    handles = add_sbot(builder, base_xform, with_gripper=True)
    set_pd_gains(builder, handles)
    arm_home = (tuple(math.radians(a) for a in args.arm_home_deg)
                if args.arm_home_deg is not None else SBOT_HOME)
    set_arm_home(builder, handles, arm_home)
    if args.arm_home_deg is not None:
        print(f"[scene] arm home (deg) = {args.arm_home_deg}")
    init_theta = GRIPPER_THETA_CLOSED if args.grip == "closed" else GRIPPER_THETA_OPEN
    set_gripper(builder, handles, init_theta)

    # Table world AABB (splat bounds + placement) -> drives the collision box, jack/camera framing.
    tlo, thi = ply_bounds(args.table_ply)
    tp = np.asarray(args.table_pos, float)
    tbl_wlo, tbl_whi = tlo + tp, thi + tp
    tbl_top = float(args.table_top_z) if args.table_top_z is not None else float(tbl_whi[2])  # ply_bounds top

    # jack default: centre of the table top (auto-tracks --table-pos)
    jack_pos = (args.jack_pos if args.jack_pos is not None
                else [float((tbl_wlo[0] + tbl_whi[0]) / 2),
                      float((tbl_wlo[1] + tbl_whi[1]) / 2), tbl_top])

    if args.table_collision:
        # Static collision slab at the table top (body=-1 -> world-fixed; never moves/falls).
        # Arm shapes stay collidable (self-collisions already off in add_sbot), so the arm
        # collides with the table but not itself.
        hz = args.table_box_thick / 2.0
        cx, cy = (tbl_wlo[0] + tbl_whi[0]) / 2, (tbl_wlo[1] + tbl_whi[1]) / 2
        hx, hy = (tbl_whi[0] - tbl_wlo[0]) / 2, (tbl_whi[1] - tbl_wlo[1]) / 2
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(cx, cy, tbl_top - hz), wp.quat_identity()),
            hx=hx, hy=hy, hz=hz, label="table_collision",
        )
        print(f"[scene] table collision box: top z={tbl_top:.3f} "
              f"xy=[{tbl_wlo[0]:.2f},{tbl_wlo[1]:.2f}]..[{tbl_whi[0]:.2f},{tbl_whi[1]:.2f}]")
    else:
        for s in range(len(builder.shape_body)):
            builder.shape_flags[s] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

    model = finalize_for_vbd(builder)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()
    solver = SolverVBD(model, iterations=12)

    # ── object set: [table, jack, connector, *arm_links] (SETUP order == render order)
    base_pose = xform_pose(base_xform)
    static_links = {"base_link": base_pose}
    dyn_bodies = {n: resolve_body(model, n) for n in LINK_SPLATS if n not in static_links}
    wrist3 = dyn_bodies[GRIPPER_LINK]

    table_c = ply_centroid(args.table_ply)            # table placed absolutely -> no recenter
    cord_c = ply_centroid(args.connector_ply)
    # Jack calibration mirrors the eval's mount: the plug trajectory is re-based in the jack BODY
    # frame (jack_q, default identity == eval socket body), while the jack SPLAT gets the obj->newton
    # calibration (jack_align, eval --mount-rot 90 0 -180) and is anchored by its cavity MOUTH
    # (jack_anchor, eval --mount-anchor 0 0 0.030) -- NOT the geometric centroid, which mis-seated it.
    jack_q = euler_deg_to_quat_wxyz(*args.jack_rpy)            # physical jack body orientation -> jqw
    jack_align = euler_deg_to_quat_wxyz(*args.jack_align_rpy)  # jack splat obj->newton calibration
    jack_anchor = np.array(args.jack_anchor, float)
    jack_splat_q = quat_mul_wxyz(list(jack_q), jack_align)     # rendered jack splat orientation
    grasp_align = euler_deg_to_quat_wxyz(*args.grasp_rpy)

    # Home pose -> connector grasp offset. Auto = midpoint of the two fingertips expressed in
    # wrist_3's frame, so the connector's centroid seats between the (closed) jaws.
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    bq0 = state_0.body_q.numpy()
    if args.grasp_offset is not None:
        grasp_offset = list(args.grasp_offset)
    else:
        tip1 = resolve_body(model, "finger1_finger_tip_link")
        tip2 = resolve_body(model, "finger2_finger_tip_link")
        grasp_world = 0.5 * (bq0[tip1][:3] + bq0[tip2][:3])
        p3, q3 = bq0[wrist3][:3], newton_pose(bq0, wrist3)[1]
        base = quat_rotate_wxyz(quat_conj_wxyz(q3), np.asarray(grasp_world) - np.asarray(p3))
        axis = base / (np.linalg.norm(base) + 1e-9)  # wrist->fingertips (the tool axis)
        grasp_offset = (base + args.grasp_protrude * axis).tolist()
    print(f"[scene] connector grasp offset (wrist_3 frame) = {np.round(grasp_offset, 3)} "
          f"(protrude {args.grasp_protrude} past fingertips)")

    grip_dir = args.gripper_gs_dir or HOST_SBOT_GS   # finger splats from here; arm links from HOST_SBOT_GS

    def _link_ply(n):
        if n == GRIPPER_LINK and args.wrist3_ply:     # wrist_3: optional gripper-cropped override
            return args.wrist3_ply
        return f"{(grip_dir if 'finger' in n else HOST_SBOT_GS)}/{n}.ply"

    gonly = [n for n in LINK_SPLATS if "finger" in n]          # the 8 gripper finger links
    dummy_bg = None
    if args.gripper_only:                                       # DEBUG: only the gripper splats
        obj_plys = [host_to_container(_link_ply(n)) for n in gonly]
        if args.compare_uncut:                                  # second static copy = the uncut splat, offset
            obj_plys += [host_to_container(_link_ply(n)) for n in gonly]
        # the renderer treats splat[0] as a STATIC background -> prepend an invisible 1-pt dummy so
        # all 8 fingers get live transforms (else finger1_knuckle, the drive link, would freeze).
        dummy_bg = os.path.join(HOST_GSVLA, "objects", "_gonly_bg.ply")
        _fp = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
               "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
        _db = np.zeros((1, 14), "<f4"); _db[0, 0:3] = [0.0, 0.0, 50.0]; _db[0, 3:6] = -6.0
        _db[0, 6] = 1.0; _db[0, 13] = -30.0                    # far, tiny, invisible
        with open(dummy_bg, "wb") as _f:
            _f.write(("ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                      + "".join(f"property float {p}\n" for p in _fp) + "end_header\n").encode("ascii"))
            _f.write(_db.tobytes())
        print(f"[scene] GRIPPER-ONLY: {len(gonly)} finger splats + invisible bg (all fingers articulate)")
    else:
        obj_plys = [host_to_container(p) for p in (args.table_ply, args.jack_ply, args.connector_ply)]
        obj_plys += [host_to_container(_link_ply(n)) for n in LINK_SPLATS]

    ped_pose = None
    if args.mount and not args.gripper_only:                    # pedestal LAST -> connector stays index 2
        base_z = float(args.base_pos[2])
        ped_host = os.path.join(HOST_GSVLA, "objects", "pedestal_gen.ply")
        ng = make_pedestal_ply(ped_host, args.mount_size, args.mount_size, base_z, color=args.mount_color)
        obj_plys.append(host_to_container(ped_host))
        ped_pose = ([float(args.base_pos[0]), float(args.base_pos[1]), base_z / 2.0], [1.0, 0.0, 0.0, 0.0])
        print(f"[scene] pedestal: {args.mount_size * 100:.0f}x{args.mount_size * 100:.0f}cm column floor->"
              f"z={base_z:.3f} ({ng} gaussians) under base {np.round(args.base_pos, 3)}")

    def scene_poses(bq):
        if args.gripper_only:                                                  # DEBUG: just the fingers
            poses = [newton_pose(bq, dyn_bodies[n]) for n in gonly]
            if args.compare_uncut:                       # static copy at capture pose (bq0, theta=-0.6), offset
                for n in gonly:
                    p, q = newton_pose(bq0, dyn_bodies[n])
                    poses.append(([p[0] + uncut_off[0], p[1] + uncut_off[1], p[2] + uncut_off[2]], q))
            return poses
        poses = [
            (list(args.table_pos), [3.0, 0.0, 0.0, 0.0]),                       # table (absolute)
            static_pose(jack_pos, jack_splat_q, jack_anchor),                    # jack (table centre, eval-calibrated)
            ride_pose(bq[wrist3][:3], newton_pose(bq, wrist3)[1],               # connector (in jaws)
                      align=grasp_align, centroid=cord_c, offset=grasp_offset),
        ]
        poses += [
            static_links[n] if n in static_links else newton_pose(bq, dyn_bodies[n])
            for n in LINK_SPLATS
        ]
        if ped_pose is not None:
            poses.append(ped_pose)                                              # pedestal (static, last)
        return poses

    # ── camera: frame arm + jack + table-top (bq0 already from eval_fk above) ─────────
    # Warn if any arm link starts inside the table box (would fight the contact at t=0).
    if args.table_collision:
        m = 0.05
        hit = [n for n, idx in dyn_bodies.items()
               if tbl_wlo[0] - m <= bq0[idx][0] <= tbl_whi[0] + m
               and tbl_wlo[1] - m <= bq0[idx][1] <= tbl_whi[1] + m
               and tbl_top - args.table_box_thick - m <= bq0[idx][2] <= tbl_top + m]
        if hit:
            print(f"[scene] WARNING: links {hit} start inside the table box -> expect contact "
                  f"jitter at t=0; nudge --table-pos forward (+y).")

    uncut_off = np.zeros(3)
    if args.gripper_only and args.compare_uncut:        # sideways offset for the uncut reference copy
        _a, _e = math.radians(args.azim), math.radians(args.elev)
        _dir = np.array([math.cos(_e) * math.cos(_a), math.cos(_e) * math.sin(_a), math.sin(_e)])
        _right = np.cross(_dir, [0.0, 0.0, 1.0])
        uncut_off = _right / (np.linalg.norm(_right) + 1e-9) * args.uncut_gap
    if args.gripper_only:                               # zoom tight on just the gripper fingers
        gpts = np.array([bq0[dyn_bodies[n]][:3] for n in gonly], float)
        if args.compare_uncut:                          # include the offset copy in the frame
            gpts = np.vstack([gpts, gpts + uncut_off])
        lo, hi = gpts.min(0) - 0.03, gpts.max(0) + 0.03
    elif args.frame == "jack":
        r = args.frame_radius
        lo, hi = np.asarray(jack_pos) - r, np.asarray(jack_pos) + r
    elif args.frame == "table":
        pad = 0.05
        lo = np.array([tbl_wlo[0] - pad, tbl_wlo[1] - pad, tbl_top - 0.05])
        hi = np.array([tbl_whi[0] + pad, tbl_whi[1] + pad, tbl_top + 0.25])
    else:  # scene: arm + table + jack
        arm_pts = [base_pose[0]] + [bq0[dyn_bodies[n]][:3] for n in dyn_bodies]
        framing = np.array(arm_pts + [jack_pos,
                                      [(tbl_wlo[0] + tbl_whi[0]) / 2, (tbl_wlo[1] + tbl_whi[1]) / 2, tbl_top]], float)
        lo, hi = framing.min(0) - 0.15, framing.max(0) + 0.15
    center = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo))
    if args.target is not None:
        center = np.asarray(args.target, float)
    if args.eye is not None:
        eye = np.asarray(args.eye, float)
    else:
        a, e = math.radians(args.azim), math.radians(args.elev)
        direction = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        eye = center + direction * args.dist_scale * diag
    if args.eye_offset is not None:                         # relative nudge from the placed eye
        eye = eye + np.asarray(args.eye_offset, float)
    if args.target_offset is not None:                     # relative nudge of the aim point
        center = center + np.asarray(args.target_offset, float)
    cam_K = make_intrinsics(args.width, args.height, args.fov_deg)
    print(f"[scene] FRONT cam: eye={np.round(eye, 3)} aim={np.round(center, 3)}")
    cam_quat = look_at_quat(eye.tolist(), center.tolist(), up=(0.0, 0.0, 1.0), convention="ros")
    # perpendicular side camera (+x): same aim/distance as the front cam, azimuth 0 (looks -x
    # toward the table centre), elevation --side-elev (default = --elev).
    extra_cams = None
    if args.side_cam:
        se = math.radians(args.side_elev if args.side_elev is not None else args.elev)
        sdir = np.array([math.cos(se), 0.0, math.sin(se)])       # +x, raised by side_elev
        side_eye = center + sdir * args.dist_scale * diag
        if args.side_z is not None:                              # override the eye's world z (raise/lower)
            side_eye[2] = args.side_z
        side_tgt = center.copy()                                 # aim point: scene centre by default ...
        if args.side_target_z is not None:                       # ... lower it (< eye z) to tilt DOWN
            side_tgt[2] = args.side_target_z
        side_quat = look_at_quat(side_eye.tolist(), side_tgt.tolist(), up=(0.0, 0.0, 1.0), convention="ros")
        extra_cams = [(side_eye.tolist(), side_quat)]
        print(f"[scene] side cam (+x): eye={np.round(side_eye, 3)} aim={np.round(side_tgt, 3)} "
              f"elev={math.degrees(se):.0f}deg -> stitched front|side")
    if args.mirror_cam:                                          # diagonally opposite: reflect the FRONT
        mirror_eye = 2.0 * center - eye                          # eye THROUGH the aim (gripper), look back
        mirror_quat = look_at_quat(mirror_eye.tolist(), center.tolist(), up=(0.0, 0.0, 1.0), convention="ros")
        extra_cams = (extra_cams or []) + [(mirror_eye.tolist(), mirror_quat)]
        print(f"[scene] mirror cam: eye={np.round(mirror_eye, 3)} aim={np.round(center, 3)} "
              f"(antipode of front through the gripper)")

    # wrist eye-in-hand camera (DYNAMIC: recomputed every frame from wrist_3's live pose)
    wrist_K = make_intrinsics(args.width, args.height, args.wrist_fov) if args.wrist_cam else None
    cam_labels = (["FRONT"] + (["SIDE +x"] if args.side_cam else [])
                  + (["MIRROR"] if args.mirror_cam else []) + (["WRIST"] if args.wrist_cam else []))

    _wprint = [True]
    _wrigid = [None]                                            # cached (eye, quat) in wrist_3 LOCAL frame

    def wrist_cams(bq):
        if not args.wrist_cam:
            return None
        wp, wq = newton_pose(bq, wrist3)                        # wrist_3 world pose (wxyz)
        if args.wrist_rigid and _wrigid[0] is not None:         # rigid mount: constant local pose
            eye_l, quat_l = _wrigid[0]
            eye = np.asarray(wp) + quat_rotate_wxyz(wq, eye_l)
            quat = quat_mul_wxyz(list(wq), quat_l)
            return [(eye.tolist(), quat, wrist_K)]
        g = np.asarray(grasp_offset, float)
        grasp_pt = np.asarray(wp) + quat_rotate_wxyz(wq, g)     # connector / grasp point (world)
        axis = quat_rotate_wxyz(wq, g)
        axis = axis / (np.linalg.norm(axis) + 1e-9)            # wrist->connector (tool axis, world)
        side = np.cross(axis, [0.0, 0.0, 1.0])                 # horizontal, perpendicular to tool axis
        sn = np.linalg.norm(side)
        side = side / sn if sn > 1e-6 else np.array([1.0, 0.0, 0.0])
        perp2 = np.cross(axis, side)                           # completes the perpendicular basis
        th = math.radians(args.wrist_orbit)                   # orbit the lateral offset around the tool axis
        lat = math.cos(th) * side + math.sin(th) * perp2
        # sit beside + behind + above the connector so the black gripper doesn't fill the view
        eye = (grasp_pt - args.wrist_back * axis + args.wrist_side * lat
               + np.array([0.0, 0.0, args.wrist_up]))
        aim = grasp_pt - args.wrist_aim_back * axis            # shift aim toward the fingers (tilt up)
        quat = look_at_quat(eye.tolist(), aim.tolist(), up=(0.0, 0.0, 1.0), convention="ros")
        if args.wrist_rigid and _wrigid[0] is None:             # bake this FIRST pose into wrist_3's frame
            eye_l = quat_rotate_wxyz(quat_conj_wxyz(wq), np.asarray(eye) - np.asarray(wp))
            quat_l = quat_mul_wxyz(quat_conj_wxyz(wq), quat)
            _wrigid[0] = (np.asarray(eye_l, float), list(quat_l))
            print("[wrist] RIGID mount: first pose baked into wrist_3's frame (gripper frozen in view)")
        if _wprint[0]:
            print(f"[wrist] eye={np.round(eye, 3)} -> grasp={np.round(grasp_pt, 3)} (table z={tbl_top:.3f})")
            _wprint[0] = False
        return [(eye.tolist(), quat, wrist_K)]
    if args.wrist_cam:
        print(f"[scene] wrist cam: eye-in-hand on wrist_3 fov={args.wrist_fov:.0f} "
              f"(back {args.wrist_back} up {args.wrist_up}) -> stitched last")
    print(f"[scene] wrist_3 home @ {np.round(bq0[wrist3][:3], 3)}  (place jack/connector near here)")
    print(f"[scene] frame center={np.round(center, 3)} diag={diag:.3f} eye={np.round(eye, 3)}")
    print(
        f"[scene] {len(obj_plys)} splats: table, jack, connector "
        f"+ {len(LINK_SPLATS)} robot links (7 arm + 8 gripper)"
    )
    print(f"[scene] jack @ {np.round(jack_pos, 3)} (table centre top); table top z={tbl_top:.3f}")

    client = NewtonGSClient(
        ply_paths=obj_plys, cam_K=cam_K, cam_pos=eye.tolist(), cam_quat=cam_quat,
        bg_ply=(host_to_container(dummy_bg) if args.gripper_only
                else (host_to_container(args.bg_ply) if args.bg_ply else None)),
        bg_pose=(list(args.bg_pos), [1.0, 0.0, 0.0, 0.0]),
        show_viewer=args.show_viewer, dry_run=args.dry_run,
        extra_cameras=extra_cams,
    )

    def render_gs(poses, dyn_cameras=None):
        # GS render + optional global brightness: gamma>1 lifts shadows (e.g. plug interior),
        # gain scales overall. Both default 1.0 = untouched. rgb is float in [0,1].
        rgb = client.render(poses, dyn_cameras=dyn_cameras)
        if args.gain == 1.0 and args.gamma == 1.0:
            return rgb
        a = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
        if args.gamma != 1.0:
            a = a ** (1.0 / args.gamma)
        return np.clip(a * args.gain, 0.0, 1.0)

    # Optional Newton mesh view (headless GL) at the same camera, stitched left of the GS frame.
    gl = None
    if args.side_by_side:
        from newton.viewer import ViewerGL

        gl = ViewerGL(width=args.width, height=args.height, headless=True, vsync=False)
        gl.set_model(model)
        pitch, yaw = eye_target_to_pitch_yaw(eye, center)
        gl.set_camera(wp.vec3(*[float(v) for v in eye]), pitch, yaw)
        gl.camera.fov = args.gl_fov if args.gl_fov is not None else args.fov_deg
        print(f"[scene] side-by-side: Newton GL @ pitch={pitch:.1f} yaw={yaw:.1f} fov={gl.camera.fov:.1f}")

    def _white_bg(u8):
        if not args.bg_white:
            return u8
        u8 = u8.copy()
        u8[u8.max(axis=2) < args.bg_white_thr] = 255           # uncovered (near-black) bg -> white
        return u8

    def make_frame(state, t):
        bq = state.body_q.numpy()
        gs_u8 = _white_bg(compose_multicam(render_gs(scene_poses(bq), dyn_cameras=wrist_cams(bq)), cam_labels))
        if gl is None:
            return gs_u8
        gl.begin_frame(t)
        gl.log_state(state)
        gl.end_frame()
        return composite_lr(gl.get_frame().numpy(), gs_u8)

    # ── policy-rollout replay: connector (+ arm via IK) follows a recorded plug trajectory ──
    if args.plug_traj:
        traj = np.load(args.plug_traj)                          # (T,7) [pos3, quat4 wxyz] in SOCKET frame
        if args.trim_settle_mm > 0.0 and len(traj) > 2:        # v2: drop the physics reset transient at the
            step_mm = 1000.0 * np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1)   # trajectory start. The
            qn = traj[:, 3:] / (np.linalg.norm(traj[:, 3:], axis=1, keepdims=True) + 1e-9)   # plug ORIENTATION
            step_deg = 2.0 * np.degrees(np.arccos(                                    # settles a few frames
                np.abs(np.sum(qn[:-1] * qn[1:], axis=1)).clip(-1.0, 1.0)))            # after its position, so
            k = 0                                                                     # gate on BOTH channels
            while k < len(step_mm) and (step_mm[k] > args.trim_settle_mm or step_deg[k] > args.trim_settle_deg):
                k += 1
            if 0 < k < len(traj) - 1:                          # keep >=2 frames; leave clean trajectories alone
                print(f"[replay] trim reset transient: dropping {k} leading frame(s) "
                      f"(|dpos| {np.round(step_mm[:k], 1)}mm / |drot| {np.round(step_deg[:k], 1)}deg exceed "
                      f"{args.trim_settle_mm}mm / {args.trim_settle_deg}deg) -> settled grasp start")
                traj = traj[k:]
        jqw = list(jack_q)                                      # jack world orientation (wxyz)
        conn_align = euler_deg_to_quat_wxyz(*args.conn_rpy)     # eval plug-splat calibration (-90 0 0)
        conn_anchor = np.array(args.conn_anchor, float)        # plug mating-face anchor (= eval --plug-anchor)
        # socket-frame plug pose -> sbot-scene world: place the recorded insertion at the jack
        conn_world = [(np.asarray(jack_pos, float) + quat_rotate_wxyz(jqw, s[:3]),
                       quat_mul_wxyz(quat_mul_wxyz(jqw, [float(x) for x in s[3:7]]), conn_align)) for s in traj]
        # --grasp-along-cord: the CORD axis, taken EMPIRICALLY from the trajectory itself (the
        # insertion advance direction, head-ward, in world) -- no quat-convention assumptions.
        # The HAND's IK targets shift back along it by `along` (jaws land that far behind the
        # plug's anchor, toward the cable end); the plug still DRAWS at conn_world unchanged.
        cord_world = quat_rotate_wxyz(jqw, (traj[-1, :3] - traj[0, :3]))
        cord_world = np.asarray(cord_world, float)
        cord_world /= np.linalg.norm(cord_world) + 1e-9
        ik_tgt = [np.asarray(cp, float) - args.grasp_along_cord * cord_world for cp, _ in conn_world]
        if args.stop_after_seat is not None:                   # drop the dead post-seat tail
            seated = traj[:, 1] >= 0.011                       # within 0.8mm of full depth (11.8mm)
            if seated.any():
                n_keep = min(len(conn_world), int(np.argmax(seated)) + args.stop_after_seat + 1)
                conn_world, ik_tgt = conn_world[:n_keep], ik_tgt[:n_keep]
                print(f"[replay] stop-after-seat: seat at traj frame {int(np.argmax(seated))}, "
                      f"rendering {n_keep}/{len(traj)} insertion frames")
        print(f"[replay] {len(conn_world)} plug poses from {args.plug_traj}; connector at jack {np.round(jack_pos, 3)}; "
              f"arm IK {'OFF' if args.no_arm_ik else 'ON'}")
        if args.grasp_along_cord:
            print(f"[replay] along-cord {args.grasp_along_cord:+.3f}m: cord axis (world) = "
                  f"{np.round(cord_world, 2)}; hand IK targets shifted toward the cable end")

        ik_solver = pos_obj = rot_obj = joint_q_ik = None
        arm_coords = list(handles.arm_joints)
        if not args.no_arm_ik:
            from newton import ik  # noqa: PLC0415
            ik_builder = new_vbd_builder(gravity=0.0)
            add_sbot(ik_builder, base_xform, with_gripper=True)
            ik_model = ik_builder.finalize()
            w3 = resolve_body(ik_model, GRIPPER_LINK)
            ga = wp.quat(grasp_align[1], grasp_align[2], grasp_align[3], grasp_align[0])  # wxyz->xyzw
            pos_obj = ik.IKObjectivePosition(
                link_index=w3, link_offset=wp.vec3(*[float(v) for v in grasp_offset]),
                target_positions=wp.array([wp.vec3(*ik_tgt[0])], dtype=wp.vec3))
            rot_obj = ik.IKObjectiveRotation(
                link_index=w3, link_offset_rotation=ga,
                target_rotations=wp.array([wp.vec4(*conn_world[0][1])], dtype=wp.vec4))
            joint_q_ik = wp.array(model.joint_q.numpy().reshape(1, -1).astype(np.float32), dtype=wp.float32)
            ik_solver = ik.IKSolver(model=ik_model, n_problems=1, objectives=[pos_obj, rot_obj],
                                    lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC)
            # HOME-RELATIVE GRASP: the gripper grasps the (flexible) cable at its HOME orientation and
            # then rides the cable's RELATIVE rotation along the rollout — it does NOT slew to the
            # plug's absolute orientation (that forced the arm ~120deg off home). rot target for the
            # gripper (wrist ∘ grasp_align) at frame i = cq_i ∘ cq0^-1 ∘ G_home, where G_home is the
            # home gripper orientation. At i=0 this is exactly G_home -> the grasp pose is ~home joints.
            _R_home_wrist = newton_pose(bq0, wrist3)[1]                       # home wrist_3 orientation (wxyz)
            _G_home = quat_mul_wxyz(list(_R_home_wrist), list(grasp_align))   # home GRIPPER orientation (wxyz)
            _grip_R_offset = quat_mul_wxyz(quat_conj_wxyz(conn_world[0][1]), _G_home)  # fixed cable->gripper

        def grip_rot_xyzw(cq):                                # desired gripper orientation at a frame -> XYZW
            g = (_G_home if args.insert_hold_home_rot          # hold home orientation (cord flexes), OR
                 else quat_mul_wxyz(list(cq), _grip_R_offset)) # ride the cable's relative rotation
            return wp.vec4(g[1], g[2], g[3], g[0])            # IKObjectiveRotation wants XYZW

        frames = []
        jq = model.joint_q.numpy().copy()
        if args.dump:
            os.makedirs(args.dump, exist_ok=True)
        dump_eef, dump_grip, dump_phase = [], [], []   # full-episode data (hold+approach+insert)
        import time as _time                            # per-stage profile (printed at the end)
        prof = {"newton (IK+FK)": 0.0, "gs render": 0.0, "compose/stitch": 0.0, "dump io": 0.0}
        ik_err = []
        pen = {n: [0, 0.0] for n in dyn_bodies} if args.diag else None   # [frames, max depth m]
        parent_of = {}
        if pen is not None:                                              # body -> parent body (for shaft sampling)
            jp, jc = model.joint_parent.numpy(), model.joint_child.numpy()
            parent_of = {int(jc[j]): int(jp[j]) for j in range(len(jc))}

        # ── CLOSED-LOOP POLICY EVAL: the trained pi05 drives the EEF from home; the plug waits at
        # its start pose until the policy closes the gripper, then rides the hand. Mirrors the
        # training data's conventions exactly (base-frame 10-D state, 7-D delta actions, 224px cams).
        if args.policy_server:
            if ik_solver is None:
                raise SystemExit("--policy-server needs arm IK (drop --no-arm-ik)")
            from PIL import Image  # noqa: PLC0415
            from openpi_client import websocket_client_policy  # noqa: PLC0415
            from scipy.spatial.transform import Rotation as _Rot  # noqa: PLC0415
            host, port = args.policy_server.rsplit(":", 1)
            pol = websocket_client_policy.WebsocketClientPolicy(host=host, port=int(port))
            PROMPT = "pick up the ethernet cable and plug it into the jack"
            SIZE = args.dump_size or 224
            thb = math.radians(args.base_yaw_deg) / 2.0
            base_quat = [math.cos(thb), 0.0, 0.0, math.sin(thb)]
            jinv = quat_conj_wxyz(jqw)

            def cam224(rgb_out, label):
                arr = np.asarray(rgb_out)
                view = arr if arr.ndim == 3 else arr[cam_labels.index(label)]
                im = Image.fromarray((view * 255.0).clip(0, 255).astype("uint8"))
                return np.asarray(im.resize((SIZE, SIZE), Image.LANCZOS))

            # EEF starts at the HOME hand grasp point (= training frame 0); track it in BASE frame
            hp0, hq0 = newton_pose(bq0, wrist3)
            eef_pw = np.asarray(hp0) + quat_rotate_wxyz(hq0, grasp_offset)
            eef_qw = quat_mul_wxyz(list(hq0), list(grasp_align))
            p_b, q_b = eef_world_to_base(eef_pw, eef_qw, args.base_pos, base_quat)
            grip, grasped, seated = 0.0, False, False
            grasp_step = seat_step = -1
            cp0, cq0 = conn_world[0]
            chunk, k = None, 0
            tr = {n: [] for n in ("eef_w", "plug_w", "y_sock", "grip", "dpos", "drot")}  # motion trace
            print(f"[eval] policy {args.policy_server} | plug start {np.round(cp0, 3)} | "
                  f"max {args.eval_max_steps} steps")
            for step in range(args.eval_max_steps):
                p_w = np.asarray(quat_rotate_wxyz(base_quat, p_b)) + np.asarray(args.base_pos)
                q_w = quat_mul_wxyz(base_quat, list(q_b))
                pos_obj.set_target_positions(wp.array([wp.vec3(*p_w)], dtype=wp.vec3))
                # IKObjectiveRotation targets are XYZW (probe-verified: wxyz here flings the arm)
                rot_obj.set_target_rotations(
                    wp.array([wp.vec4(q_w[1], q_w[2], q_w[3], q_w[0])], dtype=wp.vec4))
                ik_solver.step(joint_q_ik, joint_q_ik, iterations=24)
                ikq = joint_q_ik.numpy()[0]
                for j in arm_coords:
                    jq[j] = ikq[j]
                theta = GRIPPER_THETA_OPEN + grip * (GRIPPER_THETA_CLOSED - GRIPPER_THETA_OPEN)
                for idx, ratio in handles.gripper_coupling:
                    jq[idx] = ratio * theta
                model.joint_q.assign(jq)
                newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
                bqe = state_0.body_q.numpy()
                if not grasped and grip > 0.5:
                    # LATCH grasp: only if the hand is actually AT the plug (proximity gate), and
                    # keep the plug's current pose relative to the hand (no snap/teleport/flip).
                    expect = p_w + args.grasp_along_cord * cord_world   # plug pos if perfectly grasped
                    gdist = float(np.linalg.norm(np.asarray(cp0) - expect))
                    if gdist < 0.03:
                        grasped, grasp_step = True, step
                        qinv_w = quat_conj_wxyz(q_w)
                        latch_p = np.asarray(quat_rotate_wxyz(qinv_w, np.asarray(cp0) - p_w))
                        latch_q = quat_mul_wxyz(qinv_w, list(cq0))
                        print(f"[eval] GRASP latched at step {step} (dist {gdist * 1000:.1f}mm)")
                elif grasped and grip < 0.5:                            # release: plug stays put
                    cp0 = np.asarray(cp_p)
                    cq0 = list(cq_p)
                    grasped = False
                    print(f"[eval] RELEASED at step {step}")
                if grasped:
                    cp_p = p_w + np.asarray(quat_rotate_wxyz(q_w, latch_p))
                    cq_p = quat_mul_wxyz(list(q_w), latch_q)
                else:
                    cp_p, cq_p = cp0, cq0
                poses = scene_poses(bqe)
                poses[2] = static_pose(cp_p, cq_p, conn_anchor)
                rgb_out = render_gs(poses, dyn_cameras=wrist_cams(bqe))
                if not args.no_preview:
                    frames.append(_white_bg(compose_multicam(rgb_out, cam_labels)))
                state10 = np.concatenate([p_b, quat_to_rot6d(q_b), [grip]]).astype(np.float32)
                if chunk is None or k >= len(chunk):
                    res = pol.infer({"observation/image": cam224(rgb_out, "FRONT"),
                                     "observation/wrist_image": cam224(rgb_out, "WRIST"),
                                     "observation/state": state10, "prompt": PROMPT})
                    chunk, k = np.asarray(res["actions"]), 0
                a = np.array(chunk[k], dtype=np.float64); k += 1   # copy: msgpack arrays are read-only
                y_sock = float(quat_rotate_wxyz(jinv, np.asarray(cp_p) - np.asarray(jack_pos))[1])
                tr["eef_w"].append(np.asarray(p_w, float))         # pose BEFORE this step's action
                tr["plug_w"].append(np.asarray(cp_p, float))
                tr["y_sock"].append(y_sock)
                tr["grip"].append(grip)
                tr["dpos"].append(a[:3].copy())
                tr["drot"].append(a[3:6].copy())
                p_b = np.asarray(p_b) + a[:3]
                xq = _Rot.from_rotvec(a[3:6]).as_quat()            # xyzw
                q_b = quat_mul_wxyz(list(q_b), [xq[3], xq[0], xq[1], xq[2]])
                grip = float(np.clip(a[6], 0.0, 1.0))
                if grasped and y_sock >= 0.011 and not seated:
                    seated, seat_step = True, step
                    break
                if step % 30 == 0:
                    print(f"[eval] step {step}: grip {grip:.2f} grasped={grasped} "
                          f"plug y_sock {y_sock * 1000:+.1f}mm", flush=True)
            T = {n: np.asarray(v) for n, v in tr.items()}
            # motion-quality metrics: table penetration (inside the table xy footprint, below its
            # top) + jerk (delta of commanded dpos between consecutive steps)
            def _pen(P):
                inside = ((P[:, 0] >= tbl_wlo[0]) & (P[:, 0] <= tbl_whi[0])
                          & (P[:, 1] >= tbl_wlo[1]) & (P[:, 1] <= tbl_whi[1]))
                depth = np.where(inside, tbl_top - P[:, 2], -np.inf)
                return float(np.maximum(depth, 0).max()), int((depth > 0.002).sum())
            plug_pen_mm, plug_pen_frames = _pen(T["plug_w"]) if len(T["plug_w"]) else (0.0, 0)
            eef_pen_mm, eef_pen_frames = _pen(T["eef_w"]) if len(T["eef_w"]) else (0.0, 0)
            step_mm = np.linalg.norm(T["dpos"], axis=1) * 1000 if len(T["dpos"]) else np.zeros(1)
            jerk_mm = (np.linalg.norm(np.diff(T["dpos"], axis=0), axis=1) * 1000
                       if len(T["dpos"]) > 1 else np.zeros(1))
            clean = bool(seated and plug_pen_mm * 1000 < 5.0)
            result = {"seated": bool(seated), "clean_seat": clean, "seat_step": seat_step,
                      "grasp_step": grasp_step, "steps": step + 1,
                      "final_y_sock_mm": round(y_sock * 1000, 2),
                      "plug_pen_mm": round(plug_pen_mm * 1000, 1), "plug_pen_frames": plug_pen_frames,
                      "eef_pen_mm": round(eef_pen_mm * 1000, 1), "eef_pen_frames": eef_pen_frames,
                      "path_len_mm": round(float(step_mm.sum()), 1),
                      "step_mm_mean": round(float(step_mm.mean()), 2),
                      "step_mm_max": round(float(step_mm.max()), 2),
                      "jerk_mm_mean": round(float(jerk_mm.mean()), 2)}
            print(f"[eval] RESULT: {result}")
            import json  # noqa: PLC0415
            with open(os.path.join(args.out, "result.json"), "w") as f:
                json.dump(result, f, indent=2)
            np.savez(os.path.join(args.out, "trace.npz"), tbl_top=tbl_top, tbl_lo=tbl_wlo,
                     tbl_hi=tbl_whi, jack=np.asarray(jack_pos), **T)
            if not args.no_preview and frames:
                _save(frames, args.out, smoke=False, fps=args.fps)
            if gl is not None:
                gl.close()
            return

        # ── home -> grasp approach (prepended): ease the arm from its fixed home pose to the
        # frame-0 grasp pose with the jaws opening->closing, so it doesn't snap into the insertion.
        # IK only; connector waits at its start pose until the jaws reach it. (skip with --no-approach)
        if ik_solver is not None and (args.grasped_only
                                      or (not args.no_approach and args.approach_frames > 0)):
            q_home = jq.copy()
            cp0, cq0 = conn_world[0]
            pos_obj.set_target_positions(wp.array([wp.vec3(*ik_tgt[0])], dtype=wp.vec3))
            rot_obj.set_target_rotations(                             # home-relative grasp (frame 0 -> G_home)
                wp.array([grip_rot_xyzw(cq0)], dtype=wp.vec4))
            ik_solver.step(joint_q_ik, joint_q_ik, iterations=48)     # solve the grasp pose once
            q_grasp = joint_q_ik.numpy()[0].copy()
            _qg_deg = [math.degrees(q_grasp[j]) for j in arm_coords]   # solved grasp-pose arm joints
            _hm_deg = [math.degrees(a) for a in arm_home]
            print(f"[scene] grasp-pose arm joints (deg) = {np.round(_qg_deg, 1)}")
            print(f"[scene]   vs arm home        (deg) = {np.round(_hm_deg, 1)}  "
                  f"(max |Δ| {np.max(np.abs(np.array(_qg_deg) - np.array(_hm_deg))):.1f} deg)")
            nap = 0 if args.grasped_only else args.approach_frames    # v2 --grasped-only: no home-hold/approach
            for f in range(0 if args.grasped_only else args.home_hold + nap):
                if f < args.home_hold:
                    alpha, theta = 0.0, GRIPPER_THETA_OPEN            # hold home, jaws open
                else:
                    a = (f - args.home_hold) / max(nap - 1, 1)
                    alpha = a * a * a * (a * (a * 6 - 15) + 10)       # smootherstep ease
                    cfrac = min(max((a - 0.66) / 0.34, 0.0), 1.0)     # close jaws over the last third
                    theta = GRIPPER_THETA_OPEN + (GRIPPER_THETA_CLOSED - GRIPPER_THETA_OPEN) * cfrac
                for j in arm_coords:
                    jq[j] = (1.0 - alpha) * q_home[j] + alpha * q_grasp[j]
                for idx, ratio in handles.gripper_coupling:
                    jq[idx] = ratio * theta
                model.joint_q.assign(jq)
                newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
                bqa = state_0.body_q.numpy()
                poses = scene_poses(bqa)
                if not args.gripper_only:
                    poses[2] = static_pose(cp0, cq0, conn_anchor)     # connector waits at the start pose
                wc = wrist_cams(bqa)
                if args.dump:                                         # full-episode data: EEF = grasp point
                    hp, hqw = newton_pose(bqa, wrist3)
                    dump_eef.append((np.asarray(hp) + quat_rotate_wxyz(hqw, grasp_offset),
                                     quat_mul_wxyz(list(hqw), list(grasp_align))))
                    dump_grip.append((theta - GRIPPER_THETA_OPEN)
                                     / (GRIPPER_THETA_CLOSED - GRIPPER_THETA_OPEN))
                    dump_phase.append(0 if f < args.home_hold else 1)
                if not args.dry_run:
                    rgb_a = render_gs(poses, dyn_cameras=wc)
                    if not args.no_preview:
                        frames.append(_white_bg(compose_multicam(rgb_a, cam_labels)))
                    if args.dump:
                        dump_cams(rgb_a, cam_labels, args.dump, len(dump_eef) - 1, size=args.dump_size)
                if f % args.fps == 0:
                    print(f"[approach] frame {f}/{args.home_hold + nap}", flush=True)
            for j in arm_coords:                                     # hand off to insertion at the grasp pose,
                jq[j] = q_grasp[j]                                    # jaws closed on the connector
            for idx, ratio in handles.gripper_coupling:
                jq[idx] = ratio * GRIPPER_THETA_CLOSED
            if args.grasped_only:
                print("[approach] --grasped-only: arm at grasp pose, jaws closed (approach NOT recorded)")
            else:
                print(f"[approach] home->grasp done ({args.home_hold}+{nap} frames), jaws closed")

        _home_deg = np.array([math.degrees(a) for a in arm_home]) if ik_solver is not None else None
        _jdrift = 0.0                                          # max |arm joint - home| (deg) over insertion
        for i, (cp, cq) in enumerate(conn_world):
            _t0 = _time.perf_counter()
            if ik_solver is not None:                          # solve arm so the grasp point reaches the plug
                pos_obj.set_target_positions(wp.array([wp.vec3(*ik_tgt[i])], dtype=wp.vec3))
                rot_obj.set_target_rotations(                       # home-relative grasp: ride cable's
                    wp.array([grip_rot_xyzw(cq)], dtype=wp.vec4))   # relative rotation, not plug's absolute
                ik_solver.step(joint_q_ik, joint_q_ik, iterations=24)
                ikq = joint_q_ik.numpy()[0]
                for j in arm_coords:
                    jq[j] = ikq[j]
                _jdrift = max(_jdrift, float(np.max(np.abs(
                    np.array([math.degrees(ikq[j]) for j in arm_coords]) - _home_deg))))
                model.joint_q.assign(jq)
                newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
                gp, gq = newton_pose(state_0.body_q.numpy(), dyn_bodies[GRIPPER_LINK])  # achieved grasp pt
                ik_err.append(float(np.linalg.norm(np.asarray(gp) + quat_rotate_wxyz(gq, grasp_offset) - ik_tgt[i])))
                if args.diag and i % 60 == 0:                  # decompose fingertip->plug into cord axes
                    bqd = state_0.body_q.numpy()
                    tmid = 0.5 * (bqd[dyn_bodies["finger1_finger_tip_link"]][:3]
                                  + bqd[dyn_bodies["finger2_finger_tip_link"]][:3])
                    v = np.asarray(cp, float) - tmid           # tips midpoint -> plug anchor (world)
                    a = float(v @ cord_world)
                    perp = v - a * cord_world
                    print(f"[grasp-diag] frame {i}: tips->plug  along-cord {a * 1000:+6.1f}mm  "
                          f"perp {np.linalg.norm(perp) * 1000:6.1f}mm  (perp vec {np.round(perp, 3)})")
            prof["newton (IK+FK)"] += _time.perf_counter() - _t0
            if pen is not None:                                # table-penetration: sample each link's SHAFT
                bqf = state_0.body_q.numpy()
                nb = bqf.shape[0]
                for n, idx in dyn_bodies.items():
                    pts = [np.asarray(bqf[idx][:3], float)]     # link origin + points toward its parent joint
                    par = parent_of.get(idx, -1)
                    if 0 <= par < nb:
                        a, b = np.asarray(bqf[idx][:3], float), np.asarray(bqf[par][:3], float)
                        pts += [a + (b - a) * t for t in (0.25, 0.5, 0.75, 1.0)]
                    dmax = 0.0
                    for p in pts:
                        if (tbl_wlo[0] <= p[0] <= tbl_whi[0] and tbl_wlo[1] <= p[1] <= tbl_whi[1]
                                and p[2] < tbl_top):
                            dmax = max(dmax, tbl_top - float(p[2]))
                    if dmax > 0:
                        pen[n][0] += 1
                        pen[n][1] = max(pen[n][1], dmax)
            poses = scene_poses(state_0.body_q.numpy())
            if not args.gripper_only:
                poses[2] = static_pose(cp, cq, conn_anchor)    # index 2 = connector -> follow the trajectory
            wc = wrist_cams(state_0.body_q.numpy())
            if args.dump:                                      # EEF = grasp point (continuous across phases)
                bqr = state_0.body_q.numpy()
                hp, hqw = newton_pose(bqr, wrist3)
                dump_eef.append((np.asarray(hp) + quat_rotate_wxyz(hqw, grasp_offset),
                                 quat_mul_wxyz(list(hqw), list(grasp_align))))
                dump_grip.append(1.0)
                dump_phase.append(2)
            if not args.dry_run:
                _t0 = _time.perf_counter()
                rgb_out = render_gs(poses, dyn_cameras=wc)
                _t1 = _time.perf_counter()
                prof["gs render"] += _t1 - _t0
                if not args.no_preview:
                    frames.append(_white_bg(compose_multicam(rgb_out, cam_labels)))
                _t2 = _time.perf_counter()
                prof["compose/stitch"] += _t2 - _t1
                if args.dump:
                    dump_cams(rgb_out, cam_labels, args.dump, len(dump_eef) - 1, size=args.dump_size)
                    prof["dump io"] += _time.perf_counter() - _t2
            if i % args.fps == 0:
                print(f"[replay] frame {i}/{len(conn_world)}", flush=True)
        if ik_solver is not None:
            print(f"[replay] arm stays within {_jdrift:.1f} deg of home over the {len(conn_world)} "
                  f"insertion frames (home-relative grasp)")
        if args.dump:                                              # FULL episode: hold + approach + insertion
            ss, aa = dump_episode(args.dump, dump_eef, args.base_pos, args.base_yaw_deg,
                                  dump_grip, args.fps, phases=dump_phase)
            print(f"[dump] FULL episode: state{tuple(ss)} + action{tuple(aa)} + {len(dump_eef)} "
                  f"img/wrist frames (hold {dump_phase.count(0)} + approach {dump_phase.count(1)} "
                  f"+ insert {dump_phase.count(2)}) -> {args.dump}")
        if ik_err:
            print(f"[replay] IK reach error: mean {np.mean(ik_err) * 1000:.1f}mm  max {np.max(ik_err) * 1000:.1f}mm"
                  f"  ({'OK, arm reaches it' if np.max(ik_err) < 0.01 else 'OUT OF REACH -> move --jack-pos toward the arm'})")
        if pen is not None:
            hit = {n: v for n, v in pen.items() if v[0] > 0}
            struct = {n: v for n, v in hit.items() if not ("finger" in n or "wrist_3" in n)}
            if not hit:
                print(f"[diag] table penetration: NONE over {len(conn_world)} frames -- no link dips "
                      f"below the tabletop. clean.")
            else:
                print(f"[diag] table penetration (link origin below tbl_top={tbl_top:.3f} within footprint):")
                for n, (c, d) in sorted(hit.items(), key=lambda kv: -kv[1][1]):
                    tag = "TOOL (expected at surface)" if ("finger" in n or "wrist_3" in n) else "STRUCTURAL"
                    print(f"[diag]   {n:26s} {c:3d}/{len(conn_world)} frames  max {d * 1000:5.1f}mm  [{tag}]")
                print(f"[diag] verdict: {'STRUCTURAL links clip the table -> needs a fix' if struct else 'only TOOL links touch the surface -> fine (thats the insertion contact)'}")
        if args.dry_run:
            print("[replay] dry-run: IK checked, renderer skipped")
            if gl is not None:
                gl.close()
            return
        _t0 = _time.perf_counter()
        if not args.no_preview:
            _save(frames, args.out, smoke=False, fps=args.fps)
        prof["preview io (_save)"] = _time.perf_counter() - _t0
        nfr = max(len(frames), len(dump_eef), 1)
        ptot = sum(prof.values())
        print(f"[profile] {nfr} frames | " + " | ".join(
            f"{k}: {v:.1f}s ({v / nfr * 1000:.0f}ms/fr, {100 * v / max(ptot, 1e-9):.0f}%)"
            for k, v in prof.items()))
        print(f"[replay] done: {len(frames)} frame(s) -> {args.out}")
        if gl is not None:
            gl.close()
        return

    if args.smoke:
        _save([make_frame(state_0, 0.0)], args.out, smoke=True)
        print(f"[scene] smoke frame -> {args.out}/frame_0000.png")
        if gl is not None:
            gl.close()
        return

    viewer = open_rrd_recorder(args.rrd) if args.rrd else None
    if viewer is not None:
        viewer.set_model(model)

    frame_dt = 1.0 / args.fps
    sim_dt = frame_dt / SIM_SUBSTEPS
    num_frames = int(args.duration * args.fps)
    target_q = model.joint_target_q.numpy().copy()
    frames = []
    for frame in range(num_frames):
        t = frame * frame_dt
        for j, qj in zip(handles.arm_joints, arm_home):        # hold the chosen home (--arm-home-deg / SBOT_HOME)
            target_q[j] = qj
        if args.grip == "cycle":
            theta = gripper_theta(t, args.duration, still=args.still)
        else:
            theta = GRIPPER_THETA_CLOSED if args.grip == "closed" else GRIPPER_THETA_OPEN
        for idx, ratio in handles.gripper_coupling:
            target_q[idx] = ratio * theta
        control.joint_target_q.assign(target_q)
        for _ in range(SIM_SUBSTEPS):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
        frames.append(make_frame(state_0, t))
        if viewer is not None:
            viewer.begin_frame(t); viewer.log_state(state_0); viewer.end_frame()
        if frame % args.fps == 0:
            print(f"t={t:4.1f}s", flush=True)

    _save(frames, args.out, smoke=False, fps=args.fps)
    if viewer is not None:
        auto_blueprint(args.rrd.replace(".rrd", ".rbl"), model)
    if gl is not None:
        gl.close()
    print(f"[scene] done: {len(frames)} frame(s) -> {args.out}")


def _save(frames, out_dir, *, smoke, fps=30):
    from PIL import Image
    if smoke or len(frames) == 1:
        Image.fromarray(frames[0]).save(os.path.join(out_dir, "frame_0000.png"))
        return
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(out_dir, f"frame_{i:04d}.png"))
    path = os.path.join(out_dir, "sbot_scene.mp4")
    try:
        import imageio.v2 as imageio
        # libx264/yuv420p needs even H,W -> crop the odd last row/col; macro_block_size=1
        # keeps exact dims (no auto-pad to multiples of 16).
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1, pixelformat="yuv420p") as w:
            for f in frames:
                h, ww = f.shape[:2]
                w.append_data(np.ascontiguousarray(f[: h - h % 2, : ww - ww % 2]))
        print(f"[scene] wrote {path} ({len(frames)} frames @ {fps} fps)")
    except Exception as e:
        print(f"[scene] mp4 skipped: {e}")


if __name__ == "__main__":
    main()
