"""Shared scene-render core for the RO1 + ethernet GS renderers.

Everything here was byte-identical across scripts/record_sbot_scene_gs.py (the rigid-plug
replay track) and scripts/record_sbot_scene_gs_cable.py (the Newton-simulated cable track),
which were copy-paste forks. Only their main() differs, so this module holds the
constants and helpers both drive.

Calibration constants live HERE and nowhere else -- that is the point. A fix to a
splat anchor, a camera intrinsic, or a host path now reaches every renderer.
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
    SBOT_USD,
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


def _K_from_intr(d: dict, out_w: int, out_h: int) -> np.ndarray:
    """3x3 pinhole K from a calibrated-intrinsics dict (fx, fy, cx, cy), SCALED from the
    calibration size (d['width']xd['height']) to the render size (out_w x out_h). Scaling
    is per-axis so it stays exact even if the aspect differs slightly; gsplat honors the
    resulting fx!=fy and off-center cx/cy. Rendering at out_w x out_h and resizing back to
    the calibration size (or 224) reproduces the calibrated rays."""
    sx, sy = out_w / d["width"], out_h / d["height"]
    return np.array([[d["fx"] * sx, 0.0, d["cx"] * sx],
                     [0.0, d["fy"] * sy, d["cy"] * sy],
                     [0.0, 0.0, 1.0]], np.float64)


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


def mat_to_quat_wxyz(R):
    """3x3 rotation matrix (columns = axes) -> scalar-first quaternion [w,x,y,z]."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return (q / np.linalg.norm(q)).tolist()


def load_usd_wrist_cam(usd_path, cam_prim="/sbot/wrist_3_link/Camera", link_prim="/sbot/wrist_3_link"):
    """Read the eye-in-hand Camera authored on wrist_3 in the RO1 USD. Returns its pose RELATIVE to
    wrist_3 as (t_cam[3] metres, q_local[4] wxyz in the renderer's ROS-optical frame, hfov_deg).
    add_sbot imports the USD at scale 1.0 (1 unit = 1 m, verified), so the raw translation is metres."""
    from pxr import Usd, UsdGeom, Gf  # noqa: PLC0415
    st = Usd.Stage.Open(str(usd_path))
    xc = UsdGeom.XformCache()
    Tc = Gf.Matrix4d(*np.array(xc.GetLocalToWorldTransform(st.GetPrimAtPath(cam_prim))).flatten().tolist())
    Tw = Gf.Matrix4d(*np.array(xc.GetLocalToWorldTransform(st.GetPrimAtPath(link_prim))).flatten().tolist())
    Trel = np.array(Tc * Tw.GetInverse()).reshape(4, 4)          # camera in wrist_3 frame (USD row-major)
    t_cam = Trel[3, :3].copy()                                   # translation (metres, scale=1)
    R_usd = Trel[:3, :3].T.copy()                                # columns = camera axes (OpenGL: -z fwd, +y up)
    R_ros = R_usd @ np.diag([1.0, -1.0, -1.0])                   # -> renderer ROS-optical (+z fwd, +y down)
    q_local = mat_to_quat_wxyz(R_ros)
    cam = UsdGeom.Camera(st.GetPrimAtPath(cam_prim))
    fl = cam.GetFocalLengthAttr().Get(); ha = cam.GetHorizontalApertureAttr().Get()
    hfov = math.degrees(2.0 * math.atan(ha / (2.0 * fl))) if (fl and ha) else 69.1
    return t_cam, q_local, hfov


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
