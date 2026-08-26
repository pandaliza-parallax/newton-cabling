"""Render what the USD eye-in-hand camera sees, using ONLY Newton's headless GL viewer.

This is the no-GS-renderer path: no DalusSimCore container, no /dev/shm, no sudo.
It builds the RO1 in Newton, reads /sbot/wrist_3_link/Camera out of standardbot.usd,
places Newton's GL camera at that pose, and writes a PNG of the mesh view.

    .venv/bin/python tools/render_wrist_cam_newton.py                    # jaws open + closed pair
    .venv/bin/python tools/render_wrist_cam_newton.py --open 0.0         # single frame, jaws closed
    .venv/bin/python tools/render_wrist_cam_newton.py --usd path/to.usd  # check an edited USD
    # a recorded episode: arm posed to the logged wrist_3, cable + plug + jack in the scene
    .venv/bin/python tools/render_wrist_cam_newton.py \
        --traj /path/ep_0000 --frame 88 --grip-theta -0.0152 --out wrist_f88_grip.png

WHY THIS EXISTS BEYOND FRAMING CHECKS: the GS wrist render shows the cable seemingly
FLOATING between the jaws rather than pinched by them. The GS renderer poses the finger
splats at GRIPPER_THETA_CLOSED = 0.0 while the physics grips at THETA_CABLE = -0.0152.
Rendering the SAME frame from Newton at both angles decides whether the float is a
rendering artifact (jaws drawn too closed) or the actual physics. ``--clearance`` prints
the measured pad-to-cable-surface gap so the answer is numeric, not just visual.

TWO LIMITATIONS of Newton's stock GL camera, both from newton.viewer.Camera storing
only (pos, pitch, yaw):

  * NO ROLL. The up vector is derived from world +Z, so roll is always zero. The
    real D415 pose carries ~8 deg of roll, so this view is that much rotated versus
    the GS render / the real camera. Framing and coverage are right; the horizon not.
  * PITCH IS CLAMPED to +-89 deg, so a straight-down tool axis cannot be expressed
    exactly. We warn when the requested pitch is clipped.

Both are lifted by render(up=...): the GL renderer only ever consumes the camera's
get_front()/get_up(), so pinning the true USD basis on the instance renders the exact
6-DOF pose (this is what tools/compare_gs_newton_cams.py does). This CLI keeps the
stock camera -- it judges whether the pads touch the cable, a relative-geometry
question the camera roll does not affect.

Use this to iterate on WHERE the camera sits and WHAT it covers. Use the GS path
(tools/check_wrist_cam.sh) when you need the photometric view the policy consumes.

--- HOW A TRAJECTORY FRAME IS POSED -------------------------------------------------

The recorded episode (rl/gen_cable_traj.py layout) stores SEAT-RELATIVE poses as (T,7)
= [pos3, quat wxyz]: ``eef_traj`` (wrist_3), ``face_traj`` (the plug MATING FACE, which
is also the /World/Plug mesh origin), ``conn_traj`` (the gripped cable body). We render
IN THE SEAT FRAME -- the camera is derived from the same trajectory, so the wrist-cam
view is identical to the world-frame one up to a rigid transform. The seat->world map
used elsewhere in the pipeline (world = JACK_POS + Rz(180) @ p, JACK_POS = the GS
scene's jack) is a pure yaw + translation, so it leaves world +Z -- and therefore
Newton's up-vector convention and the no-roll artifact -- untouched. Choosing the seat
frame just skips a transform that cannot change the picture.

The arm is NOT solved by IK. ``add_sbot`` is called with a base transform chosen so the
FK'd wrist_3 lands exactly on ``eef_traj[frame]`` with the arm still at HOME joint
angles: base = T_target . inv(T_wrist3_at_home_in_base). The wrist, the gripper and the
camera are therefore bit-exact; only the arm links behind the camera sit somewhere
unphysical, and they are out of frame.

Scene geometry (all measured from rl/cable_env.py, which generated the episodes):

  * plug   = /World/Plug of newton_cabling/assets/scan_rj45.usd, mesh origin AT the
             mating face, body extending 40.2 mm back along -y. Posed at face_traj.
  * cable  = one capsule, radius 3.25 mm (6.5 mm dia), spanning face-78 mm .. face-9 mm
             along the face frame's +y insertion axis. That is the rear grip-anchor
             segment through the rod's front tip -- the stretch that passes between the
             jaws (the grip point is face-68 mm, i.e. conn_traj). The free cable behind
             the jaws (out to ~face-128 mm) droops and is NOT reconstructed here: only
             one rod body is logged (rods_traj is (T,1,3)).
  * jack   = /World/Socket, mouth at its mesh origin, cavity 16.1 mm deep along +y.
             Placed at meta.json's ``jack_pos_seatrel`` (= 12 mm behind the seated face,
             the SEAT_AIM_DY cavity-depth aim) with identity orientation.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import newton
import warp as wp

from newton_cabling.render.gs_bridge import (
    newton_pose,
    quat_mul_wxyz,
    quat_rotate_wxyz,
)
from newton_cabling.render.scene_gs_common import load_usd_wrist_cam
from newton_cabling.sim.safe_vbd import new_vbd_builder
from newton_cabling.sim.sbot import (
    FINGER_LINK_BODIES,
    SBOT_USD,
    add_sbot,
    gripper_theta_for_opening,
    set_arm_home,
    set_gripper,
)

# Same pose check_wrist_cam.py uses, so the two tools agree.
HOME_DEG = [4.0, -19.5, -113.0, 43.5, -268.9, -178.0]
BASE_POS = (0.426, -0.106, 0.87)

# D415 RGB intrinsics at 640x480 (configs/cameras.yaml, wrist == front).
FY_640x480 = 605.2867431640625

# Connector meshes (check meta.json connector_usd for what an episode was generated
# against: the servoD/roll3-era sets use cad_rj45.usd, the head55 era used scan_rj45.usd).
CONNECTOR_USD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "newton_cabling", "assets", "scan_rj45.usd",
)
CAD_CONNECTOR_USD = os.path.join(os.path.dirname(CONNECTOR_USD), "cad_rj45.usd")
PLUG_PRIM = "/World/Plug"
SOCKET_PRIM = "/World/Socket"

# Raw headA scan (plug head + boot + ~60 mm of curved cord, one fused mesh) and the
# scan->plug-frame registration build_scan_rj45.py derived when cropping /World/Plug
# out of it: rows 0-2 = plug axes in scan coords, row 3 = tip; p_plug = R @ (p_scan - tip).
_ETH_DIR = os.path.join(os.path.dirname(CONNECTOR_USD), "ethernet")
HEADA_MESH_USD = os.path.join(_ETH_DIR, "assets", "objects", "Ethernet_cable_headA", "mesh.usd")
HEADA_FRAME_R = os.path.join(_ETH_DIR, "headA_scan_frame_R.npy")
HEADA_PRIM = "/root/mesh/mesh"

# The gripped cable, in the plug FACE frame (+y = insertion). See the module docstring.
CABLE_RADIUS = 0.00325          # m -- ConnectorSpec.cable_radius_meters (6.5 mm dia)
CABLE_Y_BACK = -0.078           # m -- rear grip-anchor segment (grip point is at -0.068)
CABLE_Y_FRONT = -0.009          # m -- front tip of the rod, just behind the plug boot

# Driver angle the SIM grips the 6.5 mm cable at, versus the angle the GS renderer draws
# the finger splats at. This is the hypothesis under test.
THETA_CABLE = -0.0152           # rl/cable_env-style cable grip: ~6.35 mm measured pad gap
THETA_GS_CLOSED = 0.0           # GRIPPER_THETA_CLOSED, what the GS renderer poses: ~2.4 mm

IDENT_Q = [1.0, 0.0, 0.0, 0.0]


# ── tiny pose algebra (scalar-first quats, matching gs_bridge) ────────────────────
def _qconj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def _compose(a, b):
    """Pose a ∘ b: apply b in a's frame."""
    return (np.asarray(a[0], float) + quat_rotate_wxyz(a[1], b[0]),
            quat_mul_wxyz(list(a[1]), list(b[1])))


def _inv(a):
    qi = _qconj(a[1])
    return (-quat_rotate_wxyz(qi, a[0]), qi)


def _wp_xform(pose):
    p, q = pose
    return wp.transform(wp.vec3(*[float(v) for v in p]),
                        wp.quat(float(q[1]), float(q[2]), float(q[3]), float(q[0])))


def vfov_from_usd(usd_path, width, height):
    """Vertical FOV in deg for Newton's camera.

    Newton's Camera.fov is VERTICAL (camera.py builds the ray as
    front + right*u*alpha*aspect + up*v*alpha, alpha = tan(fov/2)), but
    load_usd_wrist_cam returns the HORIZONTAL FOV. Convert through the USD
    apertures when they are authored, else fall back to the D415 intrinsics.
    """
    from pxr import Usd, UsdGeom

    st = Usd.Stage.Open(str(usd_path))
    cam = UsdGeom.Camera(st.GetPrimAtPath("/sbot/wrist_3_link/Camera"))
    fl = cam.GetFocalLengthAttr().Get()
    va = cam.GetVerticalApertureAttr().Get()
    if fl and va:
        return math.degrees(2.0 * math.atan(va / (2.0 * fl)))
    return math.degrees(2.0 * math.atan((height / 2.0) / (FY_640x480 * height / 480.0)))


# ── trajectory ───────────────────────────────────────────────────────────────────
def load_frame(traj_dir, frame):
    """Seat-frame poses of wrist_3 / plug face / jack at ``frame`` of a recorded episode."""
    def pose(name):
        arr = np.load(os.path.join(traj_dir, name))
        if not -len(arr) <= frame < len(arr):
            raise SystemExit(f"--frame {frame} out of range for {name} ({len(arr)} frames)")
        row = np.asarray(arr[frame], float)
        return (row[0:3], row[3:7].tolist())

    with open(os.path.join(traj_dir, "meta.json")) as f:
        meta = json.load(f)
    return {
        "eef": pose("eef_traj.npy"),
        "face": pose("face_traj.npy"),
        "conn": pose("conn_traj.npy"),
        # The jack is world-fixed and axis-aligned with the seat, 12 mm behind the seated
        # face (SEAT_AIM_DY, the cavity-depth aim) -- meta records the offset directly.
        "jack": (np.asarray(meta.get("jack_pos_seatrel", [0.0, -0.012, 0.0]), float), IDENT_Q),
        "meta": meta,
    }


def _usd_mesh(prim_path, usd_path=CONNECTOR_USD):
    """Load a USD mesh for RENDERING only -- no SDF (this tool never runs contact)."""
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)  # keep the stage alive: the prim handle dies with it
    m = newton.usd.get_mesh(stage.GetPrimAtPath(prim_path), load_normals=True)
    normals = np.array(m.normals, np.float32) if m.normals is not None else None
    return newton.Mesh(np.array(m.vertices, np.float32), np.array(m.indices, np.int32),
                       normals=normals)


def _headA_mesh():
    """The raw headA scan mesh, re-expressed in the plug frame (face at origin, +y in)."""
    from pxr import Usd

    stage = Usd.Stage.Open(HEADA_MESH_USD)
    m = newton.usd.get_mesh(stage.GetPrimAtPath(HEADA_PRIM), load_normals=True)
    Rt = np.load(HEADA_FRAME_R)
    R, tip = Rt[:3], Rt[3]
    verts = ((np.array(m.vertices, np.float64) - tip) @ R.T).astype(np.float32)
    normals = (np.array(m.normals, np.float64) @ R.T).astype(np.float32) \
        if m.normals is not None else None
    return newton.Mesh(verts, np.array(m.indices, np.int32), normals=normals)


def add_scene_objects(builder, poses, plug_mesh="scan"):
    """Add plug mesh, jack mesh and the gripped-cable capsule as static (body -1) shapes.

    ``plug_mesh="headA"`` swaps the /World/Plug crop + capsule for the raw headA scan
    mesh, which carries the boot and ~60 mm of REAL curved cord -- more lifelike, but
    the cord's bend is frozen in the scanned pose, not the trajectory's actual droop.
    ``plug_mesh="cad"`` uses cad_rj45.usd's clean parametric plug + socket (what the
    servoD/roll3-era episodes actually simulate against).
    """
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False,
                                          has_particle_collision=False)
    usd = CAD_CONNECTOR_USD if plug_mesh == "cad" else CONNECTOR_USD
    builder.add_shape_mesh(-1, xform=_wp_xform(poses["jack"]),
                           mesh=_usd_mesh(SOCKET_PRIM, usd),
                           cfg=cfg, color=(0.15, 0.45, 0.20), label="jack")
    if plug_mesh == "cad":
        # bench fixture, identity on the jack pose — exactly how the env mounts it
        # (jack_fixture_rj45.usd is pre-baked into the socket frame)
        fixture_usd = os.path.join(os.path.dirname(CAD_CONNECTOR_USD), "jack_fixture_rj45.usd")
        if os.path.isfile(fixture_usd):
            builder.add_shape_mesh(-1, xform=_wp_xform(poses["jack"]),
                                   mesh=_usd_mesh("/World/Fixture", fixture_usd),
                                   cfg=cfg, color=(0.45, 0.45, 0.50), label="fixture")
    if plug_mesh == "headA":
        builder.add_shape_mesh(-1, xform=_wp_xform(poses["face"]), mesh=_headA_mesh(),
                               cfg=cfg, color=(0.75, 0.30, 0.30), label="plug")
        return
    builder.add_shape_mesh(-1, xform=_wp_xform(poses["face"]),
                           mesh=_usd_mesh(PLUG_PRIM, usd),
                           cfg=cfg, color=(0.20, 0.22, 0.26), label="plug")
    # Newton capsules extend along their local +Z; the cable runs along the face frame's
    # +y, so rotate -90 deg about x (z -> +y) and centre it on the span.
    half = 0.5 * (CABLE_Y_FRONT - CABLE_Y_BACK)
    rt2 = math.sqrt(0.5)
    local = (np.array([0.0, CABLE_Y_BACK + half, 0.0]), [rt2, -rt2, 0.0, 0.0])
    builder.add_shape_capsule(-1, xform=_wp_xform(_compose(poses["face"], local)),
                              radius=CABLE_RADIUS, half_height=half, cfg=cfg,
                              color=(0.95, 0.45, 0.10), label="cable")


# ── model build ──────────────────────────────────────────────────────────────────
def build(usd, theta, *, base_pose=None, poses=None, plug_mesh="scan"):
    """Finalized model + FK'd state: arm at HOME on ``base_pose``, jaws at ``theta``."""
    b = new_vbd_builder(gravity=0.0)
    if base_pose is None:
        base_pose = (np.asarray(BASE_POS, float), IDENT_Q)
    h = add_sbot(b, _wp_xform(base_pose), usd_path=usd, with_gripper=True)
    set_arm_home(b, h, tuple(np.radians(HOME_DEG)))
    set_gripper(b, h, theta)
    if poses is not None:
        add_scene_objects(b, poses, plug_mesh=plug_mesh)
    m = b.finalize()
    s = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s)
    return m, s


def _wrist_index(model):
    return next(i for i, lbl in enumerate(model.body_label) if "wrist_3" in lbl)


def base_for_wrist(usd, theta, target):
    """Base transform that puts the HOME-pose wrist_3 exactly on ``target``.

    No IK: the arm stays at HOME and the whole robot is rigidly relocated, so the wrist,
    the gripper linkage and the eye-in-hand camera are bit-exact for the recorded frame.
    Only the arm links -- all behind the camera -- end up somewhere unphysical.
    """
    home_base = (np.asarray(BASE_POS, float), IDENT_Q)
    model, state = build(usd, theta)
    w3 = newton_pose(state.body_q.numpy(), _wrist_index(model))
    base = _compose(target, _inv(_compose(_inv(home_base), w3)))
    print(f"  base relocated to {np.round(base[0], 4)} (arm still at HOME; wrist_3 exact)")
    return base


def wrist_cam_pose(model, state, t_cam, q_local):
    """World (eye, forward) of the USD camera, given the FK'd wrist_3 pose."""
    pos, quat = newton_pose(state.body_q.numpy(), _wrist_index(model))
    eye = np.asarray(pos) + quat_rotate_wxyz(quat, t_cam)
    # q_local is in the renderer's ROS-optical convention: +z is forward.
    fwd = quat_rotate_wxyz(quat_mul_wxyz(list(quat), q_local), [0.0, 0.0, 1.0])
    return eye, np.asarray(fwd, float)


# ── the numeric answer: how far are the pads from the cable surface? ──────────────
def _seg_distance(points, a, b):
    """Min distance from each row of ``points`` to the segment ab."""
    ab = b - a
    t = np.clip(((points - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def pad_clearance(model, state, poses):
    """Per-finger-link gap from the cable SURFACE (negative = the mesh overlaps the cable).

    Every vertex of every finger-link mesh is pushed to world through
    body_q ∘ shape_transform, then measured against the cable capsule's axis segment.
    This is the closest approach of any gripper geometry to the cable, so a value <= 0
    means the jaws are physically on the cable at this driver angle.
    """
    face = poses["face"]
    a = _compose(face, (np.array([0.0, CABLE_Y_BACK, 0.0]), IDENT_Q))[0]
    b = _compose(face, (np.array([0.0, CABLE_Y_FRONT, 0.0]), IDENT_Q))[0]
    bq = state.body_q.numpy()
    st = model.shape_transform.numpy()
    sb = model.shape_body.numpy()
    sc = model.shape_scale.numpy()
    out = {}
    for s, mesh in enumerate(model.shape_source):
        body = int(sb[s])
        if mesh is None or body < 0:
            continue
        label = model.body_label[body]
        if not any(label.endswith(n) for n in FINGER_LINK_BODIES):
            continue
        local = (st[s][0:3], [st[s][6], st[s][3], st[s][4], st[s][5]])
        world = _compose(newton_pose(bq, body), local)
        verts = np.asarray(mesh.vertices, float) * np.asarray(sc[s], float)
        pts = world[0] + quat_rotate_wxyz(world[1], verts)  # broadcasts over the last axis
        gap = float(_seg_distance(pts, a, b).min()) - CABLE_RADIUS
        out[label.rsplit("/", 1)[-1]] = gap
    return out


# ── rendering ────────────────────────────────────────────────────────────────────
# One cached viewer per output size, reused across calls: creating a ViewerGL per call
# leaks its GL context (dies with 0x502 ~12 frames in), while gl.close() after the first
# render tears down pyglet's shared GL state and every later context renders black.
_GL_VIEWERS = {}


def render(model, state, eye, fwd, vfov, width, height, zoom=1.0, up=None):
    """One headless-GL frame from (eye, fwd). Returns (H,W,3) uint8.

    ``up=None`` uses Newton's stock no-roll camera (up = world +Z projected off the
    view axis, pitch clamped to +-89). Passing the TRUE camera up vector renders the
    exact 6-DOF pose instead: the Camera only stores pitch/yaw, but the GL renderer
    consumes get_front()/get_up() alone (view matrix, rays, shadow light), so pinning
    those two on the instance restores the missing roll DOF and lifts the pitch clamp.
    """
    from newton.viewer import ViewerGL

    # Newton's Camera is Z-up: pitch = asin(z), yaw = atan2(y, x). Mirrors
    # record_sbot_scene_gs.eye_target_to_pitch_yaw.
    d = fwd / (np.linalg.norm(fwd) + 1e-9)
    pitch = math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0))))
    yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
    if abs(pitch) > 89.0 and up is None:
        print(f"  WARNING: pitch {pitch:+.1f} deg exceeds Newton's +-89 clamp; "
              f"view will be tilted up")

    gl = _GL_VIEWERS.get((width, height))
    if gl is None:
        gl = ViewerGL(width=width, height=height, headless=True, vsync=False)
        _GL_VIEWERS[(width, height)] = gl
    if gl.model is not model:
        gl.set_model(model)  # ViewerBase.set_model clears the previous model first
    gl.camera.__dict__.pop("get_front", None)  # drop a pinned basis from an earlier call
    gl.camera.__dict__.pop("get_up", None)
    gl.set_camera(wp.vec3(*[float(v) for v in eye]), pitch, yaw)
    if up is not None:
        from pyglet.math import Vec3 as PyVec3

        u = np.asarray(up, float)
        u = u / (np.linalg.norm(u) + 1e-9)
        front_v, up_v = PyVec3(*[float(v) for v in d]), PyVec3(*[float(v) for v in u])
        gl.camera.get_front = lambda: front_v
        gl.camera.get_up = lambda: up_v
    # Newton's camera is centred on its optical axis, so shrinking the FOV by `zoom` is
    # exactly a centre crop of the true wrist view -- no re-aiming, no parallax change.
    gl.camera.fov = math.degrees(2.0 * math.atan(math.tan(math.radians(vfov) / 2.0) / zoom))
    gl.begin_frame(0.0)
    gl.log_state(state)
    gl.end_frame()
    frame = gl.get_frame().numpy()
    print(f"  eye={np.round(eye, 3)} pitch={pitch:+.1f} yaw={yaw:+.1f} vfov={vfov:.1f}")
    return (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame


def annotate(u8, lines):
    """Yellow caption + centre crosshair, matching the old wrist_cam_*.png look."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(u8)
    d = ImageDraw.Draw(im)
    for i, text in enumerate([lines] if isinstance(lines, str) else lines):
        d.text((6, 6 + 12 * i), text, fill=(255, 220, 0))
    w, h = im.size
    d.line([(w // 2 - 9, h // 2), (w // 2 + 9, h // 2)], fill=(255, 220, 0))
    d.line([(w // 2, h // 2 - 9), (w // 2, h // 2 + 9)], fill=(255, 220, 0))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", default=str(SBOT_USD), help="USD to read the Camera from")
    ap.add_argument("--out", default="wrist_cam_newton.png", help="output PNG")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--open", type=float, default=None,
                    help="jaw opening 0=closed 1=open; omit to render an open|closed pair")
    ap.add_argument("--grip-theta", type=float, default=None,
                    help=f"AG-145 driver angle directly (clearer than --open, which maps "
                         f"theta = -0.6*frac). SIM cable grip = {THETA_CABLE}, "
                         f"GS renderer draws {THETA_GS_CLOSED}")
    ap.add_argument("--traj", default=None,
                    help="recorded episode dir (eef_traj.npy/face_traj.npy/meta.json); "
                         "poses the arm to that frame and adds cable + plug + jack")
    ap.add_argument("--frame", type=int, default=0, help="frame index within --traj")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="digital centre-crop zoom about the optical axis (1 = true wrist FOV)")
    ap.add_argument("--clearance", action="store_true",
                    help="print the measured finger-pad gap from the cable surface")
    args = ap.parse_args()

    from PIL import Image

    t_cam, q_local, hfov = load_usd_wrist_cam(args.usd)
    vfov = vfov_from_usd(args.usd, args.width, args.height)
    print(f"USD: {args.usd}")
    print(f"cam offset rel wrist_3 (m): {np.round(t_cam, 4)}  hfov={hfov:.1f} vfov={vfov:.1f} deg")

    poses = None
    if args.traj:
        poses = load_frame(args.traj, args.frame)
        print(f"traj: {args.traj} frame {args.frame}  "
              f"eef={np.round(poses['eef'][0], 4)} face={np.round(poses['face'][0], 4)} "
              f"(seat frame, {poses['meta'].get('frames', '?')} frames)")

    if args.grip_theta is not None:
        thetas = [args.grip_theta]
    elif args.open is not None:
        thetas = [gripper_theta_for_opening(args.open)]
    else:
        thetas = [gripper_theta_for_opening(1.0), gripper_theta_for_opening(0.0)]

    panels = []
    for theta in thetas:
        print(f"[theta={theta:+.4f}]  (== --open {theta / -0.6:.4f})")
        base_pose = base_for_wrist(args.usd, theta, poses["eef"]) if poses else None
        model, state = build(args.usd, theta, base_pose=base_pose, poses=poses)
        caption = [f"RO1 wristcam  NEWTON MESH (no roll)  vfov={vfov:.1f}deg",
                   f"grip theta={theta:+.4f}"]
        if poses:
            caption[1] += f"   {os.path.basename(args.traj.rstrip('/'))} frame {args.frame}"
            if args.clearance:
                gaps = pad_clearance(model, state, poses)
                for name, gap in sorted(gaps.items()):
                    print(f"    {name:42s} gap to cable surface {gap * 1000:+7.2f} mm")
                lo = min(gaps.values())
                caption.append(f"min pad->cable-surface gap {lo * 1000:+.2f} mm "
                               f"({'CONTACT' if lo <= 0 else 'CLEAR'})")
        eye, fwd = wrist_cam_pose(model, state, t_cam, q_local)
        if args.zoom != 1.0:
            caption[0] += f"  zoom x{args.zoom:g}"
        u8 = render(model, state, eye, fwd, vfov, args.width, args.height, zoom=args.zoom)
        panels.append(annotate(u8, caption))

    out = panels[0] if len(panels) == 1 else np.concatenate(panels, axis=1)
    Image.fromarray(out).save(args.out)
    print(f"wrote {args.out}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
