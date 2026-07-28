"""Render what the USD eye-in-hand camera sees, using ONLY Newton's headless GL viewer.

This is the no-GS-renderer path: no DalusSimCore container, no /dev/shm, no sudo.
It builds the RO1 in Newton, reads /sbot/wrist_3_link/Camera out of standardbot.usd,
places Newton's GL camera at that pose, and writes a PNG of the mesh view.

    .venv/bin/python tools/render_wrist_cam_newton.py                    # jaws open + closed pair
    .venv/bin/python tools/render_wrist_cam_newton.py --open 0.0         # single frame, jaws closed
    .venv/bin/python tools/render_wrist_cam_newton.py --usd path/to.usd  # check an edited USD

TWO LIMITATIONS of Newton's GL camera, both inherent to newton.viewer.Camera:

  * NO ROLL. The camera is parameterised by (pos, pitch, yaw) and derives its up
    vector from world +Z, so roll is always zero. The real D415 pose carries ~8 deg
    of roll, so this view is that much rotated versus the GS render / the real
    camera. Framing and coverage are right; the horizon is not.
  * PITCH IS CLAMPED to +-89 deg, so a straight-down tool axis cannot be expressed
    exactly. We warn when the requested pitch is clipped.

Use this to iterate on WHERE the camera sits and WHAT it covers. Use the GS path
(tools/check_wrist_cam.sh) when you need the photometric view the policy consumes.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import newton  # noqa: E402
import warp as wp  # noqa: E402

import record_sbot_scene_gs as R  # noqa: E402  (reuse load_usd_wrist_cam)
from newton_cabling.render.gs_bridge import (  # noqa: E402
    newton_pose, quat_mul_wxyz, quat_rotate_wxyz,
)
from newton_cabling.sim.safe_vbd import new_vbd_builder  # noqa: E402
from newton_cabling.sim.sbot import (  # noqa: E402
    SBOT_USD, add_sbot, gripper_theta_for_opening, set_arm_home, set_gripper,
)

# Same pose check_wrist_cam.py uses, so the two tools agree.
HOME_DEG = [4.0, -19.5, -113.0, 43.5, -268.9, -178.0]
BASE_POS = (0.426, -0.106, 0.87)

# D415 RGB intrinsics at 640x480 (configs/cameras.yaml, wrist == front).
FY_640x480 = 605.2867431640625


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


def build(usd, open_frac):
    """Finalized model + FK'd state for the arm at HOME with the jaws at open_frac."""
    b = new_vbd_builder(gravity=0.0)
    h = add_sbot(b, wp.transform(wp.vec3(*BASE_POS), wp.quat_identity()),
                 usd_path=usd, with_gripper=True)
    set_arm_home(b, h, tuple(np.radians(HOME_DEG)))
    set_gripper(b, h, gripper_theta_for_opening(open_frac))
    m = b.finalize()
    s = m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s)
    return m, s


def wrist_cam_pose(model, state, t_cam, q_local):
    """World (eye, forward) of the USD camera, given the FK'd wrist_3 pose."""
    w3 = [i for i, lbl in enumerate(model.body_label) if "wrist_3" in lbl][0]
    pos, quat = newton_pose(state.body_q.numpy(), w3)
    eye = np.asarray(pos) + quat_rotate_wxyz(quat, t_cam)
    # q_local is in the renderer's ROS-optical convention: +z is forward.
    fwd = quat_rotate_wxyz(quat_mul_wxyz(list(quat), q_local), [0.0, 0.0, 1.0])
    return eye, np.asarray(fwd, float)


def render(model, state, eye, fwd, vfov, width, height):
    """One headless-GL frame from (eye, fwd). Returns (H,W,3) uint8."""
    from newton.viewer import ViewerGL

    # Newton's Camera is Z-up: pitch = asin(z), yaw = atan2(y, x). Mirrors
    # record_sbot_scene_gs.eye_target_to_pitch_yaw.
    d = fwd / (np.linalg.norm(fwd) + 1e-9)
    pitch = math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0))))
    yaw = math.degrees(math.atan2(float(d[1]), float(d[0])))
    if abs(pitch) > 89.0:
        print(f"  WARNING: pitch {pitch:+.1f} deg exceeds Newton's +-89 clamp; view will be tilted up")

    gl = ViewerGL(width=width, height=height, headless=True, vsync=False)
    gl.set_model(model)
    gl.set_camera(wp.vec3(*[float(v) for v in eye]), pitch, yaw)
    gl.camera.fov = vfov
    gl.begin_frame(0.0)
    gl.log_state(state)
    gl.end_frame()
    frame = gl.get_frame().numpy()
    print(f"  eye={np.round(eye, 3)} pitch={pitch:+.1f} yaw={yaw:+.1f} vfov={vfov:.1f}")
    return (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame


def annotate(u8, text):
    """Yellow caption + centre crosshair, matching the old wrist_cam_*.png look."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(u8)
    d = ImageDraw.Draw(im)
    d.text((6, 6), text, fill=(255, 220, 0))
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
    args = ap.parse_args()

    from PIL import Image

    t_cam, q_local, hfov = R.load_usd_wrist_cam(args.usd)
    vfov = vfov_from_usd(args.usd, args.width, args.height)
    print(f"USD: {args.usd}")
    print(f"cam offset rel wrist_3 (m): {np.round(t_cam, 4)}  hfov={hfov:.1f} vfov={vfov:.1f} deg")

    fracs = [args.open] if args.open is not None else [1.0, 0.0]
    panels = []
    for f in fracs:
        label = "OPEN" if f > 0.5 else "CLOSED"
        print(f"[{label}] jaws={f:.2f}")
        model, state = build(args.usd, f)
        eye, fwd = wrist_cam_pose(model, state, t_cam, q_local)
        u8 = render(model, state, eye, fwd, vfov, args.width, args.height)
        panels.append(annotate(u8, f"RO1 wristcam  /sbot/wrist_3_link/Camera  jaws {label}  "
                                   f"vfov={vfov:.1f}deg  NEWTON MESH (no roll)"))

    out = panels[0] if len(panels) == 1 else np.concatenate(panels, axis=1)
    Image.fromarray(out).save(args.out)
    print(f"wrote {args.out}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
