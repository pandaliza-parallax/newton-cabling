"""Quick readout of the USD eye-in-hand wrist camera (/sbot/wrist_3_link/Camera) after you edit it.
Run:  .venv/bin/python tools/check_wrist_cam.py
Prints the camera's offset/FOV and, at the home pose, how squarely it looks at the gripper.
"""
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
from newton_cabling.sim.sbot import SBOT_USD, add_sbot, set_arm_home  # noqa: E402

USD = sys.argv[1] if len(sys.argv) > 1 else str(SBOT_USD)
HOME = [4.0, -19.5, -113.0, 43.5, -268.9, -178.0]
GRASP = np.array([0.291, -0.904, 0.848])   # ~grasp point at home (base 0.426 -0.106 0.87)

t_cam, q_local, fov = R.load_usd_wrist_cam(USD)
print(f"USD: {USD}")
print(f"cam offset rel wrist_3 (m): {np.round(t_cam, 4)}   FOV(h): {fov:.1f} deg")

b = new_vbd_builder(gravity=0.0)
h = add_sbot(b, wp.transform(wp.vec3(0.426, -0.106, 0.87), wp.quat_identity()), with_gripper=True)
set_arm_home(b, h, tuple(np.radians(HOME)))
m = b.finalize(); s = m.state(); newton.eval_fk(m, m.joint_q, m.joint_qd, s)
bq = s.body_q.numpy()
w3 = [i for i, l in enumerate(m.body_label) if "wrist_3" in l][0]
wp_, wq_ = newton_pose(bq, w3)
eye = np.asarray(wp_) + quat_rotate_wxyz(wq_, t_cam)
fwd = quat_rotate_wxyz(quat_mul_wxyz(list(wq_), q_local), [0.0, 0.0, 1.0])
tg = (GRASP - eye) / np.linalg.norm(GRASP - eye)
print(f"eye (world): {np.round(eye, 3)}   dist to grasp: {np.linalg.norm(GRASP - eye) * 100:.0f} cm")
print(f"dot(view, to_gripper) = {float(np.dot(fwd, tg)):.3f}   (1.0 = looking straight at the gripper)")
