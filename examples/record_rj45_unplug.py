# Headless, scripted RJ45 unplug demo, recorded to a rerun .rrd file.
#
# Extends the insertion demo with latch actuation via the revolute joint's
# position target (control.joint_target_q), which the VBD solver's return
# spring drives toward. Timeline:
#   0.0-1.5s   settle (cable sags)
#   1.5-4.5s   insert until the latch clicks in
#   4.5-5.5s   hold
#   5.5-7.5s   CONTROL: pull WITHOUT pressing the latch -- must stay seated
#   7.5-8.5s   re-seat
#   8.5-9.5s   press the latch tab (drive joint target to its inward limit)
#   9.5-12.0s  pull while pressed -- plug should slide out
#   12.0-13.0s release the latch

import newton
import newton._src.viewer.viewer_rerun as viewer_rerun_module
import numpy as np
import warp as wp
from newton.examples.contacts.example_contacts_rj45_plug import Example
from newton.viewer import ViewerRerun

# Keep the .rrd file sink: skip ViewerRerun's server launch (it replaces the sink).
viewer_rerun_module.is_jupyter_notebook = lambda: True

FPS = 60
DURATION_SECONDS = 13.0
INSERT_DEPTH = 0.035  # meters past rest pose (plug starts 25mm out of socket)
PULL_DEPTH = -0.06
PRESS_ANGLE = -0.3  # joint target [rad]; travel limit clamps at -0.2
PRESS_TARGET_KE = 1.0  # stiffer spring while pressing so the tab stays down


def plug_target_offset_y(t: float) -> float:
    if t < 1.5:
        return 0.0
    if t < 4.5:
        return INSERT_DEPTH * (t - 1.5) / 3.0
    if t < 5.5:
        return INSERT_DEPTH
    if t < 7.5:  # pull without pressing (control test)
        return INSERT_DEPTH + (PULL_DEPTH - INSERT_DEPTH) * (t - 5.5) / 2.0
    if t < 8.5:  # re-seat
        return PULL_DEPTH + (INSERT_DEPTH - PULL_DEPTH) * (t - 7.5) / 1.0
    if t < 9.5:  # hold while pressing latch
        return INSERT_DEPTH
    if t < 12.0:  # pull while pressed
        return INSERT_DEPTH + (PULL_DEPTH - INSERT_DEPTH) * (t - 9.5) / 2.5
    return PULL_DEPTH


def latch_target_angle(t: float) -> float:
    if t < 8.5:
        return 0.0
    if t < 9.5:
        return PRESS_ANGLE * (t - 8.5) / 1.0
    if t < 12.0:
        return PRESS_ANGLE
    return 0.0


def latch_angle_from_bodies(body_q: np.ndarray, plug_idx: int, latch_idx: int) -> float:
    """Rotation of the latch relative to the plug about the hinge, for logging."""
    qp = body_q[plug_idx][3:7]  # (x, y, z, w)
    ql = body_q[latch_idx][3:7]
    # q_rel = conj(qp) * ql
    px, py, pz, pw = -qp[0], -qp[1], -qp[2], qp[3]
    lx, ly, lz, lw = ql
    rx = pw * lx + px * lw + py * lz - pz * ly
    rw = pw * lw - px * lx - py * ly - pz * lz
    # Hinge axis is -X in the plug frame.
    return float(-2.0 * np.arctan2(rx, rw))


viewer = ViewerRerun(
    record_to_rrd="rj45_unplug.rrd",
    keep_historical_data=True,
)
example = Example(viewer, args=None)

model = example.model

joint_types = model.joint_type.numpy()
revolute_joints = np.flatnonzero(joint_types == int(newton.JointType.REVOLUTE))
assert len(revolute_joints) == 1, f"expected exactly one revolute joint, got {len(revolute_joints)}"
latch_joint = int(revolute_joints[0])
latch_dof = int(model.joint_qd_start.numpy()[latch_joint])

# Newer Newton: coord-shaped control.joint_target_q; older releases: DOF-shaped
# control.joint_target_pos. Index accordingly.
joint_target_array = getattr(example.control, "joint_target_q", None)
if joint_target_array is not None:
    start = getattr(model, "joint_target_q_start", None) or model.joint_q_start
    latch_coord = int(start.numpy()[latch_joint])
else:
    joint_target_array = example.control.joint_target_pos
    latch_coord = latch_dof
assert joint_target_array is not None

target_q_host = joint_target_array.numpy().copy()
target_ke_host = model.joint_target_ke.numpy().copy()
rest_target_ke = float(target_ke_host[latch_dof])

rest = example._rest_pos
num_frames = int(DURATION_SECONDS * FPS)
checkpoints = {}

for frame in range(num_frames):
    t = example.sim_time
    target = wp.vec3(rest[0], rest[1] + plug_target_offset_y(t), rest[2])

    example._pick_body.assign([-1])
    example._pick_target.assign([target])
    example.gizmo_tf = wp.transform(target, wp.quat_identity())

    target_q_host[latch_coord] = latch_target_angle(t)
    joint_target_array.assign(target_q_host)

    # Stiffen the latch return spring only while pressing.
    desired_ke = PRESS_TARGET_KE if 8.5 <= t < 12.0 else rest_target_ke
    if float(target_ke_host[latch_dof]) != desired_ke:
        target_ke_host[latch_dof] = desired_ke
        model.joint_target_ke.assign(target_ke_host)

    if example.graph:
        wp.capture_launch(example.graph)
    else:
        example.simulate()
    example.sim_time += example.frame_dt
    example.render()

    body_q = example.state_0.body_q.numpy()
    plug_y = float(body_q[example._plug_body][1])
    latch_angle = latch_angle_from_bodies(body_q, example._plug_body, example._latch_body)

    if frame % 30 == 0:
        print(
            f"t={t:5.2f}s plug_dy_target={plug_target_offset_y(t):+.3f} "
            f"latch_target={latch_target_angle(t):+.2f} plug_y={plug_y:+.4f} latch={latch_angle:+.3f}",
            flush=True,
        )
    for name, t_check in (
        ("seated", 5.4),
        ("after_pull_no_press", 7.4),
        ("end", DURATION_SECONDS - 0.05),
    ):
        if name not in checkpoints and t >= t_check:
            checkpoints[name] = plug_y

seated_y = checkpoints["seated"]
held_y = checkpoints["after_pull_no_press"]
end_y = checkpoints["end"]
rest_y = float(rest[1])

latch_held_without_press = abs(held_y - seated_y) < 0.005
extracted_after_press = end_y <= rest_y + 0.002

print(f"\nseated plug_y          = {seated_y:+.4f}")
print(f"after pull w/o press   = {held_y:+.4f}  (latch_held={latch_held_without_press})")
print(
    f"after press-and-pull   = {end_y:+.4f}  (rest_y={rest_y:+.4f}, extracted={extracted_after_press})"
)
print("recording complete: rj45_unplug.rrd")
