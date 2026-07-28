# Full arm-driven cabling cycle, built on the newton_cabling package.
#
# A Franka FR3 runs the complete loop: approach, insert (latch clicks), release
# and retreat (plug held by the latch alone), return, re-grasp, press the tab,
# and pull out -- starting and ending at the same retracted pose so the recording
# loops seamlessly.
#
# The reusable pieces now come from newton_cabling/:
#   - timeline.proven_cycle_timeline(): the declarative 19s schedule (one ordered
#     list of phases) replacing the three hand-synced functions this script used
#     to carry. The phase names appear in the telemetry.
#   - sim.safe_vbd: the footgun-free VBD setup (collapse_fixed_joints, soft
#     actuated joints, eval_fk-into-model, correct color/finalize order).
#   - sim.grasp.GraspSpring: the proven force-coupling (anti-gravity + position
#     spring), driven by the timeline's 0..1 grasp weight.
#   - sim.recording: open_rrd_recorder (keeps the .rrd sink) and
#     write_focused_blueprint (auto-excludes the ground plane).
#
# What stays here: the connector/cable rig, the IK + arm-follows-plug control,
# and the closed-loop seat detection. Requires Newton git HEAD (.venv-head).

import dataclasses
import json
import pathlib
import sys

import newton
import numpy as np
import warp as wp

newton.use_coord_layout_targets = True

import newton.examples  # noqa: E402
import newton.ik as ik  # noqa: E402
import newton.utils  # noqa: E402
from newton.examples.contacts.example_contacts_rj45_plug import _sync_cable_anchors  # noqa: E402
from newton.solvers import SolverVBD  # noqa: E402

# Local reusable package: connector spec + scene builder, the hard-won VBD setup,
# grasp spring, recording helpers, the declarative cycle timeline, and the
# structured cycle report.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root
from newton_cabling.cable import route_cable_from_boot  # noqa: E402
from newton_cabling.connector import rj45_connector  # noqa: E402
from newton_cabling.report import CycleMeasurements, evaluate_cycle, outcome_to_dict  # noqa: E402
from newton_cabling.sim.grasp import GraspSpring  # noqa: E402
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: E402
from newton_cabling.sim.safe_vbd import (  # noqa: E402
    add_actuated_urdf,
    finalize_for_vbd,
    find_body_index,
    new_vbd_builder,
)
from newton_cabling.sim.scene import add_connector_rig, load_connector_meshes  # noqa: E402
from newton_cabling.timeline import proven_cycle_timeline  # noqa: E402

FPS = 60
DURATION_SECONDS = 19.0
SIM_SUBSTEPS = 8

Z_LIFT = np.array([0.0, 0.0, 0.35])
GRASP_BEHIND = 0.018  # TCP this far behind the plug center (grip the boot)
INSERT_TRAVEL_CAP = 0.035
SEAT_PLUG_Y = -0.0050  # trigger early; preload momentum lands at the true seat
START_DY = -0.050  # retracted pose: cycle starts AND ends here (seamless loop)
ARM_TARGET_KE = 5000.0
ARM_TARGET_KD = 5.0e-2  # was 1e-2; more damping against shake
FINGER_OPENING = 0.006  # closed on the boot
FINGER_OPEN = 0.025  # released
PRESS_ANGLE = -0.3
PRESS_TARGET_KE = 1.0

TCP_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi / 2.0)
PANDA_HOME = [-3.68e-03, 2.39e-02, 3.68e-03, -2.368, -1.29e-04, 2.392, 0.785]
PANDA_URDF = "urdf/fr3_franka_hand.urdf"


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rotate(q, v):
    qv = np.array([v[0], v[1], v[2], 0.0])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[:3]


def tf_mul(t1, t2):
    p1, q1 = t1
    p2, q2 = t2
    return (p1 + quat_rotate(q1, p2), quat_mul(q1, q2))


def tf_inv(t):
    p, q = t
    qi = quat_conj(q)
    return (-quat_rotate(qi, p), qi)


def add_panda(builder: newton.ModelBuilder, base_xform: wp.transform) -> None:
    # safe_vbd.add_actuated_urdf bakes in collapse_fixed_joints=True (without it
    # the arm does not move at all under VBD).
    add_actuated_urdf(
        builder, newton.utils.download_asset("franka_emika_panda") / PANDA_URDF, base_xform
    )


def main() -> None:
    connector = rj45_connector()
    meshes = load_connector_meshes(connector)
    sc = meshes.socket.base_position
    pc = meshes.plug.base_position
    lc = meshes.latch.base_position
    # Fewer kinematic segments so the cable flexes right at the boot (not a stub).
    cable_kinematic_count = 2

    plug_y_offset = np.array([0.0, -0.025, 0.0])
    socket_pos = sc + Z_LIFT
    plug_start = pc + plug_y_offset + Z_LIFT  # dy=0 reference
    latch_start = lc + plug_y_offset + Z_LIFT
    start_offset = np.array([0.0, START_DY, 0.0])
    plug_init = plug_start + start_offset  # retracted: initial AND final pose
    latch_init = latch_start + start_offset

    panda_base = wp.transform(wp.vec3(plug_start[0], plug_start[1] - 0.55, 0.0), wp.quat_identity())

    # TCP frame relative to link7, from an uncollapsed probe model.
    probe = newton.ModelBuilder()
    probe.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / PANDA_URDF,
        xform=panda_base,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        collapse_fixed_joints=False,
    )
    probe.joint_q[:9] = [*PANDA_HOME, FINGER_OPENING, FINGER_OPENING]
    probe_model = probe.finalize()
    probe_state = probe_model.state()
    newton.eval_fk(probe_model, probe_model.joint_q, probe_model.joint_qd, probe_state)
    probe_q = probe_state.body_q.numpy()
    l7_idx = find_body_index(probe_model.body_label, "/fr3_link7")
    tcp_idx = find_body_index(probe_model.body_label, "/fr3_hand_tcp")
    T_l7 = (probe_q[l7_idx][:3].copy(), probe_q[l7_idx][3:7].copy())
    T_tcp = (probe_q[tcp_idx][:3].copy(), probe_q[tcp_idx][3:7].copy())
    T_l7_tcp = tf_mul(tf_inv(T_l7), T_tcp)

    # TCP grips the boot, BEHIND the plug head.
    tcp_rot = np.array([TCP_ROT[0], TCP_ROT[1], TCP_ROT[2], TCP_ROT[3]])
    tcp_start = plug_start + np.array([0.0, -GRASP_BEHIND, 0.0])
    T_tcp_world = (tcp_start.copy(), tcp_rot)
    T_l7_world = tf_mul(T_tcp_world, tf_inv(T_l7_tcp))
    T_l7_plug = tf_mul(tf_inv(T_l7_world), (plug_start, np.array([0.0, 0.0, 0.0, 1.0])))

    # ---------------- IK model ----------------
    ik_builder = newton.ModelBuilder()
    add_panda(ik_builder, panda_base)
    l7_ik = find_body_index(ik_builder.body_label, "/fr3_link7")
    ik_model = ik_builder.finalize()

    l7_init = np.asarray(T_l7_world[0]) + start_offset
    pos_obj = ik.IKObjectivePosition(
        link_index=l7_ik,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([wp.vec3(*l7_init)], dtype=wp.vec3),
    )
    rot_obj = ik.IKObjectiveRotation(
        link_index=l7_ik,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([wp.vec4(*T_l7_world[1])], dtype=wp.vec4),
    )
    limit_obj = ik.IKObjectiveJointLimit(
        joint_limit_lower=ik_model.joint_limit_lower,
        joint_limit_upper=ik_model.joint_limit_upper,
    )
    seed = np.zeros(ik_model.joint_coord_count, dtype=np.float32)
    seed[:7] = PANDA_HOME
    joint_q_ik = wp.array(seed.reshape(1, -1), dtype=wp.float32)
    ik_solver = ik.IKSolver(
        model=ik_model,
        n_problems=1,
        objectives=[pos_obj, rot_obj, limit_obj],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    for _ in range(20):
        ik_solver.step(joint_q_ik, joint_q_ik, iterations=24)
    arm_q0 = joint_q_ik.numpy()[0].copy()
    arm_q0[7] = FINGER_OPENING
    arm_q0[8] = FINGER_OPENING

    # ---------------- sim model ----------------
    builder = new_vbd_builder(gravity=-9.81)
    builder.rigid_gap = 0.005

    add_panda(builder, panda_base)
    link7_body = find_body_index(builder.body_label, "/fr3_link7")
    panda_shape_count = len(builder.shape_body)
    for shape_idx in range(panda_shape_count):
        builder.shape_flags[shape_idx] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

    builder.joint_q[:9] = list(arm_q0[:9])
    builder.joint_target_q[:9] = list(arm_q0[:9])
    builder.joint_target_ke[:9] = [ARM_TARGET_KE] * 9
    builder.joint_target_kd[:9] = [ARM_TARGET_KD] * 9
    builder.joint_limit_ke[:9] = [0.0] * 9
    builder.joint_limit_kd[:9] = [1.0e-4] * 9

    # No ground plane in recordings: the large plane mesh obscures the view, and
    # the viewer instances shapes so its rendered index can't be matched to a
    # model index reliably (this is what caused recordings to open to "just the
    # floor"). Nothing here needs a floor -- the arm is position-controlled and
    # the cable is pinned at both ends.

    # The whole socket/plug/latch rig (world-anchored d6 plug, latch revolute,
    # SDF contact) is built from the connector spec.
    rig = add_connector_rig(
        builder,
        connector,
        meshes,
        socket_pos=socket_pos,
        plug_pos=plug_init,
        latch_pos=latch_init,
        plug_anchor_pos=plug_init,
    )
    plug_body = rig.plug_body
    latch_body = rig.latch_body
    connector_shapes = rig.connector_shapes

    # Custom cable routing instead of the USD asset's authored loop (which
    # coils right through the gripper): straight out of the boot between the
    # fingers, then swept to the +x side and down to a floor anchor, with
    # slack for the retreat. All points relative to the plug start pose.
    # Cable out the back (boot) of the plug, then swept to the +x side and down,
    # away from the arm -- with slack for the retreat. Routing from the boot
    # (route_cable_from_boot) keeps it from clipping through the plug body.
    arm_cable_drape = (
        (0.00, 0.000, 0.000),  # boot
        (0.02, -0.030, -0.020),  # out the back and to the side
        (0.08, -0.055, -0.090),
        (0.17, -0.065, -0.190),
        (0.27, -0.055, -0.290),  # tail, off to the side and down (slack for the retreat)
    )
    cable_points = [
        wp.vec3(*point) for point in route_cable_from_boot(plug_init, drape=arm_cable_drape)
    ]
    cable_quats = newton.utils.create_parallel_transport_cable_quaternions(cable_points)
    rod_bodies, _ = builder.add_rod(
        positions=cable_points,
        quaternions=cable_quats,
        radius=connector.cable_radius_meters,
        cfg=dataclasses.replace(
            builder.default_shape_cfg, ke=connector.contact.stiffness, kd=connector.contact.damping, mu=connector.cable_friction
        ),
        bend_stiffness=1.0e1,
        bend_damping=3.0e-1,  # settle the hang (no table to rest on)
        label="cable",
    )
    for body_idx in rod_bodies[:cable_kinematic_count]:
        for cable_shape in builder.body_shapes[body_idx]:
            for conn_shape in connector_shapes:
                builder.add_shape_collision_filter_pair(cable_shape, conn_shape)
    for idx in (*rod_bodies[:cable_kinematic_count], rod_bodies[-1]):
        builder.body_mass[idx] = 0.0
        builder.body_inv_mass[idx] = 0.0
        builder.body_inertia[idx] = wp.mat33(0.0)
        builder.body_inv_inertia[idx] = wp.mat33(0.0)

    anchor_body_ids = tuple(rod_bodies[:cable_kinematic_count])
    anchor_offsets = tuple(
        wp.vec3(
            cable_points[i][0] - plug_init[0],
            cable_points[i][1] - plug_init[1],
            cable_points[i][2] - plug_init[2],
        )
        for i in range(cable_kinematic_count)
    )
    anchor_rots = tuple(cable_quats[:cable_kinematic_count])

    # safe_vbd.finalize_for_vbd: colour -> finalize -> soften actuated joints ->
    # eval_fk into the model (the exact ordering VBD needs; see safe_vbd.py).
    model = finalize_for_vbd(builder)
    joint_types = model.joint_type.numpy()

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()

    solver = SolverVBD(
        model,
        iterations=12,
        rigid_contact_hard=False,
        rigid_body_contact_buffer_size=256,
    )

    viewer = open_rrd_recorder("panda_cycle.rrd")
    viewer.set_model(model)

    cable_anchor_indices = wp.array(anchor_body_ids, dtype=int, device=model.device)
    cable_anchor_offsets = wp.array(anchor_offsets, dtype=wp.vec3, device=model.device)
    cable_anchor_rotations = wp.array(anchor_rots, dtype=wp.quat, device=model.device)

    # Latch joint dof/coord indices (the revolute whose child is the latch).
    joint_children = model.joint_child.numpy()
    latch_joint = next(
        j
        for j in range(model.joint_count)
        if joint_types[j] == int(newton.JointType.REVOLUTE) and joint_children[j] == latch_body
    )
    latch_coord = int(model.joint_q_start.numpy()[latch_joint])
    latch_dof = int(model.joint_qd_start.numpy()[latch_joint])

    grasp_spring = GraspSpring(
        model, plug_body, latch_body, tuple(plug_init), stiffness=50.0, damping=10.0
    )
    target_q_host = control.joint_target_q.numpy().copy()
    target_ke_host = model.joint_target_ke.numpy().copy()
    latch_rest_ke = float(target_ke_host[latch_dof])

    frame_dt = 1.0 / FPS
    sim_dt = frame_dt / SIM_SUBSTEPS
    num_frames = int(DURATION_SECONDS * FPS)
    sim_time = 0.0
    checkpoints = {}
    seat_dy = None
    max_tracking_error = 0.0
    max_plug_pitch = 0.0
    # The declarative schedule: one ordered list of phases (plug offset +
    # grasp/latch state) sampled at any time, replacing the three hand-synced
    # functions this script used to carry. loop=True guarantees start == end.
    timeline = proven_cycle_timeline(
        start_offset_meters=START_DY, insert_cap_meters=INSERT_TRAVEL_CAP
    )

    def simulate_frame():
        nonlocal state_0, state_1
        for _ in range(SIM_SUBSTEPS):
            state_0.clear_forces()
            grasp_spring.apply(state_0)
            wp.launch(
                kernel=_sync_cable_anchors,
                dim=cable_kinematic_count,
                inputs=(
                    state_0.body_q,
                    state_0.body_qd,
                    plug_body,
                    cable_anchor_indices,
                    cable_anchor_offsets,
                    cable_anchor_rotations,
                ),
                device=model.device,
            )
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    graph = None
    if wp.get_device().is_cuda:
        with wp.ScopedCapture() as capture:
            simulate_frame()
        graph = capture.graph

    l7_target0 = np.asarray(T_l7_world[0], dtype=np.float64)
    rot_desired = np.asarray(T_l7_world[1], dtype=np.float64)
    T_l7_plug_p = np.asarray(T_l7_plug[0], dtype=np.float64)
    rot_obj.set_target_rotations(wp.array([wp.vec4(*rot_desired)], dtype=wp.vec4))

    for frame in range(num_frames):
        t = sim_time
        sample = timeline.sample(t)
        grasp_weight = sample.grasp_weight

        # Latch return-spring boost while the tab is actively pressed.
        desired_latch_ke = PRESS_TARGET_KE if sample.latch_weight > 0.01 else latch_rest_ke
        if float(target_ke_host[latch_dof]) != desired_latch_ke:
            target_ke_host[latch_dof] = desired_latch_ke
            model.joint_target_ke.assign(target_ke_host)

        body_q_now = state_0.body_q.numpy()

        # Closed-loop seat: stop commanding deeper once the plug reaches the seat.
        dy_cmd = sample.plug_offset_meters
        if seat_dy is not None and t < 8.5 and dy_cmd > seat_dy:
            dy_cmd = seat_dy

        # The grasp spring target IS the trajectory (exact socket axis) --
        # plug alignment does not depend on arm accuracy at all.
        plug_traj = plug_start + np.array([0.0, dy_cmd, 0.0])
        grasp_spring.set_target(tuple(plug_traj))
        grasp_spring.set_strength(grasp_weight)

        # Arm target: while grasping, follow the MEASURED plug so the hand stays
        # glued to the boot (no visible sliding as the spring loads); when
        # released, follow the trajectory.
        l7_from_traj = l7_target0 + np.array([0.0, dy_cmd, 0.0])
        l7_from_plug = body_q_now[plug_body][:3] - quat_rotate(rot_desired, T_l7_plug_p)
        l7_target = grasp_weight * l7_from_plug + (1.0 - grasp_weight) * l7_from_traj
        pos_obj.set_target_positions(wp.array([wp.vec3(*l7_target)], dtype=wp.vec3))
        ik_solver.step(joint_q_ik, joint_q_ik, iterations=24)
        ik_q = joint_q_ik.numpy()[0]
        target_q_host[:7] = ik_q[:7]
        finger_target = FINGER_OPEN - (FINGER_OPEN - FINGER_OPENING) * grasp_weight
        target_q_host[7] = finger_target
        target_q_host[8] = finger_target
        target_q_host[latch_coord] = PRESS_ANGLE * sample.latch_weight
        control.joint_target_q.assign(target_q_host)

        if graph:
            wp.capture_launch(graph)
        else:
            simulate_frame()
        sim_time += frame_dt

        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.end_frame()

        body_q = state_0.body_q.numpy()
        plug_y = float(body_q[plug_body][1])
        if seat_dy is None and plug_y >= SEAT_PLUG_Y and 4.5 <= t < 8.0 and dy_cmd > 0.01:
            seat_dy = dy_cmd + 0.0003
            print(f"t={t:5.2f}s SEATED at dy={dy_cmd:+.4f}", flush=True)
        # Per-frame metrics for the cycle report (cheap numpy).
        l7_err = float(np.linalg.norm(body_q[link7_body][:3] - l7_target))
        plug_pose = body_q[plug_body]
        plug_y_axis = quat_rotate(plug_pose[3:7], np.array([0.0, 1.0, 0.0]))
        plug_pitch_deg = float(np.degrees(np.arcsin(np.clip(plug_y_axis[2], -1.0, 1.0))))
        max_tracking_error = max(max_tracking_error, l7_err)
        max_plug_pitch = max(max_plug_pitch, abs(plug_pitch_deg))
        if frame % 30 == 0:
            q_rel = quat_mul(quat_conj(plug_pose[3:7]), body_q[latch_body][3:7])
            latch_angle = float(-2.0 * np.arctan2(q_rel[0], q_rel[3]))
            print(
                f"t={t:5.2f}s {sample.phase_name:<13} dy={dy_cmd:+.4f} plug_y={plug_y:+.4f} "
                f"pitch={plug_pitch_deg:+.2f}deg latch={latch_angle:+.3f} "
                f"grasp={grasp_weight:.2f} l7_err={l7_err * 1000:5.1f}mm",
                flush=True,
            )
        for name, t_check in (
            ("start", 3.9),
            ("seated", 8.4),
            ("after_retreat_released", 12.9),
            ("end", DURATION_SECONDS - 0.05),
        ):
            if name not in checkpoints and t >= t_check:
                checkpoints[name] = plug_y

    start_y = checkpoints["start"]
    seated_y = checkpoints["seated"]
    retreat_y = checkpoints["after_retreat_released"]
    end_y = checkpoints["end"]
    # Single source of truth for the verdict (report.evaluate_cycle).
    measurements = CycleMeasurements(
        plug_y_start_meters=start_y,
        plug_y_seated_meters=seated_y,
        plug_y_after_release_meters=retreat_y,
        plug_y_end_meters=end_y,
        max_tracking_error_meters=max_tracking_error,
        max_plug_pitch_degrees=max_plug_pitch,
    )
    outcome = evaluate_cycle(measurements)
    print(
        f"\nplug_y start={start_y:+.4f} seated={seated_y:+.4f} "
        f"retreat={retreat_y:+.4f} end={end_y:+.4f}"
    )
    print(f"outcome: {type(outcome).__name__}  metrics: {outcome.metrics}")
    print(json.dumps(outcome_to_dict(outcome)))
    print("recording complete: panda_cycle.rrd")

    blueprint_path = auto_blueprint("panda.rbl", model)
    print(f"wrote blueprint {blueprint_path}")


if __name__ == "__main__":
    main()
