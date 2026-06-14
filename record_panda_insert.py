# Stage 1 of arm-driven cabling: a Franka FR3 inserts the RJ45 plug.
#
# Architecture notes (hard-won, see README):
#   - VBD's rigid-joint handling silently freezes chains containing FIXED
#     joints; import the URDF with collapse_fixed_joints=True.
#   - Actuated joints must be SOFT under VBD (vbd:joint_is_hard=0); hard mode
#     locks all relative motion. Set post-finalize via model.vbd.joint_is_hard.
#   - VBD measures joint angles against model.body_q; call
#     eval_fk(model, ..., model) before constructing the solver.
#   - The plug mesh is attached directly to the link7 body (true weld, no
#     joint), so insertion reaction forces act through the arm. The latch is a
#     revolute joint parented to link7. Panda's own URDF shapes are
#     non-colliding in stage 1; plug/latch/socket/cable contact is real.
#
# Timeline:
#   0.0-2.0s   settle at start pose (plug 25mm out of the socket)
#   2.0-6.0s   advance hand +Y, plug inserts, latch clicks in
#   6.0-7.5s   hold
#   7.5-10.0s  retract hand target 45mm WITHOUT pressing the latch --
#              the plug must stay seated, the arm strains against it
#   10.0-12.0s return target to seated pose (relax)

import dataclasses

import newton
import numpy as np
import warp as wp

newton.use_coord_layout_targets = True  # before any model building

import newton._src.viewer.viewer_rerun as viewer_rerun_module  # noqa: E402
import newton.examples  # noqa: E402
import newton.ik as ik  # noqa: E402
import newton.utils  # noqa: E402
from newton.examples.contacts.example_contacts_rj45_plug import (  # noqa: E402
    CABLE_KINEMATIC_COUNT,
    CABLE_MU,
    CABLE_RADIUS,
    CONTACT_KD,
    CONTACT_KE,
    LATCH_LIMIT_KD,
    LATCH_LIMIT_LOWER,
    LATCH_LIMIT_UPPER,
    LATCH_SPRING_KD,
    LATCH_SPRING_KE,
    SHAPE_CFG,
    _load_cable_centerline,
    _load_mesh,
    _sync_cable_anchors,
)
from newton.solvers import SolverVBD  # noqa: E402
from newton.viewer import ViewerRerun  # noqa: E402
from pxr import Usd  # noqa: E402

viewer_rerun_module.is_jupyter_notebook = lambda: True  # keep the .rrd file sink

FPS = 60
DURATION_SECONDS = 12.0
SIM_SUBSTEPS = 6

Z_LIFT = np.array([0.0, 0.0, 0.35])
INSERT_TRAVEL = 0.035  # cap; the advance is closed-loop and freezes at the seat
SEAT_PLUG_Y = -0.0036  # proven seat depth from the spring-driven demos
RETRACT_TRAVEL = -0.045
ARM_TARGET_KE = 5000.0
ARM_TARGET_KD = 1.0e-2
FINGER_OPENING = 0.006

# TCP +Z (toward/through fingertips) aligned with world +Y (insertion axis).
TCP_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi / 2.0)
PANDA_HOME = [-3.68e-03, 2.39e-02, 3.68e-03, -2.368, -1.29e-04, 2.392, 0.785]

PANDA_URDF = "urdf/fr3_franka_hand.urdf"


# --- tiny host-side quaternion/transform helpers (x, y, z, w convention) ---
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


def quat_fraction(q, fraction):
    """A rotation of `fraction` of q's angle about q's axis."""
    q = q / np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    angle = 2.0 * np.arccos(np.clip(q[3], -1.0, 1.0))
    axis = q[:3]
    norm = np.linalg.norm(axis)
    if norm < 1.0e-12 or angle < 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis / norm
    half = 0.5 * angle * fraction
    return np.array([*(np.sin(half) * axis), np.cos(half)])


def hand_target_offset_y(t: float) -> float:
    if t < 2.0:
        return 0.0
    if t < 6.0:
        return INSERT_TRAVEL * (t - 2.0) / 4.0
    if t < 7.5:
        return INSERT_TRAVEL
    if t < 10.0:
        return INSERT_TRAVEL + (RETRACT_TRAVEL - INSERT_TRAVEL) * (t - 7.5) / 2.5
    return RETRACT_TRAVEL + (INSERT_TRAVEL - RETRACT_TRAVEL) * (t - 10.0) / 2.0


def add_panda(builder: newton.ModelBuilder, base_xform: wp.transform) -> None:
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / PANDA_URDF,
        xform=base_xform,
        enable_self_collisions=False,
        parse_visuals_as_colliders=True,
        collapse_fixed_joints=True,
    )


def find_body(labels, name: str) -> int:
    return next(i for i, lbl in enumerate(labels) if lbl.endswith(name))


def main() -> None:
    usd_path = newton.examples.get_asset("rj45_plug.usd")
    stage = Usd.Stage.Open(usd_path)

    socket_mesh, sc = _load_mesh(stage, "/World/Socket")
    plug_mesh, pc = _load_mesh(stage, "/World/Plug")
    latch_mesh, lc = _load_mesh(stage, "/World/Latch")

    plug_y_offset = np.array([0.0, -0.025, 0.0])  # PLUG_Y_OFFSET
    socket_pos = np.array([sc[0], sc[1], sc[2]]) + Z_LIFT
    plug_start = np.array([pc[0], pc[1], pc[2]]) + plug_y_offset + Z_LIFT
    latch_start = np.array([lc[0], lc[1], lc[2]]) + plug_y_offset + Z_LIFT

    panda_base = wp.transform(wp.vec3(plug_start[0], plug_start[1] - 0.55, 0.0), wp.quat_identity())

    # --- TCP frame relative to link7, from an UNcollapsed throwaway model ---
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
    l7_idx = find_body(probe_model.body_label, "/fr3_link7")
    tcp_idx = find_body(probe_model.body_label, "/fr3_hand_tcp")
    T_l7 = (probe_q[l7_idx][:3].copy(), probe_q[l7_idx][3:7].copy())
    T_tcp = (probe_q[tcp_idx][:3].copy(), probe_q[tcp_idx][3:7].copy())
    T_l7_tcp = tf_mul(tf_inv(T_l7), T_tcp)

    # Desired world TCP at start: at the plug center, oriented TCP_ROT.
    tcp_rot = np.array([TCP_ROT[0], TCP_ROT[1], TCP_ROT[2], TCP_ROT[3]])
    T_tcp_world = (plug_start.copy(), tcp_rot)
    T_l7_world = tf_mul(T_tcp_world, tf_inv(T_l7_tcp))
    T_l7_plug = tf_mul(tf_inv(T_l7_world), (plug_start, np.array([0.0, 0.0, 0.0, 1.0])))

    # ---------------- IK model: collapsed panda only ----------------
    ik_builder = newton.ModelBuilder()
    add_panda(ik_builder, panda_base)
    l7_ik = find_body(ik_builder.body_label, "/fr3_link7")
    ik_model = ik_builder.finalize()

    pos_obj = ik.IKObjectivePosition(
        link_index=l7_ik,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([wp.vec3(*T_l7_world[0])], dtype=wp.vec3),
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

    ik_state = ik_model.state()
    ik_model.joint_q.assign(arm_q0)
    newton.eval_fk(ik_model, ik_model.joint_q, ik_model.joint_qd, ik_state)
    l7_fk = ik_state.body_q.numpy()[l7_ik][:3]
    residual = np.linalg.norm(l7_fk - T_l7_world[0])
    print(f"IK start-pose residual: {residual * 1000:.1f}mm", flush=True)

    # ---------------- sim model ----------------
    builder = newton.ModelBuilder(gravity=-9.81)
    SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
    builder.rigid_gap = 0.005

    add_panda(builder, panda_base)
    link7_body = find_body(builder.body_label, "/fr3_link7")
    panda_shape_count = len(builder.shape_body)

    # Stage 1: URDF shapes are visual-only; the plug shape added below collides.
    for shape_idx in range(panda_shape_count):
        builder.shape_flags[shape_idx] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

    builder.joint_q[:9] = list(arm_q0[:9])
    builder.joint_target_q[:9] = list(arm_q0[:9])
    builder.joint_target_ke[:9] = [ARM_TARGET_KE] * 9
    builder.joint_target_kd[:9] = [ARM_TARGET_KD] * 9
    builder.joint_limit_ke[:9] = [0.0] * 9  # away from limits for this task
    builder.joint_limit_kd[:9] = [1.0e-4] * 9

    builder.add_ground_plane()

    socket_shape = builder.add_shape_mesh(
        -1,
        mesh=socket_mesh,
        xform=wp.transform(wp.vec3(*socket_pos), wp.quat_identity()),
        cfg=SHAPE_CFG,
        label="socket",
    )

    # True weld: the plug mesh is a shape on link7 itself.
    plug_shape = builder.add_shape_mesh(
        link7_body,
        mesh=plug_mesh,
        xform=wp.transform(wp.vec3(*T_l7_plug[0]), wp.quat(*T_l7_plug[1])),
        cfg=SHAPE_CFG,
        label="plug",
    )

    latch_body = builder.add_link(
        xform=wp.transform(wp.vec3(*latch_start), wp.quat_identity()), label="latch"
    )
    latch_shape = builder.add_shape_mesh(latch_body, mesh=latch_mesh, cfg=SHAPE_CFG)
    connector_shapes = (socket_shape, plug_shape, latch_shape)

    # Latch hinge, expressed in the link7 frame via the plug frame.
    latch_in_plug = (
        np.array([lc[0] - pc[0], lc[1] - pc[1], lc[2] - pc[2]]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    T_l7_latch = tf_mul(T_l7_plug, latch_in_plug)
    rev_joint = builder.add_joint_revolute(
        parent=link7_body,
        child=latch_body,
        axis=(1.0, 0.0, 0.0),
        parent_xform=wp.transform(wp.vec3(*T_l7_latch[0]), wp.quat(*T_l7_latch[1])),
        child_xform=wp.transform_identity(),
        target_ke=LATCH_SPRING_KE,
        target_kd=LATCH_SPRING_KD,
        limit_lower=LATCH_LIMIT_LOWER,
        limit_upper=LATCH_LIMIT_UPPER,
        limit_kd=LATCH_LIMIT_KD,
        collision_filter_parent=True,
        custom_attributes={"vbd:joint_is_hard": 0},
    )
    builder.add_articulation([rev_joint])

    cable_points = [wp.vec3(p[0], p[1], p[2] + Z_LIFT[2]) for p in _load_cable_centerline(stage)]
    cable_quats = newton.utils.create_parallel_transport_cable_quaternions(cable_points)
    rod_bodies, _ = builder.add_rod(
        positions=cable_points,
        quaternions=cable_quats,
        radius=CABLE_RADIUS,
        cfg=dataclasses.replace(
            builder.default_shape_cfg, ke=CONTACT_KE, kd=CONTACT_KD, mu=CABLE_MU
        ),
        bend_stiffness=1.0e1,
        bend_damping=1.0e-1,
        label="cable",
    )

    for body_idx in rod_bodies[:CABLE_KINEMATIC_COUNT]:
        for cable_shape in builder.body_shapes[body_idx]:
            for conn_shape in connector_shapes:
                builder.add_shape_collision_filter_pair(cable_shape, conn_shape)

    for idx in (*rod_bodies[:CABLE_KINEMATIC_COUNT], rod_bodies[-1]):
        builder.body_mass[idx] = 0.0
        builder.body_inv_mass[idx] = 0.0
        builder.body_inertia[idx] = wp.mat33(0.0)
        builder.body_inv_inertia[idx] = wp.mat33(0.0)

    # Cable anchors follow the plug; the plug frame is link7 ∘ T_l7_plug, so
    # express the anchor offsets/rotations in the link7 frame.
    anchor_body_ids = tuple(rod_bodies[:CABLE_KINEMATIC_COUNT])
    anchor_offsets = []
    anchor_rots = []
    for i in range(CABLE_KINEMATIC_COUNT):
        p_plug = np.array(
            [
                cable_points[i][0] - plug_start[0],
                cable_points[i][1] - plug_start[1],
                cable_points[i][2] - plug_start[2],
            ]
        )
        q_plug = np.array(
            [cable_quats[i][0], cable_quats[i][1], cable_quats[i][2], cable_quats[i][3]]
        )
        p_l7, q_l7 = tf_mul(T_l7_plug, (p_plug, q_plug))
        anchor_offsets.append(wp.vec3(*p_l7))
        anchor_rots.append(wp.quat(*q_l7))

    builder.color()
    model = builder.finalize()

    # Soft mode for all actuated joints; VBD hard joints lock all motion.
    joint_is_hard = model.vbd.joint_is_hard.numpy()
    joint_types = model.joint_type.numpy()
    for j in range(model.joint_count):
        if joint_types[j] in (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)):
            joint_is_hard[j] = 0
    model.vbd.joint_is_hard.assign(joint_is_hard)

    # Bake start config into the rest pose BEFORE constructing the solver.
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()

    solver = SolverVBD(
        model,
        iterations=12,
        rigid_contact_hard=False,
        rigid_body_contact_buffer_size=256,
    )

    viewer = ViewerRerun(record_to_rrd="panda_insert.rrd", keep_historical_data=True)
    viewer.set_model(model)

    cable_anchor_indices = wp.array(anchor_body_ids, dtype=int, device=model.device)
    cable_anchor_offsets = wp.array(anchor_offsets, dtype=wp.vec3, device=model.device)
    cable_anchor_rotations = wp.array(anchor_rots, dtype=wp.quat, device=model.device)

    target_q_host = control.joint_target_q.numpy().copy()
    frame_dt = 1.0 / FPS
    sim_dt = frame_dt / SIM_SUBSTEPS
    num_frames = int(DURATION_SECONDS * FPS)
    sim_time = 0.0
    checkpoints = {}
    T_l7_plug_p = np.asarray(T_l7_plug[0], dtype=np.float64)
    T_l7_plug_q = np.asarray(T_l7_plug[1], dtype=np.float64)
    seat_dy = None  # dy at which the plug reached the seat (closed-loop freeze)

    def simulate_frame():
        nonlocal state_0, state_1
        for _ in range(SIM_SUBSTEPS):
            state_0.clear_forces()
            wp.launch(
                kernel=_sync_cable_anchors,
                dim=CABLE_KINEMATIC_COUNT,
                inputs=(
                    state_0.body_q,
                    state_0.body_qd,
                    link7_body,
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
    # Steady-state offset (gravity sag + IK residual) is pose-dependent, so a
    # one-shot correction leaves several mm. Integrate it away during the
    # settle phase, then freeze the correction for the rest of the run.
    tracking_correction = np.zeros(3)
    rot_correction = np.array([0.0, 0.0, 0.0, 1.0])
    rot_desired = np.asarray(T_l7_world[1], dtype=np.float64)
    correction_frozen = False
    for frame in range(num_frames):
        t = sim_time
        if 0.5 <= t < 1.95:
            # Settle phase: calibrate the full position + rotation offset.
            l7_pose_now = state_0.body_q.numpy()[link7_body]
            tracking_correction += 0.08 * (l7_target0 - l7_pose_now[:3])
            q_err = quat_mul(rot_desired, quat_conj(l7_pose_now[3:7]))
            rot_correction = quat_mul(quat_fraction(q_err, 0.08), rot_correction)
            rot_correction /= np.linalg.norm(rot_correction)
        elif 1.95 <= t < 7.5:
            # Insertion/hold: keep compliantly centering laterally (x, z) and in
            # rotation -- the software equivalent of a remote-center-of-compliance
            # fixture. The y (insertion axis) correction stays frozen.
            l7_pose_now = state_0.body_q.numpy()[link7_body]
            nominal = l7_target0 + np.array([0.0, hand_target_offset_y(t), 0.0])
            lateral_err = nominal - l7_pose_now[:3]
            tracking_correction[0] += 0.04 * lateral_err[0]
            tracking_correction[2] += 0.04 * lateral_err[2]
            q_err = quat_mul(rot_desired, quat_conj(l7_pose_now[3:7]))
            rot_correction = quat_mul(quat_fraction(q_err, 0.04), rot_correction)
            rot_correction /= np.linalg.norm(rot_correction)
        elif t >= 7.5 and not correction_frozen:
            correction_frozen = True
            rc_deg = np.degrees(2.0 * np.arccos(np.clip(abs(rot_correction[3]), -1.0, 1.0)))
            print(
                f"final correction: {tracking_correction * 1000} mm, rot {rc_deg:.2f} deg",
                flush=True,
            )
        dy_cmd = hand_target_offset_y(t)
        if seat_dy is not None and dy_cmd > seat_dy:
            dy_cmd = seat_dy  # seated: stop pushing, hold a light preload
        l7_target = l7_target0 + tracking_correction + np.array([0.0, dy_cmd, 0.0])
        pos_obj.set_target_positions(wp.array([wp.vec3(*l7_target)], dtype=wp.vec3))
        rot_target = quat_mul(rot_correction, rot_desired)
        rot_obj.set_target_rotations(wp.array([wp.vec4(*rot_target)], dtype=wp.vec4))
        ik_solver.step(joint_q_ik, joint_q_ik, iterations=24)
        ik_q = joint_q_ik.numpy()[0]
        target_q_host[:7] = ik_q[:7]
        target_q_host[7] = FINGER_OPENING
        target_q_host[8] = FINGER_OPENING
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
        l7_pose = body_q[link7_body]
        plug_pos = l7_pose[:3] + quat_rotate(l7_pose[3:7], T_l7_plug_p)
        plug_y = float(plug_pos[1])
        if seat_dy is None and plug_y >= SEAT_PLUG_Y and t < 7.5:
            seat_dy = dy_cmd + 0.0005  # freeze with a light preload
            print(f"t={t:5.2f}s SEATED at dy={dy_cmd:+.4f}, freezing advance", flush=True)
        plug_q = quat_mul(l7_pose[3:7], T_l7_plug_q)
        q_rel = quat_mul(quat_conj(plug_q), body_q[latch_body][3:7])
        latch_angle = float(-2.0 * np.arctan2(q_rel[0], q_rel[3]))
        if frame % 30 == 0:
            l7_err = float(np.linalg.norm(l7_pose[:3] - l7_target))
            print(
                f"t={t:5.2f}s dy_cmd={dy_cmd:+.4f} plug_y={plug_y:+.4f} "
                f"latch={latch_angle:+.3f} l7_err={l7_err * 1000:5.1f}mm",
                flush=True,
            )
        for name, t_check in (
            ("start", 1.9),
            ("seated", 7.4),
            ("after_retract", 9.9),
            ("end", DURATION_SECONDS - 0.05),
        ):
            if name not in checkpoints and t >= t_check:
                checkpoints[name] = plug_y

    start_y = checkpoints["start"]
    seated_y = checkpoints["seated"]
    retract_y = checkpoints["after_retract"]
    inserted = (seated_y - start_y) > 0.02
    latch_held = abs(retract_y - seated_y) < 0.006

    print(f"\nplug_y start={start_y:+.4f} seated={seated_y:+.4f} after_retract={retract_y:+.4f}")
    print(
        f"inserted={inserted} (travel={(seated_y - start_y) * 1000:.1f}mm), latch_held_vs_arm_pull={latch_held}"
    )
    print("recording complete: panda_insert.rrd")


if __name__ == "__main__":
    main()
