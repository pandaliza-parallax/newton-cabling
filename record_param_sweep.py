"""Batched parameter sweep over connector-insertion physics.

Builds N connector worlds (socket / plug / latch, no arm) side by side in ONE
Newton model, each from a different ConnectorSpec, drives the SAME scripted
insert -> hold -> release -> press -> extract on all of them via the cycle
timeline, steps them together, and scores each with report.evaluate_cycle.

This is the force multiplier: the GPU runs every variant in parallel, so a sweep
that would be N serial 5-minute runs becomes one pass. Each world is driven by a
batched force-spring (the same coupling as GraspSpring, vectorised over worlds);
the latch is pressed via its joint target, exactly as in the unplug demo.

Requires Newton git HEAD (.venv-head). Run from the repo root.
"""

import json
import pathlib
import sys

import newton
import numpy as np
import warp as wp

newton.use_coord_layout_targets = True

from newton.solvers import SolverVBD  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from newton_cabling.connector import rj45_connector  # noqa: E402
from newton_cabling.report import CycleMeasurements, evaluate_cycle, outcome_to_dict  # noqa: E402
from newton_cabling.sim.safe_vbd import finalize_for_vbd, new_vbd_builder  # noqa: E402
from newton_cabling.sim.scene import add_connector_rig, load_connector_meshes  # noqa: E402
from newton_cabling.timeline import proven_cycle_timeline  # noqa: E402

FPS = 60
DURATION_SECONDS = 19.0
SIM_SUBSTEPS = 8
Z_LIFT = np.array([0.0, 0.0, 0.35])
INSERT_TRAVEL_CAP = 0.035
START_DY = -0.050
PRESS_TARGET_KE = 1.0
WORLD_SPACING = 0.4  # metres between worlds along x

# The parameter being swept: how far the latch tab is pressed during extraction.
# 0.0 = not pressed (the tab stays caught on the socket lip, so the plug cannot
# come out); more negative = pressed further toward the inward travel limit.
# This is the threshold the sweep discovers: the minimum press that frees the plug.
SWEEP_PRESS_ANGLE = [0.0, -0.075, -0.15, -0.225, -0.30]


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _quat_mul(_quat_mul(q, np.array([v[0], v[1], v[2], 0.0])), _quat_conj(q))[:3]


@wp.kernel
def _batched_grasp_spring(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    plug_indices: wp.array(dtype=int),
    latch_indices: wp.array(dtype=int),
    targets: wp.array(dtype=wp.vec3),
    strength: wp.array(dtype=float),
    stiffness: float,
    damping: float,
    gravity: wp.vec3,
):
    world = wp.tid()
    plug = plug_indices[world]
    latch = latch_indices[world]
    wp.atomic_add(body_f, plug, wp.spatial_vector(-gravity * body_mass[plug], wp.vec3(0.0)))
    wp.atomic_add(body_f, latch, wp.spatial_vector(-gravity * body_mass[latch], wp.vec3(0.0)))

    s = strength[0]
    if s > 0.0:
        target = targets[world]
        plug_position = wp.transform_get_translation(body_q[plug])
        plug_velocity = wp.spatial_top(body_qd[plug])
        plug_mass = body_mass[plug]
        scale = 10.0 + plug_mass
        plug_force = scale * (stiffness * (target - plug_position) - damping * plug_velocity)
        wp.atomic_add(body_f, plug, wp.spatial_vector(plug_force * s, wp.vec3(0.0)))

        latch_velocity = wp.spatial_top(body_qd[latch])
        latch_mass = body_mass[latch]
        acceleration = (target - plug_position) * (scale * stiffness / plug_mass)
        latch_force = acceleration * latch_mass - latch_velocity * ((10.0 + latch_mass) * damping)
        wp.atomic_add(body_f, latch, wp.spatial_vector(latch_force * s, wp.vec3(0.0)))


def main() -> None:
    base_spec = rj45_connector()
    meshes = load_connector_meshes(base_spec)  # shared across all worlds
    # All worlds use the same connector; only the commanded press depth differs.
    specs = [base_spec for _ in SWEEP_PRESS_ANGLE]
    num_worlds = len(specs)

    socket_base = meshes.socket.base_position
    plug_base = meshes.plug.base_position
    latch_base = meshes.latch.base_position
    plug_y_offset = np.array([0.0, -0.025, 0.0])
    start_offset = np.array([0.0, START_DY, 0.0])

    builder = new_vbd_builder(gravity=-9.81)
    builder.rigid_gap = 0.005

    rigs = []
    plug_trajectory_origins = []  # plug pose at dy=0, per world
    for world_index, spec in enumerate(specs):
        world_shift = np.array([world_index * WORLD_SPACING, 0.0, 0.0])
        socket_pos = socket_base + Z_LIFT + world_shift
        plug_origin = plug_base + plug_y_offset + Z_LIFT + world_shift
        latch_origin = latch_base + plug_y_offset + Z_LIFT + world_shift
        plug_init = plug_origin + start_offset
        latch_init = latch_origin + start_offset
        rig = add_connector_rig(
            builder,
            spec,
            meshes,
            socket_pos=socket_pos,
            plug_pos=plug_init,
            latch_pos=latch_init,
            plug_anchor_pos=plug_init,
        )
        rigs.append(rig)
        plug_trajectory_origins.append(plug_origin)

    builder.add_ground_plane()
    model = finalize_for_vbd(builder)

    # Per-world latch joint coordinate / dof, and its rest target stiffness.
    joint_types = model.joint_type.numpy()
    joint_children = model.joint_child.numpy()
    joint_q_start = model.joint_q_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    target_ke_host = model.joint_target_ke.numpy().copy()
    latch_coords: list[int] = []
    latch_dofs: list[int] = []
    for rig in rigs:
        latch_joint = next(
            j
            for j in range(model.joint_count)
            if joint_types[j] == int(newton.JointType.REVOLUTE) and joint_children[j] == rig.latch_body
        )
        latch_coords.append(int(joint_q_start[latch_joint]))
        latch_dofs.append(int(joint_qd_start[latch_joint]))
    latch_rest_ke = [float(target_ke_host[dof]) for dof in latch_dofs]

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()
    solver = SolverVBD(
        model, iterations=12, rigid_contact_hard=False, rigid_body_contact_buffer_size=256
    )

    device = model.device
    plug_index_array = wp.array([rig.plug_body for rig in rigs], dtype=int, device=device)
    latch_index_array = wp.array([rig.latch_body for rig in rigs], dtype=int, device=device)
    target_array = wp.array(
        [wp.vec3(*(plug_trajectory_origins[w] + start_offset)) for w in range(num_worlds)],
        dtype=wp.vec3,
        device=device,
    )
    strength_array = wp.array([1.0], dtype=float, device=device)
    gravity_vec = wp.vec3(0.0, 0.0, -9.81)
    target_q_host = control.joint_target_q.numpy().copy()

    timeline = proven_cycle_timeline(
        start_offset_meters=START_DY, insert_cap_meters=INSERT_TRAVEL_CAP
    )
    frame_dt = 1.0 / FPS
    sim_dt = frame_dt / SIM_SUBSTEPS
    num_frames = int(DURATION_SECONDS * FPS)

    checkpoints: list[dict[str, float]] = [{} for _ in range(num_worlds)]
    max_pitch_degrees = [0.0] * num_worlds

    def simulate_frame() -> None:
        nonlocal state_0, state_1
        for _ in range(SIM_SUBSTEPS):
            state_0.clear_forces()
            wp.launch(
                _batched_grasp_spring,
                dim=num_worlds,
                inputs=(
                    state_0.body_q,
                    state_0.body_qd,
                    state_0.body_f,
                    model.body_mass,
                    plug_index_array,
                    latch_index_array,
                    target_array,
                    strength_array,
                    50.0,
                    10.0,
                    gravity_vec,
                ),
                device=device,
            )
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    graph = None
    if wp.get_device().is_cuda:
        with wp.ScopedCapture() as capture:
            simulate_frame()
        graph = capture.graph

    sim_time = 0.0
    for frame in range(num_frames):
        t = sim_time
        sample = timeline.sample(t)

        strength_array.assign([sample.grasp_weight])
        target_array.assign(
            [
                wp.vec3(*(plug_trajectory_origins[w] + np.array([0.0, sample.plug_offset_meters, 0.0])))
                for w in range(num_worlds)
            ]
        )

        boost_latch = sample.latch_weight > 0.01
        ke_changed = False
        for w in range(num_worlds):
            # Each world presses to its own swept depth (scaled by the press ramp).
            target_q_host[latch_coords[w]] = SWEEP_PRESS_ANGLE[w] * sample.latch_weight
            desired_ke = PRESS_TARGET_KE if boost_latch else latch_rest_ke[w]
            if float(target_ke_host[latch_dofs[w]]) != desired_ke:
                target_ke_host[latch_dofs[w]] = desired_ke
                ke_changed = True
        control.joint_target_q.assign(target_q_host)
        if ke_changed:
            model.joint_target_ke.assign(target_ke_host)

        if graph:
            wp.capture_launch(graph)
        else:
            simulate_frame()
        sim_time += frame_dt

        body_q = state_0.body_q.numpy()
        for w in range(num_worlds):
            plug_pose = body_q[rigs[w].plug_body]
            plug_y = float(plug_pose[1])
            y_axis = _quat_rotate(plug_pose[3:7], np.array([0.0, 1.0, 0.0]))
            pitch = abs(float(np.degrees(np.arcsin(np.clip(y_axis[2], -1.0, 1.0)))))
            max_pitch_degrees[w] = max(max_pitch_degrees[w], pitch)
            for name, t_check in (
                ("start", 3.9),
                ("seated", 8.4),
                ("after_release", 12.9),
                ("end", DURATION_SECONDS - 0.05),
            ):
                if name not in checkpoints[w] and t >= t_check:
                    checkpoints[w][name] = plug_y
        if frame % 60 == 0:
            print(f"t={t:5.2f}s {sample.phase_name}", flush=True)

    print(f"\n=== sweep over press depth ({num_worlds} worlds, one GPU pass) ===")
    results = []
    for w in range(num_worlds):
        cp = checkpoints[w]
        measurements = CycleMeasurements(
            plug_y_start_meters=cp["start"],
            plug_y_seated_meters=cp["seated"],
            plug_y_after_release_meters=cp["after_release"],
            plug_y_end_meters=cp["end"],
            max_tracking_error_meters=0.0,
            max_plug_pitch_degrees=max_pitch_degrees[w],
        )
        outcome = evaluate_cycle(measurements)
        value = SWEEP_PRESS_ANGLE[w]
        results.append({"press_angle_radians": value, **outcome_to_dict(outcome)})
        print(
            f"  press_angle={value:+.3f}rad -> {type(outcome).__name__:20s} "
            f"seated={cp['seated']:+.4f} retreat={cp['after_release']:+.4f} end={cp['end']:+.4f}"
        )

    pathlib.Path("sweep_results.json").write_text(json.dumps(results, indent=2))
    print("wrote sweep_results.json")


if __name__ == "__main__":
    main()
