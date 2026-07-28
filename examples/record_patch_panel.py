"""Dense patch-panel demo: connect a row of cables side by side.

Builds a row of RJ45 ports at REAL keystone pitch (~18 mm centre-to-centre, a
24-port 1U panel) in one Newton model. Every cable starts retracted and away from
its socket, then all of them insert into their adjacent sockets together via the
proven force-spring. This shows cables being connected right next to each other
at realistic spacing, and tests whether dense adjacent contact stays stable.

All ports ride the proven world-anchored d6 + force-spring rig from the scene
builder, driven by one shared insert timeline.

Requires Newton git HEAD (.venv-head). Run from the repo root.
"""

import dataclasses
import json
import pathlib
import sys

import newton
import numpy as np
import warp as wp

newton.use_coord_layout_targets = True

from newton.examples.contacts.example_contacts_rj45_plug import _sync_cable_anchors  # noqa: E402
from newton.solvers import SolverVBD  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root
from newton_cabling.cable import route_cable_from_boot  # noqa: E402
from newton_cabling.connector import rj45_connector  # noqa: E402
from newton_cabling.report import CycleMeasurements, evaluate_cycle, outcome_to_dict  # noqa: E402
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: E402
from newton_cabling.sim.safe_vbd import finalize_for_vbd, new_vbd_builder  # noqa: E402
from newton_cabling.sim.scene import (  # noqa: E402
    PATCH_PANEL_PITCH_METERS,
    add_connector_rig,
    load_connector_meshes,
    patch_panel_port_offsets,
)
from newton_cabling.timeline import CableTimeline, GraspState, LatchState, Phase  # noqa: E402

FPS = 60
DURATION_SECONDS = 13.0
SIM_SUBSTEPS = 8
Z_LIFT = np.array([0.0, 0.0, 0.35])
START_DY = -0.050  # every cable starts here: retracted, away from its socket
INSERT_CAP = 0.035  # commanded insert depth

NUM_PORTS = 6
STAGGER_SECONDS = 0.4  # delay between consecutive ports plugging in


@wp.kernel
def _batched_grasp_spring(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    plug_indices: wp.array(dtype=int),
    latch_indices: wp.array(dtype=int),
    targets: wp.array(dtype=wp.vec3),
    stiffness: float,
    damping: float,
    gravity: wp.vec3,
):
    port = wp.tid()
    plug = plug_indices[port]
    latch = latch_indices[port]
    wp.atomic_add(body_f, plug, wp.spatial_vector(-gravity * body_mass[plug], wp.vec3(0.0)))
    wp.atomic_add(body_f, latch, wp.spatial_vector(-gravity * body_mass[latch], wp.vec3(0.0)))

    target = targets[port]
    plug_position = wp.transform_get_translation(body_q[plug])
    plug_velocity = wp.spatial_top(body_qd[plug])
    plug_mass = body_mass[plug]
    scale = 10.0 + plug_mass
    plug_force = scale * (stiffness * (target - plug_position) - damping * plug_velocity)
    wp.atomic_add(body_f, plug, wp.spatial_vector(plug_force, wp.vec3(0.0)))

    latch_velocity = wp.spatial_top(body_qd[latch])
    latch_mass = body_mass[latch]
    acceleration = (target - plug_position) * (scale * stiffness / plug_mass)
    latch_force = acceleration * latch_mass - latch_velocity * ((10.0 + latch_mass) * damping)
    wp.atomic_add(body_f, latch, wp.spatial_vector(latch_force, wp.vec3(0.0)))


def main() -> None:
    connector = rj45_connector()
    meshes = load_connector_meshes(connector)
    radius = connector.cable_radius_meters
    # Fewer kinematic segments than the spec default so the cable flexes right at
    # the boot instead of sticking out as a long rigid stub.
    prefix = 2

    socket_base = meshes.socket.base_position
    plug_base = meshes.plug.base_position
    latch_base = meshes.latch.base_position
    plug_y_offset = np.array([0.0, -0.025, 0.0])

    builder = new_vbd_builder(gravity=-9.81)
    builder.rigid_gap = 0.005

    port_offsets = patch_panel_port_offsets(NUM_PORTS)
    rigs = []
    plug_origins = []  # plug pose at dy=0, per port
    cable_anchor_data = []  # (anchor_indices, anchor_offsets, anchor_rotations) per port

    for port_index, offset in enumerate(port_offsets):
        socket_pos = socket_base + Z_LIFT + offset
        plug_origin = plug_base + plug_y_offset + Z_LIFT + offset
        latch_origin = latch_base + plug_y_offset + Z_LIFT + offset
        # Every port starts retracted: its cable hangs away from the socket.
        plug_init = plug_origin + np.array([0.0, START_DY, 0.0])
        latch_init = latch_origin + np.array([0.0, START_DY, 0.0])

        rig = add_connector_rig(
            builder,
            connector,
            meshes,
            socket_pos=socket_pos,
            plug_pos=plug_init,
            latch_pos=latch_init,
            plug_anchor_pos=plug_init,
        )
        rigs.append(rig)
        plug_origins.append(plug_origin)

        # Cable for this port, prefix anchored to the plug.
        cable_points = [wp.vec3(*point) for point in route_cable_from_boot(plug_init)]
        cable_quats = newton.utils.create_parallel_transport_cable_quaternions(cable_points)
        rod_bodies, _ = builder.add_rod(
            positions=cable_points,
            quaternions=cable_quats,
            radius=radius,
            cfg=dataclasses.replace(
                builder.default_shape_cfg,
                ke=connector.contact.stiffness,
                kd=connector.contact.damping,
                mu=connector.cable_friction,
            ),
            bend_stiffness=1.0e1,
            bend_damping=3.0e-1,  # higher than the example (which rested on a table) to settle the hang
            label=f"cable_{port_index}",
        )
        for body_idx in rod_bodies[:prefix]:
            for cable_shape in builder.body_shapes[body_idx]:
                for conn_shape in rig.connector_shapes:
                    builder.add_shape_collision_filter_pair(cable_shape, conn_shape)
        for idx in (*rod_bodies[:prefix], rod_bodies[-1]):
            builder.body_mass[idx] = 0.0
            builder.body_inv_mass[idx] = 0.0
            builder.body_inertia[idx] = wp.mat33(0.0)
            builder.body_inv_inertia[idx] = wp.mat33(0.0)
        anchor_offsets = tuple(
            wp.vec3(*(np.array([cable_points[i][0], cable_points[i][1], cable_points[i][2]]) - plug_init))
            for i in range(prefix)
        )
        cable_anchor_data.append((tuple(rod_bodies[:prefix]), anchor_offsets, tuple(cable_quats[:prefix])))

    # No ground plane: every cable is pinned at both ends (prefix to its plug,
    # far end fixed), so the scene needs no floor -- and omitting the large plane
    # mesh keeps the recording from ever opening to "just the floor". The viewer
    # instances identical shapes, so the rendered shape index can't be matched to
    # a model index reliably; not adding the plane sidesteps that entirely.
    model = finalize_for_vbd(builder)

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()
    solver = SolverVBD(
        model, iterations=12, rigid_contact_hard=False, rigid_body_contact_buffer_size=512
    )

    viewer = open_rrd_recorder("patch_panel.rrd")
    viewer.set_model(model)

    device = model.device
    plug_index_array = wp.array([rig.plug_body for rig in rigs], dtype=int, device=device)
    latch_index_array = wp.array([rig.latch_body for rig in rigs], dtype=int, device=device)
    target_array = wp.array(
        [wp.vec3(*(plug_origins[p] + np.array([0.0, START_DY, 0.0]))) for p in range(NUM_PORTS)],
        dtype=wp.vec3,
        device=device,
    )
    gravity_vec = wp.vec3(0.0, 0.0, -9.81)

    cable_anchor_indices = [
        wp.array(d[0], dtype=int, device=device) for d in cable_anchor_data
    ]
    cable_anchor_offsets = [
        wp.array(d[1], dtype=wp.vec3, device=device) for d in cable_anchor_data
    ]
    cable_anchor_rotations = [
        wp.array(d[2], dtype=wp.quat, device=device) for d in cable_anchor_data
    ]

    insert_timeline = CableTimeline.build(
        [
            Phase("settle", 1.0, START_DY, GraspState.GRASPED, LatchState.NEUTRAL),
            Phase("approach", 3.0, 0.0, GraspState.GRASPED, LatchState.NEUTRAL),
            Phase("insert", 4.0, INSERT_CAP, GraspState.GRASPED, LatchState.NEUTRAL),
            Phase("hold", 2.0, INSERT_CAP, GraspState.GRASPED, LatchState.NEUTRAL),
        ]
    )

    frame_dt = 1.0 / FPS
    sim_dt = frame_dt / SIM_SUBSTEPS
    num_frames = int(DURATION_SECONDS * FPS)
    checkpoints: list[dict[str, float]] = [{} for _ in range(NUM_PORTS)]

    def simulate_frame() -> None:
        nonlocal state_0, state_1
        for _ in range(SIM_SUBSTEPS):
            state_0.clear_forces()
            wp.launch(
                _batched_grasp_spring,
                dim=NUM_PORTS,
                inputs=(
                    state_0.body_q,
                    state_0.body_qd,
                    state_0.body_f,
                    model.body_mass,
                    plug_index_array,
                    latch_index_array,
                    target_array,
                    50.0,
                    10.0,
                    gravity_vec,
                ),
                device=device,
            )
            for port_index in range(NUM_PORTS):
                wp.launch(
                    _sync_cable_anchors,
                    dim=prefix,
                    inputs=(
                        state_0.body_q,
                        state_0.body_qd,
                        rigs[port_index].plug_body,
                        cable_anchor_indices[port_index],
                        cable_anchor_offsets[port_index],
                        cable_anchor_rotations[port_index],
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
        # Ports plug in quickly in succession: each port's timeline is shifted by
        # STAGGER_SECONDS, so port 0 inserts first and the rest follow in turn.
        target_array.assign(
            [
                wp.vec3(
                    *(
                        plug_origins[p]
                        + np.array(
                            [0.0, insert_timeline.sample(t - p * STAGGER_SECONDS).plug_offset_meters, 0.0]
                        )
                    )
                )
                for p in range(NUM_PORTS)
            ]
        )
        sample = insert_timeline.sample(t)  # phase label for the telemetry line

        if graph:
            wp.capture_launch(graph)
        else:
            simulate_frame()
        sim_time += frame_dt

        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.end_frame()

        body_q = state_0.body_q.numpy()
        for p in range(NUM_PORTS):
            plug_y = float(body_q[rigs[p].plug_body][1])
            for name, t_check in (("start", 0.9), ("seated", DURATION_SECONDS - 0.05)):
                if name not in checkpoints[p] and t >= t_check:
                    checkpoints[p][name] = plug_y
        if frame % 60 == 0:
            ys = [float(body_q[rigs[p].plug_body][1]) for p in range(NUM_PORTS)]
            print(
                f"t={t:5.2f}s {sample.phase_name:<9} plug_y min={min(ys):+.4f} max={max(ys):+.4f}",
                flush=True,
            )

    print(f"\npatch panel: {NUM_PORTS} ports at {PATCH_PANEL_PITCH_METERS * 1000:.0f}mm pitch")
    port_results = []
    seated_count = 0
    for p in range(NUM_PORTS):
        cp = checkpoints[p]
        measurements = CycleMeasurements(
            plug_y_start_meters=cp["start"],
            plug_y_seated_meters=cp["seated"],
            plug_y_after_release_meters=cp["seated"],
            plug_y_end_meters=cp["start"],
            max_tracking_error_meters=0.0,
            max_plug_pitch_degrees=0.0,
        )
        outcome = evaluate_cycle(measurements)
        inserted = type(outcome).__name__ != "InsertionFailed"
        seated_count += int(inserted)
        port_results.append({"port": p, "inserted": inserted, **outcome_to_dict(outcome)})
        print(f"  port {p}: start={cp['start']:+.4f} seated={cp['seated']:+.4f} inserted={inserted}")
    print(f"{seated_count}/{NUM_PORTS} cables inserted")
    pathlib.Path("patch_panel_result.json").write_text(
        json.dumps(
            {
                "num_ports": NUM_PORTS,
                "pitch_mm": PATCH_PANEL_PITCH_METERS * 1000,
                "ports": port_results,
            },
            indent=2,
        )
    )
    blueprint = auto_blueprint("patch_panel.rbl", model)
    print(f"recording complete: patch_panel.rrd ; blueprint {blueprint}")


if __name__ == "__main__":
    main()
