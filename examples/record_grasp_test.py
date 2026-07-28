"""Milestone 1 of the real friction grasp: can kinematic fingers grip a free body?

Following Newton's cloth_franka pattern: the arm (including fingers) is driven
KINEMATICALLY -- its body poses come from forward kinematics on commanded joint
coordinates, so contact can't push it around (no shaking) -- while the gripped
object is a free VBD rigid body held by finger contact + friction (a real grip,
not a force-spring).

This minimal test drops everything else (socket, latch, cable, insertion) to
validate just the grip: place a free box at the grip centre, close the fingers,
and check it is held against gravity. If it doesn't fall, the grip works and we
build insertion on top.

Requires Newton git HEAD (.venv-head). Run from the repo root.
"""

import pathlib
import sys

import newton
import numpy as np
import warp as wp

newton.use_coord_layout_targets = True

import newton.utils  # noqa: E402
from newton.solvers import SolverVBD  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: E402
from newton_cabling.sim.safe_vbd import add_actuated_urdf, find_body_index, finalize_for_vbd  # noqa: E402

FPS = 60
DURATION_SECONDS = 5.0
SIM_SUBSTEPS = 8
PANDA_URDF = "urdf/fr3_franka_hand.urdf"
PANDA_HOME = [-3.68e-03, 2.39e-02, 3.68e-03, -2.368, -1.29e-04, 2.392, 0.785]

FINGER_OPEN = 0.035  # fingers wide open
FINGER_GRIP = 0.004  # closed tighter than the box half-width so they squeeze
Z_LIFT = np.array([0.0, 0.0, 0.35])
PLUG_HALF = (0.007, 0.020, 0.005)  # ~14 x 40 x 10 mm box, like the RJ45 plug body


def smoothstep(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return a * a * (3.0 - 2.0 * a)


def new_builder() -> newton.ModelBuilder:
    builder = newton.ModelBuilder(gravity=-9.81)
    SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
    return builder


def main() -> None:
    builder = new_builder()
    panda_base = wp.transform(wp.vec3(0.0, -0.55, 0.35), wp.quat_identity())
    add_actuated_urdf(
        builder, newton.utils.download_asset("franka_emika_panda") / PANDA_URDF, panda_base
    )

    finger_left = find_body_index(builder.body_label, "/fr3_leftfinger")
    finger_right = find_body_index(builder.body_label, "/fr3_rightfinger")
    hand = find_body_index(builder.body_label, "/fr3_link7")
    grip_bodies = {finger_left, finger_right, hand}

    builder.joint_q[:9] = [*PANDA_HOME, FINGER_OPEN, FINGER_OPEN]

    # Gripper shapes collide (SDF) with friction; the rest of the arm does not.
    for shape_idx in range(len(builder.shape_body)):
        on_gripper = builder.shape_body[shape_idx] in grip_bodies and builder.shape_type[
            shape_idx
        ] == int(newton.GeoType.MESH)
        if on_gripper:
            mesh = builder.shape_source[shape_idx]
            if mesh is not None and getattr(mesh, "sdf", None) is None:
                mesh.build_sdf(max_resolution=64, narrow_band_range=(-0.01, 0.01), margin=0.004)
            builder.shape_material_mu[shape_idx] = 1.0
        else:
            builder.shape_flags[shape_idx] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

    # The whole arm is kinematic: poses set from joint_q via eval_fk each substep,
    # so contact cannot move it (no shaking).
    for body_idx in range(builder.body_count):
        builder.body_flags[body_idx] = int(newton.BodyFlags.KINEMATIC)

    # FK the home pose to find the grip centre (between the fingers), drop a free
    # box there to be grasped.
    fk_model = builder.finalize()
    fk_state = fk_model.state()
    newton.eval_fk(fk_model, fk_model.joint_q, fk_model.joint_qd, fk_state)
    fk_bq = fk_state.body_q.numpy()
    grip_center = (fk_bq[finger_left][:3] + fk_bq[finger_right][:3]) / 2.0

    plug_body = builder.add_body(
        xform=wp.transform(wp.vec3(*grip_center), wp.quat_identity()), label="plug"
    )
    plug_cfg = newton.ModelBuilder.ShapeConfig(mu=0.9, density=1500.0, ke=5.0e4, kd=0.0, gap=0.002)
    builder.add_shape_box(
        plug_body, hx=PLUG_HALF[0], hy=PLUG_HALF[1], hz=PLUG_HALF[2], cfg=plug_cfg
    )
    builder.add_joint_free(child=plug_body)

    model = finalize_for_vbd(builder)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()
    solver = SolverVBD(
        model, iterations=12, rigid_contact_hard=False, rigid_body_contact_buffer_size=256
    )

    viewer = open_rrd_recorder("grasp_test.rrd")
    viewer.set_model(model)

    frame_dt = 1.0 / FPS
    sim_dt = frame_dt / SIM_SUBSTEPS
    num_frames = int(DURATION_SECONDS * FPS)
    arm_joint_q = model.joint_q.numpy().copy()
    box_start_z = float(grip_center[2])

    def finger_opening(t: float) -> float:
        if t < 1.0:  # settle, fingers open around the box
            return FINGER_OPEN
        if t < 2.0:  # close to grip
            return FINGER_OPEN + (FINGER_GRIP - FINGER_OPEN) * smoothstep(t - 1.0)
        return FINGER_GRIP  # hold -- gravity tests the grip

    for frame in range(num_frames):
        t = frame * frame_dt
        opening = finger_opening(t)
        arm_joint_q[7] = opening
        arm_joint_q[8] = opening
        joint_q_wp = wp.array(arm_joint_q, dtype=float, device=model.device)

        for _ in range(SIM_SUBSTEPS):
            newton.eval_fk(model, joint_q_wp, model.joint_qd, state_0)
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        if frame % 15 == 0:
            bz = float(state_0.body_q.numpy()[plug_body][2])
            print(
                f"t={t:4.1f}s finger={opening:.4f} box_z={bz:+.4f} (start {box_start_z:+.4f}, "
                f"drop {(box_start_z - bz) * 1000:+.1f}mm)",
                flush=True,
            )

    final_z = float(state_0.body_q.numpy()[plug_body][2])
    held = (box_start_z - final_z) < 0.03  # fell less than 30mm -> gripped
    print(f"\nbox start_z={box_start_z:+.4f} final_z={final_z:+.4f} -> gripped={held}")
    auto_blueprint("grasp_test.rbl", model)
    print("recording complete: grasp_test.rrd")


if __name__ == "__main__":
    main()
