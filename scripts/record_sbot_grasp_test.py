"""Milestone A of the sbot friction grasp: can the AG-145's kinematic fingers grip the plug?

Port of examples/record_grasp_test.py (Panda) to the RO1 + AG-145, same cloth_franka pattern:
every robot body is KINEMATIC (posed by eval_fk from commanded joint_q each substep, so
contact cannot shake it), only the finger link meshes collide (SDF + friction), and the
plug is a free VBD rigid body held purely by finger contact + friction.

The AG-145 caveat in sim/sbot.py (independent mimic joints -> no force closure under PD)
does NOT apply here: kinematic fingers cannot be pushed back, so the coupling holds by
construction.

Pad-gap calibration (tools/sbot cut geometry, measured): gap(theta) is ~linear near closed,
11.68mm (RJ45 plug width) at theta ~= -0.035; THETA_GRIP = -0.028 gives a ~2mm squeeze.
The plug spawns already lightly pinched (kinematic close from open would let it free-fall).

Timeline: 0-2s static hold (hand points down -> gravity shears the pads: the pure friction
test) | 2-4s +-5 deg base slew (inertial shake) | 4-5s hold. PASS = the plug tracks the
hand (relative drift < 5mm) instead of falling.

Run from the repo root (newton .venv):
    .venv/bin/python scripts/record_sbot_grasp_test.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverVBD

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root
from newton_cabling.connector import cad_rj45_connector  # noqa: E402
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: E402
from newton_cabling.sim.safe_vbd import finalize_for_vbd, new_vbd_builder  # noqa: E402
from newton_cabling.sim.sbot import add_sbot, set_arm_home, set_gripper  # noqa: E402
from newton_cabling.sim.scene import load_connector_meshes  # noqa: E402

FPS = 60
DURATION_SECONDS = 5.0
SIM_SUBSTEPS = 8

ARM_HOME_DEG = [4.0, -19.5, -113.0, 43.5, -268.9, -178.0]  # pendant pose: hand points down
THETA_GRIP = -0.028          # pad gap ~9.7mm vs 11.68mm plug -> ~1mm interference per pad

FINGER_BODIES = (
    "gripper_finger1_finger_link", "gripper_finger1_finger_tip_link",
    "gripper_finger2_finger_link", "gripper_finger2_finger_tip_link",
    "gripper_finger1_inner_knuckle_link", "gripper_finger2_inner_knuckle_link",
)


def find_body(labels: list[str], suffix: str) -> int:
    return next(i for i, lbl in enumerate(labels) if lbl.endswith(suffix))


def main() -> None:
    builder = new_vbd_builder(gravity=-9.81)
    builder.rigid_gap = 0.005      # contact-generation margin: EVERY contact script sets this
    handles = add_sbot(builder, wp.transform(wp.vec3(0.0, 0.0, 0.6), wp.quat_identity()),
                       with_gripper=True)
    set_arm_home(builder, handles, [math.radians(a) for a in ARM_HOME_DEG])
    set_gripper(builder, handles, THETA_GRIP)

    finger_idx = {find_body(builder.body_label, n) for n in FINGER_BODIES}

    # Finger meshes collide (SDF + friction); every other robot shape does not.
    pads = 0
    for s in range(len(builder.shape_body)):
        on_finger = builder.shape_body[s] in finger_idx and builder.shape_type[s] == int(
            newton.GeoType.MESH
        )
        if on_finger:
            mesh = builder.shape_source[s]
            if mesh is not None and getattr(mesh, "sdf", None) is None:
                mesh.build_sdf(max_resolution=64, narrow_band_range=(-0.01, 0.01), margin=0.004)
            builder.shape_material_mu[s] = 1.0
            pads += 1
        else:
            builder.shape_flags[s] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    print(f"[grasp] {pads} finger mesh shapes collidable (SDF, mu=1.0)")

    # The whole robot is kinematic: contact cannot move it.
    for b in range(builder.body_count):
        builder.body_flags[b] = int(newton.BodyFlags.KINEMATIC)

    # FK the grip pose to place the plug between the pads.
    fk_model = builder.finalize()
    fk_state = fk_model.state()
    newton.eval_fk(fk_model, fk_model.joint_q, fk_model.joint_qd, fk_state)
    fk_bq = fk_state.body_q.numpy()
    labels = list(fk_model.body_label)
    tip1 = find_body(labels, "gripper_finger1_finger_tip_link")
    tip2 = find_body(labels, "gripper_finger2_finger_tip_link")
    wrist = find_body(labels, "wrist_3_link")
    tip_mid = 0.5 * (fk_bq[tip1][:3] + fk_bq[tip2][:3])
    wrist_p, wrist_q = fk_bq[wrist][:3], fk_bq[wrist][3:7]  # xyzw
    from scipy.spatial.transform import Rotation
    Rw = Rotation.from_quat(wrist_q).as_matrix()
    tool_axis = tip_mid - wrist_p
    tool_axis /= np.linalg.norm(tool_axis)

    # Real CAD RJ45 plug mesh (SDF prebuilt): mesh-vs-mesh SDF is the contact pair the
    # repo's plug/socket sims are proven on (a box primitive vs SDF never made contacts).
    spec = cad_rj45_connector(friction=0.9)
    meshes = load_connector_meshes(spec)
    plug_cfg = newton.ModelBuilder.ShapeConfig(
        mu=0.9, ke=spec.contact.stiffness, kd=0.0, gap=spec.contact.gap_meters, density=1500.0
    )

    # Orient head-down between the pads: plug local x (11.68mm width) -> jaw axis (wrist y),
    # plug local y (insertion axis, leading face at y=0) -> tool axis (down). The 8P8C body
    # (first ~20mm behind the face) then spans the pad zone (0.17..0.228 along the tool axis).
    jaw_axis = Rw @ np.array([0.0, 1.0, 0.0])
    y_p = tool_axis
    x_p = jaw_axis - y_p * (jaw_axis @ y_p)
    x_p /= np.linalg.norm(x_p)
    z_p = np.cross(x_p, y_p)
    q_plug = Rotation.from_matrix(np.column_stack([x_p, y_p, z_p])).as_quat()  # xyzw
    face_pos = wrist_p + 0.225 * tool_axis             # leading face just past the tips
    print(f"[grasp] plug face {np.round(face_pos, 3)} (tool axis {np.round(tool_axis, 2)})")

    plug_body = builder.add_body(
        xform=wp.transform(wp.vec3(*face_pos), wp.quat(*q_plug)), label="plug"
    )
    builder.add_shape_mesh(plug_body, mesh=meshes.plug.mesh, cfg=plug_cfg)
    builder.add_joint_free(child=plug_body)

    model = finalize_for_vbd(builder)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()
    solver = SolverVBD(model, iterations=12, rigid_contact_hard=False,
                       rigid_body_contact_buffer_size=256)

    viewer = open_rrd_recorder("sbot_grasp_test.rrd")
    viewer.set_model(model)

    frame_dt = 1.0 / FPS
    sim_dt = frame_dt / SIM_SUBSTEPS
    plug = find_body(list(model.body_label), "plug")
    joint_q = model.joint_q.numpy().copy()
    j0 = handles.arm_joints[0]
    j0_home = joint_q[j0]

    bq = state_0.body_q.numpy()
    rel0 = bq[plug][:3] - bq[wrist][:3]                 # plug offset from the hand at spawn

    for frame in range(int(DURATION_SECONDS * FPS)):
        t = frame * frame_dt
        if 2.0 <= t < 4.0:                              # +-5 deg base slew: inertial shake
            joint_q[j0] = j0_home + math.radians(5.0) * math.sin(2.0 * math.pi * (t - 2.0) / 2.0)
        else:
            joint_q[j0] = j0_home
        joint_q_wp = wp.array(joint_q, dtype=float, device=model.device)

        for _ in range(SIM_SUBSTEPS):
            # pose ONLY the kinematic robot; the free plug keeps its solver-integrated state
            # (a full eval_fk would teleport it back to its spawn joint_q every substep)
            newton.eval_fk(model, joint_q_wp, model.joint_qd, state_0,
                           body_flag_filter=int(newton.BodyFlags.KINEMATIC))
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.end_frame()
        if frame % 30 == 0:
            bq = state_0.body_q.numpy()
            drift = np.linalg.norm((bq[plug][:3] - bq[wrist][:3]) - rel0) * 1000.0
            try:
                nc = int(contacts.rigid_contact_count.numpy()[0])
            except Exception:
                nc = -1
            print(f"t={t:4.1f}s  plug-in-hand drift {drift:6.1f}mm  contacts={nc}", flush=True)

    bq = state_0.body_q.numpy()
    drift = np.linalg.norm((bq[plug][:3] - bq[wrist][:3]) - rel0) * 1000.0
    held = drift < 5.0
    print(f"\nfinal drift {drift:.1f}mm -> gripped={held}")
    auto_blueprint("sbot_grasp_test.rbl", model)
    print("recording: sbot_grasp_test.rrd")


if __name__ == "__main__":
    main()
