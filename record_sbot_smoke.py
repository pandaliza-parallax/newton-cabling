"""Smoke test: import the StandardBots RO1 USD into the Newton env and hold it.

Proves the full robot (6-DOF arm + DH AG-145 gripper) loads via ``add_sbot``,
finalizes under SolverVBD, holds its home pose against gravity with position PD,
and that the gripper opens and closes correctly -- all eight finger joints driven
through the reconstructed AG-145 coupling from one driver angle. It records to
``sbot_smoke.rrd`` and prints the wrist sag and the measured fingertip gap so a
regression is obvious.

The gripper coupling is PD-tracked, not constraint-enforced, so this faithfully
poses the jaws but is not a force-closure grip under contact (see
``newton_cabling.sim.sbot``).

Requires Newton + a Warp device. Run from the repo root:
    .venv/bin/python record_sbot_smoke.py
"""

import pathlib
import sys

import newton
import numpy as np
import warp as wp

newton.use_coord_layout_targets = True

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from newton.solvers import SolverVBD  # noqa: E402

from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: E402
from newton_cabling.sim.safe_vbd import finalize_for_vbd, new_vbd_builder  # noqa: E402
from newton_cabling.sim.sbot import (  # noqa: E402
    GRIPPER_THETA_CLOSED,
    GRIPPER_THETA_OPEN,
    SBOT_HOME,
    add_sbot,
    set_arm_home,
    set_gripper,
    set_pd_gains,
)

FPS = 60
DURATION_SECONDS = 4.0
SIM_SUBSTEPS = 8


def smoothstep(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return a * a * (3.0 - 2.0 * a)


def gripper_theta(t: float) -> float:
    """Hold open 1.5 s, close over 1 s, then hold closed (a grasp-style motion)."""
    if t < 1.5:
        return GRIPPER_THETA_OPEN
    if t < 2.5:
        span = GRIPPER_THETA_CLOSED - GRIPPER_THETA_OPEN
        return GRIPPER_THETA_OPEN + span * smoothstep(t - 1.5)
    return GRIPPER_THETA_CLOSED


def main() -> None:
    builder = new_vbd_builder(gravity=-9.81)

    base_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    handles = add_sbot(builder, base_xform, with_gripper=True)
    set_pd_gains(builder, handles)
    set_arm_home(builder, handles, SBOT_HOME)
    set_gripper(builder, handles, GRIPPER_THETA_OPEN)  # start with jaws open

    # Import-only verification: the arm/gripper meshes are visual here, so we do
    # not pay for SDF contact. Drop all collisions to isolate "does it hold?".
    for shape_idx in range(len(builder.shape_body)):
        builder.shape_flags[shape_idx] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

    model = finalize_for_vbd(builder)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()
    solver = SolverVBD(model, iterations=12)

    viewer = open_rrd_recorder("sbot_smoke.rrd")
    viewer.set_model(model)

    target_q = model.joint_target_q.numpy().copy()
    wrist = handles.wrist_body

    def find_body(suffix: str) -> int:
        return next(i for i, lbl in enumerate(model.body_label) if lbl.endswith(suffix))

    tip1 = find_body("finger1_finger_tip_link")
    tip2 = find_body("finger2_finger_tip_link")

    def fingertip_gap_mm(bq) -> float:
        return float(np.linalg.norm(bq[tip1][:3] - bq[tip2][:3]) * 1000.0)

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    wrist_z0 = float(state_0.body_q.numpy()[wrist][2])
    print(f"home wrist height: {wrist_z0:+.4f} m", flush=True)

    frame_dt = 1.0 / FPS
    sim_dt = frame_dt / SIM_SUBSTEPS
    num_frames = int(DURATION_SECONDS * FPS)

    for frame in range(num_frames):
        t = frame * frame_dt
        theta = gripper_theta(t)
        for idx, ratio in handles.gripper_coupling:
            target_q[idx] = ratio * theta
        control.joint_target_q.assign(target_q)

        for _ in range(SIM_SUBSTEPS):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.end_frame()

        if frame % 15 == 0:
            bq = state_0.body_q.numpy()
            sag_mm = (wrist_z0 - float(bq[wrist][2])) * 1000.0
            print(
                f"t={t:4.1f}s theta={theta:+.3f} jaw_gap={fingertip_gap_mm(bq):6.1f}mm "
                f"wrist_sag={sag_mm:+6.1f}mm",
                flush=True,
            )

    bq = state_0.body_q.numpy()
    sag_mm = (wrist_z0 - float(bq[wrist][2])) * 1000.0
    held = abs(sag_mm) < 30.0
    print(f"\nfinal jaw_gap={fingertip_gap_mm(bq):.1f}mm  wrist_sag={sag_mm:+.1f}mm  held={held}")
    auto_blueprint("sbot_smoke.rbl", model)
    print("recording complete: sbot_smoke.rrd")


if __name__ == "__main__":
    main()
