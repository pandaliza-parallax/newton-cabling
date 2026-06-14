import newton
import numpy as np
import warp as wp

newton.use_coord_layout_targets = True
import newton.utils
from newton.solvers import SolverVBD

builder = newton.ModelBuilder(gravity=-9.81)
SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
builder.add_urdf(
    newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
    xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
    enable_self_collisions=False,
    parse_visuals_as_colliders=True,
)
for s in range(len(builder.shape_body)):
    builder.shape_flags[s] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

home = [-3.68e-03, 2.39e-02, 3.68e-03, -2.368, -1.29e-04, 2.392, 0.785]
builder.joint_q[:9] = [*home, 0.005, 0.005]
builder.joint_target_q[:9] = [*home, 0.005, 0.005]
builder.joint_target_ke[:9] = [5000.0] * 9
builder.joint_target_kd[:9] = [1.0e-2] * 9
builder.joint_limit_ke[:9] = [0.0] * 9  # disable limits entirely
builder.joint_limit_kd[:9] = [1.0e-4] * 9
builder.color()
model = builder.finalize()
newton.eval_fk(model, model.joint_q, model.joint_qd, model)

state_0, state_1 = model.state(), model.state()
control = model.control()
contacts = model.contacts()
solver = SolverVBD(model, iterations=12, rigid_contact_hard=False)

hand = next(i for i, l in enumerate(model.body_label) if l.endswith("/fr3_hand"))
tq = control.joint_target_q.numpy().copy()
dt = 1.0 / 360.0
for frame in range(720):  # 2 seconds at 360Hz substeps
    t = frame * dt
    tq[3] = home[3] + 0.4 * np.sin(2.0 * np.pi * 0.5 * t)
    control.joint_target_q.assign(tq)
    state_0.clear_forces()
    model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0
    if frame % 90 == 0:
        hq = state_0.body_q.numpy()[hand]
        print(f"t={t:4.2f} target_q3={tq[3]:+.3f} hand=({hq[0]:+.3f},{hq[1]:+.3f},{hq[2]:+.3f})")
