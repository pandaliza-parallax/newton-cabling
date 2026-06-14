import newton
import warp as wp

newton.use_coord_layout_targets = True
import newton.utils

builder = newton.ModelBuilder()
from newton.solvers import SolverVBD

SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
builder.add_urdf(
    newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
    xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
    enable_self_collisions=False,
    parse_visuals_as_colliders=True,
)
model = builder.finalize()
flags = model.body_flags.numpy()
inv_mass = model.body_inv_mass.numpy()
for i, lbl in enumerate(model.body_label):
    print(f"body {i:2d} {lbl:50s} flags={flags[i]:#06x} inv_mass={inv_mass[i]:.4f}")
print("joint_type:", model.joint_type.numpy().tolist())
print("joint_target_ke:", model.joint_target_ke.numpy()[:12].tolist())
print("BodyFlags:", {f.name: hex(f.value) for f in newton.BodyFlags})
