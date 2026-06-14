"""Footgun-free Newton VBD setup for actuated robots.

Every helper here bakes in a SolverVBD constraint that cost real debugging time
to discover (see the project README's "VBD + robot arm learnings"). Used together
they turn the multi-day "why won't the arm move / why is the plug rotated 90deg /
why does the latch never deflect" investigation into a couple of obviously-named
calls:

    builder = new_vbd_builder()
    add_actuated_urdf(builder, urdf_path, base_xform)
    ...                       # add shapes, joints, cable
    model = finalize_for_vbd(builder)

The individual fixes are upstreamable to Newton as a "VBD robot quickstart".
"""

from __future__ import annotations

import newton
from newton.solvers import SolverVBD

# Joint types whose relative motion is driven by control targets. Under VBD these
# must be in *soft* mode; hard mode (the default for non-cable structural joints)
# locks all relative motion, which silently freezes every actuated joint.
_ACTUATED_JOINT_TYPES: tuple[int, ...] = (
    int(newton.JointType.REVOLUTE),
    int(newton.JointType.PRISMATIC),
)


def new_vbd_builder(*, gravity: float = -9.81) -> newton.ModelBuilder:
    """A model builder with the VBD custom attributes already registered.

    ``register_custom_attributes`` must run before any shapes are added so the
    per-joint ``vbd:joint_is_hard`` slots exist; doing it here removes one easy
    way to forget it.
    """
    builder = newton.ModelBuilder(gravity=gravity)
    SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
    return builder


def add_actuated_urdf(
    builder: newton.ModelBuilder,
    urdf_path: object,
    base_xform: object,
    *,
    parse_visuals_as_colliders: bool = True,
) -> None:
    """Import a robot URDF in a form SolverVBD can actually drive.

    ``collapse_fixed_joints=True`` is mandatory: a URDF imported with its
    interleaved fixed joints (world->base, link->hand, ...) does not move at all
    under SolverVBD -- no joint drives, not even gravity -- because the fixed
    joints freeze the chain. Collapsing them was the single change that unfroze
    the Franka arm.
    """
    builder.add_urdf(
        urdf_path,
        xform=base_xform,
        enable_self_collisions=False,
        parse_visuals_as_colliders=parse_visuals_as_colliders,
        collapse_fixed_joints=True,
    )


def find_body_index(body_labels: list[str], label_suffix: str) -> int:
    """Index of the single body whose label ends with ``label_suffix``.

    URDF import prefixes body labels (e.g. ``fr3/fr3_link7``); callers match on
    the trailing segment. Raises ``StopIteration`` if no body matches, which is a
    programming error (wrong link name) rather than a recoverable outcome.
    """
    return next(index for index, label in enumerate(body_labels) if label.endswith(label_suffix))


def finalize_for_vbd(builder: newton.ModelBuilder) -> newton.Model:
    """Colour, finalize, soften actuated joints, and bake the VBD rest pose.

    The ordering matters and is the whole point of this function:

    1. ``builder.color()`` -- SolverVBD requires per-body graph colouring, and
       ``finalize()`` does not do it implicitly.
    2. ``finalize()`` -- builds the model.
    3. soften actuated joints -- ``model.vbd.joint_is_hard`` only exists after
       finalize, so this cannot be authored on the builder.
    4. ``eval_fk`` into the *model* -- VBD measures joint angles relative to
       ``model.body_q`` offset by ``model.joint_q``; baking the rest pose keeps
       the two consistent so the solver does not see a bogus initial deflection.
    """
    builder.color()
    model = builder.finalize()
    _soften_actuated_joints(model)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    return model


def _soften_actuated_joints(model: newton.Model) -> None:
    if not hasattr(model, "vbd"):
        raise RuntimeError(
            "model has no `vbd` attributes; build it with new_vbd_builder() (which calls "
            "SolverVBD.register_custom_attributes) before finalizing"
        )
    joint_is_hard = model.vbd.joint_is_hard.numpy()
    joint_types = model.joint_type.numpy()
    for joint_index in range(model.joint_count):
        if int(joint_types[joint_index]) in _ACTUATED_JOINT_TYPES:
            joint_is_hard[joint_index] = 0
    model.vbd.joint_is_hard.assign(joint_is_hard)
