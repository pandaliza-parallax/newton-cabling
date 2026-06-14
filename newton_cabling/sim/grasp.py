"""Force-spring coupling between a robot hand and the plug.

This is the architecture that finally made the full arm cycle work after every
joint-based grasp failed (rigid weld killed the latch click; d6 angular drives
rotated the plug 90deg; dual-parent grasp joints made the latch inert). The plug
rides Newton's proven world-anchored d6 (free translation, soft-locked rotation),
and the "grasp" is the upstream RJ45 example's force spring -- anti-gravity on the
plug and latch (their 1e6 density would otherwise pitch any compliant hold) plus a
position spring toward a target the arm updates each frame.

``strength`` is a 0..1 weight, so the timeline's grasp weight can drive it
directly: 1.0 fully grasped (the arm has the plug), 0.0 released (the spring is
off and the plug is held only by the latch).
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def _plug_grasp_spring_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    grasp_target: wp.array(dtype=wp.vec3),
    grasp_strength: wp.array(dtype=float),
    stiffness: float,
    damping: float,
    plug_index: int,
    latch_index: int,
    gravity: wp.vec3,
):
    # Anti-gravity on plug + latch always; only the cable should sag.
    anti_gravity_plug = -gravity * body_mass[plug_index]
    anti_gravity_latch = -gravity * body_mass[latch_index]
    wp.atomic_add(body_f, plug_index, wp.spatial_vector(anti_gravity_plug, wp.vec3(0.0)))
    wp.atomic_add(body_f, latch_index, wp.spatial_vector(anti_gravity_latch, wp.vec3(0.0)))

    strength = grasp_strength[0]
    if strength > 0.0:
        target = grasp_target[0]
        plug_position = wp.transform_get_translation(body_q[plug_index])
        plug_velocity = wp.spatial_top(body_qd[plug_index])
        plug_mass = body_mass[plug_index]
        force_scale = 10.0 + plug_mass
        plug_force = force_scale * (stiffness * (target - plug_position) - damping * plug_velocity)
        wp.atomic_add(body_f, plug_index, wp.spatial_vector(plug_force * strength, wp.vec3(0.0)))

        # Match the latch's acceleration to the plug so the assembly moves as one.
        latch_velocity = wp.spatial_top(body_qd[latch_index])
        latch_mass = body_mass[latch_index]
        spring_acceleration = (target - plug_position) * (force_scale * stiffness / plug_mass)
        latch_force = spring_acceleration * latch_mass - latch_velocity * (
            (10.0 + latch_mass) * damping
        )
        wp.atomic_add(body_f, latch_index, wp.spatial_vector(latch_force * strength, wp.vec3(0.0)))


class GraspSpring:
    """A toggleable position spring that couples the plug (and latch) to a target.

    Construct once after the model is finalized, then each substep: set the target
    to the arm's commanded plug pose, set the strength from the timeline's grasp
    weight, and call :meth:`apply` before stepping the solver.
    """

    def __init__(
        self,
        model: object,
        plug_body_index: int,
        latch_body_index: int,
        initial_target: tuple[float, float, float],
        *,
        stiffness: float = 50.0,
        damping: float = 10.0,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> None:
        self._model = model
        self._plug_body_index = plug_body_index
        self._latch_body_index = latch_body_index
        self._stiffness = stiffness
        self._damping = damping
        self._gravity = wp.vec3(*gravity)
        device = model.device
        self._target = wp.array([wp.vec3(*initial_target)], dtype=wp.vec3, device=device)
        self._strength = wp.array([1.0], dtype=float, device=device)

    def set_target(self, position: tuple[float, float, float]) -> None:
        """Aim the spring at ``position`` (the arm's commanded plug pose)."""
        self._target.assign([wp.vec3(*position)])

    def set_strength(self, weight: float) -> None:
        """Scale the spring by a 0..1 grasp weight (e.g. from the timeline)."""
        self._strength.assign([float(weight)])

    def engage(self) -> None:
        self.set_strength(1.0)

    def release(self) -> None:
        self.set_strength(0.0)

    def apply(self, state: object) -> None:
        """Add the grasp + anti-gravity forces to ``state`` for this substep."""
        wp.launch(
            _plug_grasp_spring_kernel,
            dim=1,
            inputs=(
                state.body_q,
                state.body_qd,
                state.body_f,
                self._model.body_mass,
                self._target,
                self._strength,
                self._stiffness,
                self._damping,
                self._plug_body_index,
                self._latch_body_index,
                self._gravity,
            ),
            device=self._model.device,
        )
