"""Import the StandardBots RO1 (arm + gripper) into a Newton model.

The RO1 ships as a USD (``robo_maker/sbot/assets/standardbot.usd``) rather than a
URDF, so this is the ``add_usd`` analogue of :func:`safe_vbd.add_actuated_urdf`.
The asset is authored Z-up in centimetres (``metersPerUnit = 0.01``); Newton's
``add_usd`` reads that and scales to metres, matching the env's Z-up/metres
convention -- no manual rescaling needed.

What the loader bakes in (each line was verified by loading the real asset):

* ``collapse_fixed_joints=True`` -- the same fix that unfroze the Franka under
  SolverVBD (see ``safe_vbd``). It also folds away the ``root_joint`` and the
  ``wrist3_gripper`` fixed joint, so the gripper base merges into ``wrist_3_link``.
* ``override_root_xform=True`` (with an explicit ``xform``) -- drops the base
  translation baked into the USD so the base lands exactly at ``base_xform``
  instead of ~0.83 m off in x.
* ``enable_self_collisions=False`` -- the gripper's fingers overlap their own
  knuckles at rest; self-collision would fight the solver.

Gripper: the RO1 carries a DH Robotics AG-145 -- a 1-DOF parallelogram gripper.
USD has no mimic-joint concept, so its eight finger joints import as *independent*
revolutes (the URDF mimic coupling is lost). :data:`GRIPPER_COUPLING` rebuilds it:
every finger joint is a fixed ratio of the single driver angle ``theta`` (on
``gripper_finger1_joint``). The ratios come from the AG-145 URDF
(``dh_gripper_ros/.../dh_robotics_ag145.urdf``: driver -> finger 0.5,
inner_knuckle 1.49, finger_tip mimics inner_knuckle; the finger_tip's ``0 -1 0``
axis plus the USD's driver-axis flip make the tip ratio negative here) and were
verified in Newton to keep the finger pads parallel (<0.4 deg drift) and the jaws
symmetric across the whole stroke. Use :func:`set_gripper` to drive all eight
joints from one ``theta``.

Because the joints are independent (no enforced loop closure), PD-tracking the
coupled targets is faithful for posing / rendering / data-gen but does NOT give a
force-closure grip under contact -- for that, add equality/gear constraints
between the driver and the mimic joints.
"""

from __future__ import annotations

import dataclasses
import pathlib

import newton
import warp as wp

# robo_maker lives beside newton-cabling under the parallax workspace root.
# parents[2] == the newton-cabling repo root; its parent is the workspace.
_WORKSPACE = pathlib.Path(__file__).resolve().parents[2].parent
SBOT_USD = _WORKSPACE / "robo_maker" / "sbot" / "assets" / "standardbot.usd"
SBOT_NO_GRIPPER_USD = _WORKSPACE / "robo_maker" / "sbot" / "assets" / "standardbot_no_gripper.usd"

# The six arm DOFs, in chain order (base -> wrist). Match by label suffix so the
# loader is robust to whatever the builder index happens to be.
ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint0",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
)

# AG-145 1-DOF coupling: every finger joint = ratio * driver angle (theta). The
# driver is gripper_finger1_joint (ratio 1.0). See the module docstring for the
# URDF provenance; the finger_tip ratios are negative due to the tip joint's
# flipped axis. Verified in Newton to keep the pads parallel and the jaws symmetric
# across the stroke. The first entry is the driver.
GRIPPER_DRIVER_JOINT = "gripper_finger1_joint"
GRIPPER_COUPLING: tuple[tuple[str, float], ...] = (
    ("gripper_finger1_joint", 1.0),
    ("gripper_finger1_finger_joint", 0.5),
    ("gripper_finger1_inner_knuckle_joint", 1.49),
    ("gripper_finger1_finger_tip_joint", -1.49),
    ("gripper_finger2_joint", 1.0),
    ("gripper_finger2_finger_joint", 0.5),
    ("gripper_finger2_inner_knuckle_joint", 1.49),
    ("gripper_finger2_finger_tip_joint", -1.49),
)

# Driver angle (theta on gripper_finger1_joint) at the jaw extremes. theta = 0 is
# the USD rest pose with the pads touching (closed); theta = -0.6 opens the jaws to
# ~145 mm (the AG-145's stroke). USD negated the URDF's 0..0.6 driver range.
GRIPPER_THETA_CLOSED = 0.0
GRIPPER_THETA_OPEN = -0.6

# The wrist link the gripper is welded onto (gripper_base_link collapses into it).
WRIST_BODY = "wrist_3_link"

# The six gripper links whose pad meshes actually contact a grasped object (both finger
# pads, both tips, both inner knuckles). Used by enable_finger_contact to turn their
# collision back on (add_usd imports the whole gripper with COLLIDE_SHAPES OFF).
FINGER_LINK_BODIES: tuple[str, ...] = (
    "gripper_finger1_finger_link",
    "gripper_finger1_finger_tip_link",
    "gripper_finger2_finger_link",
    "gripper_finger2_finger_tip_link",
    "gripper_finger1_inner_knuckle_link",
    "gripper_finger2_inner_knuckle_link",
)

# A gravity-loaded "ready" home: joint1 pitches the arm forward off vertical and
# joint2 folds the elbow, so the wrist reaches out ahead of the base. Unlike the
# q=0 rest pose (arm straight up, gravity along the arm), this actually loads the
# shoulder/elbow drives -- a real test that the PD gains hold the arm up.
SBOT_HOME: tuple[float, ...] = (0.0, 1.0, -1.2, 0.0, -0.4, 0.0)


@dataclasses.dataclass(frozen=True)
class SbotHandles:
    """Indices into the builder for one StandardBots instance.

    ``arm_joints`` are the six arm DOF indices. ``gripper_coupling`` pairs each of
    the eight gripper joint indices with its :data:`GRIPPER_COUPLING` ratio, so a
    runtime loop can set ``target_q[idx] = ratio * theta`` directly. All RO1 joints
    are 1-DOF revolutes, so a joint index doubles as its ``joint_q`` /
    ``joint_target_q`` index. ``wrist_body`` indexes ``body_q`` for the end frame.
    """

    arm_joints: tuple[int, ...]
    gripper_coupling: tuple[tuple[int, float], ...]
    wrist_body: int

    @property
    def gripper_joints(self) -> tuple[int, ...]:
        return tuple(idx for idx, _ in self.gripper_coupling)

    @property
    def all_actuated_joints(self) -> tuple[int, ...]:
        return self.arm_joints + self.gripper_joints


def _find_joint(builder: newton.ModelBuilder, suffix: str) -> int:
    return next(i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(suffix))


def _find_body(builder: newton.ModelBuilder, suffix: str) -> int:
    return next(i for i, lbl in enumerate(builder.body_label) if lbl.endswith(suffix))


def add_sbot(
    builder: newton.ModelBuilder,
    base_xform: wp.transform,
    *,
    usd_path: str | pathlib.Path | None = None,
    with_gripper: bool = True,
) -> SbotHandles:
    """Import the RO1 into ``builder`` at ``base_xform`` and return its handles.

    Pass ``with_gripper=False`` to load the 6-DOF arm only (``gripper_*`` handle
    tuples come back empty). ``usd_path`` overrides the bundled asset.
    """
    if usd_path is None:
        usd_path = SBOT_USD if with_gripper else SBOT_NO_GRIPPER_USD
    usd_path = pathlib.Path(usd_path)
    if not usd_path.exists():
        raise FileNotFoundError(f"StandardBots USD not found: {usd_path}")

    builder.add_usd(
        str(usd_path),
        xform=base_xform,
        collapse_fixed_joints=True,
        override_root_xform=True,
        enable_self_collisions=False,
    )

    arm = tuple(_find_joint(builder, name) for name in ARM_JOINT_NAMES)
    if with_gripper:
        coupling = tuple(
            (_find_joint(builder, name), ratio) for name, ratio in GRIPPER_COUPLING
        )
    else:
        coupling = ()
    return SbotHandles(
        arm_joints=arm,
        gripper_coupling=coupling,
        wrist_body=_find_body(builder, WRIST_BODY),
    )


def set_pd_gains(
    builder: newton.ModelBuilder,
    handles: SbotHandles,
    *,
    arm_ke: float = 6000.0,
    arm_kd: float = 60.0,
    gripper_ke: float = 200.0,
    gripper_kd: float = 5.0,
) -> None:
    """Replace the USD's huge drive gains with PD gains tuned for this env.

    The USD authors per-joint stiffness ~1e7; importing that raw makes the arm
    explosively stiff under VBD. This sets the arm and all eight gripper DOFs to
    moderate position-PD gains instead. Every gripper joint is PD-driven so it
    tracks its coupled target (the imported linkage has no enforced loop closure).
    """
    for j in handles.arm_joints:
        builder.joint_target_ke[j] = arm_ke
        builder.joint_target_kd[j] = arm_kd
    for j in handles.gripper_joints:
        builder.joint_target_ke[j] = gripper_ke
        builder.joint_target_kd[j] = gripper_kd


def set_arm_home(
    builder: newton.ModelBuilder,
    handles: SbotHandles,
    home: tuple[float, ...] = SBOT_HOME,
) -> None:
    """Set the arm's start coordinates and PD targets to ``home`` (6 values)."""
    if len(home) != len(handles.arm_joints):
        raise ValueError(f"home must have {len(handles.arm_joints)} values, got {len(home)}")
    for j, q in zip(handles.arm_joints, home, strict=True):
        builder.joint_q[j] = q
        builder.joint_target_q[j] = q


def gripper_theta_for_opening(fraction: float) -> float:
    """Map a normalized opening (0 = closed, 1 = open) to the driver angle theta."""
    fraction = min(1.0, max(0.0, fraction))
    return GRIPPER_THETA_CLOSED + (GRIPPER_THETA_OPEN - GRIPPER_THETA_CLOSED) * fraction


def set_gripper(
    builder: newton.ModelBuilder,
    handles: SbotHandles,
    theta: float,
    *,
    set_q: bool = True,
) -> None:
    """Drive the whole AG-145 linkage from one driver angle ``theta``.

    Sets every gripper joint's PD target to ``ratio * theta`` via
    :data:`GRIPPER_COUPLING` (``theta`` in ``GRIPPER_THETA_OPEN .. _CLOSED``; use
    :func:`gripper_theta_for_opening` for a normalized fraction). ``set_q=True``
    also seeds ``joint_q`` so the gripper *starts* in this pose; pass ``set_q=False``
    to issue a target without teleporting the joints. For per-frame control in a
    loop, iterate ``handles.gripper_coupling`` and write your target array directly.
    """
    for idx, ratio in handles.gripper_coupling:
        builder.joint_target_q[idx] = ratio * theta
        if set_q:
            builder.joint_q[idx] = ratio * theta


# AG-145 physical-grasp calibration (verified for the cad_rj45 plug at the pendant home pose).
# See the sbot-friction-grasp-fix memory.
# Updated 2026-07-17: grip the plug by its CABLE TAIL (~24 mm behind the face, ~5.2 mm wide) so the
# whole connector HEAD protrudes ~24 mm below the fingertips and hangs from the cable — realistic,
# and keeps the fingers well clear of the jack at full insertion (fingertips end ~12 mm ABOVE the
# jack mouth). Otherwise the plug is held at its face and the fingers travel down onto the jack (the
# "jack inside the gripper" overlap) AND jam into the cavity at reset (the decoupled ejection bug).
# Thin cable -> tight theta. Grips + holds (drift ~2 mm). Prior: body -0.028/0.190, boot -0.014/0.210.
# TRADE-OFF: the plug hangs on a long lever from the tail, so it's more compliant (may tilt at the seat).
GRIPPER_THETA_GRASP = -0.007   # ~3.7 mm pad gap for the ~5.2 mm cable (~1.5 mm squeeze)
GRASP_FACE_DIST = 0.215        # leading face 0.215 m from wrist -> fingers grip the cable tail


def enable_finger_contact(
    builder: newton.ModelBuilder,
    *,
    mu: float = 2.0,
    sdf_max_resolution: int = 96,
    narrow_band: float = 0.012,
    margin: float = 0.004,
) -> list[int]:
    """Make the AG-145 finger pad meshes collidable + frictional for a physical grasp.

    ``add_usd`` (imported with ``enable_self_collisions=False``) brings the whole gripper in
    with ``COLLIDE_SHAPES`` OFF, so a friction grasp gets ZERO finger<->object contacts until
    this re-enables them. For each MESH shape on a :data:`FINGER_LINK_BODIES` link this builds
    an SDF (needed for mesh-vs-mesh narrow phase), sets its friction ``mu``, and turns
    ``COLLIDE_SHAPES`` back on. Returns the finger shape indices. Call after ``add_sbot`` and
    before ``finalize``; it does NOT touch any other shape's flags (the caller decides whether
    to disable the rest of the robot's collision, e.g. for the kinematic-arm grasp).
    """
    # ALL finger-link bodies (every arm instance in a batched builder, not just the first).
    finger_bodies = {
        i for i, lbl in enumerate(builder.body_label)
        if any(lbl.endswith(name) for name in FINGER_LINK_BODIES)
    }
    finger_shapes: list[int] = []
    for s in range(len(builder.shape_body)):
        if builder.shape_body[s] in finger_bodies and builder.shape_type[s] == int(
            newton.GeoType.MESH
        ):
            mesh = builder.shape_source[s]
            if mesh is not None and getattr(mesh, "sdf", None) is None:
                mesh.build_sdf(
                    max_resolution=sdf_max_resolution,
                    narrow_band_range=(-narrow_band, narrow_band),
                    margin=margin,
                )
            builder.shape_material_mu[s] = mu
            builder.shape_flags[s] |= int(newton.ShapeFlags.COLLIDE_SHAPES)
            finger_shapes.append(s)
    return finger_shapes
