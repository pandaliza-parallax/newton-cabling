"""Domain specifications for a cable connector (plug + socket + latch).

A :class:`ConnectorSpec` is a pure description: which USD prims hold the meshes
and the physical parameters of the contact and the latch hinge. The Newton scene
builder (which needs a CUDA device and so lives behind the ``sim`` extra) consumes
a spec to construct the rig, so swapping RJ45 -> QSFP -> power is a new spec value
rather than a rewrite.

Nothing here imports Newton: specs are validated and tested on any machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vector3 = tuple[float, float, float]


def _is_zero_vector(vector: Vector3) -> bool:
    return math.isclose(math.sqrt(sum(component * component for component in vector)), 0.0)


@dataclass(frozen=True)
class LatchSpec:
    """The connector's latch tab, modelled as a revolute joint plug -> latch.

    Angles follow the proven RJ45 convention: ``travel_lower`` is the most-inward
    (pressed) deflection and is <= 0, ``travel_upper`` is the outward limit and is
    >= 0, and ``press_angle`` is the actively-pressed target, between the inward
    limit and neutral.
    """

    hinge_offset_meters: Vector3
    hinge_axis: Vector3
    spring_stiffness: float
    spring_damping: float
    travel_lower_radians: float
    travel_upper_radians: float
    press_angle_radians: float
    # Joint-limit damping. Under VBD's Rayleigh convention (D = kd * limit_ke) the
    # URDF/example default of ~10 is joint-freezing; keep this small.
    limit_damping: float = 1.0e-4

    def __post_init__(self) -> None:
        if _is_zero_vector(self.hinge_axis):
            raise ValueError("latch hinge_axis must be a non-zero direction")
        if self.spring_stiffness <= 0.0:
            raise ValueError(
                f"latch spring_stiffness must be positive, got {self.spring_stiffness}"
            )
        if self.spring_damping < 0.0:
            raise ValueError(
                f"latch spring_damping must be non-negative, got {self.spring_damping}"
            )
        if not (self.travel_lower_radians <= 0.0 <= self.travel_upper_radians):
            raise ValueError(
                "latch travel must bracket neutral: "
                f"lower={self.travel_lower_radians} <= 0 <= upper={self.travel_upper_radians}"
            )
        if not (self.travel_lower_radians <= self.press_angle_radians <= 0.0):
            raise ValueError(
                f"latch press_angle {self.press_angle_radians} must be within the inward "
                f"travel range [{self.travel_lower_radians}, 0]"
            )


@dataclass(frozen=True)
class ContactSpec:
    """SDF contact parameters shared by the connector surfaces."""

    stiffness: float
    damping: float
    gap_meters: float
    friction: float
    sdf_max_resolution: int
    # Body density [kg/m^3]. The plug/latch use a very high value (~1e6) so the
    # connector is effectively rigid; this is why a compliant grasp needs the
    # gravity cancellation in GraspSpring.
    density: float = 1.0e6

    def __post_init__(self) -> None:
        if self.stiffness <= 0.0:
            raise ValueError(f"contact stiffness must be positive, got {self.stiffness}")
        if self.damping < 0.0:
            raise ValueError(f"contact damping must be non-negative, got {self.damping}")
        if self.gap_meters <= 0.0:
            raise ValueError(f"contact gap must be positive, got {self.gap_meters}")
        if self.friction < 0.0:
            raise ValueError(f"contact friction must be non-negative, got {self.friction}")
        if self.sdf_max_resolution <= 0:
            raise ValueError(f"sdf_max_resolution must be positive, got {self.sdf_max_resolution}")


@dataclass(frozen=True)
class ConnectorSpec:
    """Everything the scene builder needs to instantiate a connector + cable."""

    name: str
    usd_asset_name: str
    socket_prim_path: str
    plug_prim_path: str
    latch_prim_path: str
    latch: LatchSpec
    contact: ContactSpec
    cable_radius_meters: float
    cable_kinematic_prefix: int
    # Cable surface friction (dimensionless); higher than the connector friction
    # so a routed cable grips surfaces instead of sliding freely.
    cable_friction: float = 2.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("connector name must be non-empty")
        for label, prim_path in (
            ("socket", self.socket_prim_path),
            ("plug", self.plug_prim_path),
            ("latch", self.latch_prim_path),
        ):
            if not prim_path.startswith("/"):
                raise ValueError(
                    f"{label} prim path must be absolute (start with '/'): {prim_path!r}"
                )
        if self.cable_radius_meters <= 0.0:
            raise ValueError(f"cable radius must be positive, got {self.cable_radius_meters}")
        if self.cable_kinematic_prefix < 1:
            raise ValueError(
                f"cable_kinematic_prefix must be >= 1 (segments that follow the plug), "
                f"got {self.cable_kinematic_prefix}"
            )


def rj45_connector() -> ConnectorSpec:
    """The RJ45 connector proven across the demos, as a reusable spec value.

    Mirrors the constants from Newton's ``example_contacts_rj45_plug`` that the
    project's recordings were built on.
    """
    return ConnectorSpec(
        name="rj45",
        usd_asset_name="rj45_plug.usd",
        socket_prim_path="/World/Socket",
        plug_prim_path="/World/Plug",
        latch_prim_path="/World/Latch",
        latch=LatchSpec(
            hinge_axis=(-1.0, 0.0, 0.0),
            hinge_offset_meters=(0.0, 0.0, 0.0),
            spring_stiffness=0.15,
            spring_damping=0.2,
            travel_lower_radians=-0.2,
            travel_upper_radians=0.3,
            press_angle_radians=-0.2,
            limit_damping=1.0e-4,
        ),
        contact=ContactSpec(
            stiffness=1.0e5,
            damping=0.0,
            gap_meters=0.002,
            friction=0.0,
            sdf_max_resolution=128,
            density=1.0e6,
        ),
        cable_radius_meters=0.00325,
        cable_kinematic_prefix=4,
        cable_friction=2.0,
    )


def cad_rj45_connector(gap_meters: float = 0.00005, sdf_max_resolution: int = 512) -> ConnectorSpec:
    """Real-CAD RJ45 connector from the gs-sim-vla meshes (McMaster 9953K216 plug,
    1422N17 panel jack), as a Newton-ready merged USD built by
    ``tools/cad_assets/build_cad_rj45_usd.py`` from the real STEP tessellation.

    Same prim layout as :func:`rj45_connector` (/World/Socket,/Plug,/Latch) so the rig +
    RL env load it unchanged. /World/Plug is the real molded plug cleaned to its outer
    shell (internal contact pins/debris dropped); /World/Latch is the real latch tab split
    off it, a separate body on a revolute hinge. /World/Socket is the carved bore (the
    1422N17 contact block carved out so the full-size plug seats), a snug stepped cavity
    (body pocket + latch slot). The assembly is rolled 180 about the insertion axis so the
    latch sits on -Z. Insertion is +Y, cavity floor +12 mm past the mouth. (A clean idealized
    alternative lives in ``tools/cad_assets/build_cad_rj45_clean.py``; see ASSETS.md.)
    """
    return ConnectorSpec(
        name="cad_rj45",
        usd_asset_name="cad_rj45.usd",  # resolved from newton_cabling/assets/ first
        socket_prim_path="/World/Socket",
        plug_prim_path="/World/Plug",
        latch_prim_path="/World/Latch",
        # Latch tab springs from the plug's -Z face (the keyway side). Hinge at the tab's
        # front root (encoded in the /World/Latch prim translate by build_cad_rj45_clean.py),
        # axis +x so the free end swings in z; rests flush at 0 and flexes if pushed. It
        # stays within the jack keyway depth so it slides in without jamming; the revolute
        # joint provides the spring/flex.
        latch=LatchSpec(
            hinge_axis=(1.0, 0.0, 0.0),
            # The hinge is encoded in the latch PRIM's translation (build_cad_rj45_clean
            # authors /World/Latch with translate=hinge), so the rig's latch_hinge_offset =
            # (latch.base - plug.base) + this = hinge + 0. Keep this 0; the pivot is at the
            # tab root, with no gap.
            hinge_offset_meters=(0.0, 0.0, 0.0),
            # Moderate spring so the latch rests flush (angle 0) but FLEXES when pushed during
            # insertion, then springs back. Gravity is cancelled in apply_control, so a soft
            # spring won't droop.
            spring_stiffness=3.0,
            spring_damping=0.5,
            travel_lower_radians=-0.4,  # flex range both ways (tuned with the catch)
            travel_upper_radians=0.4,
            press_angle_radians=-0.3,
            limit_damping=1.0e-4,
        ),
        contact=ContactSpec(
            stiffness=1.0e5,
            damping=0.0,
            # Real RJ45 clearance is sub-mm, so the bundled 2mm gap would inflate the
            # plug past the cavity. Use a sub-mm gap + finer SDF to resolve the fit.
            gap_meters=gap_meters,
            friction=0.0,
            sdf_max_resolution=sdf_max_resolution,
            density=1.0e6,
        ),
        cable_radius_meters=0.00325,
        cable_kinematic_prefix=4,
        cable_friction=2.0,
    )
