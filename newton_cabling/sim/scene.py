"""Build a connector rig (socket + plug + latch) from a :class:`ConnectorSpec`.

This is the spec-driven replacement for the inline rig construction the runner
scripts used to carry. Swapping RJ45 -> QSFP -> power becomes a new spec value
(plus its USD meshes) instead of editing the builder calls by hand.

The rig matches the proven layout exactly: the plug rides a world-anchored d6
(free translation, rotation soft-locked) so the grasp force-spring can drive it,
and the latch is a revolute child of the plug with a return spring and travel
limits. Contact uses an SDF ShapeConfig built from the spec.
"""

from __future__ import annotations

import dataclasses
import os

import newton
import newton.examples
import newton.usd
import numpy as np
import warp as wp
from pxr import Usd

from newton_cabling.connector import ConnectorSpec

# Repo-local asset dir for assets we author (e.g. the CAD-derived cad_rj45.usd),
# checked before falling back to Newton's bundled example assets.
_REPO_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def resolve_asset_path(name: str) -> str:
    """Locate a connector USD: absolute path, then repo assets, then Newton's bundle.

    Keeps the bundled ``rj45_plug.usd`` working (it isn't in the repo, so it falls
    through to ``newton.examples.get_asset``) while letting our own assets ship in
    ``newton_cabling/assets/``.
    """
    if os.path.isabs(name) and os.path.exists(name):
        return name
    local = os.path.join(_REPO_ASSETS, name)
    if os.path.exists(local):
        return local
    return newton.examples.get_asset(name)

Vector3 = tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class LoadedMesh:
    """A connector mesh plus the world translation of its USD prim."""

    mesh: object
    base_position: np.ndarray


@dataclasses.dataclass(frozen=True)
class ConnectorMeshes:
    socket: LoadedMesh
    plug: LoadedMesh
    latch: LoadedMesh


@dataclasses.dataclass(frozen=True)
class ConnectorRig:
    """Handles into the builder for one connector instance."""

    plug_body: int
    latch_body: int
    socket_shape: int
    plug_shape: int
    latch_shape: int

    @property
    def connector_shapes(self) -> tuple[int, int, int]:
        return (self.socket_shape, self.plug_shape, self.latch_shape)


# Real keystone patch-panel geometry (ANSI/TIA-568). A 24-port 1U panel fits the
# 19-inch rack (482.6 mm wide, 1U = 44.45 mm tall) with keystone jacks whose face
# is ~14.5 x 16 mm, placed at ~18 mm centre-to-centre pitch (≈431 mm usable width
# / 24 ports). The project's socket mesh is ~16.7 mm wide, so at this pitch the
# sockets sit ~1.3 mm apart — the realistic tight spacing of a populated panel.
PATCH_PANEL_PITCH_METERS = 0.018
RACK_UNIT_HEIGHT_METERS = 0.04445


def patch_panel_port_offsets(
    num_ports: int, *, pitch_meters: float = PATCH_PANEL_PITCH_METERS
) -> list[np.ndarray]:
    """Centred x-offsets for a single row of ``num_ports`` ports at real pitch."""
    if num_ports < 1:
        raise ValueError(f"num_ports must be >= 1, got {num_ports}")
    center = (num_ports - 1) / 2.0
    return [np.array([(index - center) * pitch_meters, 0.0, 0.0]) for index in range(num_ports)]


def _load_mesh(stage: Usd.Stage, prim_path: str, gap: float, sdf_resolution: int) -> LoadedMesh:
    prim = stage.GetPrimAtPath(prim_path)
    usd_mesh = newton.usd.get_mesh(prim, load_normals=True)
    transform = newton.usd.get_transform(prim, local=False)
    base = wp.transform_get_translation(transform)
    vertices = np.array(usd_mesh.vertices, dtype=np.float32)
    indices = np.array(usd_mesh.indices, dtype=np.int32)
    normals = np.array(usd_mesh.normals, dtype=np.float32) if usd_mesh.normals is not None else None
    mesh = newton.Mesh(vertices, indices, normals=normals)
    mesh.build_sdf(
        max_resolution=sdf_resolution,
        narrow_band_range=(-2.0 * gap, 2.0 * gap),
        margin=gap,
    )
    return LoadedMesh(mesh=mesh, base_position=np.array([base[0], base[1], base[2]]))


def load_connector_meshes(spec: ConnectorSpec) -> ConnectorMeshes:
    """Load the socket/plug/latch meshes named in the spec and build their SDFs."""
    usd_path = resolve_asset_path(spec.usd_asset_name)
    stage = Usd.Stage.Open(usd_path)
    gap = spec.contact.gap_meters
    resolution = spec.contact.sdf_max_resolution
    return ConnectorMeshes(
        socket=_load_mesh(stage, spec.socket_prim_path, gap, resolution),
        plug=_load_mesh(stage, spec.plug_prim_path, gap, resolution),
        latch=_load_mesh(stage, spec.latch_prim_path, gap, resolution),
    )


def load_fixture_mesh(
    spec: ConnectorSpec,
    *,
    asset_name: str = "jack_fixture_rj45.usd",
    sdf_resolution: int | None = None,
) -> LoadedMesh:
    """The 3D-print bench fixture, pre-baked into the socket frame by
    ``tools/cad_assets/build_jack_fixture_usd.py`` — add it to the jack body with an
    identity local transform and it sits flush around the socket mesh.

    ``sdf_resolution`` defaults to the spec's; the same voxel count over the
    fixture's ~4x larger bbox is coarser in mm, which is fine for the incidental
    pad/cable/plug contacts the fixture sees (nothing seats against it).
    """
    usd_path = resolve_asset_path(asset_name)
    stage = Usd.Stage.Open(usd_path)
    resolution = spec.contact.sdf_max_resolution if sdf_resolution is None else sdf_resolution
    return _load_mesh(stage, "/World/Fixture", spec.contact.gap_meters, resolution)


def connector_shape_config(spec: ConnectorSpec) -> object:
    """SDF ShapeConfig for the connector surfaces, from the spec's contact params."""
    return newton.ModelBuilder.ShapeConfig(
        mu=spec.contact.friction,
        ke=spec.contact.stiffness,
        kd=spec.contact.damping,
        gap=spec.contact.gap_meters,
        density=spec.contact.density,
        mu_torsional=0.0,
        mu_rolling=0.0,
    )


def add_connector_rig(
    builder: newton.ModelBuilder,
    spec: ConnectorSpec,
    meshes: ConnectorMeshes,
    *,
    socket_pos: np.ndarray,
    plug_pos: np.ndarray,
    latch_pos: np.ndarray,
    plug_anchor_pos: np.ndarray,
    lock_rotation: bool = True,
    angular_ke: float = 0.0,
    angular_kd: float = 0.0,
    plug_scale: float = 1.0,
    connector_scale: float = 1.0,
) -> ConnectorRig:
    """Add the socket/plug/latch bodies, shapes, and joints to ``builder``.

    ``plug_anchor_pos`` is where the plug's world-d6 is anchored (typically the
    plug's starting pose). The latch hinge is placed at the latch-minus-plug mesh
    offset plus the spec's ``hinge_offset_meters``.
    """
    cfg = connector_shape_config(spec)

    # connector_scale scales the WHOLE connector (socket + plug + latch) per rig about each
    # mesh origin (= the mouth, y=0, for both jack flange face and plug leading face), so the
    # cavity + plug stay concentric at the mouth and only grow/shrink. plug_scale adds the
    # per-plug FIT variation on top. The caller scales the seat reference + start distance by
    # connector_scale to match the deeper/shallower cavity.
    cs = connector_scale
    socket_shape = builder.add_shape_mesh(
        -1,
        mesh=meshes.socket.mesh,
        xform=wp.transform(wp.vec3(*socket_pos), wp.quat_identity()),
        scale=wp.vec3(cs, cs, cs),
        cfg=cfg,
        label="socket",
    )
    plug_body = builder.add_link(
        xform=wp.transform(wp.vec3(*plug_pos), wp.quat_identity()), label="plug"
    )
    ps = plug_scale * cs
    plug_shape = builder.add_shape_mesh(
        plug_body, mesh=meshes.plug.mesh, scale=wp.vec3(ps, ps, ps), cfg=cfg)
    latch_body = builder.add_link(
        xform=wp.transform(wp.vec3(*latch_pos), wp.quat_identity()), label="latch"
    )
    latch_shape = builder.add_shape_mesh(
        latch_body, mesh=meshes.latch.mesh, scale=wp.vec3(cs, cs, cs), cfg=cfg)

    joint_dof = newton.ModelBuilder.JointDofConfig
    # lock_rotation=True (default, used by the demos) gives a translation-only plug:
    # angular_axes=None locks rotation. lock_rotation=False adds 3 free angular axes for
    # a 6-DOF plug (driven by an external torque spring) — used by the RL 6-DOF env.
    angular_axes = None
    if not lock_rotation:
        # Driven angular axes (PD toward control.joint_target_q). Per the README, a d6
        # angular DRIVE can hold/move the child's orientation; a FREE angular axis
        # (ke=kd=0) is VBD-unstable (tumbles). Pass angular_ke>0 to drive it.
        angular_axes = (
            joint_dof(axis=(1.0, 0.0, 0.0), target_ke=angular_ke, target_kd=angular_kd),
            joint_dof(axis=(0.0, 1.0, 0.0), target_ke=angular_ke, target_kd=angular_kd),
            joint_dof(axis=(0.0, 0.0, 1.0), target_ke=angular_ke, target_kd=angular_kd),
        )
    world_d6 = builder.add_joint_d6(
        parent=-1,
        child=plug_body,
        linear_axes=(
            joint_dof(axis=(1.0, 0.0, 0.0)),
            joint_dof(axis=(0.0, 1.0, 0.0)),
            joint_dof(axis=(0.0, 0.0, 1.0)),
        ),
        angular_axes=angular_axes,
        parent_xform=wp.transform(wp.vec3(*plug_anchor_pos), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        custom_attributes={"vbd:joint_is_hard": 0},
    )

    latch_hinge_offset = (meshes.latch.base_position - meshes.plug.base_position) + np.asarray(
        spec.latch.hinge_offset_meters
    )
    latch_joint = builder.add_joint_revolute(
        parent=plug_body,
        child=latch_body,
        axis=spec.latch.hinge_axis,
        parent_xform=wp.transform(wp.vec3(*latch_hinge_offset), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        target_ke=spec.latch.spring_stiffness,
        target_kd=spec.latch.spring_damping,
        limit_lower=spec.latch.travel_lower_radians,
        limit_upper=spec.latch.travel_upper_radians,
        limit_kd=spec.latch.limit_damping,
        collision_filter_parent=True,
        custom_attributes={"vbd:joint_is_hard": 0},
    )
    builder.add_articulation([world_d6, latch_joint])

    return ConnectorRig(
        plug_body=plug_body,
        latch_body=latch_body,
        socket_shape=socket_shape,
        plug_shape=plug_shape,
        latch_shape=latch_shape,
    )
