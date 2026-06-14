"""Behaviour tests for connector specification validation."""

from __future__ import annotations

import dataclasses

import pytest

from newton_cabling.connector import LatchSpec, rj45_connector


def test_rj45_spec_is_valid() -> None:
    spec = rj45_connector()
    assert spec.name == "rj45"
    assert spec.cable_kinematic_prefix >= 1


def test_press_angle_outside_inward_travel_is_rejected() -> None:
    valid = rj45_connector().latch
    with pytest.raises(ValueError, match="press_angle"):
        dataclasses.replace(valid, press_angle_radians=0.5)


def test_travel_not_bracketing_neutral_is_rejected() -> None:
    valid = rj45_connector().latch
    with pytest.raises(ValueError, match="bracket neutral"):
        dataclasses.replace(valid, travel_lower_radians=0.1)


def test_zero_hinge_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero direction"):
        LatchSpec(
            hinge_axis=(0.0, 0.0, 0.0),
            hinge_offset_meters=(0.0, 0.0, 0.0),
            spring_stiffness=0.15,
            spring_damping=0.2,
            travel_lower_radians=-0.2,
            travel_upper_radians=0.3,
            press_angle_radians=-0.2,
        )


def test_relative_prim_path_is_rejected() -> None:
    spec = rj45_connector()
    with pytest.raises(ValueError, match="absolute"):
        dataclasses.replace(spec, socket_prim_path="World/Socket")


def test_zero_kinematic_prefix_is_rejected() -> None:
    spec = rj45_connector()
    with pytest.raises(ValueError, match="cable_kinematic_prefix"):
        dataclasses.replace(spec, cable_kinematic_prefix=0)
