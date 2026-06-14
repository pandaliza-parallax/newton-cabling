"""Behaviour tests for cable centerline routing."""

from __future__ import annotations

import numpy as np
import pytest

from newton_cabling.cable import (
    CABLE_SEGMENT_LENGTH_METERS,
    DEFAULT_BOOT_OFFSET_METERS,
    route_cable_from_boot,
)


def test_cable_starts_at_the_boot_behind_the_plug() -> None:
    plug = np.array([0.1, 0.2, 0.3])
    points = route_cable_from_boot(plug)
    expected_boot = plug + np.array([0.0, DEFAULT_BOOT_OFFSET_METERS, 0.0])
    assert np.allclose(points[0], expected_boot)


def test_segments_are_about_ten_millimetres() -> None:
    points = route_cable_from_boot(np.zeros(3))
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # Arc-length resampling makes segments close to the target spacing (they only
    # shrink slightly where the path bends through a waypoint corner).
    tolerance = 0.3 * CABLE_SEGMENT_LENGTH_METERS
    assert np.all(np.abs(seg_lengths - CABLE_SEGMENT_LENGTH_METERS) < tolerance)


def test_cable_routes_behind_and_below_the_plug() -> None:
    plug = np.zeros(3)
    points = np.array(route_cable_from_boot(plug))
    # Every point is behind the plug origin (negative y) and the tail drapes down.
    assert np.all(points[:, 1] <= 0.0)
    assert points[-1][2] < points[0][2]


def test_all_points_are_finite() -> None:
    points = np.array(route_cable_from_boot(np.array([1.0, -2.0, 0.5])))
    assert np.isfinite(points).all()


def test_translating_the_plug_translates_the_cable() -> None:
    base = np.array(route_cable_from_boot(np.zeros(3)))
    shifted = np.array(route_cable_from_boot(np.array([0.5, 0.0, 0.0])))
    assert np.allclose(shifted - base, np.array([0.5, 0.0, 0.0]))


def test_non_positive_segment_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="segment_length_meters"):
        route_cable_from_boot(np.zeros(3), segment_length_meters=0.0)
