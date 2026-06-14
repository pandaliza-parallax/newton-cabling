"""Cable centerline routing — the hard-won learnings, as reusable geometry.

Pure NumPy (no Newton/Warp), so it is unit-tested and type-checked anywhere. The
runner wraps the returned points in ``wp.vec3`` and feeds them to ``add_rod``.

Everything here is what it took to make a cable look right going into an RJ45
connector (matching Newton's own RJ45 example after inspecting it):

* **Exit the BACK of the plug (the boot).** The plug body spans y in [-43.6, +1] mm
  relative to its origin, so the boot is ~44 mm behind the origin. Starting the
  cable there (not at the origin) stops it clipping down through the plug body.
* **~10 mm segments**, like the example's authored curve.
* **Start near the gravity rest shape** — a gentle drape, not a steep drop — so the
  cable does not swing wildly on the first frame. (The example's cable rested flat
  on a table; a hanging cable needs to start already drooped, plus extra bend
  damping at the ``add_rod`` call.)
* **Keep the kinematic prefix SHORT and at the boot.** Threading the kinematic
  segments *inside* the plug body (as the example does, hidden) shoved the plug to
  the wrong pose in a multi-port spring rig — so fix only ~2 segments, at the boot,
  and let the cable flex right where it leaves the connector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Vector3 = tuple[float, float, float]

CABLE_SEGMENT_LENGTH_METERS = 0.010
DEFAULT_BOOT_OFFSET_METERS = -0.044

# Drape waypoints relative to the boot: out the back, then gently down to a tail,
# starting close to the gravity rest shape so the cable settles instead of swinging.
DEFAULT_DRAPE: tuple[Vector3, ...] = (
    (0.0, 0.000, 0.000),  # boot
    (0.0, -0.012, -0.006),  # flex down immediately
    (0.0, -0.028, -0.038),
    (0.0, -0.040, -0.085),
    (0.0, -0.044, -0.120),  # tail
)


def route_cable_from_boot(
    plug_position: Sequence[float] | np.ndarray,
    *,
    boot_offset_meters: float = DEFAULT_BOOT_OFFSET_METERS,
    segment_length_meters: float = CABLE_SEGMENT_LENGTH_METERS,
    drape: Sequence[Vector3] = DEFAULT_DRAPE,
) -> list[np.ndarray]:
    """Cable centerline points exiting the plug boot and draping down.

    ``plug_position`` is the plug body origin in world space; ``boot_offset_meters``
    places the first point at the boot (negative = behind the origin). The points
    are resampled to ~``segment_length_meters`` spacing. Returns NumPy points; the
    caller wraps them for Newton.
    """
    if segment_length_meters <= 0.0:
        raise ValueError(f"segment_length_meters must be positive, got {segment_length_meters}")
    if len(drape) < 2:
        raise ValueError("drape needs at least two waypoints")

    origin = np.asarray(plug_position, dtype=float)
    boot = origin + np.array([0.0, boot_offset_meters, 0.0])
    waypoints = np.array([boot + np.asarray(point, dtype=float) for point in drape])

    segments = np.diff(waypoints, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    total_length = float(lengths.sum())
    num_points = max(round(total_length / segment_length_meters) + 1, len(drape))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])

    points: list[np.ndarray] = []
    for distance in np.linspace(0.0, total_length, num_points):
        index = min(int(np.searchsorted(cumulative, distance, side="right")) - 1, len(lengths) - 1)
        fraction = (distance - cumulative[index]) / lengths[index]
        points.append(waypoints[index] + fraction * segments[index])
    return points
