"""Batched ``(n, 4)`` quaternion helpers in pure NumPy.

SciPy is deliberately avoided here. It only arrives with the ``sim`` extra (via
``newton[examples]``), while this project's declared dependency is NumPy alone —
so keeping the scripted controller SciPy-free is what lets it lint, type-check
and unit-test on a machine with no GPU and no Newton install.

Convention matches Newton / Warp / SciPy: ``q = (x, y, z, w)``, and
``quat_multiply(a, b)`` composes so that ``b`` is applied first, then ``a``.
All functions broadcast over leading axes; the last axis is the component axis.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quat_conjugate",
    "quat_multiply",
    "quat_normalize",
    "quat_rotate",
    "quat_to_rotvec",
    "rotate_by_rotvec",
    "rotvec_to_quat",
]

_EPS = 1e-12


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return `q` scaled to unit length."""
    q = np.asarray(q, dtype=np.float64)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), _EPS)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of `q`, which for a unit quaternion is its inverse."""
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ∘ b`` — the rotation that applies `b` first, then `a`."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Log map: the axis-angle vector of `q`, taking the SHORT way round.

    The ``w < 0`` sign flip matters: a quaternion and its negation are the same
    rotation, but their naive log maps differ by a full turn, so without the flip
    an error of +1 degree can read as -359 and a servo built on it would spin the
    wrong way.
    """
    q = np.asarray(q, dtype=np.float64)
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    xyz = q[..., :3]
    w = np.clip(q[..., 3], -1.0, 1.0)
    s = np.linalg.norm(xyz, axis=-1)
    small = s < 1e-8
    # angle/sin(angle/2) -> 2/w as s -> 0; the guarded divisors keep NaNs out of
    # the unused branch (np.where evaluates both).
    scale = np.where(
        small,
        2.0 / np.where(np.abs(w) < _EPS, 1.0, w),
        2.0 * np.arctan2(np.where(small, 1.0, s), w) / np.where(small, 1.0, s),
    )
    return xyz * scale[..., None]


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """Exp map: the unit quaternion of axis-angle vector `v`."""
    v = np.asarray(v, dtype=np.float64)
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    small = theta < 1e-8
    # sin(theta/2)/theta, with its Taylor limit 1/2 - theta^2/48 near zero
    s = np.where(
        small,
        0.5 - theta**2 / 48.0,
        np.sin(0.5 * theta) / np.where(small, 1.0, theta),
    )
    return np.concatenate([v * s, np.cos(0.5 * theta)], axis=-1)


def rotate_by_rotvec(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rotate vectors `x` by the rotation whose axis-angle vector is `v` (Rodrigues)."""
    v = np.asarray(v, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    small = theta < _EPS
    axis = np.where(small, 0.0, v / np.where(small, 1.0, theta))
    c, s = np.cos(theta), np.sin(theta)
    return (
        x * c + np.cross(axis, x) * s + axis * np.sum(axis * x, axis=-1, keepdims=True) * (1.0 - c)
    )


def quat_rotate(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rotate vectors `x` by unit quaternion `q`."""
    q = np.asarray(q, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    u, w = q[..., :3], q[..., 3:4]
    return x + 2.0 * np.cross(u, np.cross(u, x) + w * x)
