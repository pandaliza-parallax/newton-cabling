"""Bridge-packaging tests for the GS render side-channel.

Pure numpy: no Newton, no posix_ipc, no renderer container. Exercises the quaternion
reorder and the SETUP/UPDATE package shapes via ``dry_run=True``.
"""

from __future__ import annotations

import numpy as np
import pytest

from newton_cabling.render.gs_bridge import (
    NewtonGSClient,
    look_at_quat,
    make_intrinsics,
    newton_pose,
)


def test_newton_pose_reorders_quat_scalar_first() -> None:
    # Warp/Newton row: [px, py, pz, qx, qy, qz, qw]
    row = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]])
    pos, quat = newton_pose(row, 0)
    assert pos == [1.0, 2.0, 3.0]
    # scalar-first [w, x, y, z]
    assert quat == [0.9, 0.1, 0.2, 0.3]


def _client(**kw) -> NewtonGSClient:
    K = make_intrinsics(64, 48, 55.0)
    quat = look_at_quat([0.1, -0.1, 0.1], [0, 0, 0])
    return NewtonGSClient(
        ply_paths=["/c/mount.ply", "/c/cord.ply"],
        cam_K=K,
        cam_pos=[0.1, -0.1, 0.1],
        cam_quat=quat,
        dry_run=True,
        **kw,
    )


def test_setup_registers_plys_in_order() -> None:
    c = _client()
    s = c.last_setup
    assert s["package_type"] == "SETUP"
    assert s["point_clouds"] == ["/c/mount.ply", "/c/cord.ply"]
    assert s["transforms_to_world"] == [False, False]
    assert s["MULTI_ENV_SIM"] is True
    # one (pos, quat) seed per ply, per env
    assert len(s["PER_ENV_TRANSFORM"]["env_0"]) == 2


def test_setup_with_background_prepends_bg() -> None:
    c = _client(bg_ply="/c/bg.ply")
    assert c.last_setup["point_clouds"] == ["/c/bg.ply", "/c/mount.ply", "/c/cord.ply"]
    assert c.last_setup["transforms_to_world"] == [False, False, False]


def test_render_builds_update_and_returns_black_frame() -> None:
    c = _client()
    socket = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    plug = ([0.0, -0.025, 0.0], [1.0, 0.0, 0.0, 0.0])
    img = c.render([socket, plug])
    assert img.shape == (48, 64, 3)  # H, W from K principal point * 2
    u = c.last_update
    assert u["package_type"] == "UPDATE"
    entries = u["PER_ENV_TRANSFORM"]["env_0"]
    assert entries[0] == ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert entries[1] == ([0.0, -0.025, 0.0], [1.0, 0.0, 0.0, 0.0])
    cam = u["PER_ENV_CAMERAS_DATA_LIST"]["env_0"][0]
    assert cam["pos"] == [0.1, -0.1, 0.1]
    assert "cam_k" in cam and "quat" in cam


def test_render_with_bg_includes_bg_entry_first() -> None:
    c = _client(bg_ply="/c/bg.ply")
    socket = ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    plug = ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    c.render([socket, plug])
    entries = c.last_update["PER_ENV_TRANSFORM"]["env_0"]
    assert len(entries) == 3  # bg + 2 objects
    assert entries[0] == ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])  # bg default pose


def test_render_rejects_wrong_object_count() -> None:
    c = _client()
    with pytest.raises(ValueError):
        c.render([([0, 0, 0], [1, 0, 0, 0])])  # only 1, expected 2


def test_look_at_quat_is_unit() -> None:
    q = look_at_quat([0.2, -0.2, 0.2], [0.0, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(q), 1.0)
