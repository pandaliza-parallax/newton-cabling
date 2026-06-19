"""Drive DalusPySim's Gaussian-Splat renderer from a Newton sim over shared memory.

This is the Newton counterpart of gs-sim-vla's ``envs/gs_renderer.py`` (which drives
the same renderer from an Isaac/PhysX scene). The renderer is physics-engine-agnostic:
it consumes per-object world transforms + a camera and returns rendered RGB, so the
only Newton-specific work is pulling body poses out of ``State.body_q`` and reordering
the quaternion.

Pipeline (matches the proven Isaac path exactly):

    Newton sim (host)  ──SETUP/UPDATE over POSIX SHM──▶  parallax_sim renderer (container)
            │                                                      │
            │◀──────────────── rendered RGB ───────────────────────┘

Conventions (strict, must match the renderer):

* Quaternions are **scalar-first** ``[w, x, y, z]``. Newton/Warp ``wp.transform`` stores
  ``[x, y, z, w]`` (scalar-last) — :func:`newton_pose` does the reorder.
* Positions are **env-local metres**. Single env ⇒ env-local == world.
* Object order in every UPDATE must match ``ply_paths`` order in the SETUP.

Requirements at run time:

* ``posix_ipc`` installed in this venv (``newton-cabling[gs]`` extra).
* ``parallax_sim`` (the DalusPySim package) importable — add its repo root to
  ``PYTHONPATH``, e.g.
  ``/root/parallax/DataGenerator/sim_engine/DalusPySim`` (container) or the host path.
* The ``parallax_sim`` renderer container running (``docker compose up -d``) with
  ``ipc: host`` so the host process and the container share ``/dev/shm``.
* ``ply_paths`` must be valid **inside the renderer container** (e.g. under
  ``/root/parallax/...``), since the renderer is what opens the files.

Set ``dry_run=True`` to exercise the packaging logic without ``posix_ipc``, the
container, or a GPU — useful for tests. It records the last package on ``.last_setup``
/ ``.last_update`` and returns a black image.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Sequence

import numpy as np

Pose = tuple[Sequence[float], Sequence[float]]  # (pos[xyz], quat[wxyz])


# ── camera helpers (numpy; mirror gs-sim-vla/envs/gs_renderer.py) ───────────────
def make_intrinsics(width: int, height: int, fov_deg: float = 55.0) -> np.ndarray:
    """Pinhole 3x3 K from a horizontal FOV."""
    f = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]], np.float64)


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0), convention: str = "ros") -> list[float]:
    """Camera orientation quaternion ``[w, x, y, z]`` for an eye looking at target.

    ``convention='ros'`` is the renderer's optical frame (+z forward into the scene,
    +x right, +y down); ``'opengl'`` is -z forward, +y up.
    """
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-9
    if convention == "opengl":
        true_up = np.cross(right, fwd)
        R = np.stack([right, true_up, -fwd], axis=1)
    else:  # ros optical
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd], axis=1)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return q.tolist()


# ── Newton pose extraction ──────────────────────────────────────────────────────
def newton_pose(body_q_np: np.ndarray, body_index: int) -> Pose:
    """``(pos[xyz], quat[wxyz])`` for one body from a ``state.body_q.numpy()`` array.

    Newton rows are ``[px, py, pz, qx, qy, qz, qw]`` (Warp scalar-last); this returns
    the position and the scalar-first quaternion the renderer expects.
    """
    row = np.asarray(body_q_np[body_index], dtype=np.float64)
    pos = row[0:3].tolist()
    qx, qy, qz, qw = row[3], row[4], row[5], row[6]
    return pos, [qw, qx, qy, qz]


class NewtonGSClient:
    """SHM client that registers splats once, then renders per Newton step.

    Parameters
    ----------
    ply_paths
        Splat files **as seen by the renderer container**, in a fixed order. Every
        :meth:`render` call supplies one ``(pos, quat)`` per path, same order.
    cam_K
        3x3 intrinsics (see :func:`make_intrinsics`).
    cam_pos, cam_quat
        Camera world pose: position ``[x, y, z]`` and orientation ``[w, x, y, z]``
        (see :func:`look_at_quat`). Shared across envs (single env here).
    bg_ply
        Optional background splat, rendered first at a fixed pose.
    dry_run
        Build packages but do not touch ``posix_ipc`` / the container; render returns
        a black image. For tests on a box without the renderer.
    """

    def __init__(
        self,
        ply_paths: Sequence[str],
        cam_K: np.ndarray,
        cam_pos: Sequence[float],
        cam_quat: Sequence[float],
        *,
        num_envs: int = 1,
        bg_ply: str | None = None,
        bg_pose: Pose = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        show_viewer: bool = False,
        setup_timeout: float = 300.0,
        dry_run: bool = False,
    ):
        self.num_envs = int(num_envs)
        self.env_keys = [f"env_{i}" for i in range(self.num_envs)]
        self.ply_paths = list(ply_paths)
        self.n_obj = len(self.ply_paths)
        self.has_bg = bg_ply is not None
        self.bg_pose = bg_pose
        self.cam_K = np.asarray(cam_K, np.float64)
        self.cam_pos = [float(v) for v in cam_pos]
        self.cam_quat = [float(v) for v in cam_quat]
        self.dry_run = dry_run
        self.last_setup: dict | None = None
        self.last_update: dict | None = None

        # Lazy import so dry_run / tests work without parallax_sim or posix_ipc.
        if not dry_run:
            try:
                from parallax_sim.data_sender.ipc_sender.ipc_receiver import IPCReceiver
                from parallax_sim.data_sender.manager import SenderManager
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "Could not import parallax_sim (the DalusPySim package). Add its "
                    "repo root to PYTHONPATH and install posix_ipc "
                    "(`uv pip install posix_ipc`). Underlying error: " + str(e)
                ) from e
            self._send = SenderManager()
            self._recv = IPCReceiver()

        # Package-type / key strings are a strict contract; hard-code them so the
        # client needs no parallax_sim import in dry_run.
        self._K = dict(
            PKG="package_type",
            PLY="point_clouds",
            TTW="transforms_to_world",
            CAM_K="cam_k",
            CAM_POS="pos",
            CAM_QUAT="quat",
            VIEW="show_viewer",
            RECV="receive_rendered_rgbs",
        )

        full_plys = ([bg_ply] if self.has_bg else []) + self.ply_paths
        ttw = [False] * len(full_plys)
        init_T = {
            k: [([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]) for _ in full_plys] for k in self.env_keys
        }
        pid = secrets.token_hex(4)
        K = self._K
        setup = {
            K["PKG"]: "SETUP",
            K["PLY"]: full_plys,
            "PER_ENV_TRANSFORM": init_T,
            K["TTW"]: ttw,
            "GS_SCALE": [None] * len(full_plys),
            K["VIEW"]: bool(show_viewer),
            K["RECV"]: True,
            "MULTI_ENV_SIM": True,
            "pckg_id": pid,
        }
        self.last_setup = setup
        if dry_run:
            print(f"[gs:dry] SETUP {len(full_plys)} splats, {self.num_envs} env(s)")
            return

        print(
            f"[gs] SETUP: registering {len(full_plys)} splats "
            f"({'bg+' if self.has_bg else ''}{self.n_obj} objects); waiting up to "
            f"{setup_timeout:.0f}s for the parallax_sim handshake..."
        )
        # Resend SETUP until acked. A FRESHLY started renderer creates its return-SHM
        # sender on-demand while processing this very SETUP and sends the ack in that
        # same step; a receiver attached before the sender existed misses that one ack
        # and the renderer never resends it. Re-sending drives the handshake to
        # completion (a 2nd SETUP with a matching splat count is a cheap no-op on the
        # renderer that still echoes the pckg_id). Idempotent: same pid every attempt.
        t0 = time.time()
        attempt = 0
        while time.time() - t0 < setup_timeout:
            attempt += 1
            self._send.setup(setup)
            if self._wait_ack(pid, timeout=8.0):
                print(f"[gs] handshake OK (attempt {attempt}).")
                break
            print(f"[gs] no ack yet (attempt {attempt}); resending SETUP...")
        else:
            raise RuntimeError(
                "[gs] no handshake from parallax_sim within timeout. Is the renderer "
                "container running (docker compose up -d) with ipc: host?"
            )

    # ── SHM wait helpers ────────────────────────────────────────────────────────
    def _wait_ack(self, pid: str, timeout: float) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            data = self._recv.fetch_data()
            if data is not None and data.get("pckg_id") == pid:
                return True
            time.sleep(0.001)
        return False

    def _wait_images(self, pid: str, timeout: float = 30.0) -> np.ndarray:
        t0 = time.time()
        while time.time() - t0 < timeout:
            data = self._recv.fetch_data()
            if data is not None and "parallax-views" in data and data.get("pckg_id") == pid:
                return np.asarray(data["parallax-views"])  # (E*C, H, W, 3) in [0,1]
            time.sleep(0.0001)
        raise RuntimeError("[gs] timed out waiting for rendered views")

    # ── per-step render ─────────────────────────────────────────────────────────
    def render(self, obj_poses: Sequence[Pose]) -> np.ndarray:
        """Render one frame.

        ``obj_poses`` is one ``(pos[xyz], quat[wxyz])`` per ``ply_paths`` entry, in
        that order, in env-local metres. Returns ``(H, W, 3)`` float32 in ``[0, 1]``
        for the single-env case (``(E, H, W, 3)`` when ``num_envs > 1``).
        """
        if len(obj_poses) != self.n_obj:
            raise ValueError(
                f"expected {self.n_obj} object poses (one per ply), got {len(obj_poses)}"
            )
        poses = ([self.bg_pose] if self.has_bg else []) + list(obj_poses)
        entries = [([float(c) for c in p], [float(c) for c in q]) for p, q in poses]
        K = self._K
        T = {k: entries for k in self.env_keys}
        cams = {
            k: [{K["CAM_K"]: self.cam_K, K["CAM_POS"]: self.cam_pos, K["CAM_QUAT"]: self.cam_quat}]
            for k in self.env_keys
        }
        pid = secrets.token_hex(4)
        update = {
            "pckg_id": pid,
            K["PKG"]: "UPDATE",
            "PER_ENV_TRANSFORM": T,
            "PER_ENV_CAMERAS_DATA_LIST": cams,
        }
        self.last_update = update
        if self.dry_run:
            h = int(self.cam_K[1, 2] * 2)
            w = int(self.cam_K[0, 2] * 2)
            return np.zeros((h, w, 3), np.float32)

        self._send.update(update)
        views = self._wait_images(pid)  # (E, H, W, 3)
        views = views.astype(np.float32).reshape(self.num_envs, *views.shape[1:])
        return views[0] if self.num_envs == 1 else views
