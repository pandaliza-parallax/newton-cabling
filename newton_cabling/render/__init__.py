"""GS rendering bridge: drive DalusPySim's Gaussian-Splat renderer from Newton.

See :mod:`newton_cabling.render.gs_bridge`. Pure-Python/numpy; importing this package
does NOT import newton or warp, so it lints/tests off-GPU. The runnable demo is the
top-level ``record_rj45_insert_gs.py``.
"""

from .gs_bridge import NewtonGSClient, look_at_quat, make_intrinsics, newton_pose

__all__ = ["NewtonGSClient", "look_at_quat", "make_intrinsics", "newton_pose"]
