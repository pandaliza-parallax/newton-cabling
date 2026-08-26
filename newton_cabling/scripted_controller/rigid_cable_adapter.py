"""Bind :class:`~.align_insert.AlignInsertController` to ``rl/rigid_cable_env.py``.

Duck-typed on purpose: nothing here imports Newton, Warp or Torch, so the module
stays inside the no-GPU gate with the rest of the package. The environment object
is only ever read.

    from newton_cabling.scripted_controller import (
        AlignInsertController, config_for_rigid_cable_env, observe_rigid_cable_env)

    import rigid_cable_env
    env = rigid_cable_env.RigidCableVecEnv(n, cable_tilt_deg=(0.0, 8.0))
    ctrl = AlignInsertController(n, config_for_rigid_cable_env(rigid_cable_env))
    obs = env.reset(); ctrl.reset()
    for _ in range(steps):
        env.step(ctrl.act(observe_rigid_cable_env(env)))
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from newton_cabling.scripted_controller.align_insert import AlignInsertConfig, InsertionObs
from newton_cabling.scripted_controller.quaternion import (
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_rotate,
)

__all__ = ["config_for_rigid_cable_env", "observe_rigid_cable_env"]


def observe_rigid_cable_env(env: Any) -> InsertionObs:
    """Read the current face / wrist / seat poses out of a ``RigidCableVecEnv``.

    ``_face_pose`` and the seat attributes are the same privileged quantities the
    environment's own ``servo_action`` teacher and ``record_cable_env.py`` read.

    The ``cmd_*`` fields carry the env's open command integrator ``wrist_tgt_p/q``
    — always present on the parent env too, where it simply IS the kinematic
    target — mapped onto the plug face through the rigid-grasp assumption: the
    face rides the command by the same pose delta the wrist is being asked to
    close. They feed ``servo_source="commanded"`` and are ignored otherwise.
    """
    bqn = env.state_0.body_q.numpy()
    face, face_q = env._face_pose(bqn)
    face = np.asarray(face, dtype=np.float64)
    face_q = np.asarray(face_q, dtype=np.float64)
    wrist_p = np.asarray(bqn[env.wrist_body, :3], dtype=np.float64)
    wrist_q = quat_normalize(np.asarray(bqn[env.wrist_body, 3:7], dtype=np.float64))
    cmd_p = np.asarray(env.wrist_tgt_p, dtype=np.float64)
    cmd_q = np.asarray(env.wrist_tgt_q, dtype=np.float64)
    # pose delta commanded-vs-measured wrist; under a rigid grasp the face is carried
    # by the same delta: cmd_face = cmd_p + q_d * (face - wrist), cmd_quat = q_d * face_q
    q_d = quat_normalize(quat_multiply(cmd_q, quat_conjugate(wrist_q)))
    return InsertionObs(
        face_pos=face,
        face_quat=face_q,
        wrist_pos=wrist_p,
        seat_pos=np.asarray(env.seat_pos, dtype=np.float64),
        seat_quat=np.asarray(env.seat_q, dtype=np.float64),
        ins=np.asarray(env.ins, dtype=np.float64),
        cmd_wrist_pos=cmd_p,
        cmd_face_pos=cmd_p + quat_rotate(q_d, face - wrist_p),
        cmd_face_quat=quat_multiply(q_d, face_q),
    )


def config_for_rigid_cable_env(module: Any, **overrides: Any) -> AlignInsertConfig:
    """Build a config whose geometry and action caps track the env module's constants.

    Pass the imported ``rigid_cable_env`` module. Reading ``SEAT_AIM_DY`` / ``MAX_DPOS`` /
    ``MAX_DROT`` from it rather than duplicating the numbers means retuning the environment
    cannot silently desynchronise the controller from the cavity depth it aims at.
    """
    base = AlignInsertConfig(
        mouth_depth_m=float(module.SEAT_AIM_DY),
        max_dpos_m=float(module.MAX_DPOS),
        max_drot_rad=float(module.MAX_DROT),
    )
    return dataclasses.replace(base, **overrides) if overrides else base
