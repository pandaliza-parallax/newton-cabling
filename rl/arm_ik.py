"""Damped-least-squares IK for the RO1 arm, batched over a Newton model.

The arm is position-controlled (kinematic): the EEF-delta action becomes a target wrist
pose, and ArmIK.solve() finds the 6 arm joint angles that reach it, via a finite-difference
Jacobian evaluated with newton.eval_fk. Batched: one eval_fk per joint perturbation covers
every env at once; the small 6x6 damped solves run on the host (cheap vs the GPU eval_fk).

Used by rl/arm_connector_env.py. See the sbot-arm-insertion-rl memory for the architecture.
"""
from __future__ import annotations

import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation


class ArmIK:
    def __init__(self, model, arm_qcoords, wrist_bodies, *, eps: float = 1e-4, damping: float = 0.05):
        """arm_qcoords: (n_env, 6) int indices into model.joint_q for the 6 arm DOFs per env.
        wrist_bodies: (n_env,) wrist_3 body indices per env."""
        self.model = model
        self.qc = np.asarray(arm_qcoords, dtype=np.int64)
        self.wb = np.asarray(wrist_bodies, dtype=np.int64)
        self.n = self.qc.shape[0]
        self.eps = eps
        self.lam2 = damping * damping
        self.state = model.state()
        self.jqd = model.joint_qd

    def _wrist_poses(self, joint_q_np):
        jq = wp.array(joint_q_np, dtype=float, device=self.model.device)
        newton.eval_fk(self.model, jq, self.jqd, self.state)
        bq = self.state.body_q.numpy()
        return bq[self.wb, :3].copy(), bq[self.wb, 3:7].copy()

    @staticmethod
    def _rotvec_err(q_cur_xyzw, q_tgt_xyzw):
        Rc = Rotation.from_quat(q_cur_xyzw)
        Rt = Rotation.from_quat(q_tgt_xyzw)
        return (Rt * Rc.inv()).as_rotvec()

    def solve(self, joint_q_np, tgt_pos, tgt_quat_xyzw, *, iters: int = 8,
              tol_pos: float = 5e-4, tol_rot: float = 5e-3, step_cap: float = 0.3):
        """Return updated full joint_q_np so each env's wrist reaches (tgt_pos, tgt_quat_xyzw),
        plus the final (pos_err, rot_err) per env."""
        jq = joint_q_np.copy()
        e_pos = np.zeros((self.n, 3))
        e_rot = np.zeros((self.n, 3))
        for _ in range(iters):
            p0, q0 = self._wrist_poses(jq)
            e_pos = tgt_pos - p0
            e_rot = self._rotvec_err(q0, tgt_quat_xyzw)
            if (np.linalg.norm(e_pos, axis=1).max() < tol_pos
                    and np.linalg.norm(e_rot, axis=1).max() < tol_rot):
                break
            err = np.concatenate([e_pos, e_rot], axis=1)
            J = np.zeros((self.n, 6, 6))
            for j in range(6):
                jqp = jq.copy()
                jqp[self.qc[:, j]] += self.eps
                pp, qp = self._wrist_poses(jqp)
                J[:, 0:3, j] = (pp - p0) / self.eps
                J[:, 3:6, j] = self._rotvec_err(q0, qp) / self.eps
            JT = np.transpose(J, (0, 2, 1))
            A = J @ JT + self.lam2 * np.eye(6)[None]
            y = np.linalg.solve(A, err[:, :, None])
            dq = np.clip((JT @ y)[:, :, 0], -step_cap, step_cap)
            for j in range(6):
                jq[self.qc[:, j]] += dq[:, j]
        return jq, np.linalg.norm(e_pos, axis=1), np.linalg.norm(e_rot, axis=1)
