"""Align-then-insert scripted controller for RJ45 connector insertion.

The strategy, in the user's words: *wherever the connector starts, align its head
in front of the jack mouth, then simply push it in.*

    SETTLE -> ALIGN -> PUSH -> HOLD
                ^        |       |
                +- RETREAT <-----+ (jam / alignment lost / held plug twisted out)

Everything is expressed in the ``along`` coordinate: the signed distance of the
plug FACE from the fully-seated pose, measured along the insertion axis, positive
= deeper. The jack mouth therefore sits at ``-mouth_depth_m`` and the pre-dock
pose at ``-(mouth_depth_m + align_standoff_m)``.

Four properties of the plant drove this design; each is load-bearing.

* **The command moves the WRIST, the plug follows through a friction grasp.**
  Wrist-to-face is ~0.21 m, so a full-scale rotation step (0.75 deg) swings the
  face ~2.7 mm — MORE than a full-scale translation step (2 mm). "Fix the angle,
  then fix the position" therefore cannot work as separate phases: correcting
  orientation destroys position. Instead the servo picks a desired rigid motion
  of the FACE and converts it to the wrist command it implies
  (:meth:`AlignInsertController._to_action`), so translation and rotation are
  solved together.

* **Align OUTSIDE the contact band.** Newton wraps shapes in a contact shell
  (``rigid_gap``, 1 mm in the rigid cable env) and the measured SDF contacts fire
  at several mm of separation. Aligning a millimetre off the mouth would mean
  converging while the chamfer pushes back, so the pre-dock pose parks the face a
  configurable standoff (default 5 mm) clear of the mouth, where the alignment
  can settle in free space. Only then does the push commit.

* **Rotation must be rate-limited well below the action cap.** The environment's
  own build-time levelling ramps the wrist at ~0.3 deg/frame because the friction
  grasp is only ~1.5 mm deep and cannot follow faster; the action cap is 0.75
  deg/frame, 2.5x that. ``align_rot_rate_rad`` defaults to the validated rate,
  not the cap.

* **A push that jams stays jammed.** There is no force sensing, so progress is
  watched directly: if ``along`` stops improving for ``jam_window`` steps the
  controller retreats straight back out along the insertion axis (never twisting
  while inside the mouth) and re-aligns from the pre-dock pose.

* **Under a lagging arm, servo on the COMMANDED pose — gate on the MEASURED one.**
  The action lands in the environment's open command integrator
  (``wrist_tgt_p/q``), which the calibrated servo arm trails by its transport
  delay plus jerk-limited braking distance. Closing the servo loop on the
  measured face then means integrating corrections against stale feedback, and
  only halving every rate stabilises it (23.3% seat vs the kinematic arm's 100%
  — ``tests/gates/REPORT.md``, Wave 2c/R11). The command state is exactly
  known, so nothing needs predicting: ``servo_source="commanded"`` computes the
  dp/dr drives from where the COMMAND is — a loop the plant cannot lag, because
  it *is* the command — while every phase transition (docked, seated, jammed,
  lost, retreat-arrival) still fires on the measured face, so no commitment is
  ever made on anything but ground truth. ``stable_steps`` naturally waits out
  the physical settle after the command parks.

Pure NumPy — no Newton, Warp, Torch or SciPy — so it unit-tests anywhere. Bind it
to a simulation with an adapter; see :mod:`.rigid_cable_adapter`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from newton_cabling.scripted_controller.quaternion import (
    quat_conjugate,
    quat_multiply,
    quat_to_rotvec,
    rotate_by_rotvec,
)

__all__ = ["AlignInsertConfig", "AlignInsertController", "InsertPhase", "InsertionObs"]


class InsertPhase(IntEnum):
    """Per-environment state of the insertion machine."""

    SETTLE = 0
    ALIGN = 1
    PUSH = 2
    HOLD = 3
    RETREAT = 4


@dataclass(frozen=True)
class AlignInsertConfig:
    """Tuning for :class:`AlignInsertController`. Distances in metres, angles in radians.

    The defaults are sized for the rigid RJ45 cable environment: a 12 mm cavity,
    a 2 mm / 0.75 deg per-step action cap, and a friction grasp that tolerates
    ~0.8 mm and ~0.3 deg of commanded motion per control step.
    """

    # ── task geometry, in the `along` coordinate (0 = seated, negative = out) ──
    mouth_depth_m: float = 0.012
    """How far past the jack mouth the seated pose lies (the cavity depth aim)."""
    align_standoff_m: float = 0.005
    """Pre-dock parking distance OUTSIDE the mouth. Keep clear of the contact band."""
    seat_target_m: float = 0.0
    """`along` the push drives toward."""
    hold_target_m: float = 0.0
    """`along` the hold phase regulates around."""

    # ── the environment's per-step action caps (the action is a fraction of these) ──
    max_dpos_m: float = 0.002
    max_drot_rad: float = math.radians(0.75)

    # ── align servo ──
    kp_lin: float = 0.5
    kp_rot: float = 0.5
    align_lin_rate_m: float = 0.0008
    """Face translation rate cap. 0.4x the action cap, matching the proven teacher advance."""
    align_rot_rate_rad: float = math.radians(0.3)
    """Face rotation rate cap — the rate the friction grasp is known to follow."""

    # ── when the pre-dock pose counts as reached ──
    lat_tol_m: float = 0.0005
    ang_tol_rad: float = math.radians(1.0)
    along_tol_m: float = 0.001
    stable_steps: int = 3
    """Consecutive in-tolerance steps required, so the push cannot start on a fly-through."""

    # ── push ──
    push_rate_m: float = 0.0008
    push_correction: float = 0.3
    """Gain on the lateral/angular correction kept live during the push, as a fraction of
    the align gains and rate caps. 0.0 gives a strictly open-loop straight push."""
    hold_enter_m: float = -0.002
    """`along` at which the push is considered seated and hands over to HOLD."""
    hold_regain_m: float = 0.003
    """If a held plug backs out this far past `hold_enter_m`, push again."""

    # ── jam detection and recovery ──
    jam_window: int = 20
    """Steps without progress that count as a jam."""
    jam_progress_m: float = 0.0005
    """`along` gain that resets the jam counter."""
    abort_lat_m: float = 0.004
    abort_ang_rad: float = math.radians(10.0)
    """Alignment this bad mid-push means back out and start over rather than force it."""
    retreat_rate_m: float = 0.001

    # ── misc ──
    settle_steps: int = 2
    """Zero-action steps after a reset, letting the grasp come to rest before measuring."""
    hold_kp: float = 0.3
    gripper_cmd: float = 0.0
    """Value written to the gripper channel. The rigid cable env ignores it; it is part of
    the pi05 rj45_sbot 7-D action layout and is carried through for datagen parity."""
    servo_source: str = "measured"
    """Which face pose the SERVO terms track: ``"measured"`` (the simulated plant — the
    default, bit-identical to the original controller) or ``"commanded"`` (the env's open
    command integrator, for a lagging servo arm; requires the ``cmd_*`` observation
    fields). Phase transitions gate on the MEASURED pose in both modes."""

    def __post_init__(self) -> None:
        if self.align_standoff_m <= 0.0:
            raise ValueError("align_standoff_m must be > 0 (align outside the mouth)")
        if self.max_dpos_m <= 0.0 or self.max_drot_rad <= 0.0:
            raise ValueError("action caps must be > 0")
        if not 0.0 <= self.push_correction <= 1.0:
            raise ValueError("push_correction must be in [0, 1]")
        if self.jam_window < 1 or self.stable_steps < 1:
            raise ValueError("jam_window and stable_steps must be >= 1")
        if self.servo_source not in ("measured", "commanded"):
            raise ValueError('servo_source must be "measured" or "commanded"')

    @property
    def align_along_m(self) -> float:
        """`along` coordinate of the pre-dock pose (outside the jack mouth)."""
        return -(self.mouth_depth_m + self.align_standoff_m)

    @property
    def mouth_along_m(self) -> float:
        """`along` coordinate of the jack mouth plane."""
        return -self.mouth_depth_m


@dataclass(frozen=True)
class InsertionObs:
    """What the controller needs to see, all world-frame, all shape ``(n, ...)``.

    Quaternions are ``(x, y, z, w)``. This is exactly the information the learned
    policy gets (its 22-D observation carries the same face-vs-seat position and
    orientation errors), so a scripted-vs-learned comparison is like for like.
    """

    face_pos: np.ndarray
    """(n, 3) world position of the plug face."""
    face_quat: np.ndarray
    """(n, 4) world orientation of the plug face."""
    wrist_pos: np.ndarray
    """(n, 3) world position of the wrist the action commands — sets the lever arm."""
    seat_pos: np.ndarray
    """(n, 3) world position the face must reach to be seated."""
    seat_quat: np.ndarray
    """(n, 4) world orientation the face must match."""
    ins: np.ndarray
    """(n, 3) unit insertion axis, pointing INTO the jack."""
    cmd_wrist_pos: np.ndarray | None = field(default=None)
    """(n, 3) COMMANDED wrist position — the env's open command integrator
    (``wrist_tgt_p``). Optional; required only for ``servo_source="commanded"``."""
    cmd_face_pos: np.ndarray | None = field(default=None)
    """(n, 3) where the plug face WILL sit once the arm has tracked the command,
    under the rigid-grasp assumption. Optional; see ``cmd_wrist_pos``."""
    cmd_face_quat: np.ndarray | None = field(default=None)
    """(n, 4) commanded-face orientation, same derivation as ``cmd_face_pos``."""


def _cap(v: np.ndarray, limit: float) -> np.ndarray:
    """Scale each row of `v` so its norm is at most `limit`, preserving direction.

    Per-component clipping would bend the commanded direction — a diagonal
    correction would come out axis-aligned — so the whole vector is scaled.
    """
    if limit <= 0.0:
        return np.zeros_like(v)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, limit / np.maximum(norm, 1e-12))


class AlignInsertController:
    """Vectorised align-then-insert controller: one independent state machine per env.

        ctrl = AlignInsertController(n, AlignInsertConfig())
        ctrl.reset()
        action = ctrl.act(obs)          # (n, 7) in [-1, 1]

    The returned action is ``[dpos(3), drotvec(3), gripper(1)]`` as a fraction of
    the configured per-step caps — the pi05 ``rj45_sbot`` layout.
    """

    def __init__(self, n: int, config: AlignInsertConfig | None = None) -> None:
        self.n = int(n)
        self.config = config if config is not None else AlignInsertConfig()
        self.reset()

    def reset(self) -> None:
        """Restart every environment's machine at SETTLE."""
        n = self.n
        self.phase = np.full(n, int(InsertPhase.SETTLE), dtype=np.int64)
        self.phase_steps = np.zeros(n, dtype=np.int64)
        # how many times a jammed push retreated and re-aligned, per env
        self.attempts = np.zeros(n, dtype=np.int64)
        self._stable = np.zeros(n, dtype=np.int64)
        self._best_along = np.full(n, -np.inf)
        self._stall = np.zeros(n, dtype=np.int64)
        self.along = np.zeros(n)
        self.latn = np.zeros(n)
        self.ang = np.zeros(n)
        # commanded-face errors, populated only when servo_source="commanded"
        self.cmd_along: np.ndarray | None = None
        self.cmd_latn: np.ndarray | None = None
        self.cmd_ang: np.ndarray | None = None

    # ── measurement ───────────────────────────────────────────────────────────
    def _face_errors(
        self, face_pos: np.ndarray, face_quat: np.ndarray, obs: InsertionObs
    ) -> tuple[np.ndarray, ...]:
        """Errors of an arbitrary face pose (measured or commanded) against the seat."""
        e = np.asarray(face_pos, dtype=np.float64) - np.asarray(obs.seat_pos, dtype=np.float64)
        ins = np.asarray(obs.ins, dtype=np.float64)
        along = np.sum(e * ins, axis=1)
        lat = e - along[:, None] * ins
        # rotation taking the seat frame ONTO the face — its negation is the correction
        rot_err = quat_to_rotvec(quat_multiply(face_quat, quat_conjugate(obs.seat_quat)))
        return along, lat, np.linalg.norm(lat, axis=1), rot_err, np.linalg.norm(rot_err, axis=1)

    def _errors(self, obs: InsertionObs) -> tuple[np.ndarray, ...]:
        """Face-vs-seat error, split into the axial / lateral / angular parts."""
        return self._face_errors(obs.face_pos, obs.face_quat, obs)

    # ── the servo ─────────────────────────────────────────────────────────────
    def act(self, obs: InsertionObs) -> np.ndarray:
        """Advance every state machine one step and return the ``(n, 7)`` action."""
        c = self.config
        ins = np.asarray(obs.ins, dtype=np.float64)
        along, lat, latn, rot_err, ang = self._errors(obs)
        self.along, self.latn, self.ang = along, latn, ang

        # Which face pose the servo terms drive. "measured" closes the loop on the
        # plant; "commanded" closes it on the env's open command integrator (see the
        # module docstring) — while `_advance` below stays on the measured errors, so
        # every phase commitment is still made on ground truth.
        if c.servo_source == "commanded":
            if obs.cmd_wrist_pos is None or obs.cmd_face_pos is None or obs.cmd_face_quat is None:
                raise ValueError(
                    'servo_source="commanded" requires the cmd_wrist_pos / cmd_face_pos / '
                    "cmd_face_quat observation fields (the rigid-cable adapter derives them "
                    "from env.wrist_tgt_p / env.wrist_tgt_q)"
                )
            s_along, s_lat, s_latn, s_rot_err, s_ang = self._face_errors(
                obs.cmd_face_pos, obs.cmd_face_quat, obs
            )
            self.cmd_along, self.cmd_latn, self.cmd_ang = s_along, s_latn, s_ang
        else:
            s_along, s_lat, s_rot_err = along, lat, rot_err
        ph = self.phase.copy()

        dp = np.zeros((self.n, 3))
        dr = np.zeros((self.n, 3))

        # ALIGN — full 6-DOF drive of the face onto the pre-dock pose. Lateral and
        # axial error are corrected together with the rotation, because the rotation
        # itself swings the face and a sequential scheme would chase its own tail.
        m = (ph == InsertPhase.ALIGN)[:, None]
        d_along = c.align_along_m - s_along
        dp = np.where(m, _cap(c.kp_lin * (-s_lat + d_along[:, None] * ins), c.align_lin_rate_m), dp)
        dr = np.where(m, _cap(-c.kp_rot * s_rot_err, c.align_rot_rate_rad), dr)

        # PUSH — advance along the insertion axis. `push_correction` keeps a softened
        # version of the align servo live so the chamfer can guide without the
        # controller fighting it; at 0.0 this is a strictly straight push.
        k = c.push_correction
        m = (ph == InsertPhase.PUSH)[:, None]
        adv = np.clip(c.seat_target_m - s_along, 0.0, c.push_rate_m)
        corr = _cap(k * c.kp_lin * (-s_lat), k * c.align_lin_rate_m)
        dp = np.where(m, adv[:, None] * ins + corr, dp)
        dr = np.where(m, _cap(-k * c.kp_rot * s_rot_err, k * c.align_rot_rate_rad), dr)

        # HOLD — same softened correction, but regulating depth instead of driving in,
        # so the seat is maintained without ramming the plug deeper.
        m = (ph == InsertPhase.HOLD)[:, None]
        reg = np.clip(c.hold_kp * (c.hold_target_m - s_along), -c.push_rate_m, c.push_rate_m)
        dp = np.where(m, reg[:, None] * ins + corr, dp)
        dr = np.where(m, _cap(-k * c.kp_rot * s_rot_err, k * c.align_rot_rate_rad), dr)

        # RETREAT — straight back out, no rotation and no lateral correction: twisting a
        # plug that is jammed inside the mouth levers it against the cavity walls.
        m = (ph == InsertPhase.RETREAT)[:, None]
        back = np.clip(c.align_along_m - s_along, -c.retreat_rate_m, 0.0)
        dp = np.where(m, back[:, None] * ins, dp)
        dr = np.where(m, 0.0, dr)

        # SETTLE leaves dp/dr at zero.
        self._advance(ph, along, latn, ang)
        return self._to_action(dp, dr, obs)

    def _advance(
        self, ph: np.ndarray, along: np.ndarray, latn: np.ndarray, ang: np.ndarray
    ) -> None:
        """Update phases from the CURRENT measurement (`ph` is the phase just acted on)."""
        c = self.config
        self.phase_steps += 1

        # jam bookkeeping: watch `along` for real progress rather than a force signal
        in_push = ph == InsertPhase.PUSH
        improved = in_push & (along > self._best_along + c.jam_progress_m)
        self._best_along = np.where(improved, along, self._best_along)
        self._stall = np.where(improved, 0, np.where(in_push, self._stall + 1, self._stall))
        jammed = in_push & (self._stall >= c.jam_window)

        nxt = ph.copy()
        nxt = np.where(
            (ph == InsertPhase.SETTLE) & (self.phase_steps >= c.settle_steps),
            int(InsertPhase.ALIGN),
            nxt,
        )

        docked = (
            (latn <= c.lat_tol_m)
            & (ang <= c.ang_tol_rad)
            & (np.abs(along - c.align_along_m) <= c.along_tol_m)
        )
        self._stable = np.where((ph == InsertPhase.ALIGN) & docked, self._stable + 1, 0)
        nxt = np.where(
            (ph == InsertPhase.ALIGN) & (self._stable >= c.stable_steps), int(InsertPhase.PUSH), nxt
        )

        seated = in_push & (along >= c.hold_enter_m)
        lost = (latn > c.abort_lat_m) | (ang > c.abort_ang_rad)
        nxt = np.where(seated, int(InsertPhase.HOLD), nxt)
        nxt = np.where(in_push & ~seated & (jammed | lost), int(InsertPhase.RETREAT), nxt)

        nxt = np.where(
            (ph == InsertPhase.RETREAT) & (along <= c.align_along_m + c.along_tol_m),
            int(InsertPhase.ALIGN),
            nxt,
        )
        nxt = np.where(
            (ph == InsertPhase.HOLD) & (along < c.hold_enter_m - c.hold_regain_m),
            int(InsertPhase.PUSH),
            nxt,
        )
        # HOLD abort — a held plug that twists or shears out of tolerance must back out,
        # not grind. Without this, HOLD regulates DEPTH forever while attitude runs away:
        # observed on the r180 grasp under the calibrated arm, where contact + gravity
        # ratchet the friction grasp ~0.2 deg/frame to a ~50 deg wedge (the mid-push
        # `lost` abort never fires because the plug is already seated). Same thresholds
        # as the push abort; placed AFTER the regain rule so abort wins when both hold.
        # RETREAT's straight-back-out drive is the least-bad move for a twisted plug.
        nxt = np.where((ph == InsertPhase.HOLD) & lost, int(InsertPhase.RETREAT), nxt)

        changed = nxt != ph
        retried = (ph == InsertPhase.RETREAT) & (nxt == int(InsertPhase.ALIGN))
        self.attempts += retried.astype(np.int64)
        entering_push = changed & (nxt == int(InsertPhase.PUSH))
        self._best_along = np.where(entering_push, along, self._best_along)
        self._stall = np.where(entering_push, 0, self._stall)
        self.phase_steps = np.where(changed, 0, self.phase_steps)
        self._stable = np.where(changed, 0, self._stable)
        self.phase = nxt

    def _to_action(self, dp_face: np.ndarray, dr_face: np.ndarray, obs: InsertionObs) -> np.ndarray:
        """Convert a desired rigid motion of the FACE into the wrist action that causes it.

        Treating the grasp as rigid over one step, rotating the body by ``dr`` about the
        face origin and translating it by ``dp`` moves the wrist by
        ``dp + (R(dr) - I) @ (wrist - face)``. The residual grasp compliance and slip are
        not modelled — they do not need to be, because the servo re-measures the face every
        step and simply corrects whatever the last command failed to achieve.

        The lever arm comes from the SAME frame the dp/dr drives were computed in:
        measured wrist minus measured face by default, commanded wrist minus commanded
        face under ``servo_source="commanded"`` — the motion is of the command state, so
        its geometry must be command-space too.

        The two commands are scaled by a COMMON factor when either exceeds its cap, so a
        clamp slows the motion down instead of bending it into a different screw axis.
        """
        c = self.config
        if c.servo_source == "commanded":
            wrist, face = obs.cmd_wrist_pos, obs.cmd_face_pos
        else:
            wrist, face = obs.wrist_pos, obs.face_pos
        r = np.asarray(wrist, dtype=np.float64) - np.asarray(face, dtype=np.float64)
        dp_w = dp_face + rotate_by_rotvec(dr_face, r) - r
        lin = np.linalg.norm(dp_w, axis=1, keepdims=True)
        rot = np.linalg.norm(dr_face, axis=1, keepdims=True)
        s = np.minimum(
            np.minimum(1.0, c.max_dpos_m / np.maximum(lin, 1e-12)),
            c.max_drot_rad / np.maximum(rot, 1e-12),
        )
        action = np.zeros((self.n, 7))
        action[:, 0:3] = dp_w * s / c.max_dpos_m
        action[:, 3:6] = dr_face * s / c.max_drot_rad
        action[:, 6] = c.gripper_cmd
        return np.clip(action, -1.0, 1.0)

    # ── reporting ─────────────────────────────────────────────────────────────
    def phase_counts(self) -> dict[str, int]:
        """How many environments sit in each phase right now."""
        return {p.name: int(np.sum(self.phase == int(p))) for p in InsertPhase}
