"""Servo-plant variant of the rigid cable env: the arm is driven by a CALIBRATED PLANT.

`RigidCableVecEnv` drives the (kinematic) arm from an IK solution by LINEARLY INTERPOLATING
the joint coordinates across the SUBSTEPS of a control frame — an ideal, infinitely stiff,
zero-latency arm. Real RO2-core joints have transport delay, finite bandwidth and per-joint
lag; a policy trained against the ideal arm learns a timing that the hardware cannot honour.
This subclass replaces ONLY the source of the per-substep joint positions with a
`ServoArmBank` (the hardware-calibrated plant from parallax-demo-newton `sysid/` + `control/`,
see newton_cabling/servo_arm/CONTRACT.md): goal -> retimer -> 500 Hz feedforward -> transport delay -> plant.

    from servo_cable_env import ServoCableVecEnv
    env = ServoCableVecEnv(4, cable_tilt_deg=8.0)          # default bank, lazily imported
    env.set_stage(0); obs = env.reset()
    obs, rew, done, success, depth_mm = env.step(action)   # identical interface to the parent

EVERYTHING else is the parent's: scene build, align/settle, snapshots, reset, curriculum,
reward, violation counting, obs/action, the servo_action() teacher. Gripper and jack joint
coordinates keep the parent's behaviour EXACTLY — the bank owns the 6 arm DOFs and nothing
else (contract pin 3).

── why this file duplicates the parent's step() ──────────────────────────────────────────
`RigidCableVecEnv.step()` has no extractable seam: the IK call and `_sim(jq_prev)` sit in the
middle of a straight-line body, and the parent file is off-limits (it carries uncommitted user
work). Orchestrator ruling (newton_cabling/servo_arm/CONTRACT.md, Addendum 1): duplicate the body, swap ONLY
the jq/_sim segment, and pay for it with two guards, both of which live here:

  1. PARENT_SOURCE_SHA256 pins the exact parent source this copy was written against.
     tests/test_golden_kinematic.py asserts it still matches, so a future parent edit fails
     loudly ("re-sync the subclass copy") instead of silently drifting.
  2. `_IdealBank` (below) implements the bank contract as EXACT linear interpolation — i.e.
     the parent's own arm model. With it, this class must reproduce the parent's rollout to
     numerical tolerance; the same test proves it. That makes the copy auditable independently
     of whatever the calibrated plant does.

Everything numeric (MAX_DPOS, MARGIN, reward weights, ...) is IMPORTED from the parent module,
never re-typed: retuning the parent must not silently desynchronise this subclass.

── the revolution BRANCH OFFSET (contract Addendum 3) ────────────────────────────────────
The env's joints are continuous and `rl/arm_ik.py` is unbounded, so a build slot can settle a
whole revolution outside the bank's +/-360 deg MuJoCo limits (measured joint5 = -391.4 deg on
one slot of a 2-arm build while the other sat at -31.0 deg — the identical pose, one turn
away; which slot wraps is build-dependent, so it presents as an intermittent failure).
The bank's refusal is correct and stays untouched. This class owns the bookkeeping: at reset
it establishes a per-(env, joint) CONSTANT offset, an exact multiple of 2*pi, subtracts it
from jq0 and from every goal on the way in, and adds it back to every sample on the way out.
Pose-exact (FK is 2*pi-periodic in a revolute joint), and — the reason it is an offset rather
than a wrap — no sample stream ever interpolates ACROSS a 2*pi jump, so the arm never sweeps a
full unwind inside a frame. A goal that needs a DIFFERENT branch than reset established is a
real full-turn command, not bookkeeping, and RAISES.

── the COMMAND-LEAD CLAMP (contract Addendum 3, optional) ────────────────────────────────
`wrist_tgt_p` is an OPEN command integrator in the parent (`+= dpos` every frame, no
feedback), which assumes an arm that tracks within the frame. A real plant does not: measured
median wrist following error 0.18 mm with the ideal arm vs 28.5 mm (max 56.9) with the
calibrated plant, same teacher, same seed — the target simply runs away. `max_cmd_lead_m`
bounds it, like a real controller's following-error fault limit. Default None = strict parent
parity, which is what keeps the golden and equivalence guards meaningful.

WARNING (Wave 2c, REPORT.md R9): as implemented this clamp is MEASURED NET-NEGATIVE — it
rescales the whole 3-D lead vector isotropically, and since the lead is dominated by the
axial push component, it attenuates the lateral correction by the same factor: the command
degenerates into an axial pull with no lateral authority and the grasp drags the plug
sideways (lat 3.97 -> 39.38 mm at 5 mm, tuned profile). Do NOT enable it until the governor
is anisotropic (clamp the along-`ins` component; budget axial and lateral separately).

── PLAN CHURN: re-goal suppression (contract Addendum 4, R2 — DEFAULT OFF) ───────────────
`set_goal` is the expensive half of the bank (a full retimer re-plan per env; advance-only is
~2.3x cheaper) and this env re-goals every env every frame at 60 Hz. Two optional knobs thin
that stream: `regoal_deadband_rad` suppresses a send whose every-joint delta from the LAST
SENT goal is inside the deadband, and `regoal_every` forces a send at least every k frames so
a suppressed goal can never go stale. Both default OFF (0.0 / 1), and at those values the
suppression clauses are skipped outright — the default configuration is bit-identical to the
pre-R2 exact-inequality rule by construction, not by measurement.
CAVEAT, and the reason they stay off: a deadband QUANTIZES the commanded pose. This task's
lateral tolerance is 0.5 mm, and a deadband coarse enough to save real planning work can
easily exceed the joint motion 0.5 mm represents. These are perf knobs to be measured after
the bank's OTG work (Addendum 4 R1), never a behavioural default.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import os
import sys

import newton
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation as Rot

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from newton_cabling.sim.sbot import ARM_JOINT_NAMES  # noqa: E402
from rigid_cable_env import (  # noqa: E402
    DEV, EJECT_DIST, HOLD_STEPS, MARGIN, MAX_DPOS, MAX_DROT, R_SUCCESS, SEAT_ANGLE,
    SEAT_DEPTH_TOL, SEAT_OFFSET, SUBSTEPS, W_PROG, W_VEL, W_VIOL,
    RigidCableVecEnv, _sync_pad_anchors)

# Control-frame rate of the env: SUBSTEPS solver substeps of self.dt = 1/60/SUBSTEPS each.
# The bank is a per-CONTROL-FRAME generator, so this is the frame_dt it must be built with.
FRAME_DT = 1.0 / 60.0

TWO_PI = 2.0 * math.pi
LIMIT_EPS = 1e-9      # rad of slack when testing a value against the bank's limits, so a goal
#                       that sits exactly ON a limit is not rejected by a rounding hair.


def bank_joint_limits(bank):
    """(lo, hi) radian arrays of the bank's 6 arm joint limits, or None if it has none.

    None is a real answer, not a fallback: a bank without limits (the `_IdealBank` guard stub)
    accepts every branch, so there is nothing to offset and the branch bookkeeping switches
    off entirely — which is exactly what keeps the identity-bank equivalence bit-exact.
    The private `_jnt_lo/_jnt_hi` probe is deliberate coupling to `newton_cabling.servo_arm`:
    the contract's API does not expose limits, and Addendum 3 puts the offset on THIS side of
    the seam, so the limits have to be read from somewhere. Promoting them to a public
    property on ServoArmBank would let the first branch here do the job.
    """
    lim = getattr(bank, "joint_limits", None)
    if lim is None:
        lo, hi = getattr(bank, "_jnt_lo", None), getattr(bank, "_jnt_hi", None)
        lim = None if lo is None or hi is None else (lo, hi)
    if lim is None:
        return None
    lo, hi = (np.asarray(x, dtype=np.float64) for x in lim)
    if lo.shape != (6,) or hi.shape != (6,):
        raise ValueError(f"bank {type(bank).__name__} exposes joint limits of shape "
                         f"{lo.shape}/{hi.shape}, expected (6,)/(6,)")
    return lo, hi

# ── guard 1: parent-source drift ──────────────────────────────────────────────────────────
# sha256 of the parent methods whose bodies are duplicated below. `step` is copied wholesale
# with the jq/_sim segment swapped; `_sim`'s substep loop is copied into _sim_from_stream.
# Recompute with:  hashlib.sha256(inspect.getsource(RigidCableVecEnv.step).encode()).hexdigest()
# If a test reports a mismatch: DIFF the parent method, port the change here, then update the
# hash. Never update the hash alone — that is exactly the drift the guard exists to catch.
PARENT_SOURCE_SHA256 = {
    "step": "995be5882cf126ae97606ad97a82e59f33a9ff0101fc7ffb7bf599f7923bcbac",
    "_sim": "8ce8d183bd542c5242df52f5259affda6eabddaa240cd2dbf7b57b6406633989",
}


def parent_source_sha256() -> dict[str, str]:
    """Current sha256 of the duplicated parent methods (for the drift-guard test)."""
    return {
        nm: hashlib.sha256(
            inspect.getsource(getattr(RigidCableVecEnv, nm)).encode("utf-8")).hexdigest()
        for nm in PARENT_SOURCE_SHA256
    }


# ── guard 2: the identity bank ────────────────────────────────────────────────────────────
class _IdealBank:
    """The `ServoArmBank` contract implemented as the PARENT'S arm: exact linear interpolation
    from the current joint positions to the goal across the frame, no delay, no dynamics.

    Its only purpose is the equivalence guard: `ServoCableVecEnv` driven by this bank must
    reproduce `RigidCableVecEnv` to numerical tolerance. The sample formula is deliberately
    written as `q + a * (goal - q)` with `a = (s + 1) / substeps` — CHARACTER-FOR-CHARACTER the
    parent's `_sim` interpolation — so the two agree in IEEE arithmetic, not just in the limit.
    Any other algebraically equal form (e.g. `(1 - a) * q + a * goal`) rounds differently and
    would turn a bit-exact guard into a fuzzy one.

    No RNG, no torch, no warp — same rules as the real bank (contract pin 5).
    """

    def __init__(self, n: int, *, frame_dt: float = FRAME_DT):
        self.n = int(n)
        self.frame_dt = float(frame_dt)
        self._q = np.zeros((self.n, 6))
        self._qd = np.zeros((self.n, 6))
        self._goal = np.zeros((self.n, 6))
        self._ready = False

    def reset(self, jq0: np.ndarray) -> None:
        jq0 = np.asarray(jq0, dtype=np.float64)
        if jq0.shape != (self.n, 6):
            raise ValueError(f"_IdealBank.reset expects (n, 6) = {(self.n, 6)}, got {jq0.shape}")
        if not np.isfinite(jq0).all():
            raise ValueError("_IdealBank.reset: non-finite jq0")
        self._q = jq0.copy()
        self._qd = np.zeros((self.n, 6))
        self._goal = jq0.copy()
        self._ready = True

    def set_goal(self, env_ids: np.ndarray, q_star: np.ndarray) -> None:
        if not self._ready:
            raise RuntimeError("_IdealBank.set_goal before reset()")
        env_ids = np.asarray(env_ids, dtype=np.int64)
        q_star = np.asarray(q_star, dtype=np.float64)
        if q_star.shape != (env_ids.shape[0], 6):
            raise ValueError(f"_IdealBank.set_goal expects ({env_ids.shape[0]}, 6), "
                             f"got {q_star.shape}")
        if not np.isfinite(q_star).all():
            raise ValueError("_IdealBank.set_goal: non-finite goal")
        self._goal[env_ids] = q_star

    def advance(self, substeps: int) -> np.ndarray:
        if not self._ready:
            raise RuntimeError("_IdealBank.advance before reset()")
        q = self._q
        out = np.empty((substeps, self.n, 6))
        for s in range(substeps):
            a = (s + 1) / substeps
            out[s] = q + a * (self._goal - q)
        # measured state at frame end == the last emitted sample (contract), NOT self._goal:
        # the two can differ by an ulp and the difference is the parent's, so keep the parent's.
        self._q = out[-1].copy()
        self._qd = (self._goal - q) / self.frame_dt
        return out

    @property
    def jq(self) -> np.ndarray:
        return self._q

    @property
    def jqd(self) -> np.ndarray:
        return self._qd


class ServoCableVecEnv(RigidCableVecEnv):
    """`RigidCableVecEnv` with the per-substep arm joint stream sourced from a servo plant."""

    def __init__(self, n: int, *, bank=None, max_cmd_lead_m: float | None = None,
                 regoal_deadband_rad: float = 0.0, regoal_every: int = 1, **kwargs):
        """bank: an object implementing the ServoArmBank contract (reset/set_goal/advance/jq).
        None -> build the calibrated `newton_cabling.servo_arm.ServoArmBank` for n arms.

        max_cmd_lead_m: following-error governor on the wrist command, metres. None (default)
        = strict parent parity: the wrist target stays an open integrator and this env
        reproduces `RigidCableVecEnv` exactly under an identity bank. Set it and the target is
        held within that distance of the MEASURED wrist. LEAVE IT None: Wave 2c measured the
        isotropic clamp NET-NEGATIVE for the task (it strips lateral authority and causes
        sideways ramming — tests/gates/REPORT.md, R9). Blocked until anisotropic.

        regoal_deadband_rad / regoal_every: re-goal suppression, both OFF by default
        (0.0 / 1) — see the module docstring's PLAN-CHURN section. These are PERF knobs and
        must stay off until measured; a coarse deadband quantizes fine alignment, and the
        task's lateral tolerance is 0.5 mm.

        The import is LAZY (inside __init__, not at module scope) on purpose: this module must
        stay importable — and the _IdealBank equivalence guard must stay runnable — on a box
        where the calibrated plant package is absent or half-installed. Passing an explicit
        bank never touches the import at all.

        Bank resolution and validation happen BEFORE super().__init__: building the scene costs
        ~25 s per call, and nobody should wait that long to be told the bank is the wrong shape.
        """
        if bank is None:
            from newton_cabling.servo_arm import ServoArmBank  # noqa: PLC0415  (lazy: see above)
            bank = ServoArmBank(n, frame_dt=FRAME_DT)
        for meth in ("reset", "set_goal", "advance"):
            if not callable(getattr(bank, meth, None)):
                raise TypeError(f"bank {type(bank).__name__} does not implement .{meth}() "
                                f"— see newton_cabling/servo_arm/CONTRACT.md")
        # Probe the CLASS, not the instance: `.jq` is a property that (correctly) RAISES on a
        # bank that has not been reset yet, and hasattr() only swallows AttributeError — an
        # instance probe here turned "validate the API" into "trip the fail-loud guard".
        if not hasattr(type(bank), "jq"):
            raise TypeError(f"bank {type(bank).__name__} has no .jq property "
                            f"— see newton_cabling/servo_arm/CONTRACT.md")
        if max_cmd_lead_m is not None and not (max_cmd_lead_m > 0.0):
            raise ValueError(f"max_cmd_lead_m must be positive or None, got {max_cmd_lead_m}")
        if regoal_deadband_rad < 0.0:
            raise ValueError(f"regoal_deadband_rad must be >= 0, got {regoal_deadband_rad}")
        if int(regoal_every) != regoal_every or regoal_every < 1:
            raise ValueError(f"regoal_every must be an integer >= 1, got {regoal_every}")
        limits = bank_joint_limits(bank)
        super().__init__(n, **kwargs)
        self.bank = bank
        self.max_cmd_lead_m = max_cmd_lead_m
        self.regoal_deadband_rad = float(regoal_deadband_rad)
        self.regoal_every = int(regoal_every)
        self._bank_limits = limits
        # Per-(env, joint) revolution offset, established at reset. Zero until then, and zero
        # for good against a bank that declares no limits.
        self._branch = np.zeros((n, 6))
        # set_goal() is per-env-id by contract; this env always commands the whole batch.
        self._bank_ids = np.arange(n, dtype=np.int64)
        # Land the plant on the built arm pose immediately: an env that has been constructed
        # but not yet reset() must never hold a bank in an undefined state (fail-loud house
        # rule — a zero-initialised plant would silently fling the arm on the first step).
        self.sync_bank_to_jq()

    # ── bank <-> env joint-coordinate plumbing ────────────────────────────────────────────
    # self.jq is the model's FULL joint-coordinate vector (arm + gripper + the free joints of
    # the cable body, the pad proxies and the jack). self.arm_qc, built by the parent, is the
    # (n, 6) int index array of the 6 ARM coordinates per env, so `self.jq[self.arm_qc]` reads
    # an (n, 6) arm block and `self.jq[self.arm_qc] = block` writes one. That slice is the
    # ENTIRE bank interface; every other coordinate stays the parent's business (contract
    # pin 3), which is why the substep vectors below are built from a jq_prev COPY.

    def _arm_context(self, q, what: str) -> str:
        """Per-joint table in degrees — what a bank-side range/shape raise is missing."""
        rows = "\n".join(
            f"    env {i}: " + "  ".join(f"{nm}={math.degrees(v):9.2f}"
                                         for nm, v in zip(ARM_JOINT_NAMES, q[i]))
            for i in range(q.shape[0]))
        turns = np.round(self._branch / TWO_PI).astype(int)
        return (f"{what} handed to the servo bank (degrees):\n{rows}\n"
                f"    branch offset (turns, established at reset): {turns.tolist()}")

    def _establish_branch(self, q0):
        """Pick the per-(env, joint) revolution offset that lands `q0` inside the bank's range.

        Minimal by construction: joints already in range keep offset 0, so the plant runs on
        the branch its calibration was fitted on. An out-of-range joint is moved by the whole
        number of turns that brings it nearest the middle of its range — the only choice that
        is both pose-preserving and range-restoring. A joint that is STILL out of range after
        that (e.g. a +/-135 deg joint at 200 deg) is genuinely unreachable, not a branch
        problem: leave it, and let the bank raise the honest error.
        """
        if self._bank_limits is None:
            return np.zeros_like(q0)
        lo, hi = self._bank_limits
        out = (q0 < lo - LIMIT_EPS) | (q0 > hi + LIMIT_EPS)
        if not out.any():
            return np.zeros_like(q0)
        turns = np.where(out, np.round((q0 - 0.5 * (lo + hi)) / TWO_PI), 0.0)
        assert np.array_equal(turns, np.round(turns)), "branch offset must be whole turns"
        return TWO_PI * turns

    def _to_bank(self, q_env, what: str):
        """Env joint frame -> bank joint frame. Raises if `q_env` needs a different branch."""
        q_bank = q_env - self._branch
        if self._bank_limits is not None:
            lo, hi = self._bank_limits
            bad = (q_bank < lo - LIMIT_EPS) | (q_bank > hi + LIMIT_EPS)
            if bad.any():
                rc = list(zip(*(x.tolist() for x in np.nonzero(bad))))
                raise ValueError(
                    f"{what} needs a DIFFERENT revolution branch than reset established, at "
                    f"(env, joint) {rc} — that is a real full-turn command, not bookkeeping: "
                    f"the arm is being asked to unwind a whole turn mid-episode. "
                    f"{self._arm_context(q_bank, what + ' in the BANK frame')}")
        return q_bank

    def sync_bank_to_jq(self) -> None:
        """Land the plant exactly on the env's current arm pose, zero velocity.

        Must follow every non-plant rewrite of the arm coordinates — reset()'s snapshot
        restore, set_stage()'s re-align — for the same reason `_sync_teleport` must follow
        every body_q rewrite: the plant integrates from its own remembered state, and a stale
        one turns a teleport into a commanded motion.
        """
        q0 = self.jq[self.arm_qc]
        # The branch offset is (re-)established HERE and only here, so it is constant for the
        # episode as Addendum 3 requires. Re-deriving it per step would let the arm creep a
        # turn at a time with nothing ever raising.
        self._branch = self._establish_branch(q0)
        try:
            self.bank.reset(self._to_bank(q0, "reset pose"))
        except ValueError as e:
            raise ValueError(f"{e}\n{self._arm_context(q0, 'reset pose (ENV frame)')}") from e
        # Force the next step() to issue a goal: the plant was just re-placed, so no in-flight
        # plan survives and the re-goal suppression below has nothing valid to compare against.
        self._last_goal = None                          # last goal ACTUALLY SENT, per env
        self._since_send = np.zeros(self.n, dtype=np.int64)

    def reset(self):
        obs = super().reset()
        self.sync_bank_to_jq()
        return obs

    def set_stage(self, stage: int):
        # The align/settle machinery (_align_fn, driven from set_stage) is AUTHORING, not
        # control: it ramps the wrist to a level hang while building the episode snapshot, and
        # it deliberately uses the parent's ideal interpolation. Only the policy loop goes
        # through the plant. Re-sync afterwards because the re-align rewrites self.jq.
        super().set_stage(stage)
        if hasattr(self, "bank"):     # parent __init__ calls set_stage(0) before self.bank exists
            self.sync_bank_to_jq()

    # ── the seam ──────────────────────────────────────────────────────────────────────────
    def _sim_from_stream(self, jq_prev, arm_stream):
        """`RigidCableVecEnv._sim(jq_prev)` with the interpolation replaced by `arm_stream`.

        arm_stream: (SUBSTEPS, n, 6) arm joint positions from the bank, one per substep.
        Every non-arm coordinate is carried from jq_prev unchanged — which is exactly what the
        parent's interpolation does to them (they are equal in jq_prev and self.jq, so
        `x + a * 0.0 == x` bit-for-bit). The substep body below is the parent's verbatim.
        """
        for s in range(SUBSTEPS):
            jq_s = jq_prev.copy()
            jq_s[self.arm_qc] = arm_stream[s]
            jqw = wp.array(jq_s, dtype=float, device=self.device)
            newton.eval_fk(self.model, jqw, self.model.joint_qd, self.state_0,
                           body_flag_filter=int(newton.BodyFlags.KINEMATIC))
            wp.launch(_sync_pad_anchors, dim=self._n_anchors,
                      inputs=(self.state_0.body_q, self.state_0.body_qd,
                              self._anc_par, self._anc_idx, self._anc_p, self._anc_q),
                      device=self.device)
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self, action):
        # ── COPY of RigidCableVecEnv.step (guarded by PARENT_SOURCE_SHA256) ───────────────
        # Only the block marked SEAT differs. Lines are kept verbatim, dead stores included
        # (the first face/_terms read is overwritten below without being used in the parent
        # too), so this body can be diffed against the parent line-by-line.
        a = np.asarray(action.detach().cpu() if hasattr(action, "detach") else action,
                       dtype=np.float64)
        a = np.clip(a, -1.0, 1.0)
        bqn = self.state_0.body_q.numpy()
        face, faceq = self._face_pose(bqn)
        along, lat, latn, ang = self._terms(face, faceq)
        # PURE RL: the action IS the gripper motion (user spec — no scripted base in the loop).
        # The only env-side modification is the finger<->jack STANDOFF clamp: never advance the
        # finger front past the jack mouth minus MARGIN. That is a physical constraint (a real
        # gripper cannot cross the panel), not a controller; retreating is always free.
        # The clamp runs BEFORE the bank (contract pin 4): the plant must be handed a goal the
        # gripper is allowed to reach, never a forbidden one it would chase against the panel.
        dpos = a[:, 0:3] * MAX_DPOS
        d_along = np.sum(dpos * self.ins, axis=1)
        room = np.maximum(0.0, (self.front_room_ep - MARGIN) - self.advanced)
        excess = np.maximum(0.0, d_along - room)
        dpos = dpos - excess[:, None] * self.ins
        self.advanced += np.sum(dpos * self.ins, axis=1)
        self.wrist_tgt_p = self.wrist_tgt_p + dpos
        drot = a[:, 3:6] * MAX_DROT
        self.wrist_tgt_q = (Rot.from_rotvec(drot) * Rot.from_quat(self.wrist_tgt_q)).as_quat()
        # ── SUBCLASS-ONLY: command-lead clamp (contract Addendum 3) ──────────────────────
        # Not in the parent, and skipped entirely when unset — that `is not None` is what keeps
        # the default configuration bit-identical to RigidCableVecEnv. `bqn` is the body state
        # read at the top of this step, i.e. the wrist as MEASURED before the arm moves.
        # NOTE the deliberate asymmetry: `self.advanced` above still counts the UNCLAMPED
        # command. It only feeds the finger<->jack standoff wall, and over-counting there makes
        # that wall trip EARLIER — the safe direction. Reconciling the two would couple the
        # panel constraint to the plant's tracking error, which is a different design.
        if self.max_cmd_lead_m is not None:
            wnow = bqn[self.wrist_body, :3]
            lead = self.wrist_tgt_p - wnow
            mag = np.linalg.norm(lead, axis=1, keepdims=True)
            over = mag > self.max_cmd_lead_m
            self.wrist_tgt_p = np.where(
                over, wnow + lead * (self.max_cmd_lead_m / np.maximum(mag, 1e-12)),
                self.wrist_tgt_p)
        # ── SEAT: parent lines 942-945 (jq_prev / ik.solve -> self.jq / _sim(jq_prev)) ────
        jq_prev = self.jq.copy()
        # IK from the PLANT'S measured pose (self.jq holds it, see below), so the joint goal is
        # the one that closes the CURRENT tracking error rather than the ideal arm's.
        jq_goal, _, _ = self.ik.solve(self.jq, self.wrist_tgt_p, self.wrist_tgt_q,
                                      iters=self.ik_iters)
        # env joint frame -> bank joint frame (revolution offset off); samples come back the
        # other way below. Everything the bank ever sees lives in the bank frame.
        q_star = self._to_bank(jq_goal[self.arm_qc], "IK joint goal")
        # Re-goal only the envs whose joint goal actually MOVED. set_goal is the expensive half
        # of the bank (a full retimer re-plan per env; advance-only is ~2.3x cheaper), and a
        # bank whose goal is unchanged holds its in-flight plan, which is what an unchanged
        # command means anyway. Exact inequality on purpose — a tolerance here would be a
        # silent "close enough to not bother" knob on the commanded trajectory.
        if self._last_goal is None:
            send = np.ones(self.n, dtype=bool)          # first frame after a plant re-placement
        else:
            send = np.any(q_star != self._last_goal, axis=1)
            # ── R2 suppression (contract Addendum 4). BOTH clauses are SKIPPED at their
            # defaults, so `send` is then literally the exact-inequality mask above and the
            # default configuration is bit-identical by construction, not by measurement.
            if self.regoal_deadband_rad > 0.0:
                # Compared against the last goal ACTUALLY SENT, never against last frame's
                # candidate: that is what lets a slow drift accumulate past the deadband and
                # eventually fire. Comparing to the candidate would re-baseline every frame
                # and freeze a creeping goal forever.
                send &= np.any(np.abs(q_star - self._last_goal) > self.regoal_deadband_rad,
                               axis=1)
            if self.regoal_every > 1:
                # Staleness floor: nothing may go more than regoal_every frames unsent, no
                # matter how small the deltas. regoal_every == 1 needs no floor — with no
                # deadband the only suppressed frames are bit-identical goals — so the clause
                # is skipped rather than made trivially true, keeping the default path exact.
                send |= self._since_send >= self.regoal_every - 1
        changed = self._bank_ids[send]
        if changed.size:
            try:
                self.bank.set_goal(changed, q_star[changed])
            except ValueError as e:
                raise ValueError(
                    f"{e}\n{self._arm_context(q_star, 'IK joint goal (BANK frame)')}") from e
            if self._last_goal is None:
                self._last_goal = q_star.copy()
            else:
                # ONLY the rows that went out: an unsent row must keep the goal the bank is
                # actually holding. (At the defaults this is a distinction without a
                # difference — an unsent row's candidate is bit-equal to its stored goal —
                # which is why it does not perturb the equivalence guard.)
                self._last_goal[changed] = q_star[changed]
        self._since_send = np.where(send, 0, self._since_send + 1)
        arm_stream = self.bank.advance(SUBSTEPS)
        arm_stream = np.asarray(arm_stream, dtype=np.float64)
        if arm_stream.shape != (SUBSTEPS, self.n, 6):
            raise ValueError(f"bank.advance({SUBSTEPS}) returned {arm_stream.shape}, expected "
                             f"{(SUBSTEPS, self.n, 6)} — see newton_cabling/servo_arm/CONTRACT.md")
        if not np.isfinite(arm_stream).all():
            raise ValueError("bank.advance returned non-finite joint positions")
        # bank joint frame -> env joint frame, BEFORE anything drives Newton or self.jq. With
        # the offset zero (every bank without limits, and every in-range build) `x + 0.0 == x`
        # bit-for-bit, so the default path is untouched.
        self._sim_from_stream(jq_prev, arm_stream + self._branch)
        # self.jq must end at where the ARM ACTUALLY IS, not at the commanded goal: it is the
        # next step's IK seed and the FK source, and a plant that lags its goal would otherwise
        # accumulate a phantom offset. ik.solve only ever writes arm coordinates, so taking
        # jq_goal and overwriting the arm block leaves every other coordinate untouched.
        self.jq = jq_goal
        self.jq[self.arm_qc] = self.bank.jq + self._branch
        # ── end SEAT ──────────────────────────────────────────────────────────────────────

        bqn = self.state_0.body_q.numpy()
        bqd = self.state_0.body_qd.numpy()
        face, faceq = self._face_pose(bqn)
        along, lat, latn, ang = self._terms(face, faceq)
        d = self._dist(face, faceq)
        rew = W_PROG * (self.prev_dist - d)
        seated = (along >= -SEAT_DEPTH_TOL) & (latn <= SEAT_OFFSET) & (ang <= SEAT_ANGLE)
        speed = (np.linalg.norm(bqd[self.rod_front, 0:3], axis=1)
                 + np.linalg.norm(bqd[self.rod_front, 3:6], axis=1))
        rew = rew + np.where(seated, R_SUCCESS - W_VEL * speed, 0.0)
        viol = self._violations()
        self.viol_last = viol
        rew = rew - W_VIOL * (viol > 0)
        self.hold = np.where(seated, self.hold + 1, 0)
        success = (self.hold >= HOLD_STEPS).astype(np.float32)
        dist_seat = np.linalg.norm(face - self.seat_pos, axis=1)
        bad = (~np.isfinite(face).all(axis=1)) | (dist_seat > EJECT_DIST)
        rew = np.where(bad, -1.0, rew)
        success = np.where(bad, 0.0, success)
        d = np.where(bad, self.prev_dist, d)
        self.prev_dist = d
        self.seated_inst_frac = float(np.mean(np.where(bad, False, seated)))
        depth_mm = np.clip(np.linalg.norm(face - self.face_start, axis=1), 0, 0.2) * 1000.0
        obs = self._obs(bqn)
        z = torch.zeros(self.n, device=DEV)
        to_t = lambda x: torch.as_tensor(np.nan_to_num(x), dtype=torch.float32, device=DEV)  # noqa: E731
        return obs, to_t(rew), z, to_t(success), to_t(depth_mm)
