"""ServoArmBank — n hardware-calibrated RO2-core servo-arm plants, vectorized.

Implements the frozen ServoArmBank contract: put the calibrated servo model
(parallax-demo-newton ``sysid/`` + ``control/``) in the loop of the cable envs
as a joint-trajectory generator. The env keeps driving its Newton arm
kinematically per substep — only the SOURCE of the per-substep joint positions
changes, from "linear interpolation to the IK result" to "what the real,
calibrated arm would actually do when handed that IK result as a goal".

One calibration, one command path (pinned)
------------------------------------------
goal -> robot retimer (control.retime, VERBATIM port: ruckig point-to-point)
     -> 500 Hz position stream with transport delay (control.stream)
     -> calibrated servo plant (sysid.plant, gravity-compensated MuJoCo).

There is no controller here and no direct setpoint path: the arm's onboard
servo closes the loop and the plant models it (sysid/README.md). Goals follow
two paths (contract Addendum 4, ruling R1):

  * COLD START (commanded state at rest — hold, or a plan fully streamed):
    ``plan_move``, the ported robot retimer, rest-to-rest, stamped so the
    transport delay applies as dead time. Identical to the reference
    integration's ``move_to``.
  * INTERRUPT (a plan is in flight): a ruckig OTG replan seeded with the
    currently DELIVERED commanded position, VELOCITY and ACCELERATION
    (command-side state — a robot's controller re-plans from what it
    commanded, not from measured joint state), targeting the new goal at
    rest, under the same VMAX/AMAX/JMAX imported from the verbatim port.
    Without the seed, a 60 Hz re-goal stream replaces every plan inside its
    own initial jerk ramp and the arm achieves 0.12% of commanded motion
    (tests/gates/REPORT.md, Gate 2) — the plant is fine; rest-to-rest
    replanning is not.

Interrupt continuity therefore means position AND velocity continuity of the
commanded stream; both are pinned by tests.

FIDELITY CAVEAT (pinned by Addendum 4): no hardware recording exists of the
real arm under high-rate re-goals. The OTG path yields a PLAUSIBLE arm, not a
validated one — the calibration itself was fitted on discrete task-mediated
moves seconds apart. The next hardware ask is one recording of the arm
streamed goals at policy rate.

Timing model
------------
The plant steps at its own MJCF timestep (2 ms), which equals the robot's
500 Hz stream tick — asserted at construction, because the calibration artifact
was fitted at that timestep and one-stream-sample-per-plant-step is the cadence
the fit validated. Each tick samples the (delay-shifted) stream and writes the
position ctrl, ZOH until the next tick, exactly like the reference integration
writes the streamed setpoint once per physics substep.

The transport delay lives INSIDE the bank, as ``SetpointStreamer(delay_s=...)``
— a time shift on the continuous trajectory, exact at any delay (the robot-side
implementation this mirrors keeps an in-flight buffer of maxlen d+1; the shift
is the continuous-time equivalent, see control/stream.py's module docstring).
The env must not add its own.

One stamping subtlety makes the shift equal the real buffer in BOTH regimes.
The streamer clamps a plan to its start for ``delay_s`` after its stamp. For a
COLD start that is exactly the d+1 buffer: the samples in flight are the hold —
constant — so "frozen at plan start" IS "old samples keep executing", and the
new plan's motion lands one delay late, as it must. For an INTERRUPT the
samples in flight are MOVING, so stamping the replan at ``now`` would freeze
delivery for 10 of every 16.7 ms under per-frame re-goals — an O(v * delay)
artifact per re-goal that caps a 60 Hz march at 26% of commanded motion
(measured). The OTG replan is therefore seeded from the DELIVERED commanded
state and stamped at ``now - delay_s``, so it takes over mid-flight exactly
where the old plan's in-flight samples leave off; the delivered command then
matches the true buffer's continuation to O(jmax * delay^3) ~ 5e-7 rad (both
evolve jerk-limited from the same position/velocity/acceleration). The delay
mechanism itself — one scalar from the artifact, applied by the streamer — is
unchanged.

``advance(substeps)`` runs one control frame of ``frame_dt`` and returns the
measured joint positions resampled on the env's substep grid: sample k lands at
``frame_start + (k+1) * frame_dt / substeps``, so the last sample is the frame
end. Plant ticks (2 ms) need not divide ``frame_dt`` (the envs run 60 Hz and
30 Hz frames — neither divides evenly): the bank keeps the plant on its own
uninterrupted 2 ms grid, always integrated to at-or-past the frame boundary,
and the returned samples are linear interpolation of the tick-grid measured
trajectory. The interpolation error is O(dt^2 * qdd) ~ 1e-6 rad, two orders
below the model's own 0.07 deg accuracy — while the alternative (warping the
plant timestep to fit the frame) would silently perturb the calibrated
dynamics. Both clocks are computed as integer multiples (never accumulated), so
there is no drift and runs are bit-reproducible.

Plant configuration: gripper ON, payload compensated
----------------------------------------------------
``Plant()`` defaults to gripper-OFF because the sysid recordings had no gripper
fitted; this bank runs the DEMO configuration, ``Plant(with_gripper=True)``
(total mass 31.23 kg, not 30.01), with the calibrated per-joint gains applied
through ``Plant.apply`` — the constructor alone loads the MJCF's placeholder
gains (kp=6000 everywhere; the calibrated values span 15.9 .. 161218).

One extra flag beyond ``Plant(with_gripper=True)``: the AG-145 housing mass
rides in the joint-less body ``ag145_payload`` (rigidly welded to wrist_3, so
the dynamics already match the demo's fused point mass), but
``sysid.plant.enable_gravity_compensation`` only covers the six ARM_BODIES. In
the demo Newton build the housing is fused INTO wrist_3, whose gravcomp=1
covers it — so for parity the bank sets ``body_gravcomp=1`` on ``ag145_payload``
too (measured at the calibration's working pose: J4 parks 0.040 deg off goal
without the flag, 0.004 deg with it; the remainder is the eight PD-held finger
links, which the demo also leaves uncompensated). ``mj_setConst`` after the
write is NOT optional: on a compiled model the write is a silent no-op until
the derived constants are recomputed (sysid/README.md's one MuJoCo trap).

Gripper/finger coordinates are entirely the env's business (pinned): here the
fingers exist only as calibrated payload/reaction mass, PD-held at the model's
neutral pose by the MJCF's own finger actuators.

Determinism (pinned)
--------------------
No RNG anywhere; single-threaded MuJoCo; clocks from integer multiples. Same
(jq0, goal sequence, advance sequence) -> bit-identical outputs. Fail loudly
on anything off-contract: missing calibration, wrong shapes, non-finite or
out-of-joint-range goals, calls before reset(). No silent defaults.
"""
from __future__ import annotations

import mujoco
import numpy as np

from control.retime import (AMAX_RAD_S2, JMAX_RAD_S3, STREAM_DT,  # single source
                            VMAX_RAD_S)                           # of truth: limits
from control.stream import SetpointStreamer, plan_move
from sysid.params import N_ARM, load_servo_params, servo_transport_delay_s
from sysid.plant import Plant

__all__ = ["ServoArmBank"]

_PAYLOAD_BODY = "ag145_payload"
# Boundary tolerance for the tick/frame clock comparisons — absorbs the ~1-ulp
# mismatch between k*dt and m*frame_dt when frame_dt IS a tick multiple, and is
# six orders below the 2 ms tick, so it can never eat a real tick.
_EPS = 1e-9


class _PositionStream(SetpointStreamer):
    """SetpointStreamer with a position-only sampling fast path.

    ``sample()`` derives velocity+acceleration feedforward by running
    ``np.gradient`` over the ENTIRE (M, 6) trajectory on every call — ~100 us
    for a typical plan — to return values this plant never consumes (sample()'s
    own docstring: only ``q`` reaches the plant; the feedforward's effect is
    already inside the fitted gains). At n=64 arms x 500 Hz that is ~50 ms of
    pure waste per 33 ms control frame, an order of magnitude over the physics.

    ``sample_q`` is the position leg of the parent's ``sample``, nothing else:
    same hold/interrupt/delay/past-the-end semantics through the same parent
    state (``set_trajectory``/``set_hold`` are untouched). Pinned against drift
    by tests/test_servo_arm.py::test_position_fastpath_matches_stream_sample,
    which asserts bit-equality with ``sample()[0]`` across every branch.
    """

    def sample_q(self, now_s: float) -> np.ndarray:
        """Position to command at ``now_s``. Same branches as ``sample()``.

        Returns a reference (not a copy) — callers here only read it into
        ``ctrl`` / hand it to ``plan_move``, which copies.
        """
        if self._traj is None:
            if self._hold is None:
                raise RuntimeError(
                    "stream sampled with neither a trajectory nor a hold — "
                    "the bank must be reset() before it is advanced")
            return self._hold
        t = (float(now_s) - self._t0) - self.delay_s
        if t >= self._traj.duration_s:
            return self._traj.end_q
        return self._traj.at(t)


class _OTGTrajectory:
    """Duck-typed stand-in for ``control.stream.Trajectory``, backed directly
    by a ruckig ``Trajectory`` object (the OTG replan's own profile).

    Implements the exact surface ``SetpointStreamer`` consumes (``duration_s``,
    ``end_q``, ``at``, ``velocity_at``, ``acceleration_at``, same [0, duration]
    clamping) with ruckig's closed-form ``at_time`` instead of a dense
    STREAM_DT position grid. Two reasons this is a stand-in and not a sampled
    port-style grid:

      * the interrupt path needs the commanded (q, qd, qdd) EXACTLY at the
        next interrupt to seed the next replan — ``at_time`` is analytic,
        where the grid's np.gradient feedforward is smoothed and ~50x more
        expensive per query;
      * OTG plans are replaced every frame under policy-rate re-goals, so the
        one-time cost of densifying a multi-second plan (hundreds of at_time
        calls) would be paid 64x per frame for samples never consumed.

    The ruckig profile is solved in a delta frame anchored at the seed
    position (same trick, same reason as the verbatim ``_ruckig_retime``:
    ruckig hits float edge cases for large absolute positions with small
    deltas), and shifted back on every query.
    """

    __slots__ = ("_rt", "_anchor", "duration_s", "end_q")

    def __init__(self, rt, anchor: np.ndarray):
        self._rt = rt
        self._anchor = anchor
        self.duration_s = float(rt.duration)
        self.end_q = np.asarray(rt.at_time(self.duration_s)[0], dtype=np.float64) + anchor

    def _clamp(self, t_s: float) -> float:
        return min(max(float(t_s), 0.0), self.duration_s)

    def at(self, t_s: float) -> np.ndarray:
        p = self._rt.at_time(self._clamp(t_s))[0]
        return np.asarray(p, dtype=np.float64) + self._anchor

    def velocity_at(self, t_s: float) -> np.ndarray:
        return np.asarray(self._rt.at_time(self._clamp(t_s))[1], dtype=np.float64)

    def acceleration_at(self, t_s: float) -> np.ndarray:
        return np.asarray(self._rt.at_time(self._clamp(t_s))[2], dtype=np.float64)

    def state_at(self, t_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(q, qd, qdd) in one at_time call — the replan seed."""
        p, v, a = self._rt.at_time(self._clamp(t_s))
        return (np.asarray(p, dtype=np.float64) + self._anchor,
                np.asarray(v, dtype=np.float64),
                np.asarray(a, dtype=np.float64))


def _otg_replan(start_q: np.ndarray, start_qd: np.ndarray, start_qdd: np.ndarray,
                goal_q: np.ndarray) -> _OTGTrajectory:
    """Ruckig replan from a MOVING commanded state to ``goal_q`` at rest.

    The interrupt half of Addendum 4's R1. Limits come from the verbatim port
    (VMAX/AMAX/JMAX — one source of truth), the profile is jerk-limited like
    every plan the real task streams, and the seed is the command-side state.

    No toppra fallback here, deliberately: toppra parameterizes a geometric
    path FROM REST and cannot honor a nonzero seed velocity — falling back
    would silently teleport the commanded velocity to zero, which is the exact
    0.12%-tracking bug this path exists to fix. A failed solve raises.
    """
    from ruckig import InputParameter, Ruckig  # lazy, mirroring control.retime
    from ruckig import Trajectory as _RuckigTrajectory

    inp = InputParameter(N_ARM)
    inp.current_position = [0.0] * N_ARM
    inp.current_velocity = [float(x) for x in start_qd]
    inp.current_acceleration = [float(x) for x in start_qdd]
    inp.target_position = [float(x) for x in (goal_q - start_q)]
    inp.target_velocity = [0.0] * N_ARM
    inp.target_acceleration = [0.0] * N_ARM
    inp.max_velocity = [VMAX_RAD_S] * N_ARM
    inp.max_acceleration = [AMAX_RAD_S2] * N_ARM
    inp.max_jerk = [JMAX_RAD_S3] * N_ARM
    otg = Ruckig(N_ARM)
    rt = _RuckigTrajectory(N_ARM)
    try:
        result = otg.calculate(inp, rt)
    except Exception as exc:
        raise RuntimeError(f"OTG replan failed: {exc}") from exc
    if int(result) < 0:
        raise RuntimeError(
            f"OTG replan failed: ruckig result {result} for seed qd={start_qd} "
            f"qdd={start_qdd} delta={goal_q - start_q}")
    return _OTGTrajectory(rt, np.asarray(start_q, dtype=np.float64).copy())


def _clip_seed(x: np.ndarray, limit: float, name: str) -> np.ndarray:
    """Clip a replan seed to its limit, loudly if the overage is real.

    Seeds from an OTG plan are within limits by construction (ruckig respects
    its own bounds) up to float round-off; seeds from a cold plan's derived
    feedforward carry np.gradient discretization noise of O(jmax * dt) ~ 3e-3
    at the acceleration peaks. Both are noise, not state — clipped. Anything
    beyond 2% of the limit means the command-state bookkeeping is broken and
    raises instead of feeding ruckig garbage.
    """
    x = np.asarray(x, dtype=np.float64)
    over = float(np.abs(x).max()) - limit
    if over > 0.02 * limit:
        raise RuntimeError(
            f"commanded {name} exceeds its limit by {over:.3e} "
            f"(limit {limit}) — command-state bookkeeping is broken")
    return np.clip(x, -limit, limit)


def _enable_payload_gravcomp(model: mujoco.MjModel) -> None:
    """Compensate the AG-145 housing's weight, matching the demo build.

    The demo Newton path fuses the housing mass into wrist_3
    (SbotRO2CoreLearned._fuse_point_mass) and wrist_3 carries gravcomp=1, so
    the payload's weight is compensated by construction. The MJCF twin keeps
    the same mass in the welded child body ``ag145_payload`` — dynamically
    identical, but ``enable_gravity_compensation`` only flags the six arm
    links, leaving 1.103 kg for the wrist servos to hold (J4: 0.040 deg of
    steady droop at the working pose). One flag restores parity. The finger
    bodies stay uncompensated on purpose: the demo leaves them uncompensated
    too, and parity — not perfection — is the requirement.

    The trailing ``mj_setConst`` is load-bearing: writing ``body_gravcomp`` on
    a compiled model is a SILENT NO-OP until the derived constants are
    recomputed — ``qfrc_gravcomp`` stays exactly 0.0 and the servo absorbs
    gravity, which reads as plausible droop rather than an error.
    """
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _PAYLOAD_BODY)
    if b < 0:
        raise KeyError(
            f"body {_PAYLOAD_BODY!r} not in model — the MJCF twin changed; "
            "re-derive the payload-compensation parity against the demo build")
    model.body_gravcomp[b] = 1.0
    mujoco.mj_setConst(model, mujoco.MjData(model))


class ServoArmBank:
    """n independent calibrated arms. Pure CPU: numpy + mujoco + ruckig/toppra.

    NO torch, NO warp, NO newton imports — must run in the no-GPU test gate.
    See the module docstring for the full semantics; the per-method docstrings
    below restate only what each call pins.
    """

    def __init__(self, n: int, *, frame_dt: float, params=None):
        """Build the bank: one calibrated MjModel shared by n MjData plants.

        ``params`` defaults to ``sysid.params.load_servo_params()``, which
        FAILS LOUDLY if the calibration artifact is absent or incomplete — that
        behavior is preserved, never wrapped in a fallback. A custom dict must
        cover all six joints (checked here, so a partial dict cannot silently
        leave XML placeholder gains on the missing joints).

        The transport delay is always read from the calibration artifact
        (``servo_transport_delay_s``): it is a property of the calibrated comms
        path, not a per-joint gain, so custom gain dicts do not carry it.
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
            raise TypeError(f"n must be an int, got {type(n).__name__}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        frame_dt = float(frame_dt)
        if not np.isfinite(frame_dt) or frame_dt <= 0.0:
            raise ValueError(f"frame_dt must be finite and > 0, got {frame_dt}")
        if params is None:
            params = load_servo_params()          # raises if uncalibrated
        missing = [j for j in range(N_ARM) if j not in params]
        if missing:
            raise KeyError(
                f"params is missing joints {missing} — a partial gain set would "
                "silently leave the MJCF placeholder gains on those joints")

        self.n = int(n)
        self.frame_dt = frame_dt
        self._delay_s = servo_transport_delay_s()

        # The demo configuration: gripper ON (AG-145 payload), calibrated gains
        # applied per joint — Plant() alone carries placeholder gains — and the
        # payload's gravity compensated for parity with the demo's fused build.
        self._plant = Plant(with_gripper=True)
        for j in range(N_ARM):
            self._plant.apply(j, params[j])
        _enable_payload_gravcomp(self._plant.model)
        self._model = self._plant.model

        self._dt = float(self._model.opt.timestep)
        if abs(self._dt - STREAM_DT) > 1e-12:
            raise RuntimeError(
                f"plant timestep {self._dt} != 500 Hz stream tick {STREAM_DT} — "
                "the calibration was fitted at the MJCF timestep and the bank "
                "writes one stream sample per plant step; a changed timestep is "
                "a recalibration event, not a knob to absorb here")

        self._qadr = np.asarray(self._plant.qadr, dtype=int)
        self._qdof = np.asarray(self._plant.dof, dtype=int)
        self._act = np.asarray(self._plant.act, dtype=int)
        # Arm joint limits, for the never-hand-the-bank-an-unreachable-goal
        # pin: goal clamping happens BEFORE the bank, so an out-of-range goal
        # (or start) here is a caller bug and raises.
        jids = [next(jid for jid in range(self._model.njnt)
                     if int(self._model.jnt_qposadr[jid]) == int(qa))
                for qa in self._qadr]
        self._jnt_lo = np.array([self._model.jnt_range[j, 0] for j in jids])
        self._jnt_hi = np.array([self._model.jnt_range[j, 1] for j in jids])

        self._data = [mujoco.MjData(self._model) for _ in range(self.n)]
        self._streams = [_PositionStream(delay_s=self._delay_s)
                         for _ in range(self.n)]

        # Clocks as integer counters — times are always computed as k * dt /
        # m * frame_dt products, never accumulated, so long runs cannot drift.
        self._frame = 0        # control frames advanced since reset
        self._tick = 0         # plant steps taken since reset
        self._ready = False    # reset() must run before anything else

        # Measured-state history at tick times, spanning the current frame
        # boundary, for the substep-grid resampling (trimmed to the last two
        # entries after every advance — enough because the newest entry is
        # always at-or-past the frame boundary and ticks are appended in order).
        self._hist_t: list[float] = []
        self._hist_q: list[np.ndarray] = []
        self._hist_qd: list[np.ndarray] = []
        self._jq = np.zeros((self.n, N_ARM))
        self._jqd = np.zeros((self.n, N_ARM))

    # ------------------------------------------------------------- validation
    def _require_reset(self, what: str) -> None:
        if not self._ready:
            raise RuntimeError(
                f"{what} called before reset() — the bank has no defined state "
                "until reset(jq0) places every plant")

    def _check_qmat(self, x, rows: int, name: str) -> np.ndarray:
        q = np.asarray(x, dtype=np.float64)
        if q.shape != (rows, N_ARM):
            raise ValueError(
                f"{name} must have shape ({rows}, {N_ARM}), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError(f"{name} contains non-finite values")
        bad = (q < self._jnt_lo - _EPS) | (q > self._jnt_hi + _EPS)
        if bad.any():
            rows_bad, cols_bad = np.nonzero(bad)
            raise ValueError(
                f"{name} outside joint limits at (row, joint) "
                f"{list(zip(rows_bad.tolist(), cols_bad.tolist()))} — goal "
                "clamping happens BEFORE the bank; the bank must never chase "
                "an unreachable goal")
        return q

    # -------------------------------------------------------------------- API
    def reset(self, jq0: np.ndarray) -> None:
        """(n, 6) arm joint positions. Plant lands exactly at jq0, zero
        velocity, stream holding jq0 (no trajectory, nothing in flight)."""
        jq0 = self._check_qmat(jq0, self.n, "jq0")
        for i in range(self.n):
            d = self._data[i]
            mujoco.mj_resetData(self._model, d)      # gripper at neutral, ctrl 0
            d.qpos[self._qadr] = jq0[i]
            mujoco.mj_forward(self._model, d)
            self._streams[i].set_hold(jq0[i])        # interrupts + empties
        self._frame = 0
        self._tick = 0
        self._hist_t = [0.0]
        self._hist_q = [jq0.copy()]
        self._hist_qd = [np.zeros((self.n, N_ARM))]
        self._jq = jq0.copy()
        self._jqd = np.zeros((self.n, N_ARM))
        self._ready = True

    def _command_state(self, i: int, now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Commanded (q, qd, qdd) DELIVERED to plant i at ``now`` — the replan seed.

        Command-side state, never plant-measured qvel (Addendum 4: a robot's
        controller re-plans from what it commanded). Exact (analytic at_time)
        when the active plan is an OTG replan; derived via the parent
        Trajectory's own feedforward methods when it is a cold ``plan_move``
        plan; rest when holding or fully streamed.
        """
        s = self._streams[i]
        traj = s._traj
        zero = np.zeros(N_ARM)
        if traj is None:                              # holding
            return s.sample_q(now).copy(), zero, zero.copy()
        tc = (now - s._t0) - s.delay_s
        if tc >= traj.duration_s:                     # fully streamed: parked at rest
            return np.asarray(traj.end_q, dtype=np.float64).copy(), zero, zero.copy()
        tc = max(tc, 0.0)
        if isinstance(traj, _OTGTrajectory):
            return traj.state_at(tc)
        return traj.at(tc), traj.velocity_at(tc), traj.acceleration_at(tc)

    def set_goal(self, env_ids: np.ndarray, q_star: np.ndarray) -> None:
        """New joint goal per listed env, INTERRUPTING any in-flight plan.

        Two paths, per contract Addendum 4 (R1):

          * commanded state at rest -> COLD START: ``plan_move`` (the ported
            robot retimer, rest-to-rest), stamped at ``now`` so the transport
            delay applies as dead time — for a hold, the in-flight samples are
            constant and the streamer's clamp reproduces the real d+1 buffer
            exactly.
          * plan in flight -> OTG REPLAN: ruckig seeded with the delivered
            commanded position/velocity/acceleration, stamped at
            ``now - delay_s`` so it takes over exactly where the old plan's
            in-flight samples leave off (see the module docstring: stamping at
            ``now`` would freeze delivery for a delay after every re-goal —
            measured 26% tracking under 60 Hz re-goals — while the splice
            matches the true in-flight buffer to O(jmax * delay^3)).

        Commanded position AND velocity stay continuous across the interrupt;
        both are pinned by tests.
        """
        self._require_reset("set_goal")
        ids = np.asarray(env_ids)
        if ids.ndim != 1:
            raise ValueError(f"env_ids must be 1-D, got shape {ids.shape}")
        if ids.size == 0:
            return
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(
                f"env_ids must be an integer array, got dtype {ids.dtype}")
        if np.any((ids < 0) | (ids >= self.n)):
            raise IndexError(
                f"env_ids out of range [0, {self.n}): {ids.tolist()}")
        if np.unique(ids).size != ids.size:
            raise ValueError(
                f"env_ids contains duplicates: {ids.tolist()} — ambiguous "
                "(two goals for one env in a single call)")
        q_star = self._check_qmat(q_star, ids.size, "q_star")
        now = self.frame_dt * self._frame
        for k, i in enumerate(ids):
            i = int(i)
            s = self._streams[i]
            q0, v0, a0 = self._command_state(i, now)
            if not (np.any(v0 != 0.0) or np.any(a0 != 0.0)):
                s.set_trajectory(plan_move(q0, q_star[k]), now)
            else:
                v0 = _clip_seed(v0, VMAX_RAD_S, "velocity")
                a0 = _clip_seed(a0, AMAX_RAD_S2, "acceleration")
                s.set_trajectory(_otg_replan(q0, v0, a0, q_star[k]),
                                 now - self._delay_s)

    def advance(self, substeps: int) -> np.ndarray:
        """Advance one control frame (``frame_dt``): tick the 500 Hz stream
        through the delay into the plants, then return the measured joint
        positions resampled on the env's substep grid.

        Returns (substeps, n, 6); sample k is the state at
        ``frame_start + (k+1) * frame_dt / substeps`` — the last sample is the
        frame end and equals ``self.jq`` bit-exactly. Because ``substeps`` only
        selects the OUTPUT grid (the physics runs on its own tick regardless),
        advancing with 4 vs 8 substeps yields nested, bit-identical samples.
        """
        self._require_reset("advance")
        if isinstance(substeps, bool) or not isinstance(substeps, (int, np.integer)):
            raise TypeError(f"substeps must be an int, got {type(substeps).__name__}")
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")

        t_end = self.frame_dt * (self._frame + 1)
        # Integrate every plant to at-or-past the frame boundary. Each tick
        # samples its env's stream at the tick time (ZOH) and steps 2 ms. A
        # goal issued at the frame boundary first reaches the first tick at or
        # after that boundary — the bank never pre-consumes a future command,
        # because ticks past the previous boundary were stepped with controls
        # written BEFORE it.
        while self._dt * self._tick < t_end - _EPS:
            t_tick = self._dt * self._tick
            q_t = np.empty((self.n, N_ARM))
            qd_t = np.empty((self.n, N_ARM))
            for i in range(self.n):
                d = self._data[i]
                d.ctrl[self._act] = self._streams[i].sample_q(t_tick)
                mujoco.mj_step(self._model, d)
                q_t[i] = d.qpos[self._qadr]
                qd_t[i] = d.qvel[self._qdof]
            self._tick += 1
            self._hist_t.append(self._dt * self._tick)
            self._hist_q.append(q_t)
            self._hist_qd.append(qd_t)

        T = np.asarray(self._hist_t)
        Q = np.asarray(self._hist_q)                  # (K, n, 6)
        Qd = np.asarray(self._hist_qd)
        taus = self.frame_dt * (self._frame + np.arange(1, substeps + 1) / substeps)
        if taus[-1] > T[-1] + 1e-7:
            raise RuntimeError(
                f"clock invariant broken: frame end {taus[-1]!r} is beyond the "
                f"last plant tick {T[-1]!r} — the tick loop under-integrated")
        tq = np.minimum(taus, T[-1])                  # absorb <=_EPS boundary slack
        idx = np.clip(np.searchsorted(T, tq, side="left"), 1, len(T) - 1)
        w = ((tq - T[idx - 1]) / (T[idx] - T[idx - 1]))[:, None, None]
        out = Q[idx - 1] + w * (Q[idx] - Q[idx - 1])
        qd = Qd[idx - 1] + w * (Qd[idx] - Qd[idx - 1])

        self._jq = out[-1].copy()
        self._jqd = qd[-1].copy()
        self._frame += 1
        # Keep the last two tick entries: the newest is at-or-past the new
        # boundary, the one before is strictly before it — together they
        # bracket every sample time the next frame can ask for.
        self._hist_t = self._hist_t[-2:]
        self._hist_q = self._hist_q[-2:]
        self._hist_qd = self._hist_qd[-2:]
        return out

    @property
    def jq(self) -> np.ndarray:
        """(n, 6) measured joint positions at the current frame boundary."""
        self._require_reset("jq")
        return self._jq.copy()

    @property
    def jqd(self) -> np.ndarray:
        """(n, 6) measured joint velocities at the current frame boundary."""
        self._require_reset("jqd")
        return self._jqd.copy()
