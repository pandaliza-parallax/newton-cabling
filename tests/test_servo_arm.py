"""Unit gate for newton_cabling.servo_arm.ServoArmBank (no GPU, < 5 s).

What is pinned here, per the frozen ServoArmBank contract:

  * step response  — no overshoot; dead time = the artifact's transport delay,
                     asserted explicitly; tracking lag consistent with each
                     joint's fitted T_lag = kv/kp (~14-18 ms)
  * interrupt      — re-goal mid-plan produces no position discontinuity in
                     the output samples (mirrors control's own interrupt gate)
  * determinism    — same (reset, goals, advance) script -> bit-identical
                     outputs, across fresh banks AND across re-reset
  * resampling     — (substeps, n, 6), monotone stitching across frames, last
                     sample bit-equal to bank.jq, substep count only selects
                     the output grid (4-substep samples nest inside 8-substep)
  * plant config   — gripper ON (31.23 kg, not the stripped 30.01), calibrated
                     gains applied (not the MJCF's 6000 placeholders), AG-145
                     payload gravity-compensated THROUGH the derived constants
                     (the mj_setConst silent-no-op trap)
  * fail loudly    — wrong shapes, non-finite / out-of-range goals, bad env
                     ids, use before reset: all raise

Run from the repo root:  .venv/bin/python -m pytest tests/test_servo_arm.py -q
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from control.retime import VMAX_RAD_S
from control.stream import SetpointStreamer, plan_move
from newton_cabling.servo_arm import ServoArmBank
from newton_cabling.servo_arm.bank import _PositionStream
from sysid.params import N_ARM, NOMINAL_Q, load_servo_params, servo_transport_delay_s

FRAME = 1.0 / 30.0
SUB = 8
SUB_DT = FRAME / SUB

# Steady droop bound: the eight finger links are PD-held but not gravity-
# compensated (demo parity — see bank.py), which parks the wrist joints up to
# ~0.005 deg (~9e-5 rad) off the commanded position. Settle/overshoot
# tolerances sit above that floor and well below any real servo pathology.
DROOP_RAD = 2e-4


def _bank(n: int) -> ServoArmBank:
    return ServoArmBank(n, frame_dt=FRAME)


def _tile(n: int) -> np.ndarray:
    return np.tile(NOMINAL_Q, (n, 1))


# ---------------------------------------------------------------- step response
def test_step_response_dead_time_no_overshoot_lag():
    """One env per joint, each stepping its own joint by +0.25 rad."""
    delay = servo_transport_delay_s()
    params = load_servo_params()
    n = N_ARM
    bank = _bank(n)
    jq0 = _tile(n)
    bank.reset(jq0)

    # Pre-settle the finger-gravity transient so the dead-time window is
    # measured against a truly static plant (the stream still holds jq0
    # exactly, so the later plan starts from jq0, not from the drooped pose).
    # 1.0 s: the creep is ~1e-7 rad/s at 0.5 s, ~1e-11 at 1.0 s.
    for _ in range(30):
        bank.advance(SUB)
    q_eq = bank.jq
    assert np.abs(q_eq - jq0).max() < DROOP_RAD
    assert np.abs(bank.jqd).max() < 1e-8

    step = 0.25
    goal = jq0.copy()
    for j in range(N_ARM):
        goal[j, j] += step
    bank.set_goal(np.arange(n), goal)

    n_frames = int(np.ceil(3.4 / FRAME))
    ts, qs = [], []
    for f in range(n_frames):
        qs.append(bank.advance(SUB))
        ts.append(FRAME * (f + np.arange(1, SUB + 1) / SUB))
    t = np.concatenate(ts)                   # seconds since the goal was issued
    q = np.concatenate(qs, axis=0)           # (T, n, 6)

    for j in range(N_ARM):
        qj = q[:, j, j]
        T_lag = params[j].kv / params[j].kp

        # Dead time, explicitly: bit-still through the transport delay, first
        # motion strictly after it (and not unreasonably late — the jerk-
        # limited profile leaves the start slowly, hence the loose upper edge).
        pre = t <= delay + 1e-12
        assert pre.any()
        assert np.abs(qj[pre] - q_eq[j, j]).max() < 1e-9, f"joint {j} moved inside the dead time"
        moved = np.abs(qj - q_eq[j, j]) > 1e-6
        assert moved.any(), f"joint {j} never responded"
        t_first = t[moved][0]
        assert delay < t_first < delay + 0.06, (
            f"joint {j} response began at {t_first*1e3:.1f} ms, expected just "
            f"after the {delay*1e3:.0f} ms transport delay")

        # No overshoot: never beyond the goal by more than the droop floor.
        assert (qj - goal[j, j]).max() < DROOP_RAD, f"joint {j} overshot"

        # Settled: at plan end + delay + 8 T_lag the error is at the droop
        # floor and it stays there to the end of the run.
        ref = plan_move(jq0[j], goal[j])
        t_settle = ref.duration_s + delay + 8.0 * T_lag
        late = t >= t_settle
        assert late.any()
        assert np.abs(qj[late] - goal[j, j]).max() < 1.5 * DROOP_RAD, (
            f"joint {j} not settled {8.0 * T_lag*1e3:.0f} ms after the plan ended")

        # Tracking lag consistent with T_lag: the 50%-crossing of the measured
        # response trails the commanded profile by ~ delay + T_lag.
        lvl = jq0[j, j] + 0.5 * step
        cj = ref.q[:, j]
        kc = int(np.argmax(cj >= lvl))
        t_cmd = float(np.interp(lvl, cj[kc - 1:kc + 1], ref.t[kc - 1:kc + 1]))
        kp_ = int(np.argmax(qj >= lvl))
        t_meas = float(np.interp(lvl, qj[kp_ - 1:kp_ + 1], t[kp_ - 1:kp_ + 1]))
        shift = t_meas - t_cmd
        assert delay + 0.6 * T_lag < shift < delay + 2.0 * T_lag, (
            f"joint {j}: 50% crossing shift {shift*1e3:.1f} ms inconsistent with "
            f"delay {delay*1e3:.0f} ms + T_lag {T_lag*1e3:.1f} ms")

    assert np.abs(bank.jqd).max() < 1e-4


# -------------------------------------------------------------------- interrupt
def test_interrupt_mid_plan_no_position_discontinuity():
    """Re-goal mid-move -> continuous output positions and convergence.

    Under the OTG interrupt path (contract Addendum 4, R1) the replan is
    seeded with the commanded velocity, so the arm brakes through the old
    motion and returns — it does NOT command an instantaneous stop (the old
    rest-to-rest semantics this test was first written against).
    """
    bank = _bank(1)
    jq0 = _tile(1)
    bank.reset(jq0)
    outs = [bank.advance(SUB) for _ in range(9)]          # settle 0.3 s

    goal_a = jq0.copy()
    goal_a[0, 1] += 0.35
    bank.set_goal(np.array([0]), goal_a)
    outs += [bank.advance(SUB) for _ in range(24)]        # 0.8 s: mid-plan
    assert abs(bank.jqd[0, 1]) > 0.05, "arm should be genuinely moving at the interrupt"

    bank.set_goal(np.array([0]), jq0)                     # re-goal while moving
    outs += [bank.advance(SUB) for _ in range(96)]        # 3.2 s: brake + return

    q = np.concatenate(outs, axis=0)[:, 0, :]             # (T, 6)
    dq = np.abs(np.diff(q, axis=0)).max()
    # Any interrupt bug shows as an O(0.1 rad) jump; smooth motion is bounded
    # by ~vmax per output sample interval.
    assert dq < 2.0 * VMAX_RAD_S * SUB_DT, f"position discontinuity {dq:.2e} rad"
    assert np.abs(bank.jq[0] - jq0[0]).max() < 1.5 * DROOP_RAD
    assert np.abs(bank.jqd).max() < 1e-4


def test_interrupt_commanded_velocity_continuity():
    """OTG interrupt: the COMMANDED stream is velocity-continuous.

    Pre-fix, a mid-move re-goal teleported the commanded velocity to zero
    (rest-to-rest replan) — a step of ~0.3 rad/s. With the seeded replan the
    delivered command's velocity moves by at most one accel-limited stream
    tick across the splice.
    """
    from control.retime import AMAX_RAD_S2, STREAM_DT

    bank = _bank(1)
    jq0 = _tile(1)
    bank.reset(jq0)
    for _ in range(9):
        bank.advance(SUB)
    goal = jq0.copy()
    goal[0, 1] += 0.35
    bank.set_goal(np.array([0]), goal)
    for _ in range(30):                                   # 1.0 s: near cruise
        bank.advance(SUB)
    now = bank.frame_dt * bank._frame
    _, v_before, _ = bank._command_state(0, now)
    assert abs(v_before[1]) > 0.1, "commanded stream should be moving"

    back = jq0.copy()
    back[0, 1] += 0.05
    bank.set_goal(np.array([0]), back)                    # interrupt at speed
    _, v_after, _ = bank._command_state(0, now)
    dv = np.abs(v_after - v_before).max()
    assert dv <= AMAX_RAD_S2 * STREAM_DT + 1e-9, (
        f"commanded velocity stepped {dv:.2e} rad/s across the interrupt "
        f"(bar: one accel-limited tick = {AMAX_RAD_S2 * STREAM_DT:.2e})")

    # And the measured plant never decelerates harder than the command path
    # allows: frame-to-frame velocity deltas stay accel-bounded (the old
    # rest-to-rest interrupt produced ~25 rad/s^2 servo-bandwidth braking).
    prev = bank.jqd
    dv_meas = 0.0
    for _ in range(45):
        bank.advance(SUB)
        dv_meas = max(dv_meas, np.abs(bank.jqd - prev).max())
        prev = bank.jqd
    assert dv_meas < 3.0 * AMAX_RAD_S2 * FRAME, (
        f"measured accel {dv_meas / FRAME:.2f} rad/s^2 exceeds the commanded "
        f"accel limit regime")


def test_regoal_march_tracking_regression():
    """The Gate-2 probe, pinned: goals marched incrementally with re-goals
    EVERY frame at 60 Hz for 2 s must achieve a large fraction of the motion
    of one uninterrupted goal. Pre-fix (rest-to-rest replans): 0.12%. The bar
    is >= 80%; the shortfall vs 100% is one braking distance — every replan
    intends to STOP at the currently-marched goal, so the command rides
    v^2/2a + v*a/2j (~4.6 deg at this rate) behind the march, exactly as the
    real controller would. After the march ends the plan runs out and the arm
    must land on the final goal (no motion is permanently lost).
    """
    frame_dt = 1.0 / 60.0
    frames = 120
    total = np.radians(24.0)
    joint = 1
    jq0 = _tile(1)
    goal_end = jq0.copy()
    goal_end[0, joint] += total

    # marched re-goals, one per frame
    march = ServoArmBank(1, frame_dt=frame_dt)
    march.reset(jq0)
    for f in range(frames):
        g = jq0.copy()
        g[0, joint] += total * (f + 1) / frames
        march.set_goal(np.array([0]), g)
        march.advance(SUB)
    achieved_march = march.jq[0, joint] - jq0[0, joint]

    # one uninterrupted goal, same wall time
    single = ServoArmBank(1, frame_dt=frame_dt)
    single.reset(jq0)
    single.set_goal(np.array([0]), goal_end)
    for _ in range(frames):
        single.advance(SUB)
    achieved_single = single.jq[0, joint] - jq0[0, joint]

    frac = achieved_march / achieved_single
    print(f"\n[march] 60 Hz re-goals: {np.degrees(achieved_march):.2f} deg vs "
          f"single goal {np.degrees(achieved_single):.2f} deg -> {frac * 100:.1f}%")
    assert frac >= 0.80, (
        f"per-frame re-goal tracking {frac * 100:.2f}% < 80% "
        f"({np.degrees(achieved_march):.3f} of {np.degrees(achieved_single):.3f} deg)")

    # goals stop -> the last plan completes: nothing was permanently lost
    for _ in range(60):                                   # settle 1 s
        march.advance(SUB)
    assert abs(march.jq[0, joint] - goal_end[0, joint]) < 1.5 * DROOP_RAD


# ------------------------------------------------------------------ determinism
def _script(bank: ServoArmBank) -> bytes:
    """Fixed goal/advance sequence exercising BOTH plan paths: cold starts
    (first goals from hold) and an OTG interrupt (env 0 re-goaled mid-move)."""
    jq0 = _tile(3) + np.array([[0.0], [0.02], [-0.03]])
    bank.reset(jq0)
    blobs = [bank.advance(5)]
    g = jq0[[0, 2]].copy()
    g[:, 2] += 0.2
    bank.set_goal(np.array([0, 2]), g)
    blobs += [bank.advance(5) for _ in range(3)]
    g1 = jq0[[1]].copy()
    g1[0, 4] -= 0.15
    bank.set_goal(np.array([1]), g1)
    blobs += [bank.advance(7) for _ in range(3)]
    g2 = jq0[[0]].copy()
    g2[0, 2] += 0.05
    bank.set_goal(np.array([0]), g2)          # interrupts env 0 in flight (OTG)
    blobs += [bank.advance(6) for _ in range(3)]
    blobs += [bank.jq, bank.jqd]
    return b"".join(b.tobytes() for b in blobs)


def test_determinism_bit_identical():
    a = _script(_bank(3))
    bank = _bank(3)
    b = _script(bank)
    c = _script(bank)          # re-reset on the same bank: state fully cleared
    assert a == b, "two fresh banks diverged on an identical script"
    assert a == c, "reset() did not fully clear bank state"


# ------------------------------------------------------------------- resampling
def test_advance_resampling_shape_stitching_last_sample():
    bank = _bank(2)
    jq0 = _tile(2)
    bank.reset(jq0)
    out = bank.advance(5)
    assert out.shape == (5, 2, N_ARM) and out.dtype == np.float64
    assert np.array_equal(out[-1], bank.jq)
    for _ in range(30):                  # settle the reset transient (1.0 s)
        bank.advance(SUB)

    goal = jq0.copy()
    goal[:, 0] += 0.3                    # J0: vertical axis, gravity-free, clean
    bank.set_goal(np.array([0, 1]), goal)
    prev_last = bank.jq
    for _ in range(75):                  # 2.5 s
        out = bank.advance(SUB)
        assert out.shape == (SUB, 2, N_ARM)
        assert np.array_equal(out[-1], bank.jq)
        seq = np.concatenate([prev_last[None], out], axis=0)[:, :, 0]
        d = np.diff(seq, axis=0)
        # -1e-7 floor: at the end of the move the uncompensated finger droop
        # redistributes at the new pose and J0 relaxes back by ~4e-8 rad —
        # physics, not resampling. A real ordering bug shows as O(v*dt)~1e-3.
        assert d.min() > -1e-7, "samples not monotone along a monotone move"
        assert d.max() < 2.0 * VMAX_RAD_S * SUB_DT, "sample-to-sample jump"
        prev_last = bank.jq
    assert np.abs(bank.jq[:, 0] - goal[:, 0]).max() < 1.5 * DROOP_RAD


def test_substeps_only_select_output_grid():
    """advance(4) samples nest bit-identically inside advance(8) samples: the
    plant physics runs on its own 2 ms tick regardless of the output grid.
    A mid-move re-goal is included so the OTG interrupt path is covered too."""
    b4, b8 = _bank(1), _bank(1)
    jq0 = _tile(1)
    b4.reset(jq0)
    b8.reset(jq0)
    goal = jq0.copy()
    goal[0, 1] += 0.2
    for b in (b4, b8):
        b.set_goal(np.array([0]), goal)
    goal2 = jq0.copy()
    goal2[0, 1] += 0.1
    for f in range(30):
        if f == 20:                          # interrupt in flight -> OTG plan
            for b in (b4, b8):
                b.set_goal(np.array([0]), goal2)
        o4 = b4.advance(4)
        o8 = b8.advance(8)
        assert np.array_equal(o4, o8[1::2])
    assert np.array_equal(b4.jq, b8.jq)


# ------------------------------------------------- plant configuration (the trap)
def test_gripper_on_calibrated_payload_compensated():
    bank = _bank(1)
    m = bank._plant.model

    # Gripper ON: 31.23 kg. The stripped (sysid-recording) plant is 30.01 kg —
    # Plant() DEFAULTS to that, which is the trap this test exists for.
    total = float(m.body_mass.sum())
    assert abs(total - 31.2302) < 0.02, f"total mass {total:.4f} kg != gripper-on 31.23"
    assert total > 30.5, "bank built the gripper-OFF (sysid-recording) plant"
    assert bank._plant.with_gripper

    # Calibrated gains applied — Plant() alone carries the MJCF's 6000/60
    # placeholders on every joint.
    params = load_servo_params()
    for j in range(N_ARM):
        a = bank._plant.act[j]
        assert m.actuator_gainprm[a, 0] == params[j].kp
        assert m.actuator_biasprm[a, 2] == -params[j].kv

    # Payload gravcomp reached the DERIVED constants (writing body_gravcomp on
    # a compiled model is a silent no-op until mj_setConst; qfrc_gravcomp
    # would read exactly 0.0).
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ag145_payload")
    assert b >= 0 and m.body_gravcomp[b] == 1.0
    d = mujoco.MjData(m)
    d.qpos[bank._qadr] = NOMINAL_Q
    mujoco.mj_forward(m, d)
    assert np.abs(d.qfrc_gravcomp).max() > 1.0, "gravcomp did not reach the compiled model"


# ------------------------------------------------------- stream fast path parity
def test_position_fastpath_matches_stream_sample():
    """_PositionStream.sample_q must be the position leg of SetpointStreamer.
    sample(), bit-exactly, on every branch: hold, pre-start (inside the
    transport delay), mid-trajectory, at the end boundary, past the end."""
    delay = servo_transport_delay_s()
    q0, q1 = NOMINAL_Q, NOMINAL_Q + 0.2
    traj = plan_move(q0, q1)

    fast, ref = _PositionStream(delay_s=delay), SetpointStreamer(delay_s=delay)
    for s in (fast, ref):
        s.set_hold(q0)
    assert np.array_equal(fast.sample_q(0.3), ref.sample(0.3)[0])

    for s in (fast, ref):
        s.set_trajectory(traj, 1.0)
    probes = [1.0, 1.0 + 0.5 * delay, 1.0 + delay, 1.5,
              1.0 + delay + traj.duration_s, 1.0 + delay + traj.duration_s + 5.0]
    for t in probes:
        assert np.array_equal(fast.sample_q(t), ref.sample(t)[0]), f"diverged at t={t}"

    # Degenerate zero-length plan (goal == start) must not blow up either path.
    z = plan_move(q0, q0)
    for s in (fast, ref):
        s.set_trajectory(z, 0.0)
    for t in (0.0, delay, 1.0):
        assert np.array_equal(fast.sample_q(t), ref.sample(t)[0])

    # An OTG replan trajectory (ruckig-backed duck type) must sample
    # identically through both paths as well.
    from newton_cabling.servo_arm.bank import _otg_replan
    otg = _otg_replan(q0, np.full(N_ARM, 0.2), np.zeros(N_ARM), q1)
    assert otg.duration_s > 0.0
    for s in (fast, ref):
        s.set_trajectory(otg, 2.0)
    for t in (2.0, 2.0 + delay, 2.3, 2.0 + delay + otg.duration_s,
              2.0 + delay + otg.duration_s + 5.0):
        assert np.array_equal(fast.sample_q(t), ref.sample(t)[0]), f"OTG diverged at t={t}"
    # and it lands where it aimed
    assert np.abs(otg.end_q - q1).max() < 1e-9


# ------------------------------------------------------------------ fail loudly
def test_fail_loudly():
    with pytest.raises((TypeError, ValueError)):
        ServoArmBank(0, frame_dt=FRAME)
    with pytest.raises((TypeError, ValueError)):
        ServoArmBank(2.5, frame_dt=FRAME)
    with pytest.raises(ValueError):
        ServoArmBank(1, frame_dt=0.0)
    with pytest.raises(ValueError):
        ServoArmBank(1, frame_dt=float("nan"))
    with pytest.raises(KeyError):
        ServoArmBank(1, frame_dt=FRAME, params={0: load_servo_params()[0]})

    bank = _bank(2)
    with pytest.raises(RuntimeError):
        bank.advance(SUB)                                   # before reset
    with pytest.raises(RuntimeError):
        bank.set_goal(np.array([0]), _tile(1))              # before reset
    with pytest.raises(RuntimeError):
        _ = bank.jq                                         # before reset

    with pytest.raises(ValueError):
        bank.reset(_tile(3))                                # wrong row count
    with pytest.raises(ValueError):
        bank.reset(np.full((2, N_ARM), np.nan))             # non-finite

    bank.reset(_tile(2))
    good = _tile(1)
    with pytest.raises(ValueError):
        bank.set_goal(np.array([0]), np.full((1, N_ARM), np.inf))   # non-finite goal
    with pytest.raises(ValueError):
        bank.set_goal(np.array([0]), _tile(2))              # ids/goals mismatch
    with pytest.raises(ValueError):
        bank.set_goal(np.array([[0]]), good)                # 2-D ids
    with pytest.raises(TypeError):
        bank.set_goal(np.array([0.0]), good)                # float ids
    with pytest.raises(IndexError):
        bank.set_goal(np.array([2]), good)                  # out of range
    with pytest.raises(ValueError):
        bank.set_goal(np.array([1, 1]), _tile(2))           # duplicate ids
    over = _tile(1)
    over[0, 1] = 3.0                                        # J1 limit is 2.356
    with pytest.raises(ValueError):
        bank.set_goal(np.array([0]), over)                  # unreachable goal

    with pytest.raises(ValueError):
        bank.advance(0)
    with pytest.raises((TypeError, ValueError)):
        bank.advance(2.5)
