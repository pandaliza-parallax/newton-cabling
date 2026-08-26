#!/usr/bin/env python3
"""GATE 1 — TRANSFER: does the integrated env faithfully realize the calibrated plant?

    .venv/bin/python tests/gates/gate_transfer.py [--envs 4] [--steps 60] [--seed 0]

`ServoCableVecEnv` sits between two things that must agree exactly:

    calibrated plant  --(bank.advance)-->  arm_stream  --(+branch, eval_fk)-->  Newton

Everything in that chain is a place a bug hides silently: the revolution BRANCH offset
(Addendum 3) is added and subtracted in four different places, the substep stream is
resampled onto the env's grid, `self.jq` is re-seeded from the bank for the next IK, and the
bank's goal is issued in the bank's own joint frame. A wrong sign, a stale offset or a
half-applied re-goal all produce an arm that MOVES PLAUSIBLY — it just is not the arm that
was calibrated. This gate drives the real bank with a fixed, deterministic action sequence and
checks, every frame, that what reached Newton is what the plant measured.

Four independent checks (any FAIL is a red gate):

  T1  STREAM   the last substep sample driven into Newton, minus the branch offset, equals
               the bank's own measured `jq` at the frame boundary.        <= 0.05 deg / joint
  T2  SEED     `env.jq[arm_qc]` (the next step's IK seed and FK source), minus the branch
               offset, equals the bank's measured `jq`.                   <= 0.05 deg / joint
  T3  FK       the wrist pose Newton actually holds equals the forward kinematics of the
               bank's measured joints. This is the end-to-end, joint-space-independent
               statement of the same claim, and the only one that would catch a branch offset
               that is wrong by a NON-multiple of 2*pi.        <= 1e-5 m and <= 1e-3 deg
  T4  REPLAY   a fresh `ServoArmBank`, reset to the same jq0 and handed the same goal
               sequence the env issued, reproduces the env's bank trajectory. This is the
               determinism pin (contract pin 5) and it also proves the env issues goals in
               the bank frame consistently: a replay that diverges means the env's goal
               stream carried hidden state.                              <= 0.05 deg / joint

The 0.05 deg bar is the demo repo's `validate_newton` bar for the same question (does the
Newton arm realize the MuJoCo plant), reused here so the two pipelines are held to one
standard.

Also REPORTED (not asserted): the branch offset actually established, the plant's following
error against its own goal, and the wrist command lead — the numbers Gate 2 needs to explain
behaviour. The ACTIONS are fixed (no RNG), and T4 pins the bank's own determinism; the scene
build is not bit-reproducible across processes (the VBD settle runs on the GPU), so the
reported context numbers move by a fraction of a percent between runs while the four verdicts
do not.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TWO_PI = 2.0 * math.pi

# ── the bars ──────────────────────────────────────────────────────────────────────────────
JOINT_TOL_DEG = 0.05      # demo repo validate_newton's bar
FK_POS_TOL_M = 1e-5       # 10 um: FK of identical joints through the identical model
FK_ROT_TOL_DEG = 1e-3


class RecordingBank:
    """Transparent proxy around a ServoArmBank that records the whole conversation.

    Forwards everything (``__getattr__``), so the env's private probes — ``_jnt_lo`` /
    ``_jnt_hi``, read by `servo_cable_env.bank_joint_limits` — reach the real bank unchanged.
    ``jq`` / ``jqd`` are declared as real properties on the CLASS because the env validates
    the bank with ``hasattr(type(bank), "jq")``.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "resets", [])
        object.__setattr__(self, "goals", [])       # (frame, ids, q_star) in the BANK frame
        object.__setattr__(self, "streams", [])     # (substeps, n, 6) per advance, BANK frame
        object.__setattr__(self, "jq_at_end", [])   # (n, 6) per advance, BANK frame
        object.__setattr__(self, "frame", 0)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def reset(self, jq0):
        self.resets.append((self.frame, np.array(jq0, dtype=np.float64, copy=True)))
        # a reset restarts the plant: the replay must restart with it
        object.__setattr__(self, "frame", 0)
        del self.goals[:], self.streams[:], self.jq_at_end[:]
        return self._inner.reset(jq0)

    def set_goal(self, env_ids, q_star):
        self.goals.append((self.frame,
                           np.array(env_ids, dtype=np.int64, copy=True),
                           np.array(q_star, dtype=np.float64, copy=True)))
        return self._inner.set_goal(env_ids, q_star)

    def advance(self, substeps):
        out = self._inner.advance(substeps)
        self.streams.append(np.array(out, dtype=np.float64, copy=True))
        self.jq_at_end.append(np.array(self._inner.jq, dtype=np.float64, copy=True))
        object.__setattr__(self, "frame", self.frame + 1)
        return out

    @property
    def jq(self):
        return self._inner.jq

    @property
    def jqd(self):
        return self._inner.jqd


def fixed_actions(n: int, steps: int) -> np.ndarray:
    """A deterministic (n, 7) action sequence, built to EXERCISE the seam rather than to seat.

    Smooth sinusoids so the wrist target stays bounded (a random walk would drift the arm out
    of the workspace and the failure would be the IK's, not the seam's), on three different
    periods so the six joints never move in lockstep, plus a SIGN FLIP at the halfway point:
    an abrupt reversal is what forces `set_goal` to interrupt an in-flight ruckig plan, which
    is the one code path a smooth trajectory would never reach.
    """
    t = np.arange(steps, dtype=np.float64)[:, None, None]
    i = np.arange(n, dtype=np.float64)[None, :, None]
    a = np.zeros((steps, n, 7))
    for k, (amp, period) in enumerate(((0.6, 21.0), (0.5, 17.0), (0.6, 13.0))):
        a[:, :, k] = (amp * np.sin(TWO_PI * t / period + 0.7 * i + 0.3 * k))[:, :, 0]
    for k, (amp, period) in enumerate(((0.4, 19.0), (0.4, 11.0), (0.35, 23.0))):
        a[:, :, 3 + k] = (amp * np.sin(TWO_PI * t / period + 1.1 * i + 0.5 * k))[:, :, 0]
    a[steps // 2:] *= -1.0
    return a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--tilt", type=float, default=8.0)
    args = ap.parse_args()

    import newton
    import warp as wp
    from scipy.spatial.transform import Rotation as Rot

    from newton_cabling.servo_arm import ServoArmBank
    from servo_cable_env import FRAME_DT, ServoCableVecEnv  # noqa: PLC0415
    from rigid_cable_env import SUBSTEPS

    print(f"GATE 1 — TRANSFER | envs {args.envs} steps {args.steps} seed {args.seed} "
          f"stage {args.stage} tilt {args.tilt}")
    t0 = time.perf_counter()
    bank = RecordingBank(ServoArmBank(args.envs, frame_dt=FRAME_DT))
    tilt = (0.0, args.tilt) if args.tilt > 0 else 0.0
    env = ServoCableVecEnv(args.envs, bank=bank, max_cmd_lead_m=None,
                           seed=args.seed, cable_tilt_deg=tilt)
    env.set_stage(args.stage)
    print(f"[build] {time.perf_counter() - t0:.1f}s", flush=True)

    # ── capture what actually reaches Newton ────────────────────────────────────────────
    # `_sim_from_stream` is the ONLY door between the bank and the solver: whatever array
    # arrives here is what eval_fk poses the arm with. Wrapping it (on the instance, no file
    # touched) records the ground truth for T1 without trusting any bookkeeping upstream.
    driven: list[np.ndarray] = []
    real_sim = env._sim_from_stream

    def spy(jq_prev, arm_stream):
        driven.append(np.array(arm_stream, dtype=np.float64, copy=True))
        return real_sim(jq_prev, arm_stream)

    env._sim_from_stream = spy

    obs = env.reset()
    del driven[:]                                  # reset() itself does not step
    branch = env._branch.copy()
    turns = np.round(branch / TWO_PI).astype(int)
    print(f"[branch] offset turns per (env, joint): {turns.tolist()}")
    assert np.allclose(branch, TWO_PI * turns, atol=1e-12), "branch offset is not whole turns"

    fk_state = env.model.state()                   # scratch state for the independent FK
    actions = fixed_actions(args.envs, args.steps)

    e_stream = np.zeros((args.steps, args.envs, 6))
    e_seed = np.zeros((args.steps, args.envs, 6))
    fk_pos = np.zeros((args.steps, args.envs))
    fk_rot = np.zeros((args.steps, args.envs))
    follow = np.zeros((args.steps, args.envs))     # plant joint error vs its own goal, deg
    lead_mm = np.zeros((args.steps, args.envs))    # |wrist_tgt_p - measured wrist|, mm
    n_goals = 0

    t1 = time.perf_counter()
    for t in range(args.steps):
        jq_before = env.jq.copy()
        env.step(actions[t])
        meas = bank.jq_at_end[-1]                                   # BANK frame, frame end
        stream = driven[-1]                                         # ENV frame, as driven
        assert stream.shape == (SUBSTEPS, args.envs, 6), stream.shape

        # T1 — the sample that posed Newton at the end of the frame, back in the bank frame
        e_stream[t] = np.degrees(stream[-1] - branch - meas)
        # T2 — the seed the next IK will use / the FK source
        e_seed[t] = np.degrees(env.jq[env.arm_qc] - branch - meas)

        # T3 — end-to-end: FK(measured joints) vs the wrist Newton is actually holding.
        # Non-arm coordinates come from the pre-step jq, exactly as `_sim_from_stream` does.
        jq_fk = jq_before.copy()
        jq_fk[env.arm_qc] = meas + branch
        newton.eval_fk(env.model, wp.array(jq_fk, dtype=float, device=env.device),
                       env.model.joint_qd, fk_state,
                       body_flag_filter=int(newton.BodyFlags.KINEMATIC))
        bq_fk = fk_state.body_q.numpy()
        bq_now = env.state_0.body_q.numpy()
        fk_pos[t] = np.linalg.norm(bq_fk[env.wrist_body, :3] - bq_now[env.wrist_body, :3], axis=1)
        fk_rot[t] = np.degrees((Rot.from_quat(bq_fk[env.wrist_body, 3:7])
                                * Rot.from_quat(bq_now[env.wrist_body, 3:7]).inv()).magnitude())

        # reported context: how far the plant is from the goal it was last given, and how far
        # the open command integrator has run away from the arm it commands
        if bank.goals:
            g = np.zeros((args.envs, 6))
            last = {}
            for _f, ids, q in bank.goals:
                for k, i in enumerate(ids):
                    last[int(i)] = q[k]
            for i in range(args.envs):
                g[i] = last.get(i, meas[i])
            follow[t] = np.degrees(np.abs(g - meas)).max(axis=1)
            n_goals = len(bank.goals)
        lead_mm[t] = np.linalg.norm(
            env.wrist_tgt_p - bq_now[env.wrist_body, :3], axis=1) * 1000.0
    dt_run = time.perf_counter() - t1

    # ── T4 — independent replay of the recorded goal stream ─────────────────────────────
    ref = ServoArmBank(args.envs, frame_dt=FRAME_DT)
    ref.reset(bank.resets[-1][1])
    by_frame: dict[int, list] = {}
    for f, ids, q in bank.goals:
        by_frame.setdefault(f, []).append((ids, q))
    e_replay = np.zeros((args.steps, args.envs, 6))
    for t in range(args.steps):
        for ids, q in by_frame.get(t, ()):
            ref.set_goal(ids, q)
        ref.advance(SUBSTEPS)
        e_replay[t] = np.degrees(ref.jq - bank.jq_at_end[t])

    # ── verdict ─────────────────────────────────────────────────────────────────────────
    checks = [
        ("T1 STREAM  driven[-1] - branch  vs  bank.jq",
         float(np.abs(e_stream).max()), JOINT_TOL_DEG, "deg"),
        ("T2 SEED    env.jq[arm] - branch vs  bank.jq",
         float(np.abs(e_seed).max()), JOINT_TOL_DEG, "deg"),
        ("T3 FK-pos  FK(bank.jq) vs Newton wrist",
         float(fk_pos.max()), FK_POS_TOL_M, "m"),
        ("T3 FK-rot  FK(bank.jq) vs Newton wrist",
         float(fk_rot.max()), FK_ROT_TOL_DEG, "deg"),
        ("T4 REPLAY  fresh bank, same goals",
         float(np.abs(e_replay).max()), JOINT_TOL_DEG, "deg"),
    ]
    print(f"\n{'check':<44} {'worst':>12} {'bar':>10}  verdict")
    ok = True
    for name, worst, bar, unit in checks:
        good = worst <= bar
        ok &= good
        print(f"{name:<44} {worst:>12.3e} {bar:>10.1e}  {'PASS' if good else 'FAIL'} [{unit}]")

    print(f"\nreported context ({args.steps} frames x {args.envs} envs, {dt_run:.1f}s of stepping)")
    print(f"  branch offset               : {np.abs(turns).max()} turn(s) max, "
          f"{int((turns != 0).sum())} of {turns.size} (env, joint) pairs offset")
    print(f"  frames that re-goaled       : {n_goals} of {args.steps} "
          f"(each call interrupts the in-flight ruckig plan of the envs whose goal moved)")
    print(f"  plant following error       : median {np.median(follow):6.3f} deg  "
          f"p95 {np.percentile(follow, 95):6.3f}  max {follow.max():6.3f}")
    print(f"  wrist command lead          : median {np.median(lead_mm):6.2f} mm  "
          f"p95 {np.percentile(lead_mm, 95):6.2f}  max {lead_mm.max():6.2f}")
    print(f"  per-joint worst |stream err|: "
          f"{np.array2string(np.abs(e_stream).max(axis=(0, 1)), precision=2)} deg")

    print(f"\nGATE 1 {'PASS' if ok else 'FAIL'}  ({time.perf_counter() - t0:.1f}s total)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
