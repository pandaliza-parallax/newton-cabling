#!/usr/bin/env python3
"""GATE 2 — BEHAVIOR: does the task still get done once the arm is the CALIBRATED one?

    .venv/bin/python tests/gates/gate_behavior.py                       # the A/B/C matrix
    .venv/bin/python tests/gates/gate_behavior.py --configs A,B,C5,C10
    .venv/bin/python tests/gates/gate_behavior.py --configs C5-stable8,B-slow

Gate 1 proves the plant's trajectory reaches Newton intact. That says nothing about whether
the TASK survives: the scripted `AlignInsertController` was tuned against an arm with no lag,
and every one of its thresholds (`stable_steps`, `jam_window`, the align rate caps) is a
statement about how fast the arm answers a command.

The matrix, all on the same seed, the same stage and the same per-episode jack placement
(``env._rng`` is re-seeded before every reset, so config-to-config differences are the ARM and
nothing else):

    A     RigidCableVecEnv  (kinematic, ideal arm)      — the reference, must stay ~100%
    B     ServoCableVecEnv  real bank, max_cmd_lead_m=None  (strict parent-parity semantics)
    C5    ServoCableVecEnv  real bank, max_cmd_lead_m=5 mm  (following-error governor)
    C10   ServoCableVecEnv  real bank, max_cmd_lead_m=10 mm

Primary subject is the scripted controller — it is the point of the effort. The env's own
`servo_action()` teacher runs as `A-teacher` / `B-teacher` for reference; note it does NOT
seat at stage 4 on the parent env either (see REPORT.md), so it is a lag-sensitivity probe,
not a baseline.

`-<name>` suffixes are RUNTIME AlignInsertConfig overrides (nothing is edited on disk); they
exist to answer "if B/C underperform, WHICH threshold is the one that stopped being true".
Envs are built once per kind and reused across configs, because a build costs ~30-40 s and the
budget is measured in minutes.
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

class DecimatingBank:
    """A ServoArmBank proxy that forwards only every ``decimate``-th ``set_goal``.

    NOT part of the gate — it is the RUNTIME emulation of a candidate integration change, so
    the recommendation in REPORT.md rests on a measurement instead of an argument. It changes
    nothing on disk: `bank` is an ordinary `ServoCableVecEnv` kwarg, so handing the env a
    different bank object is configuration in exactly the way `max_cmd_lead_m` is.

    Why this is the interesting knob: `ServoArmBank.set_goal` re-plans a REST-TO-REST
    jerk-limited ruckig trajectory from the currently commanded position, and the env issues a
    new goal every 16.7 ms control frame. A jerk-limited profile covers j*dt^3/6 = 1.16 urad in
    its first frame, so a plan that is replaced every frame never leaves its own initial jerk
    ramp. Dropping goals lets an in-flight plan actually run.

    ``decimate = 1`` forwards everything, i.e. is bit-identical to the bare bank — which is
    what keeps configs B / C5 / C10 honest measurements of the shipped seam.
    """

    def __init__(self, inner, decimate: int = 1):
        object.__setattr__(self, "_inner", inner)
        self.decimate = int(decimate)
        self.frame = 0
        self.issued = 0
        self.dropped = 0

    def __getattr__(self, name):                    # _jnt_lo / _jnt_hi / frame_dt / ...
        return getattr(self._inner, name)

    def reset(self, jq0):
        self.frame = self.issued = self.dropped = self.sent_envs = 0
        return self._inner.reset(jq0)

    def set_goal(self, env_ids, q_star):
        if self.decimate <= 1 or self.frame % self.decimate == 0:
            self.issued += 1
            # per-ENV sends: the env batches only the rows it re-goals, so this is the number
            # the R2 suppression knobs actually move (and the retimer cost that follows).
            self.sent_envs += int(np.asarray(env_ids).size)
            return self._inner.set_goal(env_ids, q_star)
        self.dropped += 1
        return None

    def advance(self, substeps):
        out = self._inner.advance(substeps)
        self.frame += 1
        return out

    @property
    def jq(self):
        return self._inner.jq

    @property
    def jqd(self):
        return self._inner.jqd


# ── the matrix ────────────────────────────────────────────────────────────────────────────
_SLOW = dict(align_lin_rate_m=0.0004, align_rot_rate_rad=math.radians(0.15),
             push_rate_m=0.0004)
_FAST = dict(align_lin_rate_m=0.0016, align_rot_rate_rad=math.radians(0.6),
             push_rate_m=0.0016)
_VSLOW = dict(align_lin_rate_m=0.0002, align_rot_rate_rad=math.radians(0.075),
              push_rate_m=0.0002)
# Rotation-only slowdown. The controller's own docstring says wrist-to-face is ~0.21 m, so a
# rotation the plant OVERSHOOTS lands as a lateral face error of 0.21 m x overshoot: 3 deg of
# overshoot is 11 mm of lateral, which is the scale of the residual B is stuck at. This
# isolates that channel from the linear one.
_ROTSLOW = dict(align_rot_rate_rad=math.radians(0.1), kp_rot=0.2)
# Lower proportional gain, same rate caps: attacks integrator WINDUP (the correction is
# re-commanded every frame while the previous one is still in flight) rather than rate.
_LOWKP = dict(kp_lin=0.15, kp_rot=0.15)


_R7 = dict(_SLOW, kp_lin=0.15, kp_rot=0.15)
"""The Wave-2c profile: align/push rates halved and the proportional gain cut 3.3x. It is
what made ALIGN converge against the lagging plant (lat 16.1 -> 0.92 mm), at the cost of
needing a ~2.6x longer horizon."""
_CMD = dict(servo_source="commanded")
"""Wave-4 R11: the servo terms close on the env's OWN command integrator instead of the
measured face, so the loop no longer chases a plant it cannot see arrive. Phase gates stay on
measured truth, so this cannot fake a seat."""


def S(kind="servo", lead=None, over=None, decimate=1, deadband=0.0, every=1):
    """One row of the matrix. Every field is applied at RUNTIME (attribute writes or a bank
    proxy) — no file on disk is touched by any config.

    kind      "parent" (kinematic RigidCableVecEnv) or "servo" (ServoCableVecEnv + real bank)
    lead      `env.max_cmd_lead_m`, metres, None = strict parent parity
    over      AlignInsertConfig overrides; None = drive with the env's own servo_action() teacher
    decimate  DecimatingBank proxy: forward only every N-th set_goal (the Wave-2b emulation of
              a re-goal-rate change, kept so the two waves stay comparable)
    deadband  `env.regoal_deadband_rad` — the NATIVE R2 knob (Wave 2c), joint-space suppression
    every     `env.regoal_every` — the NATIVE R2 staleness floor
    """
    return dict(kind=kind, lead=lead, over={} if over is None and kind else over,
                decimate=decimate, deadband=deadband, every=every)


SPECS: dict[str, dict] = {
    # ── the gate proper ──
    "A":            S("parent"),
    "B":            S(),
    "C5":           S(lead=0.005),
    "C10":          S(lead=0.010),
    # ── the env's own servo teacher, secondary ──
    "A-teacher":    dict(S("parent"), over=None),
    "B-teacher":    dict(S(), over=None),
    # ── controller-side timing hypotheses (AlignInsertConfig, runtime) ──
    "C2":           S(lead=0.002),
    "B-stable8":    S(over=dict(stable_steps=8)),
    "B-jam40":      S(over=dict(jam_window=40)),
    "B-slow":       S(over=dict(_SLOW)),
    "B-fast":       S(over=dict(_FAST)),
    "B-slow-s8":    S(over=dict(_SLOW, stable_steps=8)),
    "B-vslow":      S(over=dict(_VSLOW)),
    "B-rotslow":    S(over=dict(_ROTSLOW)),
    "B-lowkp":      S(over=dict(_LOWKP)),
    "B-slow-kp":    S(over=dict(_SLOW, **_LOWKP)),
    "B-vslow-kp-s8": S(over=dict(_VSLOW, **_LOWKP, stable_steps=8)),
    "C5-vslow-kp-s8": S(lead=0.005, over=dict(_VSLOW, **_LOWKP, stable_steps=8)),
    # ── the Wave-2c candidate: halved align/push rates + lowered proportional gain. This is
    # the combination that makes ALIGN actually converge under the calibrated arm's lag
    # (lat 16.1 -> 0.92 mm, ang 25.5 -> 1.45 deg at 260 steps). Needs a longer horizon: the
    # servo arm reaches pre-dock around the step count at which the ideal arm is already held.
    "B-slow-kp-s8":  S(over=dict(_SLOW, **_LOWKP, stable_steps=8)),
    "C5-slow-kp":    S(lead=0.005, over=dict(_SLOW, **_LOWKP)),
    "C5-slow-kp-s8": S(lead=0.005, over=dict(_SLOW, **_LOWKP, stable_steps=8)),
    # ── Wave 4b: the R11 verdict rows ──────────────────────────────────────────────────
    # D is the hypothesis: commanded-source servo at STOCK rates. If R11 is right, the
    # rate-halving R7 needed is unnecessary, because the reason ALIGN diverged was that the
    # loop was closed on a face that had not moved yet — not that the gains were too hot.
    "D":            S(over=dict(_CMD)),
    "D-R7":         S(over=dict(_R7, **_CMD)),
    "D-s8":         S(over=dict(_CMD, stable_steps=8)),
    "B-R7":         S(over=dict(_R7)),                 # alias of B-slow-kp, clearer name
    "C5-stable8":   S(lead=0.005, over=dict(stable_steps=8)),
    "C5-slow":      S(lead=0.005, over=dict(_SLOW)),
    "C5-slow-s8":   S(lead=0.005, over=dict(_SLOW, stable_steps=8)),
    "C5-slow-s8-j40": S(lead=0.005, over=dict(_SLOW, stable_steps=8, jam_window=40)),
    # ── Wave-2b command-path emulation (DecimatingBank proxy) ──
    "B-d6":         S(decimate=6),      # re-goal at 10 Hz
    "B-d20":        S(decimate=20),     # 3 Hz
    "B-d60":        S(decimate=60),     # 1 Hz
    "C5-d60":       S(lead=0.005, decimate=60),
    # ── Wave-2c NATIVE re-goal suppression (R2 knobs, Addendum 4) ──
    # The quantization question: does a joint-space deadband cost fine alignment, where the
    # per-frame goal delta is small by definition?
    "B-db2e3":      S(deadband=2e-3),                 # ~0.115 deg/joint
    "B-db2e3-e6":   S(deadband=2e-3, every=6),        # + a 10 Hz staleness floor
    "C5-db2e3":     S(lead=0.005, deadband=2e-3),
}
DEFAULT = "A,B,D"
# ── what the PASS/FAIL verdict is computed from (Wave 4b semantics) ──────────────────────
# A is the reference (kinematic arm, must reproduce the documented baseline), D is the
# SUBJECT (calibrated arm + servo_source="commanded", the shipping candidate), and B is an
# EXPECTED-FAIL CANARY: measured-mode on the calibrated arm is known to fail, so if it ever
# passes, something upstream changed — the plant stopped lagging, or the env got easier — and
# D's pass would no longer mean what it claims. That is a warning, not a gate failure: B
# passing is not itself a defect, it is a signal that the gate has lost its discriminating
# power. C5 (the lead clamp) is deliberately NOT gated: R9 found the isotropic governor
# net-negative and it is not a shipping config. It stays runnable as a SPECS experiment.
GATE_REFERENCE, GATE_CANARY, GATE_SUBJECT = "A", "B", "D"
GATE_CONFIGS = (GATE_REFERENCE, GATE_CANARY, GATE_SUBJECT)


def run_one(env, module, label, steps, settle, ctrl, observe, seed, phase_names):
    """One deterministic rollout. Metrics mirror rl/eval_scripted_controller.py so the numbers
    are directly comparable with the repo's own baseline table, plus the timing and lag
    diagnostics this gate exists to produce."""
    import torch

    # Re-seed the per-episode jack placement so every config sees the SAME episodes. Without
    # this, `_rng` advances across runs and A/B differences would be part scene, part arm.
    env._rng = np.random.default_rng(seed)
    obs = env.reset()
    if ctrl is not None:
        ctrl.reset()

    n = env.n
    # Start pose and total wrist travel. These are what separate "the controller chose badly"
    # from "the arm never moved": a config that ends where it started, with the command
    # integrator far away, is a TRACKING failure, not a strategy failure.
    bq0 = env.state_0.body_q.numpy()
    wrist0 = bq0[env.wrist_body, :3].copy()
    f0, fq0 = env._face_pose(bq0)
    a0, _, l0, g0 = env._terms(f0, fq0)
    start = (float(a0.mean() * 1000), float(l0.mean() * 1000), float(np.degrees(g0.mean())))

    # ── the per-env DR draw, for failure attribution ──────────────────────────────────
    # reset() places the jack at  seat = face_ref + _approach*ins + off,  where `off` is the
    # episode's random lateral offset (uniform magnitude in [0, _mag], uniform direction in
    # the jaw/tool plane). Inverting that is the only way to recover the draw: the env keeps
    # the offset nowhere. Tilt is drawn once at build and scaled by the stage.
    off_vec = (np.asarray(env.seat_pos) - np.asarray(env.face_ref)
               - env._approach * np.asarray(env.ins))
    jack_off_mm = np.linalg.norm(off_vec, axis=1) * 1000.0
    scale = env.stage / max(1, env.num_stages - 1)
    tilt_deg = np.degrees(np.asarray(env.cable_tilt) * scale)

    held, along_t, latn_t, ang_t = [], [], [], []
    ever = np.zeros(n)
    viol_any = np.zeros(n)
    ejected = np.zeros(n, dtype=bool)
    t_seat = np.full(n, -1)          # first instantaneously-seated step
    t_succ = np.full(n, -1)          # first step the HOLD_STEPS success latches
    phase_hist = np.zeros(len(phase_names), dtype=np.int64)
    lead_mm, follow_deg = [], []
    err = None

    for t in range(steps):
        with torch.no_grad():
            if ctrl is not None:
                action = ctrl.act(observe(env))
            else:
                action = env.servo_action()
        if ctrl is not None:
            phase_hist += np.bincount(ctrl.phase, minlength=len(phase_names))
        try:
            obs, _, _, succ, _ = env.step(action)
        except ValueError as e:                      # e.g. the branch-offset raise
            err = f"{type(e).__name__}: {str(e).splitlines()[0]}"
            break
        s = succ.cpu().numpy()
        ever = np.maximum(ever, s)
        viol_any += env.viol_last > 0
        bqn = env.state_0.body_q.numpy()
        face, faceq = env._face_pose(bqn)
        along, _, latn, ang = env._terms(face, faceq)
        seated = ((along >= -module.SEAT_DEPTH_TOL) & (latn <= module.SEAT_OFFSET)
                  & (ang <= module.SEAT_ANGLE))
        t_seat = np.where((t_seat < 0) & seated, t, t_seat)
        t_succ = np.where((t_succ < 0) & (s > 0), t, t_succ)
        gone = (~np.isfinite(face).all(axis=1)) | (
            np.linalg.norm(face - env.seat_pos, axis=1) > module.EJECT_DIST)
        ejected |= gone
        lead_mm.append(np.linalg.norm(env.wrist_tgt_p - bqn[env.wrist_body, :3], axis=1) * 1000.0)
        if getattr(env, "bank", None) is not None and getattr(env, "_last_goal", None) is not None:
            follow_deg.append(np.degrees(np.abs(env._last_goal - env.bank.jq)).max(axis=1))
        if t >= steps - settle:
            held.append(s)
            along_t.append(along)
            latn_t.append(latn)
            ang_t.append(ang)

    travel = float(np.linalg.norm(
        env.state_0.body_q.numpy()[env.wrist_body, :3] - wrist0, axis=1).mean() * 1000)

    if not held:                                     # aborted before the tail window
        return {"label": label, "error": err or "no tail window", "held": 0.0, "clean": 0,
                "ever": 0.0, "viol_envs": n, "along_mm": float("nan"), "latn_mm": float("nan"),
                "ang_deg": float("nan"), "ejected": int(ejected.sum()), "attempts": 0,
                "t_succ": t_succ, "t_seat": t_seat, "phase_frac": {}, "lead_mm": (0, 0, 0),
                "follow_deg": (0, 0, 0), "n": n, "start": start, "travel_mm": travel,
                "send_frac": float("nan")}

    H = np.asarray(held)
    ok = ~ejected
    m = ok if ok.any() else np.ones(n, dtype=bool)
    tot = max(1, int(phase_hist.sum()))
    out = {
        "label": label,
        "error": err,
        "n": n,
        "start": start,
        "travel_mm": travel,
        # per-env attribution: the DR draw, the outcome, and where each env ended up
        "held_per_env": H.mean(axis=0),
        # R11 predicts measured-mode rams 1.3-1.7 mm past the seat under lag while commanded
        # mode never crosses the target — a difference that should show up HERE if anywhere.
        "viol_per_env": viol_any.copy(),
        "viol_steps": int(viol_any.sum()),
        "jack_off_mm": jack_off_mm,
        "tilt_deg": tilt_deg,
        "final_along_mm": np.asarray(along_t)[-1] * 1000.0,
        "final_latn_mm": np.asarray(latn_t)[-1] * 1000.0,
        "final_ang_deg": np.degrees(np.asarray(ang_t)[-1]),
        "held": float(H.mean()),
        "held_ok": float(H[:, m].mean()),
        "ever": float(ever.mean()),
        "viol_envs": int((viol_any > 0).sum()),
        "viol_step_frac": float(viol_any.sum() / (steps * n)),
        "along_mm": float(np.mean(np.asarray(along_t)[:, m]) * 1000.0),
        "latn_mm": float(np.mean(np.asarray(latn_t)[:, m]) * 1000.0),
        "ang_deg": float(np.degrees(np.mean(np.asarray(ang_t)[:, m]))),
        "clean": int(((H.mean(axis=0) > 0.5) & (viol_any == 0) & ok).sum()),
        "ejected": int(ejected.sum()),
        "t_succ": t_succ,
        "t_seat": t_seat,
        "attempts": int(ctrl.attempts.sum()) if ctrl is not None else 0,
        "phase_frac": {nm: phase_hist[i] / tot for i, nm in enumerate(phase_names)},
        "phase_end": ctrl.phase_counts() if ctrl is not None else {},
        "lead_mm": (float(np.median(lead_mm)), float(np.percentile(lead_mm, 95)),
                    float(np.max(lead_mm))),
        # fraction of (frame, env) pairs that actually re-goaled the retimer — the number the
        # R2 suppression knobs move, and the retimer cost that rides on it
        "send_frac": (float(getattr(env.bank, "sent_envs", 0)) / max(1, steps * n)
                      if getattr(env, "bank", None) is not None else float("nan")),
        "follow_deg": ((float(np.median(follow_deg)), float(np.percentile(follow_deg, 95)),
                        float(np.max(follow_deg))) if follow_deg else (0.0, 0.0, 0.0)),
    }
    return out


def fmt_t(v):
    """Time-to-seat distribution over the envs that got there at all."""
    got = v[v >= 0]
    if got.size == 0:
        return "  never"
    return f"{np.median(got):3.0f} [{got.min():3.0f},{got.max():3.0f}] {got.size}/{v.size}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", default=DEFAULT,
                    help=f"comma-separated, from: {', '.join(SPECS)}")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--settle", type=int, default=80,
                    help="tail window the held rate is measured over (matches "
                         "rl/eval_scripted_controller.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--tilt", type=float, default=8.0)
    ap.add_argument("--standoff-mm", type=float, default=5.0)
    ap.add_argument("--push-correction", type=float, default=0.3)
    args = ap.parse_args()

    labels = [s.strip() for s in args.configs.split(",") if s.strip()]
    bad = [x for x in labels if x not in SPECS]
    if bad:
        raise SystemExit(f"unknown config(s) {bad}; choose from {list(SPECS)}")

    import rigid_cable_env as module
    from newton_cabling.scripted_controller import (
        AlignInsertController, InsertPhase, config_for_rigid_cable_env, observe_rigid_cable_env)

    phase_names = [p.name for p in InsertPhase]
    tilt = (0.0, args.tilt) if args.tilt > 0 else 0.0
    print(f"GATE 2 — BEHAVIOR | envs {args.envs} steps {args.steps} settle {args.settle} "
          f"seed {args.seed} stage {args.stage} tilt {args.tilt}")
    print(f"configs: {', '.join(labels)}")

    envs: dict[str, object] = {}

    def get_env(kind):
        if kind not in envs:
            t0 = time.perf_counter()
            if kind == "parent":
                e = module.RigidCableVecEnv(args.envs, seed=args.seed, cable_tilt_deg=tilt)
            else:
                from newton_cabling.servo_arm import ServoArmBank
                from servo_cable_env import FRAME_DT, ServoCableVecEnv
                # decimate=1 forwards every goal, so this is the shipped seam verbatim; only
                # the `-dN` configs below turn the proxy into something else.
                e = ServoCableVecEnv(
                    args.envs, bank=DecimatingBank(ServoArmBank(args.envs, frame_dt=FRAME_DT)),
                    max_cmd_lead_m=None, seed=args.seed, cable_tilt_deg=tilt)
            e.set_stage(args.stage)
            print(f"[build {kind}] {time.perf_counter() - t0:.1f}s | "
                  f"front_room {e.front_room * 1000:.1f}mm", flush=True)
            envs[kind] = e
        return envs[kind]

    rows = []
    for label in labels:
        sp = SPECS[label]
        kind, lead, over, dec = sp["kind"], sp["lead"], sp["over"], sp["decimate"]
        if kind == "parent" and (lead is not None or dec != 1 or sp["deadband"] or
                                 sp["every"] != 1):
            raise SystemExit(f"{label}: the parent env has no lead clamp and no bank")
        env = get_env(kind)
        # Every knob below is a plain attribute the seam reads fresh each frame, so switching
        # them between runs is RUNTIME configuration — no rebuild, no file edit.
        env.max_cmd_lead_m = lead
        if kind == "servo":
            env.bank.decimate = dec
            env.regoal_deadband_rad = sp["deadband"]
            env.regoal_every = sp["every"]
        ctrl = observe = None
        if over is not None:
            cfg = config_for_rigid_cable_env(
                module, align_standoff_m=args.standoff_mm / 1000.0,
                push_correction=args.push_correction, **over)
            ctrl, observe = AlignInsertController(args.envs, cfg), observe_rigid_cable_env
        t0 = time.perf_counter()
        r = run_one(env, module, label, args.steps, args.settle, ctrl, observe,
                    args.seed, phase_names)
        r["secs"] = time.perf_counter() - t0
        r["lead_set"] = lead
        r["decimate"] = dec
        r["kind"] = kind
        r["deadband"] = sp["deadband"]
        r["every"] = sp["every"]
        r["driver"] = "scripted" if over is not None else "env-teacher"
        r["over"] = over
        rows.append(r)
        note = f" | ERROR {r['error']}" if r.get("error") else ""
        print(f"  {label:<12} held {r['held']:6.1%} clean {r['clean']}/{r['n']} "
              f"ever {r['ever']:4.0%} | along {r['along_mm']:+7.2f}mm lat {r['latn_mm']:5.2f}mm "
              f"ang {r['ang_deg']:6.2f}deg | viol {r['viol_envs']}/{r['n']} "
              f"retreats {r['attempts']} | lead med {r['lead_mm'][0]:5.1f} max "
              f"{r['lead_mm'][2]:5.1f}mm | {r['secs']:.0f}s{note}", flush=True)

    # ── the table ───────────────────────────────────────────────────────────────────────
    print(f"\n{'config':<12} {'drive':<11} {'lead':>6} {'goalHz':>7} {'held':>7} {'clean':>7} "
          f"{'ever':>6} {'along':>8} {'lat':>6} {'ang':>7} {'viol':>6} {'retr':>5} "
          f"{'t-to-seat (med [min,max] got/n)':>32}")
    for r in rows:
        lead = "-" if r["lead_set"] is None else f"{r['lead_set'] * 1000:.0f}mm"
        # goal rate is a BANK property; the kinematic parent has no bank at all
        ghz = "-" if r["kind"] == "parent" else f"{60.0 / r['decimate']:.0f}"
        print(f"{r['label']:<12} {r['driver']:<11} {lead:>6} {ghz:>7} {r['held']:>6.1%} "
              f"{r['clean']:>4}/{r['n']} {r['ever']:>5.0%} {r['along_mm']:>7.2f} "
              f"{r['latn_mm']:>5.2f} {r['ang_deg']:>6.2f} "
              f"{r['viol_envs']:>3}/{r['n']}({r.get('viol_steps', 0):>4}) "
              f"{r['attempts']:>5} {fmt_t(r['t_succ']):>32}")

    print(f"\n{'config':<12} {'phase occupancy (SETTLE/ALIGN/PUSH/HOLD/RETREAT)':<52} "
          f"{'cmd lead med/p95/max mm':>26} {'follow med/max deg':>21}")
    for r in rows:
        pf = r["phase_frac"]
        occ = ("  ".join(f"{pf.get(k, 0.0):5.1%}" for k in phase_names) if pf
               else "(env teacher — no phases)")
        print(f"{r['label']:<12} {occ:<52} "
              f"{r['lead_mm'][0]:8.1f} {r['lead_mm'][1]:8.1f} {r['lead_mm'][2]:8.1f} "
              f"{r['follow_deg'][0]:10.2f} {r['follow_deg'][2]:10.2f} "
              f"{r['send_frac']:9.1%} sent")

    # Did the arm move at all? A config whose face ends where it started, while the command
    # integrator ran away, has a TRACKING failure — no controller retune can reach it.
    print(f"\n{'config':<12} {'start along/lat/ang':>26} -> {'end along/lat/ang':>24} "
          f"{'wrist travel':>13}")
    for r in rows:
        s = r["start"]
        print(f"{r['label']:<12} {s[0]:>9.2f}mm {s[1]:>6.2f}mm {s[2]:>6.2f}deg -> "
              f"{r['along_mm']:>8.2f}mm {r['latn_mm']:>6.2f}mm {r['ang_deg']:>6.2f}deg "
              f"{r['travel_mm']:>10.2f}mm")

    # ── per-env failure attribution ─────────────────────────────────────────────────────
    # The question this answers: is a non-seating env explained by its DOMAIN-RANDOMISATION
    # DRAW (a big jack offset, a steep droop) or not? If the failures are spread evenly across
    # the draw, the controller has a systematic problem and DR is a red herring.
    print(f"\nper-env attribution (servo rows; '*' = seated and held)")
    print(f"{'config':<14} {'env':>3} {'jack off':>9} {'tilt':>6} {'held':>6} "
          f"{'along':>8} {'latn':>7} {'ang':>7}")
    for r in rows:
        if r["kind"] == "parent" or "held_per_env" not in r:
            continue
        order = np.argsort(-r["jack_off_mm"])          # worst draw first
        for i in order:
            seat = "*" if r["held_per_env"][i] > 0.5 else " "
            print(f"{r['label']:<14} {i:>3} {r['jack_off_mm'][i]:>8.2f}mm "
                  f"{r['tilt_deg'][i]:>5.1f}° {r['held_per_env'][i]:>5.0%}{seat} "
                  f"{r['final_along_mm'][i]:>7.2f} {r['final_latn_mm'][i]:>6.2f} "
                  f"{r['final_ang_deg'][i]:>6.2f}")
        got = r["held_per_env"] > 0.5
        if got.any() and (~got).any():
            print(f"{'':<14} {'':>3} seated draws: off "
                  f"{r['jack_off_mm'][got].mean():.2f}mm tilt {r['tilt_deg'][got].mean():.1f}°"
                  f" | failed draws: off {r['jack_off_mm'][~got].mean():.2f}mm "
                  f"tilt {r['tilt_deg'][~got].mean():.1f}°")

    # ── verdict ─────────────────────────────────────────────────────────────────────────
    by = {r["label"]: r for r in rows}
    print()
    # A run that omits any gate config is a DIAGNOSTIC, not a gate. Saying PASS because the
    # configs that could have failed were not run is the one wrong answer this script must
    # never give, so it is refused explicitly rather than computed over whatever was present.
    missing = [c for c in GATE_CONFIGS if c not in by]
    if missing:
        print(f"GATE 2 INCOMPLETE — diagnostic run; gate configs {missing} not included. "
              f"Re-run with --configs {','.join(GATE_CONFIGS)} for a verdict.")
        return 2
    # Guaranteed present by the guard above, so no None-handling below.
    ref = by[GATE_REFERENCE]
    def at_parity(r):
        """Matches the reference within 5% on hold rate, and no worse on violations."""
        return r["held"] >= 0.95 * ref["held"] and r["viol_envs"] <= ref["viol_envs"]

    # A — reference sanity. If the kinematic arm cannot seat, nothing below means anything.
    a_ok = ref["held"] >= 0.95 and ref["viol_envs"] == 0
    print(f"  {GATE_REFERENCE:<2} reference reproduces the documented baseline : "
          f"{'PASS' if a_ok else 'FAIL'} (held {ref['held']:.1%}, viol {ref['viol_envs']})")

    # D — the subject. This is the row the gate exists to judge.
    subj = by[GATE_SUBJECT]
    d_ok = at_parity(subj)
    print(f"  {GATE_SUBJECT:<2} calibrated arm reaches reference parity      : "
          f"{'PASS' if d_ok else 'FAIL'} (held {subj['held']:.1%} vs {ref['held']:.1%}, "
          f"viol {subj['viol_envs']} vs {ref['viol_envs']})")

    # B — expected-fail canary. Never gates; only warns when it stops failing.
    canary = by[GATE_CANARY]
    canary_passed = at_parity(canary)
    print(f"  {GATE_CANARY:<2} expected-fail canary (measured mode)         : "
          f"{'fails as expected' if not canary_passed else 'PASSES — UNEXPECTED'} "
          f"(held {canary['held']:.1%})")

    ok = a_ok and d_ok
    print(f"\nGATE 2 {'PASS' if ok else 'FAIL'}")
    if canary_passed:
        print("\n  !! WARNING: measured-mode B unexpectedly passes — either the plant lost its\n"
              "  !! lag or the env changed; investigate before trusting D. The gate cannot\n"
              "  !! discriminate commanded-source servoing from measured-source while this\n"
              "  !! holds, so D's PASS above is not evidence for R11.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
