# Wave 2 gate report — calibrated servo arm in the cable-insertion env

*Authored by the Wave-2 gate agent (2026-07-31); persisted by the orchestrator.
Re-run commands at the bottom.*

| | verdict | headline |
|---|---|---|
| **Gate 1 — transfer** | **PASS** | worst joint disagreement `2.5e-14 deg` vs a `0.05 deg` bar; wrist FK error exactly `0.0` |
| **Gate 2 — behavior** | **FAIL** | reference A 100% seat / 8/8 clean / 0 viol; every servo config **0%** — and the wrist moves **0.09 mm vs A's 90.82 mm** |

Not in tension — that *is* the finding. **The integration is correct and the
plant is the problem.** The seam delivers the calibrated plant's trajectory into
Newton exactly; that trajectory just doesn't insert an RJ45, because under the
env's 60 Hz re-goal stream the calibrated command path advances the arm by
**0.12% of what it's asked for**.

---

## 0. Baseline reproduction — and the trap, resolved

Exact recipe (the repo's own canonical harness, unmodified):

```bash
.venv/bin/python rl/eval_scripted_controller.py --mode all --checkpoint /nonexistent \
    --envs 8 --stage 4 --steps 260 --settle 80 --seed 0
```

(`--checkpoint /nonexistent` is only how you ask `--mode all` to skip the PPO
leg. `--tilt` defaults to 8.)

```
stg 4 scripted  held 100.00%  clean 8/8 | along -0.00mm lat 0.01mm ang 0.01deg | viol 0/8 | phases {'HOLD': 8} retries 0
stg 4 teacher   held   0.00%  clean 0/8 | along -30.71mm lat 0.08mm ang 24.92deg | viol 0/8
```

**The documented 100%/0-viol baseline belongs to the scripted
`AlignInsertController`, not to `env.servo_action()`.** Reproduced exactly.

The previous agent's harness was NOT wrong. `env.servo_action()` seats at **no
stage, with or without tilt, on the ideal kinematic parent env** (`--mode
teacher --all-stages --tilt 0`: 0.00% at stages 0-4, parking at ang ~25-29 deg,
hold=0). Cause: it closes its orientation loop on the **wrist** (servoing to
`wq_level`) and deliberately never servos the plug FACE — its own docstring says
a face-angle servo "PUMPS" the pendulum. With the current ~45 deg tilted grasp
the face starts ~25 deg off the seat; `SEAT_ANGLE` is 8 deg. Pre-existing
property of the uncommitted `rl/rigid_cable_env.py`, unrelated to the servo
work.

---

## 1. Gate 1 — TRANSFER: PASS (`gate_transfer.py --envs 4 --steps 60 --seed 0`, 44 s)

Fixed RNG-free action sequence (three-frequency sinusoids + a sign flip at the
halfway point, so the `set_goal` **interrupt** path is exercised and the wrist
target stays bounded). What reaches Newton is captured by wrapping
`_sim_from_stream` on the instance — the single door between bank and solver —
so nothing upstream is trusted.

| check | worst | bar | |
|---|---|---|---|
| **T1 STREAM** driven substep `[-1]` − branch vs `bank.jq` | `2.54e-14` deg | 0.05 deg | PASS |
| **T2 SEED** `env.jq[arm_qc]` − branch vs `bank.jq` | `1.37e-05` deg | 0.05 deg | PASS |
| **T3 FK-pos** FK(`bank.jq`) vs the wrist Newton holds | `0.0` m | 1e-5 m | PASS |
| **T3 FK-rot** FK(`bank.jq`) vs the wrist Newton holds | `0.0` deg | 1e-3 deg | PASS |
| **T4 REPLAY** fresh bank, same jq0 + same goal sequence | `0.0` deg | 0.05 deg | PASS |

0.05 deg is the demo repo's `validate_newton` bar, reused so both pipelines
answer to one standard.

- **The branch offset is genuinely exercised, not dead code.** 3 of 4 envs
  needed joint 5 shifted a full `-1` turn. T3 would catch an offset wrong by a
  non-multiple of 2*pi, and reads exactly 0.
- **T2 is `1.4e-5` not 0** because `self.jq` is float32 and the bank is float64:
  at ~6.3 rad float32 spacing is 2.4e-7 rad = 1.4e-5 deg — matches to the digit,
  3600x under bar. Benign; the next step's IK seed carries a float32 rounding of
  the plant state.
- Context at moderate actions: following error median 1.89 deg / max 5.71 deg,
  wrist lead median 7.05 mm / max 17.69 mm, goal re-issued on 60 of 60 frames.
  Entire preview of Gate 2.

---

## 2. Gate 2 — BEHAVIOR: FAIL (`gate_behavior.py --configs A,B,C5,C10 --envs 8 --steps 260 --settle 80 --seed 0`)

Stage 4, `cable_tilt_deg=(0,8)`, `AlignInsertController` at harness defaults.
`env._rng` re-seeded before every reset, so all configs see identical episodes;
the only difference per row is the arm.

| config | arm | lead clamp | held | clean | along | lat | ang | viol | t-to-seat |
|---|---|---|---|---|---|---|---|---|---|
| **A** | kinematic parent | – | **100.0%** | **8/8** | −0.01 mm | 0.01 mm | 0.01° | 0/8 | 122 [119,124] 8/8 |
| **B** | calibrated bank | None (parity) | 0.0% | 0/8 | −47.7 mm | 15.6 mm | 25.5° | 0/8 | never |
| **C5** | calibrated bank | 5 mm | 0.0% | 0/8 | −47.7 mm | 15.7 mm | 25.5° | 0/8 | never |
| **C10** | calibrated bank | 10 mm | 0.0% | 0/8 | −47.5 mm | 15.7 mm | 25.5° | 0/8 | never |

| config | phase occupancy S/A/P/H/R | cmd lead med/p95/max | follow err med/max | wrist travel |
|---|---|---|---|---|
| A | 0.4 / 32.8 / 7.7 / **59.1** / 0.0 % | 0.0 / 0.3 / 0.5 mm | – | **90.82 mm** |
| B | 0.4 / **99.6** / 0.0 / 0.0 / 0.0 % | 154.8 / 285.9 / **376.8** mm | 34.38 / 34.38° | **0.09 mm** |
| C5 | 0.4 / **99.6** / 0.0 / 0.0 / 0.0 % | 5.0 / 5.0 / 5.0 mm | 34.38 / 34.38° | **0.09 mm** |
| C10 | 0.4 / **99.6** / 0.0 / 0.0 / 0.0 % | 10.0 / 10.0 / 10.0 mm | 34.38 / 34.38° | **0.09 mm** |

Three decisive observations:

1. **The lead clamp works and changes nothing.** C5/C10 hold the command to
   exactly 5.0/10.0 mm (vs B's runaway to 377 mm) and the trajectory is
   identical to B within 0.1 mm / 0.04 deg. Correctly implemented, behaviorally
   inert in this regime. Not the fix.
2. **`stable_steps` / `jam_window` are not the story.** The controller spends
   99.6% of the episode in ALIGN and never enters PUSH; neither threshold is
   ever evaluated, retreats = 0.
3. **The plant sits a constant 34.38 deg from its own goal** — that is
   `ArmIK`'s `iters=2 x step_cap=0.3 rad`. The IK is saturated against its step
   cap every frame: the target is so far away the solver returns a
   maximally-clipped goal, forever.

### Ruling out the controller: eleven runtime variants (no file edits)

| variant | changes (runtime only) | along | lat | ang | cmd lead med | held |
|---|---|---|---|---|---|---|
| `B` | — | −46.13 | 14.66 | 25.56 | 153.7 mm | 0% |
| `B-slow` | align/push rates ÷2 | −46.11 | 14.54 | 25.50 | 77.1 mm | 0% |
| `B-fast` | align/push rates ×2 | −46.09 | 14.69 | 25.51 | 245.9 mm | 0% |
| `B-stable8` | `stable_steps` 3→8 | −46.14 | 14.59 | 25.52 | 153.7 mm | 0% |
| `B-jam40` | `jam_window` 20→40 | −46.11 | 14.54 | 25.55 | 153.8 mm | 0% |
| `B-d6` | re-goal 60→10 Hz | −46.85 | 14.38 | 25.05 | 149.7 mm | 0% |
| `B-d20` | re-goal→3 Hz | −50.10 | 20.39 | 20.03 | 105.2 mm | 0% |
| `B-d60` | re-goal→1 Hz | **−2.05** | 17.73 | **7.95** | 93.4 mm | 0% |
| `C5-d60` | 1 Hz + 5 mm clamp | −33.69 | **72.00** | **4.18** | 5.0 mm | 0% |
| `B-teacher` | env `servo_action()` on the bank | −46.12 | 14.54 | 25.53 | 144.5 mm | 0% |

Read `B-slow`/`B`/`B-fast` together: halving or doubling the commanded rate
moves the command lead proportionally (77/154/246 mm) and moves the plug not at
all — a tracking failure, not a strategy failure. At 1 Hz re-goals the arm
finally moves (ang 25.5→7.95) but the loop is sampled 60x slower than the
controller integrates and lateral control diverges (17.7 mm vs the 3 mm
`SEAT_OFFSET`). Partial fix, not a fix.

### The mechanism, measured directly on the bank (no GPU, no env)

`ServoArmBank.set_goal` re-plans a **rest-to-rest, jerk-limited** ruckig
trajectory from the currently commanded position (`control.retime`: VMAX 0.4,
AMAX 0.7, JMAX 1.5), and `transport_delay_s = 0.010 s` = 0.60 of a 16.7 ms
control frame. A jerk-limited profile from rest covers `j*dt^3/6 = 1.16 urad`
in its first frame. The env issues a new goal every frame, so **the plan is
replaced before it ever leaves its own initial jerk ramp.**

One arm, 24 deg commanded over 120 frames:

| re-goal rate | 60 Hz (env) | 30 Hz | 10 Hz | 5 Hz | 3 Hz | 2 Hz | 1 Hz | one uninterrupted goal |
|---|---|---|---|---|---|---|---|---|
| achieved | **0.029°** | 0.011° | 0.205° | 0.962° | 2.843° | 6.444° | 17.497° | **23.98° in 2.0 s** |
| tracking | **0.12%** | 0.05% | 0.85% | 4.0% | 11.8% | 26.9% | 72.9% | **99.9%** |

And the property that closes the case — re-goaling every frame, achieved motion
is **independent of how fast the command moves** (0.05-0.80 deg/frame all
achieve 0.0005 deg total). No gain, rate, tolerance or threshold on the
controller side can change the outcome. **The plant itself is fine**: handed one
goal and left alone it tracks 24 deg to 99.9% in 2 s, as its unit tests assert.
The failure is the *regime* — a command path designed around discrete `move_to`
goals seconds apart, driven by a policy loop re-commanding at 60 Hz.

### Contract angle

`newton_cabling/servo_arm/CONTRACT.md` `set_goal` specifies "Retimed by control/ ... FROM THE
PLANT'S CURRENT streamed state (pos+vel) ... ruckig's online-trajectory-
generation mode". The implementation is honest about the gap in its own
docstring ("the verbatim retimer is rest-to-rest, so only the position seeds the
plan"). `control.retime` hard-codes zero initial velocity and is a verbatim port
that must not be edited there. So the bank implements point-to-point
replanning, not OTG. Orchestrator decision required (see Addendum 4 of the
contract for the ruling).

---

## 3. Recommendations (none applied by Wave 2)

- **R1 (the fix).** Velocity-seeded re-plan (true ruckig OTG) in
  `newton_cabling/servo_arm/bank.py` — keep `plan_move` for cold starts; the
  interrupt path passes current commanded velocity/acceleration into ruckig.
  `control/retime.py` stays verbatim-untouched.
- **R2 (cheap, partial, needed regardless).** Goal deadband / decimation knob in
  `ServoCableVecEnv` next to the existing exact-inequality suppression.
  Companion to R1, not an alternative.
- **R3.** Keep `max_cmd_lead_m` default `None` (diagnostic, not behavioral —
  demonstrated inert here). Re-evaluate after R1.
- **R4.** Do not raise `ik_iters`/`step_cap` now; the 34.38 deg saturation is a
  symptom. Revisit after R1.
- **R5.** Controller tuning: nothing to do. All varied parameters produced the
  same trajectory within 0.1 mm. `AlignInsertConfig` needs no change on this
  evidence; revisit `stable_steps` only once the plant actually tracks.
- **R6 (user-facing, outside this effort).** `env.servo_action()` does not seat
  at any stage on the ideal arm in the current uncommitted
  `rigid_cable_env.py`. Anything still treating it as "the teacher"
  (`record_cable_env.py` without `--scripted`, BC warm-start) records failing
  rollouts.

## 4. What remains UNVALIDATED

- **The calibration's high-interrupt-rate regime — the central open question.**
  Nobody has a hardware recording of the arm streamed goals at 60 Hz. Until
  someone does, R1 produces a *plausible* arm, not a *validated* one. This
  should be the next hardware ask.
- The lead clamp's behavioral effect (can't be asked until the arm tracks).
- Everything past ALIGN on the servo arm: PUSH, HOLD, RETREAT, jam detection,
  and the standoff clamp have never executed with the calibrated plant.
- Gate 1 covers 60 frames x 4 envs of sinusoids; it never triggers the
  different-branch-mid-episode raise, nor an out-of-range-not-whole-turn build.
- n=8, one seed, stage 4 only (servo rows fail uniformly to 0.1 mm across 14
  runs, so seed variance is implausible as a confound — but no sweep was run).
- Scene builds are not bit-reproducible across processes (VBD settle on GPU);
  all comparisons here are within-process by construction.

## 5. Re-run

```bash
.venv/bin/python tests/gates/gate_transfer.py                                    # ~45 s, PASS
.venv/bin/python tests/gates/gate_behavior.py                                    # ~3 min, FAIL (pre-R1)
.venv/bin/python tests/gates/gate_behavior.py --configs B,B-slow,B-d60,C5-d60    # diagnosis matrix
```

Both take `--envs/--steps/--seed/--stage/--tilt`. `--configs` accepts any subset
of `SPECS`; `-suffix` labels are runtime experiments, not gate criteria — the
verdict is computed from `A`, `B`, `C5` only.

---

# Wave 2c — delta report (post-R1 OTG fix + R2 knobs)

**Overall behavior gate: still FAIL** — but the failure moved twice and is now
quantified and bounded. R1 turned a dead arm into a live one; the remaining gap
is a controller-timing problem with a measured partial recovery (0% -> 23% held
/ 38% ever), not a structural one. Not a conditional pass: no override set
reaches parity with A.

| gate | Wave 2b | Wave 2c |
|---|---|---|
| Gate 1 — transfer | PASS | **PASS** (T1 2.544e-14 deg, T2 1.360e-05 deg, T3 0.0/0.0, T4 0.0 — bars unchanged) |
| Gate 2 — behavior | FAIL, arm frozen (0.09 mm travel) | **FAIL**, arm live (86 mm travel), 23.3% held with overrides vs A's 100% |

## R1 verified independently (march probe, re-run unchanged)

| re-goal rate | pre-R1 | post-R1 |
|---|---|---|
| 60 Hz (the env) | 0.12% | **80.42%** |
| 10 Hz | 0.85% | 81.51% |
| 2 Hz | 26.9% | 85.44% |
| one uninterrupted goal | 99.9% | 99.9% |

Tracking is now essentially re-goal-rate-independent — which retires R2 as a
correctness fix; re-goal decimation is purely a performance knob.

## Base matrix (260 steps, seed 0, stage 4, tilt 8)

| config | held | ever | along start->end | lat end | ang end | wrist travel | lead med/max | phases S/A/P/H/R |
|---|---|---|---|---|---|---|---|---|
| A | **100.0%** | 100% | −45.88 -> **+0.00** | 0.01 | 0.01 | 90.88 mm | 0.0/0.5 | 0.4/32.8/7.7/59.1/0 |
| B | 0.0% | 0% | −45.28 -> −21.10 | 12.58 | **6.33** | **60.11 mm** | 41.7/80.1 | 0.4/99.6/0/0/0 |
| C5 | 0.0% | 0% | -> **−2.24** | **83.30** | 7.12 | 26.80 mm | 4.9/5.4 | 0.4/99.6/0/0/0 |
| C10 | 0.0% | 0% | -> −5.81 | **68.51** | 7.13 | 28.98 mm | 9.9/10.4 | 0.4/99.6/0/0/0 |

- The 34.38 deg IK step-cap saturation is GONE (follow err 9.05 med / 20.65 max).
  **R4 closed — no ik_iters/step_cap change warranted.**
- PUSH/HOLD/RETREAT still never execute at stock gains (docking needs lat <=
  0.5 mm; B sits at 12.58 mm).

## Controller-timing diagnosis — rate is the lever, gain alone is not

| variant | change | lat | ang | PUSH% |
|---|---|---|---|---|
| B | — | 10.25 | 6.34 | 0.0 |
| B-stable8 | stable_steps 3->8 | 9.67 | 6.45 | 0.0 |
| B-jam40 | jam_window 20->40 | 9.80 | 6.61 | 0.0 |
| B-lowkp | kp 0.5->0.15 only | 11.34 | 6.64 | 0.0 |
| B-rotslow | rot rate 0.3->0.1 deg, kp_rot 0.2 | 4.05 | 7.26 | 0.0 |
| **B-slow** | **rates / 2** | **1.70** | **1.60** | 1.2 |
| **B-slow-kp** | **rates / 2 + kp 0.15** | **0.92** | **1.45** | **5.6** |
| B-vslow | rates / 4 | 1.93 | **10.41** (rate floor: 25.5 deg at 0.075 deg/frame needs 340+ frames) | 0.0 |
| B-db2e3 | regoal_deadband 2e-3 | 10.00 | 6.71 | 0.0 (suppresses only 0.4% of sends) |

## The lead clamp CAUSES ramming as implemented (R9)

| | lat (untuned, 260) | lat (tuned slow-kp, 700) |
|---|---|---|
| no clamp | 12.58 mm | **3.97 mm** |
| max_cmd_lead_m = 5 mm | **83.30 mm** | **39.38 mm** |
| max_cmd_lead_m = 10 mm | 68.51 mm | – |

Mechanism: the governor rescales the whole 3-D lead vector ISOTROPICALLY and
re-anchors to the measured wrist each frame. The lead is dominated by the axial
push component, so the clamp attenuates the lateral correction by the same
factor — the command degenerates into a persistent "5 mm ahead along ins" pull
with no lateral authority, and the grasp drags the plug sideways. Monotone in
clamp tightness, as the attenuation story predicts.

## Long horizon with the tuned profile (700 steps, settle 120)

| config | held | clean | ever | lat | ang | t-to-seat | phases S/A/P/H/R |
|---|---|---|---|---|---|---|---|
| A | **100.0%** | 8/8 | 100% | 0.01 | 0.01 | **122 [119,124] 8/8** | 0.1/12.2/2.9/84.8/0 |
| **B-slow-kp** | **23.3%** | **3/8** | **38%** | 3.97 | 2.74 | **319 [315,372] 3/8** | 0.1/76.4/7.2/16.3/0 |
| B-slow-kp-s8 | 15.4% | 2/8 | 25% | 4.63 | 3.01 | 322 | 0.1/83.7/5.2/10.9/0 |
| C5-slow-kp | 0.0% | 0/8 | 0% | **39.38** | 3.40 | never | 0.1/99.9/0/0/0 |

First non-zero success on the calibrated arm. stable_steps 3->8 is measurably
WORSE (15.4% vs 23.3%): under lag it delays PUSH entry and the extra frames
cost more than the fly-through protection is worth.

## Recommendations (Wave 2c; none applied)

- **R7 (adopt): a servo-arm controller PROFILE** — align_lin_rate_m
  0.0008->0.0004, align_rot_rate_rad 0.3->0.15 deg, push_rate_m 0.0008->0.0004,
  kp_lin/kp_rot 0.5->0.15. 0% -> 23.3% held / 38% ever; the only thing that
  makes ALIGN converge (lat 12.6 -> 0.92 mm). Ship as an alternate
  AlignInsertConfig preset selected by arm model, NOT a new kinematic default.
- **R8: raise the servo-arm episode budget to >= 700 steps** (A seats at 122,
  servo at 319; at 260 the tuned config scores 0% purely on horizon).
- **R9 (blocking): do NOT ship max_cmd_lead_m until the governor is
  anisotropic** (clamp the along-ins component; budget axial and lateral
  separately). Supersedes R3, which assumed it was merely inert.
- **R10: stable_steps 3->8 rejected on measurement; jam_window no effect. R5's
  revisit clause closed.**
- **R11: the remaining 77% gap is lag, not gain.** Further tuning trades
  rotation speed against lateral overshoot (B-vslow proves the trade). The
  principled next step is lag compensation — predict the face forward by the
  plant's known lag, or servo on the commanded rather than measured wrist. Untested.
- R4 closed; R2 downgraded to a perf knob.

## Still unvalidated (Wave 2c)

- The hardware question, unchanged and sharper: no recording exists of the real
  RO2 core streamed goals at 60 Hz. R1 swapped one unverified interrupt model
  for a better-motivated one. **Still the next hardware ask.**
- Deadband quantization untriggered (2e-3 rad barely engages), not refuted.
- Anisotropic lead governor: recommended, never built.
- 5 of 8 envs never seat even tuned; no per-env failure attribution vs jack
  offset magnitude.
- n=8, one seed, stage 4 only; within-process comparisons (VBD settle variance
  ~1 mm cross-process).

## Housekeeping

- gate_behavior.py verdict bug fixed: it printed GATE 2 PASS when the required
  configs were absent from a run; now prints GATE 2 INCOMPLETE and exits 2
  unless all of A,B,C5 ran. Any `GATE 2 PASS` in w2c_long.log predates the fix.
- New SPECS labels: B-slow, B-slow-kp, B-slow-kp-s8, B-vslow, B-rotslow,
  B-lowkp, C5-slow-kp, B-db2e3, B-db2e3-e6, plus the -dN decimation rows.

---

# Wave 4 — R11 verdict run: servo-on-commanded CONFIRMED, D ~ A

The controller gained `servo_source="commanded"` (servo errors from the
commanded face derived off wrist_tgt; ALL phase gates stay on measured truth).
Matrix at 700 steps / settle 120 / seed 0 / stage 4 / tilt 8, lead clamp None:

| config | mode | rates | held | clean | ever | lat | ang | viol | t-to-seat |
|---|---|---|---|---|---|---|---|---|---|
| A kinematic | – | stock | **100.0%** | 8/8 | 100% | 0.01 | 0.01 | 0/8 | **122 [119,124]** |
| B servo | measured | stock | 0.0% | 0/8 | 0% | 15.68 | 7.31 | 0/8 | never |
| B-R7 servo | measured | R7 | 21.7% | 2/8 | 38% | 6.18 | 4.26 | 0/8 | 315 |
| **D servo** | **commanded** | **stock** | **100.0%** | **8/8** | **100%** | **0.29** | **0.10** | **0/8** | **246 [197,370]** |
| D-R7 servo | commanded | R7 | 100.0% | 8/8 | 100% | 0.44 | 0.12 | 0/8 | 270 [264,275] |

- Command lead collapses by construction: median 38.8 -> **0.4 mm** (the servo
  parks the integrator on the pre-dock pose instead of running away). Follow
  error median 0.06 deg. D matches A on outcome, not speed: 246 vs 122 (2.0x).
- **R7 retired as a requirement**; keep as an option — on top of commanded mode
  it buys robustness (peak lead 43.4 -> 17.5 mm, peak follow 10.94 -> 3.89 deg,
  seat window [197,370] -> [264,275]) for ~24 median steps. Evaluate D-R7 on the
  first hardware trial.
- **Per-env attribution (first ever)**: measured-mode failure is predicted by
  the TILT draw (all >= 4.9 deg fail; the worst jack offset 7.48 mm nearly
  seats), i.e. tilt aggravated the stale-feedback instability. In commanded
  mode the DR dependence VANISHES — all 8 envs hold across the full span
  (offset 0.02-7.48 mm, tilt 0.1-7.3 deg). Measured-mode B also EJECTS an env
  (grasp destroyed, plug metres away); commanded mode ejects none.
- **Depth-ram prediction neither confirmed nor refuted**: `_violations()` counts
  finger-jack contacts, not seat overshoot — the wrong instrument. Measured
  along overshoot: A -0.00, D +0.13, D-R7 +0.18 mm. A proper test needs a
  depth-overshoot metric or the contact-force channel.
- **Shipping preset**: `config_for_rigid_cable_env(rigid_cable_env,
  servo_source="commanded")`, env at `max_cmd_lead_m=None`. Episode budget for
  the servo arm: >= 450 steps (D worst case 370). R9's clamp is now largely
  moot (median lead 0.4 mm); if kept, reshape as a fault DETECTOR, not a
  corrector.
- R11 CLOSED. Remaining open: hardware 60 Hz recording (top ask, unchanged);
  depth-ram instrumentation; seat-time variance tail on larger batches; one
  seed / one stage coverage.

## Wave 4c — formal gate redefined and GREEN

gate_behavior.py now gates on named roles: A = reference (must pass), D =
subject (must pass), B = expected-fail canary (never gates; an unexpected B
pass prints a loud warning that D's pass is no longer evidence for R11). C5
dropped from gating per R9. Bare `gate_behavior.py` runs the formal gate.

Verification run (A,B,D / 700 steps / settle 120 / seed 0):
A 100% (seat 122), B 0% (canary behaved), D 100% 8/8 0-viol (seat 249
[225,367] — reproduces Wave 4b's 246 [197,370] within cross-process VBD
variance). **Verdict: GATE 2 PASS, exit 0.** The first honest green gate on
record.
