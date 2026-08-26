# ServoArmBank contract (frozen — Wave 1 codes against this)

Purpose: put the hardware-calibrated RO2-core servo model (parallax-demo-newton
`sysid/` + `control/`) in the loop of newton-cabling's cable-insertion envs as a
**joint-trajectory generator**. The cable env keeps driving the arm kinematically
per substep — only the source of the per-substep joint positions changes.

## API (package `newton_cabling/servo_arm/`, new dir)

```python
class ServoArmBank:
    """n independent calibrated arms. Pure CPU: numpy + mujoco + ruckig/toppra.
    NO torch, NO warp, NO newton imports — must run in the no-GPU test gate."""

    def __init__(self, n: int, *, frame_dt: float, params=None):
        """params default: sysid.params.load_servo_params() — fails loudly if absent.
        Plant config: gripper-ON (AG-145 payload fused), per the demo repo path."""

    def reset(self, jq0: np.ndarray) -> None:
        """(n, 6) arm joint positions. Plant lands exactly at jq0, zero velocity,
        stream empty, delay buffer empty."""

    def set_goal(self, env_ids: np.ndarray, q_star: np.ndarray) -> None:
        """New joint goal per listed env. Retimed by control/ (ruckig point-to-point)
        FROM THE PLANT'S CURRENT streamed state (pos+vel), INTERRUPTING any in-flight
        plan — ruckig's online-trajectory-generation mode, the same interrupt
        semantics control/stream.py already implements and tests."""

    def advance(self, substeps: int) -> np.ndarray:
        """Advance one control frame (frame_dt). Internally: run the 500 Hz stream
        through the transport-delay buffer and mj_step the plant at its own dt.
        Returns (substeps, n, 6): joint positions resampled evenly across the frame,
        last sample == plant state at frame end. This feeds the env's existing
        per-substep kinematic drive unchanged."""

    @property
    def jq(self) -> np.ndarray: ...      # (n, 6) current measured positions
    @property
    def jqd(self) -> np.ndarray: ...     # (n, 6) current measured velocities
```

## Pinned semantics (not negotiable without orchestrator sign-off)

1. **One calibration, one path.** Task-mediated path only: goal -> retimer ->
   500 Hz pos+vel+accel feedforward -> servo plant. Never a direct setpoint stream.
2. **Transport delay lives INSIDE the bank** (in-flight command buffer,
   `transport_delay_s` from the params artifact, buffers maxlen = d+1). The env
   must not add its own.
3. **6 DOF only.** Gripper/finger/jack coordinates stay entirely the env's business;
   the integrator maps env `jq` slices <-> the bank's (n, 6).
4. **Goal clamping happens BEFORE the bank.** The env's finger-standoff clamp
   truncates the commanded wrist advance pre-IK; the bank must never be handed an
   unreachable/forbidden goal to chase.
5. **Deterministic.** No RNG anywhere in the bank. Same (jq0, goal sequence) ->
   bit-identical output. Gates and golden runs depend on this.
6. **Fail loudly.** Missing params, uncalibrated joint, shape mismatch, non-finite
   goal: raise. No silent defaults (house rule from sysid/).

## Builder's own unit tests (tests/test_servo_arm.py, no GPU, < 5 s)

- Step-response: command a step goal; assert no overshoot; settle time consistent
  with per-joint T_lag (~15-18 ms); first response begins transport_delay_s late.
- Interrupt mid-plan: no position jump (mirrors control's own gate).
- Determinism: two identical runs bit-equal.
- advance() resampling: monotone time base; last sample equals plant state.

## Integrator seam (rl/servo_cable_env.py, new file)

`ServoCableVecEnv(RigidCableVecEnv)`: where the parent's `step()` assigns
`self.jq` from IK and `_sim(jq_prev)` interpolates linearly, the subclass feeds
the IK result to `bank.set_goal`, calls `bank.advance(substeps)`, and drives the
same per-substep kinematic path with the bank's samples. `_sync_teleport` /
`body_q_prev` semantics preserved exactly. Parent file: ZERO diffs.

Golden regression (tests/test_golden_kinematic.py): fixed-seed parent-env rollout
hashed before/after — proves the kinematic path is bit-untouched, and guards any
future seam edit.

## File ownership (hard boundaries; stop-and-report on conflict)

| agent          | writes ONLY |
|----------------|-------------|
| Scout          | nothing (read-only) |
| Test-hardener  | tests/test_scripted_controller.py |
| Packager       | both pyproject.toml; minimal __file__-relative fixes in sysid/params.py, sysid/prep_run.py if forced |
| ServoArm builder | newton_cabling/servo_arm/** (new), tests/test_servo_arm.py (new) |
| Env integrator | rl/servo_cable_env.py (new), tests/test_golden_kinematic.py (new) |
| Gate runner    | tests/gates/** (new) + results report |
| Reviewer       | nothing (read-only) |

NOBODY touches: rl/rigid_cable_env.py, newton_cabling/scripted_controller/**,
or any file with uncommitted user modifications (rl/gen_cable_traj.py,
rl/record_cable_env.py, rl/train_cable_ppo.py, scripts/record_sbot_scene_gs_cable.py,
tools/*). No git commits by any agent.

## Addendum 1 (orchestrator ruling): the step() seam

The parent `RigidCableVecEnv.step()` has no extractable seam method and may not be
edited. The subclass is therefore ALLOWED to duplicate the parent's step() body
with only the jq/_sim segment swapped (goal -> bank.set_goal -> bank.advance ->
per-substep drive), under two mandatory guards:

1. **Drift guard**: the subclass module stores the sha256 of the parent step()
   source it was written against; a test asserts it still matches, so any future
   parent edit fails loudly with "re-sync the subclass copy" instead of drifting.
2. **Identity-bank equivalence**: an `_IdealBank` stub (advance() returns exact
   linear interpolation to the goal — the parent's own kinematic behavior) must
   make `ServoCableVecEnv` reproduce the parent env's rollout on a fixed seed to
   numerical tolerance. This proves the copy is faithful independent of the
   calibrated plant.

Golden baseline may be tolerance-based (stored npz + allclose) rather than a byte
hash if GPU nondeterminism is observed — verify determinism empirically (two runs)
and choose accordingly, documenting the choice in the test.

## Addendum 3 (orchestrator ruling): joint revolution branch + command lead

**Branch offset (the joint5 = -391 deg blocker).** The env's IK/Newton joints are
continuous; a build slot may settle a full revolution outside the bank's +-360 deg
limits. Ruling: the bank stays untouched (its refusal is correct honesty). The
branch offset is OWNED BY ServoCableVecEnv as a per-joint constant established at
reset: `offset = jq0_env - wrap(jq0_env into bank limits)` (an exact multiple of
2*pi). Env -> bank: subtract offset from every IK goal and from jq0. Bank -> env:
add offset back to every sample before it drives Newton/self.jq. This is
pose-exact, deterministic, and — critically — never interpolates across a 2*pi
jump (no full-unwind sweep inside a frame). Assert the offset is a multiple of
2*pi and constant for the episode; if a goal ever needs a DIFFERENT branch than
reset established, raise (that is a real full-turn command, not bookkeeping).

**Command-lead clamp (the 28 mm open-integrator runaway).** wrist_tgt_p is an
open command integrator in the parent; with a lagging plant it outruns the arm
(~0.5 s / 28 mm observed). Ruling: add an OPTIONAL `max_cmd_lead_m` parameter to
ServoCableVecEnv (default None = strict parent parity): when set, the wrist
target's advance is clamped so |wrist_tgt_p - measured wrist| <= max_cmd_lead_m
(a following-error governor, like the real controller's fault limit — physically
motivated, not a hack). Default stays None; Wave 2 measures teacher behavior
both ways and recommends. Action semantics (deltas-on-command) stay parent-parity
in the default configuration.

**Golden baseline vs .gitignore**: the repo's blanket `*.npz` ignore would drop
tests/golden_kinematic.npz; orchestrator adds a `!tests/golden_kinematic.npz`
negation (one line, user-visible, reported).

## Addendum 4 (orchestrator ruling, post-Gate-2): OTG interrupt is mandatory

Gate 2 measured the rest-to-rest re-plan deviation at 0.12% tracking under the
env's 60 Hz re-goal stream (see tests/gates/REPORT.md). Rulings:

**R1 approved — the bank must implement the contract's original OTG semantics
literally**: the interrupt path seeds ruckig with the in-flight COMMANDED
position, velocity, and acceleration (command-side state, not plant-measured
state — a robot's own controller re-plans from what it commanded). Rest-to-rest
`plan_move` remains for cold starts (at-rest re-goals). `control/retime.py`
stays verbatim-untouched; the OTG path lives in bank.py, importing the same
VMAX/AMAX/JMAX limits from the port (single source of truth for limits). The
calibrated plant + 500 Hz stream + transport delay are unchanged — the
calibration models the plant tracking a stream, and only the plan generator
changes.

**Fidelity caveat, pinned**: no hardware recording exists of the arm under
high-rate re-goals. R1 yields a plausible arm, not a validated one. The next
hardware ask is one recording of the real arm streamed goals at policy rate;
until then every consumer-facing doc must carry this caveat.

**R2 approved** — optional re-goal suppression knob in ServoCableVecEnv
(deadband in joint space and/or every-k-frames), DEFAULT OFF. Perf/plan-churn
aid, never a behavioral default until measured post-R1. Beware: a coarse
deadband quantizes fine alignment (0.5 mm lateral tolerance) — default off.

**R3 confirmed** — max_cmd_lead_m default stays None (diagnostic).
**R4** — ik_iters/step_cap untouched until post-R1 measurement.
**R5** — AlignInsertConfig unchanged; the scripted controller survives Wave 2
with zero recommended changes.

## Addendum 2: install state (Packager, done)

newton-cabling's venv already imports sysid/control (editable, one home).
Consumer command if a venv is ever rebuilt:
    cd newton-cabling
    uv pip install --python .venv/bin/python -e '.[servo]'
    uv pip install --python .venv/bin/python --no-deps -e ../parallax-demo-newton
(-e on the first command is load-bearing: without it, newton-cabling itself gets
silently un-editabled.) warp/newton pins verified untouched: 1.14.0 / 1.4.0.dev0.
