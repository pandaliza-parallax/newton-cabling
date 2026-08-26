"""Golden regression on the cable env's KINEMATIC PATH, and the guards for the servo subclass.

`rl/servo_cable_env.py` duplicates `RigidCableVecEnv.step()` with only the jq/_sim segment
swapped (newton_cabling/servo_arm/CONTRACT.md, Addendum 1). This file is the price of that
duplication:

  * test_parent_source_unchanged   — guard 1: the parent methods the copy was written against
                                     still hash to PARENT_SOURCE_SHA256.
  * test_golden_kinematic_path     — the parent env's own arm-drive trace, pinned to a stored
                                     baseline. Any change to the standoff clamp, the IK call,
                                     the substep interpolation or SUBSTEPS breaks it.
  * test_ideal_bank_equivalence    — guard 2: on ONE ServoCableVecEnv instance, the parent's
                                     step() and the subclass's step() driven by `_IdealBank`
                                     produce the same arm drive.

── determinism verdict (measured on this box: RTX 5090, warp 1.14, newton 1.4.0.dev0) ──────
Three repeats of the same fixed-action rollout on one env instance, 40 steps, n=2:

    channel                                            max |run-to-run diff|
    per-substep joint stream / wrist pose / wrist
    target / `advanced`  (KINEMATIC)                   0.0    — BIT-IDENTICAL
    plug-face pose, cable body_q  (DYNAMIC)            7.6e-2 m over 40 steps,
                                                       1e-4 .. 5e-4 m already at step 0

So the comparison mode is split, and deliberately so:
  * the kinematic channels are compared EXACTLY (atol=0). They are a pure function of
    (joint state, wrist target, action) evaluated on the host + a deterministic eval_fk —
    no contact solver, no atomics — so anything else is a real change.
  * the dynamic channels are NOT asserted numerically. VBD's contact reduction reduces
    contacts with atomics, so ordering (and hence the friction impulse on a 6.9 g body held
    by a ~1.5 mm-deep friction grasp) varies run to run; step-0 divergence is already 1e-4 m
    and a 40-step closed contact chain amplifies it into centimetres. Pinning them would be a
    flake generator, not a regression test. They are stored under `diag_*` and only checked
    for finiteness / non-ejection, and the observed deviation is printed for the record.

── why the rollout pins its inputs ─────────────────────────────────────────────────────────
The env's BUILD is itself dynamic: `_align()` settles the grasp and iterates wrist rotations
until the hang is level, breaking on a velocity threshold. Two builds of the same env level to
(measured) 0.98 deg and 0.89 deg, so `_snap_jq`, `ins`, `front_room` and `seat_qw` differ from
build to build and no absolute trace is reproducible across builds. The rollout therefore
OVERWRITES the six quantities the kinematic path reads — jq, wrist_tgt_p, wrist_tgt_q, ins,
front_room_ep, advanced — with values stored in the baseline. With those pinned, the arm drive
is a pure deterministic function of the action sequence and reproduces across builds, machines
and (the point of the exercise) across the two step() implementations.

Regenerate the baseline after an INTENDED change to the kinematic path:
    .venv/bin/python tests/test_golden_kinematic.py --regen

HEADS-UP for whoever commits this: `.gitignore:34` is a blanket `*.npz`, so
tests/golden_kinematic.npz (41 KB) is IGNORED and the test degrades to a loud "missing
baseline" for everyone else. It needs either `git add -f tests/golden_kinematic.npz` or a
`!tests/golden_kinematic.npz` negation line. Neither is this file's to make.
"""

from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RL = os.path.join(_ROOT, "rl")
for _p in (_ROOT, _RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# rl/ is not a package and the env needs warp + a CUDA device; keep the no-GPU gate green.
newton = pytest.importorskip("newton", reason="newton/warp/CUDA required for the cable env")
rigid_cable_env = pytest.importorskip("rigid_cable_env", reason="rl/ env needs the GPU stack")
servo_cable_env = pytest.importorskip("servo_cable_env", reason="rl/ env needs the GPU stack")

RigidCableVecEnv = rigid_cable_env.RigidCableVecEnv
ServoCableVecEnv = servo_cable_env.ServoCableVecEnv
SUBSTEPS = rigid_cable_env.SUBSTEPS

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_kinematic.npz")

N_ENVS = 2            # the env builds a full VBD scene per arm (~25 s at n=2); 2 is enough to
#                       catch any per-env indexing bug in the (n, 6) arm slicing.
STEPS = 55            # long enough for the standoff clamp to saturate (asserted below)
RESET_SEED = 7        # env._rng is re-seeded before every reset so the jack placement repeats
ACTION_SEED = 4242
ADVANCE_BIAS = 0.8    # fraction of MAX_DPOS commanded along `ins` per step. Sized so the
#                       commanded advance (STEPS * ADVANCE_BIAS * MAX_DPOS = 88 mm) OVERRUNS
#                       the measured standoff room (~53 mm) and the clamp has to bind. That
#                       rams the plug well past the seat once it docks — deliberately: this
#                       test pins the ARM DRIVE, and the drive must stay exact in the regime
#                       where the env's one piece of control logic is actually doing work.
#                       Behaviour/seating quality is the Gate runner's job, not this file's.

# The kinematic channels, compared exactly. `stream` is the whole per-substep joint vector fed
# to eval_fk — arm AND gripper AND jack AND cable free joints — so a change that leaks into a
# non-arm coordinate is caught too.
EXACT_KEYS = ("stream", "wrist", "wtgt", "advanced", "jq")
DIAG_KEYS = ("diag_face", "diag_rew", "diag_success")

# The identity-bank equivalence MEASURES bit-exact (0.0 on every channel, see the test's own
# print). The tolerance is not 0 only because the plant's frame-end state is the last emitted
# sample, `q + 1.0 * (goal - q)`, which is not guaranteed by IEEE to equal the parent's
# `self.jq = goal` for every joint value, and that seed feeds the next IK. 1e-12 rad / m is
# ~4 orders above that ulp and ~9 below anything physical (MAX_DPOS is 2e-3 m).
EQUIV_ATOL = 1e-12


# ── rollout harness ───────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _capture_substep_stream(sink):
    """Record every joint vector handed to the KINEMATIC eval_fk, i.e. the arm drive itself.

    `_sim` / `_sim_from_stream` are the only callers that pass `body_flag_filter`; ArmIK's own
    FK probes call eval_fk positionally, so the kwarg cleanly separates the drive from the IK.
    Patching the attribute on the `newton` module works because both env modules do
    `import newton` and resolve `newton.eval_fk` at call time.
    """
    real = newton.eval_fk

    def spy(*args, **kw):
        if "body_flag_filter" in kw:
            sink.append(args[1].numpy().copy())
        return real(*args, **kw)

    newton.eval_fk = spy
    try:
        yield
    finally:
        newton.eval_fk = real


def _pin(env, pins):
    """Overwrite everything the kinematic path reads with the baseline's values.

    The arm jumps from the snapshot pose to the pinned pose over the first frame's substeps
    (build-to-build spread is ~1e-3 rad, i.e. sub-millimetre at the pads), which perturbs the
    dynamic channels only — the asserted channels are pure functions of what is pinned here.
    """
    env.jq = pins["jq0"].copy()
    env.wrist_tgt_p = pins["wrist_tgt_p0"].copy()
    env.wrist_tgt_q = pins["wrist_tgt_q0"].copy()
    env.ins = pins["ins"].copy()
    env.front_room_ep = float(pins["front_room_ep"])
    env.advanced = pins["advanced0"].copy()


def _rollout(env, step_fn, acts, pins):
    """Reset -> pin -> run `acts` through `step_fn`, capturing the kinematic + diagnostic trace."""
    env._rng = np.random.default_rng(RESET_SEED)   # reset() draws the jack placement from it
    env.reset()
    _pin(env, pins)
    if hasattr(env, "sync_bank_to_jq"):
        env.sync_bank_to_jq()   # the plant must start from the PINNED arm pose, not reset()'s
    sink = []
    wrist, wtgt, adv, jq, face, rew, succ = [], [], [], [], [], [], []
    with _capture_substep_stream(sink):
        for k in range(len(acts)):
            _obs, r, _d, s, _dep = step_fn(env, acts[k])
            bqn = env.state_0.body_q.numpy()
            f, fq = env._face_pose(bqn)
            wrist.append(np.concatenate([bqn[env.wrist_body, :3],
                                         bqn[env.wrist_body, 3:7]], axis=1))
            wtgt.append(np.concatenate([env.wrist_tgt_p, env.wrist_tgt_q], axis=1))
            adv.append(env.advanced.copy())
            jq.append(env.jq.copy())
            face.append(np.concatenate([f, fq], axis=1))
            rew.append(r.cpu().numpy())
            succ.append(s.cpu().numpy())
    stream = np.asarray(sink)
    assert stream.shape[0] == len(acts) * SUBSTEPS, (
        f"expected {len(acts) * SUBSTEPS} kinematic eval_fk calls, saw {stream.shape[0]} — "
        f"the substep drive changed shape")
    return {
        "stream": stream.reshape(len(acts), SUBSTEPS, -1),
        "wrist": np.asarray(wrist), "wtgt": np.asarray(wtgt),
        "advanced": np.asarray(adv), "jq": np.asarray(jq),
        "diag_face": np.asarray(face), "diag_rew": np.asarray(rew),
        "diag_success": np.asarray(succ),
    }


def _make_actions(ins, n):
    """Fixed OPEN-LOOP action sequence: advance along the insertion axis with lateral and
    rotational dither. Open-loop is load-bearing — a state-feedback teacher would make the
    action sequence depend on the (nondeterministic) contact solve and no channel would be
    comparable. The advance bias is sized so the standoff clamp saturates part-way through.
    """
    rng = np.random.default_rng(ACTION_SEED)
    acts = np.zeros((STEPS, n, 7))
    for k in range(STEPS):
        acts[k, :, 0:3] = np.clip(ADVANCE_BIAS * ins + 0.10 * rng.standard_normal((n, 3)),
                                  -1.0, 1.0)
        acts[k, :, 3:6] = 0.08 * rng.standard_normal((n, 3))
    return acts


def _pins_from(env):
    """Snapshot the six pinned inputs from a freshly reset env (baseline generation only)."""
    return {
        "jq0": env.jq.copy(),
        "wrist_tgt_p0": env.wrist_tgt_p.copy(),
        "wrist_tgt_q0": env.wrist_tgt_q.copy(),
        "ins": env.ins.copy(),
        "front_room_ep": np.float64(env.front_room_ep),
        "advanced0": env.advanced.copy(),
    }


# ── fixtures ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def baseline():
    if not os.path.exists(BASELINE):
        pytest.fail(
            f"missing golden baseline {BASELINE} — regenerate with "
            f"`.venv/bin/python tests/test_golden_kinematic.py --regen`. If it exists on the "
            f"author's box but not here, it was swallowed by `.gitignore:34 *.npz` (see the "
            f"module docstring); the file must be force-added or the pattern negated.")
    with np.load(BASELINE) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def parent_env():
    return RigidCableVecEnv(N_ENVS, seed=0)


@pytest.fixture(scope="module")
def servo_env():
    return ServoCableVecEnv(N_ENVS, seed=0,
                            bank=servo_cable_env._IdealBank(N_ENVS,
                                                            frame_dt=servo_cable_env.FRAME_DT))


# ── guard 1: parent source drift ──────────────────────────────────────────────────────────
def test_parent_source_unchanged():
    cur = servo_cable_env.parent_source_sha256()
    stale = {k: (v, cur[k]) for k, v in servo_cable_env.PARENT_SOURCE_SHA256.items()
             if cur[k] != v}
    assert not stale, (
        "RigidCableVecEnv source changed under rl/servo_cable_env.py — RE-SYNC THE SUBCLASS "
        "COPY: diff the parent method, port the change into ServoCableVecEnv.step / "
        "_sim_from_stream, re-run the equivalence test, THEN update PARENT_SOURCE_SHA256. "
        f"Stale (recorded -> current): {stale}")


def test_the_drift_guard_has_teeth(monkeypatch):
    """Negative control: the guard must read the LIVE parent source, not a frozen copy."""
    original = RigidCableVecEnv.step

    def step(self, action):        # same behaviour, different source text
        return original(self, action)

    monkeypatch.setattr(RigidCableVecEnv, "step", step)
    assert (servo_cable_env.parent_source_sha256()["step"]
            != servo_cable_env.PARENT_SOURCE_SHA256["step"])


# ── _IdealBank contract sanity (no GPU work) ──────────────────────────────────────────────
def test_ideal_bank_reproduces_the_parent_interpolation():
    """The identity bank must equal `jq_prev + (s+1)/S * (goal - jq_prev)` BIT-for-bit."""
    rng = np.random.default_rng(0)
    q0 = rng.normal(size=(3, 6))
    goal = q0 + rng.normal(size=(3, 6)) * 0.05
    bank = servo_cable_env._IdealBank(3)
    bank.reset(q0)
    bank.set_goal(np.arange(3), goal)
    out = bank.advance(SUBSTEPS)
    for s in range(SUBSTEPS):
        want = q0 + ((s + 1) / SUBSTEPS) * (goal - q0)
        assert np.array_equal(out[s], want), f"substep {s} is not the parent's interpolation"
    assert np.array_equal(bank.jq, out[-1]), "bank.jq must be the last emitted sample"


def test_ideal_bank_fails_loudly():
    bank = servo_cable_env._IdealBank(2)
    with pytest.raises(RuntimeError):
        bank.advance(SUBSTEPS)                      # advance before reset
    with pytest.raises(ValueError):
        bank.reset(np.zeros((3, 6)))                # wrong n
    bank.reset(np.zeros((2, 6)))
    with pytest.raises(ValueError):
        bank.set_goal(np.arange(2), np.full((2, 6), np.nan))


def test_servo_env_rejects_a_bank_that_breaks_the_contract(monkeypatch):
    """Exercise the REAL validation in ServoCableVecEnv.__init__ without a 25 s scene build.

    The bank checks run before `self.bank = bank` and before any use of the parent's state, so
    stubbing the parent's __init__ reaches them and nothing else. (A hand-copied mirror of the
    checks would be one more thing that can drift — the whole disease this file treats.)
    """
    monkeypatch.setattr(RigidCableVecEnv, "__init__", lambda self, n, **kw: None)

    class NoAdvance:
        def reset(self, jq0): ...
        def set_goal(self, ids, q): ...

    class NoJq:
        def reset(self, jq0): ...
        def set_goal(self, ids, q): ...
        def advance(self, substeps): ...

    with pytest.raises(TypeError, match="advance"):
        ServoCableVecEnv(2, bank=NoAdvance())
    with pytest.raises(TypeError, match="jq"):
        ServoCableVecEnv(2, bank=NoJq())


# ── the golden regression ─────────────────────────────────────────────────────────────────
def test_golden_kinematic_path(parent_env, baseline):
    acts = baseline["acts"]
    got = _rollout(parent_env, RigidCableVecEnv.step, acts, baseline)
    for k in EXACT_KEYS:
        assert got[k].shape == baseline[k].shape, f"{k}: shape drift"
        if not np.array_equal(got[k], baseline[k]):
            d = np.abs(got[k] - baseline[k])
            pytest.fail(f"KINEMATIC PATH CHANGED in `{k}`: max |diff| = {d.max():.3e} at "
                        f"{np.unravel_index(int(d.argmax()), d.shape)}. This trace is a pure "
                        f"function of the pinned inputs and the action sequence, so this is a "
                        f"real behaviour change, not GPU noise. If intended, regenerate: "
                        f"`.venv/bin/python tests/test_golden_kinematic.py --regen`")
    # the standoff clamp must actually have been exercised (it is the one piece of env-side
    # control logic in step(), and the subclass moves the bank behind it)
    bound = float(baseline["front_room_ep"]) - rigid_cable_env.MARGIN
    assert got["advanced"].max() >= bound - 1e-12, (
        f"standoff clamp never saturated (max advanced {got['advanced'].max():.4f} vs bound "
        f"{bound:.4f}) — the golden rollout no longer covers the clamp branch")
    # dynamic channels: report, do not pin (see the module docstring)
    dev = {k: float(np.abs(got[k] - baseline[k]).max()) for k in DIAG_KEYS}
    print(f"\n[golden] dynamic-channel deviation vs baseline (NOT asserted): {dev}")
    face = got["diag_face"][..., :3]
    assert np.isfinite(face).all(), "plug face went non-finite"


# ── guard 2: identity-bank equivalence ────────────────────────────────────────────────────
def test_ideal_bank_equivalence(servo_env, baseline):
    """Parent step() vs subclass step() ON THE SAME ENV OBJECT, from the same pinned state.

    Same object on purpose: two separate builds settle to different snapshots (see the module
    docstring), so a cross-instance comparison could never be exact. Here the only difference
    between the two runs is which step() body executes.
    """
    acts = baseline["acts"]
    ref = _rollout(servo_env, RigidCableVecEnv.step, acts, baseline)
    got = _rollout(servo_env, ServoCableVecEnv.step, acts, baseline)
    worst = {}
    for k in EXACT_KEYS:
        d = float(np.abs(got[k] - ref[k]).max())
        worst[k] = d
        assert d <= EQUIV_ATOL, (
            f"ServoCableVecEnv(_IdealBank) diverges from the parent in `{k}`: max |diff| = "
            f"{d:.3e} > {EQUIV_ATOL:.0e}. The duplicated step() body is NOT faithful.")
    print(f"\n[equivalence] max |servo(_IdealBank) - parent| per channel: {worst}")
    # cross-check: the parent-step run on THIS instance must also match the stored baseline,
    # which is what proves the pinning really makes the trace build-independent.
    for k in EXACT_KEYS:
        assert np.array_equal(ref[k], baseline[k]), (
            f"parent step() on the servo instance does not reproduce the baseline in `{k}` — "
            f"the pinned inputs no longer determine the kinematic path")


class _LaggingBank(servo_cable_env._IdealBank):
    """A bank that reaches only `reach` of its goal each frame — a plausible plant, a WRONG one.

    reach=0.999 is the equivalence negative control (a hair off the parent); a small reach is
    a crude stand-in for the calibrated plant's real lag, used by the command-lead test.
    """

    def __init__(self, n, *, frame_dt=servo_cable_env.FRAME_DT, reach=0.999):
        super().__init__(n, frame_dt=frame_dt)
        self.reach = float(reach)

    def advance(self, substeps):
        goal = self._goal.copy()
        self._goal = self._q + self.reach * (goal - self._q)
        out = super().advance(substeps)
        self._goal = goal
        return out


def test_the_equivalence_check_has_teeth(servo_env, baseline):
    """Negative control: a 0.1%-lagging plant must break the comparison the ideal bank passes.

    Without this, an equivalence test that compared, say, an empty trace would pass forever.
    """
    acts = baseline["acts"]
    ideal = servo_env.bank
    try:
        ref = _rollout(servo_env, RigidCableVecEnv.step, acts, baseline)
        servo_env.bank = _LaggingBank(N_ENVS, frame_dt=servo_cable_env.FRAME_DT)
        got = _rollout(servo_env, ServoCableVecEnv.step, acts, baseline)
    finally:
        servo_env.bank = ideal
        servo_env.sync_bank_to_jq()
    dev = {k: float(np.abs(got[k] - ref[k]).max()) for k in EXACT_KEYS}
    print(f"\n[negative control] max |servo(_LaggingBank) - parent| per channel: {dev}")
    assert dev["stream"] > EQUIV_ATOL and dev["wrist"] > EQUIV_ATOL, (
        "a lagging plant produced the parent's arm drive — the equivalence test is not "
        f"actually comparing the drive ({dev})")


# ── Addendum 3: revolution branch offset ──────────────────────────────────────────────────
# The real ServoArmBank's limits, in radians. Reproduced here so the branch machinery is
# testable with NO dependency on newton_cabling.servo_arm existing.
_LIM_DEG = np.array([[-360.0, -135.0, -180.0, -360.0, -360.0, -360.0],
                     [360.0, 135.0, 180.0, 360.0, 360.0, 360.0]])
BRANCH_STEPS = 15     # short: this measures a pose identity, not a trajectory


class _LimitedIdealBank(servo_cable_env._IdealBank):
    """`_IdealBank` that refuses out-of-range joints exactly as the calibrated bank does."""

    def __init__(self, n, *, frame_dt=servo_cable_env.FRAME_DT):
        super().__init__(n, frame_dt=frame_dt)
        self._jnt_lo, self._jnt_hi = np.radians(_LIM_DEG[0]), np.radians(_LIM_DEG[1])

    def _range_check(self, q, name):
        bad = (q < self._jnt_lo) | (q > self._jnt_hi)
        if bad.any():
            rc = list(zip(*(x.tolist() for x in np.nonzero(bad)), strict=True))
            raise ValueError(f"{name} outside joint limits at (row, joint) {rc}")

    def reset(self, jq0):
        self._range_check(np.asarray(jq0, dtype=np.float64), "jq0")
        super().reset(jq0)

    def set_goal(self, env_ids, q_star):
        self._range_check(np.asarray(q_star, dtype=np.float64), "q_star")
        super().set_goal(env_ids, q_star)


_SWAPPED = ("bank", "_bank_limits", "_branch", "max_cmd_lead_m",
            "regoal_deadband_rad", "regoal_every")


@contextlib.contextmanager
def _swapped_bank(env, bank, *, max_cmd_lead_m=None, regoal_deadband_rad=0.0, regoal_every=1):
    """Install `bank` + knobs on a built env exactly as the constructor would, then put it back.

    Swapping beats constructing: a second ServoCableVecEnv costs ~25 s of scene build and adds
    nothing — the branch offset, the lead clamp and the re-goal suppression are all pure
    step()-side logic that reads these attributes and nothing else.
    """
    keep = {k: getattr(env, k) for k in _SWAPPED}
    env.bank = bank
    env._bank_limits = servo_cable_env.bank_joint_limits(bank)
    env.max_cmd_lead_m = max_cmd_lead_m
    env.regoal_deadband_rad = float(regoal_deadband_rad)
    env.regoal_every = int(regoal_every)
    try:
        yield env
    finally:
        for k, v in keep.items():
            setattr(env, k, v)
        env.sync_bank_to_jq()


def _shift_joint5(env, turns):
    """Relabel joint5 of every env by whole revolutions — pose-preserving, and exactly the
    production failure mode. Must be applied AFTER _pin, which rewrites env.jq wholesale."""
    env.jq[env.arm_qc[:, 5]] += turns * servo_cable_env.TWO_PI


def _wrist_trace(env, acts, pins, steps, shift_turns=0.0):
    env._rng = np.random.default_rng(RESET_SEED)
    env.reset()
    _pin(env, pins)
    if shift_turns:
        _shift_joint5(env, shift_turns)
    env.sync_bank_to_jq()
    branch0 = env._branch.copy()
    out = []
    for k in range(steps):
        env.step(acts[k])
        # Addendum 3: the offset is established at reset and is CONSTANT for the episode.
        assert np.array_equal(env._branch, branch0), f"branch offset drifted at step {k}"
        bqn = env.state_0.body_q.numpy()
        out.append(np.concatenate([bqn[env.wrist_body, :3], bqn[env.wrist_body, 3:7]], axis=1))
    return np.asarray(out)


def test_branch_offset_absorbs_a_full_revolution(servo_env, baseline):
    """(a) construction/reset survives an out-of-branch pose, (b) the DRIVEN POSE is unchanged.

    Without the offset this is the measured production failure: one slot settles at
    joint5 = -391.4 deg and the bank (correctly) refuses the whole batch.
    """
    acts, pins = baseline["acts"], baseline
    with _swapped_bank(servo_env, _LimitedIdealBank(N_ENVS)) as env:
        ref = _wrist_trace(env, acts, pins, BRANCH_STEPS)
        b_ref = env._branch.copy()
        # relabel joint5 one MORE turn out of the bank's range — same arm, same pose
        got = _wrist_trace(env, acts, pins, BRANCH_STEPS, shift_turns=-1.0)  # (a) must not raise
        b_got = env._branch.copy()
    # NOTE the baseline's own offset is generally NON-ZERO: the pinned pose comes from a real
    # build, and that build already put a slot's joint5 out past -360 deg. That is the
    # production failure this machinery exists for, reproduced without trying.
    turns_ref, turns_got = b_ref / servo_cable_env.TWO_PI, b_got / servo_cable_env.TWO_PI
    print(f"\n[branch] offsets in turns — baseline {turns_ref[:, 5].tolist()}, "
          f"after a -1 turn relabel {turns_got[:, 5].tolist()}")
    for t in (turns_ref, turns_got):
        assert np.array_equal(t, np.round(t)), f"offset is not whole turns: {t}"
    d = turns_got - turns_ref
    assert np.array_equal(d[:, 5], np.full(N_ENVS, -1.0)), (
        f"joint5's offset must absorb exactly the one turn added, got {d[:, 5]}")
    assert np.array_equal(np.delete(d, 5, axis=1), np.zeros((N_ENVS, 5))), (
        "no other joint's offset may move")
    # (b) pose identity. Position compares directly; the QUATERNION legitimately flips sign
    # (a 2*pi hinge shift negates it — same rotation, other cover), so compare the geodesic
    # angle instead of the raw components.
    per_step = np.abs(got[..., :3] - ref[..., :3]).max(axis=(1, 2))
    dot = np.abs(np.sum(got[..., 3:7] * ref[..., 3:7], axis=-1)).clip(0, 1)
    dang = np.degrees(2.0 * np.arccos(dot)).max()
    print(f"\n[branch] driven-pose deviation across a 2*pi relabel (um): "
          f"step0 {per_step[0] * 1e6:.4f}, step1 {per_step[1] * 1e6:.4f}, "
          f"final {per_step[-1] * 1e6:.4f}; rotation {dang * 3.6e6:.4f} micro-deg")
    # NOT bit-exact, and it cannot be: warp's `float` is float32, so every joint vector is
    # truncated to single precision before eval_fk, and -6.83 rad resolves ~8x coarser than
    # -0.55 rad. The IK then closes a loop on those float32 wrist poses (with a float32
    # finite-difference Jacobian), so the two branches diverge diffusively. Bounds below are
    # ~1 um on the FIRST frame — the pose identity itself — and a decade of headroom on the
    # 15-step accumulation. Both are orders below MAX_DPOS (2 mm) and the 0.66 mm seat slop.
    assert per_step[0] < 1e-6, f"one frame of branch offset moved the wrist {per_step[0]:.3e} m"
    assert per_step.max() < 1e-4, f"branch offset drifted the wrist {per_step.max():.3e} m"
    assert dang < 1e-4, f"branch offset rotated the wrist by {dang:.3e} deg"


def test_a_different_branch_goal_raises(servo_env, baseline):
    """(c) A goal a FURTHER turn out is a real full-turn command and must not be absorbed."""
    with _swapped_bank(servo_env, _LimitedIdealBank(N_ENVS)) as env:
        env._rng = np.random.default_rng(RESET_SEED)
        env.reset()
        _pin(env, baseline)
        _shift_joint5(env, -1.0)
        env.sync_bank_to_jq()                   # establishes offset = -1 turn on joint5
        _shift_joint5(env, -1.0)                # ...then ask for one MORE turn, mid-episode
        with pytest.raises(ValueError, match="DIFFERENT revolution branch"):
            env.step(baseline["acts"][0])


# ── Addendum 3: command-lead clamp ────────────────────────────────────────────────────────
def _max_lead(env, acts, pins, steps=30):
    env._rng = np.random.default_rng(RESET_SEED)
    env.reset()
    _pin(env, pins)
    env.sync_bank_to_jq()
    lead = []
    for k in range(steps):
        env.step(acts[k])
        wnow = env.state_0.body_q.numpy()[env.wrist_body, :3]
        lead.append(np.linalg.norm(env.wrist_tgt_p - wnow, axis=1).max())
    return float(np.max(lead))


def test_command_lead_clamp_engages(servo_env, baseline):
    """A heavily lagging plant runs the open command integrator away; the governor bounds it.

    reach=0.05 closes 5% of the remaining joint error per frame, so against a target moving
    ~1.6 mm/frame the lead settles near 1.6/0.05 = 32 mm — the same order as the 28.5 mm
    measured against the real calibrated plant, which is the point of the exercise.
    """
    acts, pins, cap = baseline["acts"], baseline, 0.005
    with _swapped_bank(servo_env, _LaggingBank(N_ENVS, reach=0.05)) as env:
        free = _max_lead(env, acts, pins)
    with _swapped_bank(servo_env, _LaggingBank(N_ENVS, reach=0.05),
                       max_cmd_lead_m=cap) as env:
        held = _max_lead(env, acts, pins)
    print(f"\n[cmd-lead] max |wrist_tgt - wrist|: unclamped {free * 1000:.2f} mm, "
          f"clamped {held * 1000:.2f} mm (cap {cap * 1000:.1f} mm)")
    assert free > cap * 2, (
        f"the lagging plant only led by {free * 1000:.2f} mm — this test no longer exercises "
        f"a runaway, so the clamp result below proves nothing")
    # the measured lead is taken AFTER the frame, so the plant has moved since the clamp;
    # it can only have moved TOWARD the target, hence <= cap up to solver noise.
    assert held <= cap + 1e-9, f"clamp did not hold: {held:.6f} m > {cap:.6f} m"


def test_command_lead_clamp_is_off_by_default(servo_env):
    assert servo_env.max_cmd_lead_m is None, (
        "default must stay None — strict parent parity is what makes the golden and "
        "equivalence guards meaningful")


# ── Addendum 4 R2: re-goal suppression ────────────────────────────────────────────────────
class _CountingBank(servo_cable_env._IdealBank):
    """`_IdealBank` that records which envs were re-goaled on each frame.

    set_goal is called at most once per step and always before advance, so draining the
    pending list in advance() buckets the sends by control frame.
    """

    def __init__(self, n, **kw):
        super().__init__(n, **kw)
        self.sent = []
        self._pending = []

    def set_goal(self, env_ids, q_star):
        super().set_goal(env_ids, q_star)
        self._pending = np.asarray(env_ids).tolist()

    def advance(self, substeps):
        self.sent.append(self._pending)
        self._pending = []
        return super().advance(substeps)


def _regoal_trace(env, acts, pins, steps, **knobs):
    """Per-frame (sent env-ids, IK goal in the bank frame) for a pinned rollout.

    The goals are captured by shadowing `_to_bank` on the INSTANCE — it is the single place
    every goal passes through on its way to the bank, and unlike the bank's own set_goal it
    also sees the goals that were suppressed, which is exactly what the deadband tests need.
    """
    bank = _CountingBank(N_ENVS)
    goals = []
    with _swapped_bank(env, bank, **knobs):
        env._rng = np.random.default_rng(RESET_SEED)
        env.reset()
        _pin(env, pins)
        env.sync_bank_to_jq()
        bank.sent.clear()            # drop the reset()'s own bookkeeping
        original = env._to_bank

        def spy(q_env, what, _o=original):
            out = _o(q_env, what)
            if what.startswith("IK"):
                goals.append(np.asarray(out).copy())
            return out

        env._to_bank = spy
        try:
            for k in range(steps):
                env.step(acts[k])
        finally:
            del env._to_bank         # drop the instance attr; the class method takes over again
    return bank.sent, goals


def _sends_per_frame(env, acts, pins, steps, **knobs):
    sent, _ = _regoal_trace(env, acts, pins, steps, **knobs)
    return np.array([len(s) for s in sent])


def test_regoal_suppression_is_off_by_default(servo_env, baseline):
    """At the defaults an env may be skipped ONLY when its goal is bit-identical to the last
    one sent — i.e. the R2 clauses are inert and the pre-R2 exact-inequality rule alone decides.

    NOTE the claim is not "nothing is ever skipped": the exact-inequality rule (pre-R2) does
    fire on its own, ~10% of frames late in this rollout, once the standoff clamp saturates
    and the IK settles on a bit-exact fixed point. Skipping a byte-for-byte identical goal is
    a no-op for the plant, which is why it was always safe.
    The end-to-end bit-identity is nailed down by test_ideal_bank_equivalence (which runs at
    these defaults and matches the parent to 0.0); this pins the mechanism behind it.
    """
    assert (servo_env.regoal_deadband_rad, servo_env.regoal_every) == (0.0, 1)
    sent, goals = _regoal_trace(servo_env, baseline["acts"], baseline, 40)
    last, skipped = {}, 0
    for frame, (ids, g) in enumerate(zip(sent, goals, strict=True)):
        for i in range(N_ENVS):
            if i in ids:
                last[i] = g[i].copy()
                continue
            skipped += 1
            assert i in last and np.array_equal(g[i], last[i]), (
                f"frame {frame} env {i} was suppressed at the DEFAULTS with a changed goal "
                f"(max delta {np.abs(g[i] - last.get(i, g[i])).max():.3e} rad) — the R2 "
                f"clauses are not inert")
    print(f"\n[regoal] defaults: {skipped} of {40 * N_ENVS} env-frames skipped, every one on a "
          f"bit-identical goal")


def test_regoal_deadband_suppresses_but_still_fires(servo_env, baseline):
    """(a) a deadband above the per-frame delta suppresses, (b) accumulated drift still fires."""
    acts, pins, steps = baseline["acts"], baseline, 40
    full = _sends_per_frame(servo_env, acts, pins, steps)
    dead = _sends_per_frame(servo_env, acts, pins, steps, regoal_deadband_rad=0.05)
    fired = np.flatnonzero(dead > 0)
    print(f"\n[regoal] sends: default {full.sum()}/{steps * N_ENVS}, deadband 0.05 rad "
          f"{dead.sum()}/{steps * N_ENVS}; frames that fired {fired.tolist()}")
    assert dead.sum() < full.sum(), "(a) the deadband suppressed nothing"
    # (b) the goal creeps ~0.01 rad/frame, so a 0.05 rad deadband must re-fire periodically —
    # this is the guard against the last-SENT-vs-last-candidate bug, which freezes forever
    # after frame 0.
    assert len(fired) > 1, (
        f"(b) only frame(s) {fired.tolist()} ever sent — accumulated drift never re-fired, "
        f"so the deadband is being measured against last frame's candidate, not the last "
        f"goal actually sent")
    assert fired.max() > steps // 2, "(b) sends died out in the second half of the rollout"


def test_regoal_every_forces_a_stale_goal_out(servo_env, baseline):
    """(c) with a deadband nothing can exceed, `regoal_every` still forces the send."""
    acts, pins, steps, k = baseline["acts"], baseline, 24, 3
    never = _sends_per_frame(servo_env, acts, pins, steps, regoal_deadband_rad=1e3)
    forced = _sends_per_frame(servo_env, acts, pins, steps, regoal_deadband_rad=1e3,
                              regoal_every=k)
    # Frame 0 ALWAYS sends and no deadband can stop it: reset() re-places the plant, so there
    # is no last-sent goal to measure a delta against and the plant needs its first command.
    assert never[0] == N_ENVS, "the first frame after a reset must always re-goal"
    assert never[1:].sum() == 0, (
        f"a 1000 rad deadband still sent {never[1:].sum()} goals after frame 0 — nothing is "
        f"being suppressed, so the forcing result below proves nothing")
    fired = np.flatnonzero(forced > 0)
    print(f"\n[regoal] regoal_every={k}: frames that fired {fired.tolist()}")
    assert forced[fired].min() == N_ENVS, "a forced frame must re-goal the whole batch"
    gaps = np.diff(np.concatenate([[-1], fired]))
    assert gaps.max() <= k, f"(c) went {gaps.max()} frames without a send, cap is {k}"


def test_regoal_deadband_accumulates_per_env_not_across_the_batch(servo_env, baseline):
    """A slow env must keep accumulating drift even while a NOISY neighbour fires every frame.

    This is the guard on `self._last_goal[changed] = q_star[changed]` being a PER-ROW write.
    The wholesale `self._last_goal = q_star.copy()` passes every other test in this file —
    n=2 with correlated motion fires both envs together, so the difference never shows — but
    it re-baselines the rows that were NOT sent to their current candidate. Any neighbour
    firing then silently erases a suppressed env's accumulated drift, and the
    drifting-goal freeze the deadband is supposed to avoid comes back in cross-env form
    (at n=64, one noisy env is enough to freeze the other 63).

    So: decouple the goals. Env 0 creeps below the deadband; every other env swings at full
    scale and fires every frame. Env 0 must still re-fire on its own accumulated drift.
    """
    steps, deadband = 30, 3.0e-4
    ins = baseline["ins"]
    acts = np.zeros((steps, N_ENVS, 7))
    acts[:, 0, 0:3] = 0.02 * ins[0]                       # env 0: slow creep, ~8.5e-5 rad/frame
    acts[:, 1:, 3:6] = np.array([0.0, 0.0, 1.0])          # everyone else: full-scale yaw,
    acts[1::2, 1:, 3:6] *= -1.0                           # alternating so the goal stays bounded
    sent, goals = _regoal_trace(servo_env, acts, baseline, steps,
                                regoal_deadband_rad=deadband)
    g = np.asarray(goals)                                 # (steps, n, 6) goals in the bank frame
    per_frame = np.abs(np.diff(g, axis=0)).max(axis=2)    # (steps-1, n)
    fired = {i: [f for f, ids in enumerate(sent) if i in ids] for i in range(N_ENVS)}
    print(f"\n[regoal] decoupled: env0 delta/frame {per_frame[:, 0].max():.2e} rad "
          f"(deadband {deadband:.1e}), env0 fired on {len(fired[0])}/{steps} frames {fired[0]}, "
          f"env1 on {len(fired[1])}/{steps}")
    # the premises: env 0 genuinely below the deadband every frame, neighbours genuinely above
    assert per_frame[:, 0].max() < deadband, (
        f"env 0 moves {per_frame[:, 0].max():.2e} rad/frame, at or above the {deadband:.1e} "
        f"deadband — it is not being per-frame suppressed, so this test proves nothing")
    assert per_frame[:, 1:].min() > deadband, "the neighbour envs are not firing every frame"
    for i in range(1, N_ENVS):
        assert fired[i] == list(range(steps)), f"env {i} must fire on every frame, got {fired[i]}"
    assert len(fired[0]) < steps, "env 0 was never suppressed; the deadband did nothing"
    # THE KILL: under the wholesale-copy mutant env 0 fires at frame 0 and never again.
    later = [f for f in fired[0] if f > 0]
    assert len(later) >= 2, (
        f"env 0 fired only at {fired[0]} — its accumulated drift never re-fired while a noisy "
        f"neighbour was sending every frame. That is the signature of `_last_goal` being "
        f"written WHOLESALE instead of per-row: the neighbour's send re-baselines env 0's "
        f"stored goal, erasing the drift the deadband is meant to accumulate.")
    gaps = np.diff([0, *later])
    assert gaps.max() <= deadband / per_frame[:, 0].min() + 2, (
        f"env 0's re-fire gaps {gaps.tolist()} exceed what its drift rate implies")


def test_regoal_knobs_reject_nonsense(monkeypatch):
    monkeypatch.setattr(RigidCableVecEnv, "__init__", lambda self, n, **kw: None)
    bank = servo_cable_env._IdealBank(2)
    with pytest.raises(ValueError, match="regoal_deadband_rad"):
        ServoCableVecEnv(2, bank=bank, regoal_deadband_rad=-1e-6)
    with pytest.raises(ValueError, match="regoal_every"):
        ServoCableVecEnv(2, bank=bank, regoal_every=0)
    with pytest.raises(ValueError, match="regoal_every"):
        ServoCableVecEnv(2, bank=bank, regoal_every=1.5)


# ── baseline generation ───────────────────────────────────────────────────────────────────
def _regen():
    env = RigidCableVecEnv(N_ENVS, seed=0)
    env._rng = np.random.default_rng(RESET_SEED)
    env.reset()
    pins = _pins_from(env)
    pins["acts"] = _make_actions(pins["ins"], N_ENVS)
    print(f"[regen] front_room_ep = {float(pins['front_room_ep']):.4f} m, "
          f"clamp bound = {float(pins['front_room_ep']) - rigid_cable_env.MARGIN:.4f} m")
    out = _rollout(env, RigidCableVecEnv.step, pins["acts"], pins)
    bound = float(pins["front_room_ep"]) - rigid_cable_env.MARGIN
    hit = "SATURATED" if out["advanced"].max() >= bound - 1e-12 else "NOT reached"
    print(f"[regen] advanced: max {out['advanced'].max():.4f} m (clamp {hit})")
    np.savez_compressed(BASELINE, **pins, **out)
    print(f"[regen] wrote {BASELINE} ({os.path.getsize(BASELINE) / 1e6:.2f} MB)")


if __name__ == "__main__":
    if "--regen" not in sys.argv:
        raise SystemExit("usage: python tests/test_golden_kinematic.py --regen")
    _regen()
