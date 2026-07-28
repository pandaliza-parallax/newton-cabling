# Cable-Insertion PPO: direct gripper-motion policy

**Status (2026-07-21): working.** PPO trained end-to-end to predict **gripper (end-effector)
motion directly** — no scripted base controller in the loop — seats a dangling-cable RJ45
into a panel jack at **94.4% held-success at the hardest curriculum stage** (30 mm approach,
8 mm jack offsets, 0–8° cable droop). Checkpoint: `runs/cable_v3/best_model.pt`.

This documents what changed relative to the earlier arm PPO (`train_arm_ppo.py`, residual on
a scripted base; see `CAD_RJ45_PPO.md` for the free-plug history) and how to reproduce it.

---

## Task

The StandardBots RO1 grips the **flexible ethernet cable** ~5 cm behind the connector; the
RJ45 connector dangles at the end of a strain-relief boot. The task is a horizontal insertion
into a world-level panel jack (face perpendicular to the ground). Episodes can start with the
hanging connector **drooping up to 8° below horizontal** — the policy must rotate the wrist
to level it before it can seat (all orientation authority is the robot's: the gripped cable
segments are rigidly anchored to the wrist, so the cable moves *only* with the hand).

Env: `cable_env.py::CableInsertVecEnv` (batched, GPU, Newton/SolverVBD).

## Action space — the headline change

The policy output **is** the gripper motion, executed verbatim (pi05 `rj45_sbot` EEF-delta
format, 60 Hz):

```
action ∈ [-1,1]^7 = [dpos(3), drotvec(3), gripper(1)]
```

| dims | meaning | scale / step |
|------|---------|--------------|
| 0–2  | world-frame wrist translation delta | × `MAX_DPOS` = 2 mm |
| 3–5  | world-frame wrist rotation delta (rotvec) | × `MAX_DROT` = 0.75° |
| 6    | gripper (pi05 compat; grasp currently fixed) | — |

The six arm joints follow via batched DLS IK (`arm_ik.py`, 2 iters/step); the arm is
kinematic (posed by `eval_fk` each substep). The **only** env-side modification of the action
is the standoff clamp — the forward component is capped so the finger front never crosses the
jack-mouth plane minus 3 mm (a physical constraint, not a controller; retreat is free).
Finger↔jack contacts cannot generate a physics response (both kinematic) so they are counted
per step as **violations** → reward penalty + logged.

Observation (22-D, pi05 state layout + privileged sim terms the VLA will replace with
images): `[eef_pos(3), eef_rot6d(6), gripper(1), face→seat error ×50 (3), face lin vel(3),
rot error to seat frame ×3 (3), face ang vel(3)]`.

Reward: dense progress on a weighted seat distance (`W_PROG=100`), `R_SUCCESS=30` while
seated (velocity-damped), violation penalty (`W_VIOL=2`). Success = seated within
5 mm depth / 3 mm lateral / 8° for **20 consecutive steps** (hold-based; position-only was
gameable by a twisted plug).

## Curriculum — tilt is a difficulty axis

`CURRICULUM[stage] = (approach, jack lateral offset)` from (10 mm, 0) to (30 mm, 8 mm), and
**cable droop scales with the stage**: `tilt_stage = stage/(num_stages−1) × tilt`, per-env
uniform in `[0, tilt]` (DR across the batch; baked into the settled snapshot, so
`set_stage()` re-runs the wrist-pitch align to the scaled droop and re-snapshots — jacks
parked during the re-align, finger standoff re-measured per snapshot).

## Training recipe (`train_cable_ppo.py`)

CleanRL-style GPU PPO (`train_ppo.py::ActorCritic`, 2×256 tanh, state-independent log_std),
plus three additions that turned out to be *necessary*, each earned by a failed run:

1. **BC warm-start** from the scripted funnel-servo teacher (`env.servo_action()` — exists
   only for demos/BC/eval, never in the training loop). 15 rollouts across all stages,
   50 epochs, MSE → ~3e-4. The teacher itself is 100%/0-violations at every stage.
2. **Critic-only warmup** (`--vf-warmup 8`): first iterations train only the value net, so
   garbage advantages from a fresh critic can't wreck the BC mean (v1 measured KL 0.13 on
   the first update — the BC policy was destroyed immediately).
3. **Scheduled exploration-noise anneal** (`--logstd-final −2.25` over 80% of the run):
   PPO never shrinks log_std on its own (entropy stayed pinned at init for 100+ iterations),
   and σ=0.29 random-walks ~2.6 mm over the 20-step hold vs the 3 mm tolerance — holds break
   stochastically no matter how good the mean is. The anneal overrides `log_std.data` each
   iteration on a linear schedule.

### The three-run arc (why each piece exists)

| run | config | result | lesson |
|-----|--------|--------|--------|
| cable_v1 | full tilt DR from iter 0, BC 6 rollouts, log_std −1.0 | **0% held, entire run**; eval depth shrank 23→10 mm | tilting swings the grip ~19 mm back (0.192 m wrist→grip lever), so "10 mm near-seated" stage 0 was top-stage difficulty → bootstrap destroyed. BC underfit (~25% relative action error). |
| cable_v2 | tilt curriculum + BC 15×50 (mse 3e-4) + critic warmup, log_std −1.25 fixed | learned 0→43% eval at stage 0, then **plateaued ~30%** | entropy pinned at init ⇒ hold-breaking noise floor. Deterministic eval > noisy training held, but promotion (rolling held ≥ 0.55) unreachable. |
| cable_v3 | resume v2 best + **log_std anneal −1.25→−2.25** | 50→65→83% stage 0-1, promoted through **all 5 stages** by ~it 140, **90–94% at stage 4**; violations 30%→1–2% | scheduled noise anneal was the missing piece. 1.84 M env steps, ~45 min at N=64 (~500–700 env-steps/s). |

The trained policy seats a 49 mm-out, 7°-drooping start in ~40 steps — ~2× faster than the
scripted teacher.

## Reproduce

```bash
# train from scratch (BC + critic warmup + anneal are defaults)
.venv/bin/python rl/train_cable_ppo.py --envs 64 --iters 400 --rollout 96 --tilt 8 \
    --run-name cable_v4

# resume / continue a checkpoint
.venv/bin/python rl/train_cable_ppo.py --envs 64 --iters 300 --tilt 8 \
    --bc-iters 0 --vf-warmup 0 --init-from rl/runs/cable_v3/best_model.pt --run-name cable_v5

# record a policy rollout (keep --steps ≤ ~130, see caveats)
.venv/bin/python rl/record_cable_env.py --checkpoint rl/runs/cable_v3/best_model.pt \
    --envs 1 --stage 4 --cable-tilt 8 --steps 130 --seed 1 --out cable_ppo_v3
uvx --from rerun-sdk rerun cable_ppo_v3.rrd cable_ppo_v3.rbl
```

Logs: `runs/<name>/log.csv` (`held`, `seated_inst`, `viol_frac`, KL, entropy, steps/s).
Checkpoints: `best_model.pt` (by eval held-success), `final_model.pt`.

## Caveats / next

- **Horizon**: trained on 96-step rollouts. Some seeds eject violently past ~t≈150 when
  rolled much longer (measured 78–125° twist at t=160–200). For datagen, **cut episodes at
  success + hold (≤130 steps)**, or train a v4 with a longer `--rollout` if long steady
  holds are needed in the data.
- **Kinematic arm**: no joint dynamics/torque limits — force realism is untested (same
  caveat as the rigid-plug pipeline); fine for image-based VLA trajectory generation.
- **Gripper dim inert**: the grasp is a fixed friction/anchor grip; the 7th action dim is
  format-compat only.
- **VBD nondeterminism**: single-seed recordings vary; validate any claim at N ≥ 4 (the
  94.4% figure is a 64-env deterministic eval).
- **Stage E (next)**: batch policy rollouts → the existing `tools/render_batch*` datagen
  pipeline → pi05 episodes (action/state format already matches `rj45_sbot`).
