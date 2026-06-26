# cad_rj45 — asset collection & PPO runner notes

Everything you need to train PPO on the **clean idealized RJ45** insertion task. The legacy
`rj45` (Newton's bundled toy) asset is still there and untouched; this doc is about `cad_rj45`.

---

## 1. Asset collection

| File | What it is |
|---|---|
| `newton_cabling/assets/cad_rj45.usd` | The shipped asset. Prims `/World/Socket` (panel jack), `/World/Plug` (8P8C body+boot+cable), `/World/Latch` (slanted spring-clip on a revolute hinge). |
| `tools/cad_assets/build_cad_rj45_clean.py` | **The builder.** Clean parametric primitives sized to the real McMaster parts (9953K216 plug, 1422N17 jack), booleaned (manifold3d) into watertight solids, authored with **unmerged verts** (flat/matte shading). Edit geometry here → rebuild. |
| `newton_cabling/connector.py` → `cad_rj45_connector()` | The `ConnectorSpec` (contact + latch params). Friction is a kwarg: `cad_rj45_connector(friction=...)`. |
| `newton_cabling/sim/scene.py` | `load_connector_meshes` + `add_connector_rig` build the rig (world-d6 plug, revolute latch, SDF contact) from the spec. |
| `tools/cad_assets/ASSETS.md` | Full history / rationale of the asset (STEP-tessellation attempts → clean idealized). |
| `cad_clean.rrd` + `rl/record_cad_latch.py` | A clean **kinematic** demo recording (plug drives straight in, latch flips). Not physics — just for showing the asset. |

**Rebuild after editing the builder:**
```bash
.venv/bin/python tools/cad_assets/build_cad_rj45_clean.py     # -> cad_rj45.usd
.venv/bin/python rl/smoke_asset.py cad_rj45                    # sanity: loads, seats, no NaN
```

The latch is a **separate articulated body** but it's effectively passive (the physical click
is rigid-VBD-capped; see ASSETS.md). RL success is depth/lateral/angle based and does **not**
read the latch — so the latch is neutral for training (its flip in `cad_clean.rrd` is scripted).

---

## 2. cad_rj45 physical params (current)

From `cad_rj45_connector()` (`newton_cabling/connector.py`):

| Param | Value | Notes for RL |
|---|---|---|
| contact `stiffness` | **1e6** | Bumped from 1e5 this session (for a non-issue). **Recommend 1e5 for PPO** — 10× stiffer contacts raise jitter/NaN risk under aggressive exploration. |
| contact `friction` (μ) | **2.0** | High. Adds real axial resistance; base-controller seating ~44%. **Recommend 0.5–1.0 (or curriculum low→high) for training**; μ=2 is a stress value. Applies to socket+plug+latch. |
| contact `gap_meters` | 1e-4 (0.1 mm) | Tight (real RJ45 clearance). SDF narrow band = ±2·gap = ±0.2 mm → deep penetration from big actions can **tunnel** (PPO may exploit it). Mitigate with an action-magnitude cap or a slightly larger gap. |
| `sdf_max_resolution` | 512 | Fine; OK. |
| `density` | 1e6 | Plug is effectively rigid. |
| latch spring | ke=0.15, kd=0.03, travel ±0.6 rad | Soft return spring; passive. |

Plug body ~11.68 × 8 mm, cavity floor +12 mm past the mouth, plug↔cavity clearance ~0.4 mm/side.

---

## 3. The RL env — `ConnectorVecEnv` (`rl/connector_env.py`)

Construct: `ConnectorVecEnv(n, seed, random_easy, asset="cad_rj45")`. N connector worlds
(socket+plug+latch, **no cable/arm**) in one batched Newton model.

- **Obs** (`OBS_DIM = 9`): `pos_err(3)`, `lin_vel(3)`, `orient_err_rotvec(3)`.
- **Action**: `ACT_DIM = 3` (translation-only plug) **or 6** with `random_easy=True` (pos + orientation; the plug rides a *driven* d6 angular axis). Action = a residual nudge to the base controller's target (`RESIDUAL_SCALE = 0.15`).
- **Plug drive**: position spring (`SPRING_KE=50, SPRING_KD=10`) toward the seat target; `random_easy` adds a strong angular drive (`ANGULAR_KE=10, ANGULAR_KD=6`) that auto-aligns the plug.
- **Reward** (`write_reward`): potential-based progress toward the seated pose (lateral-weighted distance + rotation term) + a **hold-to-success** bonus (`r_success` minus a velocity penalty) once seated. `success` = **sustained** seat (held `hold_steps`), not a transient touch.
- **Seated test** (the success/eval metric): `seat_gap ≤ seat_depth_tol` AND `lateral offset ≤ seat_offset` AND `ang_err ≤ seat_angle`.

**`cad_rj45` ASSET_PROFILE** (the geometry/reward constants for this asset):
```
plug_y=(0,0,0)        seat_aim_dy=0.012   box=0.05   rigid_gap=0.0001
re_lat=0.004   re_approach_min=0.010   re_approach_max=0.030   re_rot=RE_ROT
seat_depth_tol=0.005   seat_offset=0.003   seat_angle=SEAT_ANGLE
```
(`rigid_gap=0.0001` is the builder's rigid-contact gap — sub-mm to match the tight cavity; the
5 mm `rj45` value would hold the plug off the walls and block insertion.)

**Curriculum** (start-pose difficulty, advanced automatically by the trainer):
- translation mode: `CURRICULUM = [0.5, 1.5, 3, 5, 8, 11, 15] mm` lateral offset.
- `random_easy` mode: `RE_CURRICULUM = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]` scale on lateral + ≤20° rotation.

`SUBSTEPS=4`, `Z_LIFT=0.35 m` (worlds lifted off the origin), `SPACING=0.4 m` between worlds, `max_steps=200`.

---

## 4. Running PPO — `rl/train_ppo.py`

CleanRL-style PPO, fully on-GPU against the batched env. Needs a CUDA GPU + the `sim` extra.

```bash
# baseline cad_rj45 run (translation-only, auto-curriculum from stage 0)
uv run --extra sim python rl/train_ppo.py --asset cad_rj45 --envs 1000 --iters 400 --run-name cad_ppo

# 6-DOF (pos+orientation) with the random_easy start distribution
uv run --extra sim python rl/train_ppo.py --asset cad_rj45 --random-easy --envs 2000 --iters 400 --run-name cad_re

# warm-start from a checkpoint
uv run --extra sim python rl/train_ppo.py --asset cad_rj45 --init-from rl/runs/<name>/best_model.pt ...
```

Key args (defaults): `--envs 1000 --rollout 32 --epochs 4 --minibatches 8 --lr 3e-4
--gamma 0.99 --lam 0.95 --clip 0.2 --ent-coef 0.005 --vf-coef 0.5 --max-grad 0.5
--target-kl 0.02 --contact-buffer 64 --seed 0`.

**Curriculum auto-advance:** stage advances when window-mean depth ≥ `--advance-depth` (18 mm)
for `--advance-window` (10) iters, or seated ≥ `--advance-seated` (0.12). `--no-curriculum`
pins the hardest stage.

**Outputs:** `runs/<name>/log.csv` (iter, stage, env_steps, mean_rew, success_rate, mean_depth_mm),
`best_model.pt` (by success), `final_model.pt`. Eval every `--eval-every` (50) iters via a
deterministic mean-action rollout (the honest steady-state metric). Lots of prior runs already in
`rl/runs/` (e.g. `re_6dof_long`, `extend`, `curriculum`) — good warm-start sources.

**Record a trained policy** (physics, for viewing):
```bash
.venv/bin/python rl/record_policy.py --asset cad_rj45 --checkpoint rl/runs/<name>/best_model.pt --envs 9 --stage 5
```

---

## 5. Recommended settings before a serious PPO run

Decisions left from this session that are worth revisiting for training stability:

1. **Contact `stiffness` 1e6 → 1e5.** The 1e6 bump was for a rendering confusion (the
   "penetration" was a stale viewer, not real). 1e5 is the proven value and is gentler under
   aggressive exploration. One-line change in `cad_rj45_connector()`.
2. **`friction` 2.0 → ~0.5–1.0** for learning from scratch (or curriculum it low→high). μ≥1
   saturates the physical effect and just makes the task harder; μ=2 is a stress test.
3. **Tunneling guard.** With the 0.1 mm gap / ±0.2 mm band, watch for the policy clipping the
   plug through walls. Options: cap action magnitude, widen the gap slightly, or eyeball a
   `record_policy` rollout. The `--contact-buffer` (64) overflow warnings are a hint the plug is
   in heavy wall contact (expected at high friction) — bump it if you see drops.
4. Start with `--random-easy` only if you want orientation in the action space; the 3-DOF
   translation env is easier and a good first target.

(These are recommendations, not applied — the asset currently ships at stiffness=1e6, friction=2.0.)

---

## 6. Quick reference

```bash
.venv/bin/python tools/cad_assets/build_cad_rj45_clean.py     # rebuild asset
.venv/bin/python rl/smoke_asset.py cad_rj45                   # sanity check
uv run --extra sim python rl/train_ppo.py --asset cad_rj45 --envs 1000 --iters 400 --run-name cad_ppo
uvx --from rerun-sdk rerun cad_clean.rrd cad_clean.rbl        # view the asset demo
```

---

## 7. Run log

### `cad_re` — 2026-06-22 (6-DOF, same setup as the rj45 runs)
Command (mirrors the rj45 6-DOF run: random_easy + auto-curriculum + entropy decay):
```bash
uv run --extra sim python rl/train_ppo.py --asset cad_rj45 --random-easy \
  --envs 2000 --iters 600 --rollout 96 \
  --ent-coef 0.005 --ent-coef-final 0.0 --ent-anneal-frac 0.5 \
  --run-name cad_re
```
- **Stability:** PPO smoke (512 envs, 6 iters) ran clean — **no NaN/ejection** at the shipped
  params (stiffness 1e6, friction 2.0, gap 0.1 mm). The doc's §5 NaN concern didn't bite.
- **Asset note:** friction μ=2.0 makes this **harder than the frictionless rj45** task, so expect a
  lower plateau than the rj45 ladder (~74%).
- Throughput ~167 k steps/s (~1 s/iter); run ≈ 10–15 min. Outputs in `rl/runs/cad_re/`.
  Honest metric = the `[eval]` SR (deterministic mean-action), not the per-iter rollout SR.
- rj45 reference ladder (held): 0.6% base → 16% +curriculum → 55% +6-DOF → 63% +ent-decay → 74% @5k.

**Results** (eval = honest deterministic SR; "held" = *sustained* seat, 20 steps):

| run | μ | best eval SR | steady (stage 4) | depth | verdict |
|---|---|---|---|---|---|
| `cad_re` | 2.0 | 13.5% | ~13% | stuck 23 mm | friction caps even easy stages |
| `cad_re` extended to ~1230 effective iters | 2.0 | ~13% | ~13% | stuck 23 mm | **NOT undertraining — real plateau** |
| `cad_re_mu05` | 0.5 | **16.4%** | ~12% | stuck 22 mm | friction lifted *easy* stages (45% rollout), hard stages re-jam |

**Base controller** (zero residual = pure scripted insert, μ=0.5; `rl/base_eval.py`):

| stage | re_scale | base SR | depth |
|---|---|---|---|
| 0 (≈aligned) | 0.10 | **73.6%** | 31.6 mm |
| 1 | 0.20 | 50.7% | 29.0 mm |
| 2 | 0.35 | 22.9% | 25.2 mm |
| 3 | 0.50 | 10.2% | 23.3 mm |
| 4 | 0.70 | 3.6% | 21.7 mm |
| 5 (full tilt+offset) | 1.00 | **1.4%** | 20.7 mm |

**Success criterion (cad_rj45):** `seat_gap ≤ 5 mm` of the 12 mm seat (i.e. plug **≥7 mm into the cavity**) AND lateral ≤ 3 mm AND angle ≤ `SEAT_ANGLE`, **held 20 consecutive steps**. (The log's `depth_mm` is *travel-from-start*, not absolute insertion depth.)

**Diagnosis — two caps, geometry dominates:**
1. **Friction** (μ=2.0): caps even aligned stages; lowering to 0.5 restores easy-stage seating (base 74% @ stage 0).
2. **Square-mouth cavity + tilt (DOMINANT):** [build_cad_rj45_clean.py:134](../tools/cad_assets/build_cad_rj45_clean.py#L134) carves the bore as a plain `box` — **no lead-in chamfer**, 0.4 mm clearance/side. A plug starting at ≤15–20° tilt swings its leading corner ≫0.4 mm, catches the square mouth edge, and jams ~10 mm short. The base controller collapses **74% → 1.4%** as tilt rises; PPO (bounded ±15% residual) lifts the hard stages ~3× (3.6% → ~12% @ stage 4) but **cannot overcome the geometric jam**.

**Highest-leverage fix:** add a **lead-in chamfer/flare** to the cavity mouth (real RJ45 jacks have one) so a tilted plug self-aligns into the bore — this raises the *base* controller, and PPO builds higher on top. Secondary: pace the curriculum on *seated* rather than `depth ≥ 18 mm` (it jumped to stage 4 by iter ~73, before mastering 0–3).

### `cad_chamfer` — 2026-06-22 (lead-in chamfer added — the fix applied)
Added a funnel chamfer at the cavity mouth (`CHAMFER_W=2.5 mm` flare, `CHAMFER_D=3.0 mm` deep, ~40° lead-in) in [build_cad_rj45_clean.py](../tools/cad_assets/build_cad_rj45_clean.py) (`build_jack`), rebuilt the asset, same PPO setup as `cad_re_mu05` (μ=0.5).

**Proof the chamfer fixes the diagnosed jam** (`rl/why_base_fails.py`, base controller, final-state instantaneous-seated):

| stage | base seated **before** | base seated **after** | failed-plug median gap before → after |
|---|---|---|---|
| 2 | 48.1% | **86.8%** | 12.0 mm → **0.4 mm** |
| 4 | 12.7% | **57.4%** | 12.0 mm → **3.1 mm** |
| 5 | 5.6% | **41.0%** | 12.1 mm → **5.7 mm** |

The "stuck at gap=12 mm (the mouth)" signature is gone — failed plugs are now *inside* the bore, and the dominant miss shifts depth → **angle** (plug enters + centers but settles ~4.5–4.9° over the 3° tolerance — fine-correction territory for PPO).

**Base controller held-SR (μ=0.5), before → after chamfer:**

| stage | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| before | 73.6 | 50.7 | 22.9 | 10.2 | 3.6 | 1.4 |
| **after** | **77.9** | **65.0** | **39.4** | **25.7** | **15.0** | **6.9** |

**PPO result:**

| run | asset | best eval SR | at stage | vs base |
|---|---|---|---|---|
| `cad_re_mu05` | square mouth | 16.4% | 4 (never cleared) | — |
| **`cad_chamfer`** | **chamfered** | **32.3%** | **5 (full distribution)** | base@5 6.9% → **~4.5×** |

The chamfer let the curriculum blow through to **stage 5 (the full start distribution) by iter ~56** and PPO climbed to **~32% on the hardest stage** — roughly **2× the non-chamfered 16.4%**, and on harder starts. Verdict: the ~13–16% plateau was a **geometry artifact (square un-chamfered mouth)**, not a PPO ceiling.

**Iteration scaling (chamfered, μ=0.5, stage 5, best eval SR):**

| iters | best eval SR | Δ | run |
|---|---|---|---|
| 600 | 32.3% | — | `cad_chamfer` |
| 2,000 | 38.4% | +6.1 | `cad_chamfer_2k` |
| 5,000 | **40.5%** | +2.1 | `cad_chamfer_5k` |

Most of the climb is iters 600→2000 (entropy annealing window); 2k→5k adds only +2 (the 40.5% came in the final post-anneal stretch). So 40% *looked* like the ceiling for this configuration — but §8 shows it wasn't (the cap was action authority + a measurement artifact). Chamfer checkpoint: `rl/runs/cad_chamfer_5k/best_model.pt`.

---

## 8. Action authority + the horizon artifact — the real result (2026-06-24)

The "~40% ceiling" (and the friction caps 26%/13%) turned out to be **two things**: a real action-authority limit and a measurement artifact. Fixing both, the policy actually **solves the insertion at ~95 / 89 / 83%** (μ=0.5/1.0/2.0).

### 8a. Action authority (RESIDUAL_SCALE) is the dominant lever
The policy is a *residual* on the scripted base controller: `total = clip(base + RESIDUAL_SCALE·policy, −1, 1)`. At the old 0.15 it could only fine-tune — it couldn't shove a wedged/cocked plug free. Raising it (new `--residual-scale` CLI):

| residual (μ=0.5, 2k, **old 80-step eval**) | SR |
|---|---|
| 0.15 (baseline) | 38.4% |
| 0.5 | 48.4% |
| 1.0 | 54.4% |
| 1.0 → 5k | 55.8% |

Monotonic, no saturation. (>1.0 only amplifies gain — the clip caps the range; the true authority limit is `MAX_DELTA` = ±2 mm/step.) **Authority rescues every friction**, residual 0.15→1.0 (2k, 80-step eval): μ=0.5 38→56, μ=1.0 26→49, μ=2.0 13→42 — the "friction cap" was largely an authority cap.

### 8b. The 80-step eval was a HORIZON ARTIFACT (understated by ~40 pts)
`evaluate_policy` averaged seated-occupancy over t40–80, but at stage 5 the plug is **still travelling in** during that window (occupancy 3%@t40 → 93%@t80 at μ=0.5; later at higher friction). It was measuring insertion-in-progress, not success. Re-evaluated over the **settled tail t120–200** (`rl/eval_horizon.py`):

| μ | 80-step eval | **TRUE steady-state** | holds (no drift)? |
|---|---|---|---|
| 0.5 | 54.5% | **94.8%** | yes (95% @ step 199) |
| 1.0 | 47.8% | **88.8%** | yes (90%) |
| 2.0 | 41.0% | **82.8%** | yes (84%) |

So the policy **solves insertion at ~95/89/83% across a 4× friction range**, and once seated it **holds** (steady ≈ ever-seated ≈ end-of-rollout). Friction costs real but mild performance. **Eval fixed**: `evaluate_policy` horizon 80→200, averages the settled tail — future SR reflects reality.

### 8c. What did NOT move it
Richer observations (**ang_vel**, **contact-force** wrench — both added + tested, §swept) = **neutral**; more iters/envs = diminishing; **ANGULAR_KD 6→12** = worse (over-damped). The two real levers were **geometry (the lead-in chamfer, §7) + action authority (residual)**.

### 8d. Tooling added this study
- **CLI knobs:** `--friction`, `--residual-scale`, `--angular-kd`, `--obs-contact` (train_ppo); `--residual-scale`, `--re-scale`, `--latch-anim` (record_policy).
- **Diagnostics:** `rl/eval_horizon.py` (settled-rate eval), `rl/why_policy_fails.py` & `rl/why_base_fails.py` (failure breakdown by depth/lateral/angle), `rl/base_eval.py`.
- **Best checkpoints:** `cad_res1p0_mu05_to5k` (μ=0.5, ~95%), `cad_res1p0_mu1_2k` (μ=1.0, ~89%), `cad_res1p0_mu2_2k` (μ=2.0, ~83%) — all residual=1.0.
- **Rollouts:** `cad_stage5_grid.rrd` (9-env), `cad_stage5_hard.rrd` (single env at 1.3× stage-5 misalignment — still seats).

### 8e. Open follow-ups
Curriculum paced on *seated* (it jumps to stage 5 by iter ~56); larger `MAX_DELTA` (genuine extra authority beyond residual=1.0); the ~5–17% hard-tilt core; teacher→student distill to a vision policy for sim-to-real.

