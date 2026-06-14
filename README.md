# newton-cabling

Evaluating the [Newton physics engine](https://github.com/newton-physics/newton) for a
robotics **datacenter-cabling** task — manipulating cables and inserting/extracting RJ45
connectors — and factoring the hard-won knowledge into a small, tested package.

MuJoCo and Genesis proved inadequate for the fine cable/connector detail this task needs.
Conclusion so far: **Newton handles the hard contact and deformable-cable physics out of the
box**, so building a custom simulator is not justified. The reusable building blocks live in
[`newton_cabling/`](newton_cabling); the `record_*.py` scripts are runnable demos.

> **Status / honesty note.** The arm demo currently drives the plug with a **force-spring
> coupling**, not a real finger grasp — it's a faithful *insertion/extraction* demo, not real
> manipulation. A true friction grasp (fingers physically holding the plug) is in progress; see
> [Friction grasp](#friction-grasp-in-progress). Where a demo is scripted rather than
> contact-driven (e.g. the latch press), it says so.

## Why Newton over Isaac Sim

We evaluated both. The task's defining difficulty is **fine deformable-cable behaviour and
sub-millimetre contact** at the plug/socket interface — exactly where MuJoCo and Genesis fell
short, and the axis on which the two NVIDIA-stack options most differ.

**What pushed us to Newton:**

- **Cable fidelity is the whole task.** Newton's VBD solver simulates 1D deformables (cables)
  with self-collision and proper bend/twist transport as a first-class, GPU-batched feature,
  two-way coupled to rigid bodies. Newton 1.0 ships this as a headline capability, and NVIDIA's
  own contact-rich manipulation demos (hose-connector insertion, GPU rack assembly) target this
  problem class. We confirmed it: latch click/hold/press-extract and cable bend/twist all work.
- **Code-first, GUI-optional.** Newton is a Python library on [Warp](https://github.com/NVIDIA/warp);
  the entire workflow here is a `.py` file run headless, edit-run-inspect in minutes. Isaac Sim
  is GUI/Omniverse-centric with a large install and USD-composition overhead.
- **Cost and iteration.** Headless Newton runs on one modest GPU; recordings are small `.rrd`
  files viewed locally in [rerun](https://rerun.io).
- **Open and portable.** Apache-2.0, Linux Foundation governance, and it plugs into Isaac Lab /
  Isaac Sim as a swappable physics backend — choosing Newton now doesn't lock us out later.

**What we gave up (the honest trade):** Isaac Sim's PhysX articulations are a decade more
battle-tested. Newton 1.0's rigid-robot coupling seam is rough, and we paid for it (~15 debug
iterations on undocumented VBD constraints — now encoded as asserts in
[`newton_cabling/sim/safe_vbd.py`](newton_cabling/sim/safe_vbd.py)). Isaac Sim also has
first-class robot/gripper assets and a built-in surface gripper.

**Net:** for a cable-manipulation task where deformable fidelity is make-or-break and fast
code-driven iteration matters, Newton was the right call — with eyes open about its 1.0-era
rigid-coupling roughness. The likely endgame is the hybrid that's emerging anyway: Newton as the
physics backend inside Isaac Lab for batched policy training.
([Newton 1.0 announcement](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/),
[contact-rich manipulation](https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/).)

## The `newton_cabling` package

Pure-logic modules import no GPU code and are linted, type-checked, and unit-tested anywhere.
The GPU-backed `sim/` modules live behind the `sim` extra and are verified by execution.

| Module | What it does |
|---|---|
| `timeline.py` | Declarative `CableTimeline`: one ordered list of `Phase` (plug offset + `GraspState`/`LatchState`) sampled at any time. `loop=True` validates start == end so a recording can't jitter on wrap. Replaces three hand-synced functions. |
| `report.py` | `evaluate_cycle()` — single source of truth for pass/fail, returning an outcome variant (`CycleSucceeded` / `InsertionFailed` / `LatchSlipped` / `ExtractionIncomplete`) the tuning harness pattern-matches on, with JSON round-trip. |
| `connector.py` | `ConnectorSpec` / `LatchSpec` / `ContactSpec` domain types with validation; `rj45_connector()` is the proven instance. Swapping a connector is a new spec value. |
| `cable.py` | `route_cable_from_boot()` — cable centreline routing with the modelling learnings baked in (exit the boot, ~10mm segments, gentle drape near rest, short kinematic prefix). |
| `sim/safe_vbd.py` | The footgun-free VBD setup: `new_vbd_builder` / `add_actuated_urdf` / `finalize_for_vbd` bake in `collapse_fixed_joints`, soft actuated joints, `eval_fk`-into-model, and the color/finalize order, with build-time asserts. |
| `sim/scene.py` | `load_connector_meshes` + `add_connector_rig` build the socket/plug/latch rig (world-d6 plug, latch revolute, SDF contact) from a `ConnectorSpec`; `patch_panel_port_offsets` for a real keystone panel. |
| `sim/grasp.py` | `GraspSpring` — force-coupling (anti-gravity + position spring) driven by a 0..1 weight. The interim grasp; being replaced by a real friction grasp. |
| `sim/recording.py` | `open_rrd_recorder` (keeps the `.rrd` file sink) and `auto_blueprint` (warns if a ground plane is present — see learnings). |

## Demos (runner scripts)

Each `record_*.py` builds a scene, runs headless, and writes a rerun `.rrd` (plus a focused
`.rbl` blueprint). **Recordings are not committed** — they're large and fully regenerable.
To produce and view one (needs a CUDA GPU and the `sim` extra — see [Running it](#running-it)):

```bash
uv run --extra sim python record_patch_panel.py     # writes patch_panel.rrd + patch_panel.rbl
uvx --from rerun-sdk rerun patch_panel.rrd patch_panel.rbl
```

(Substitute any `record_*.py`; each writes a matching `.rrd`/`.rbl` pair.)

- **`record_rj45_insert.py` / `record_rj45_unplug.py`** — the connector physics, driven by the
  proven spring rig: insertion with a latch that deflects over the socket lip and clicks; a
  pull-without-press control (latch holds), then press-and-extract.
- **`record_cable_twist.py`** — three cables of increasing bend stiffness with a spinning driven
  end; twist propagation through 90° bends.
- **`record_panda_cycle.py`** — a Franka FR3 running the full insert → release → retreat → return
  → press → extract loop (force-spring grasp), as a seamless 19s loop.
- **`record_patch_panel.py`** — a row of RJ45 ports at **real keystone pitch (18mm)**; cables
  start retracted and plug in side by side.
- **`record_param_sweep.py`** — N connector worlds in one model, one GPU pass, scored per-world by
  `evaluate_cycle`; sweeping press depth discovers the extraction threshold.
- **`record_grasp_test.py`** — milestone toward the real friction grasp (see below).

## Key learnings

### VBD + robot arm (the expensive ones)

- **FIXED joints freeze VBD chains.** A URDF with interleaved fixed joints does not move *at all*
  under SolverVBD. Import with `collapse_fixed_joints=True`.
- Actuated joints must be **soft** (`model.vbd.joint_is_hard[j] = 0`, post-finalize).
- Call `eval_fk(model, model.joint_q, model.joint_qd, model)` (into the model) **before**
  constructing SolverVBD — VBD measures joint angles against the model's rest pose.
- URDF `limit_kd≈10` is joint-freezing under VBD's Rayleigh convention; use `limit_kd≈1e-4`.
- d6 angular drives can hold the child rotated ~90°; a d6 with no angular axes locks rotation.
- Seed IK from a non-singular home configuration, not the URDF zero pose.

All of the above are encoded as defaults/asserts in `newton_cabling/sim/safe_vbd.py`.

### Reliable recordings: no ground plane

Recordings kept opening to **"just the floor."** The viewer **instances** identical shapes, so the
ground plane's rendered index can't be matched to a builder or model index to exclude it from the
blueprint. The fix that works: **don't add a ground plane in anything that records** — nothing here
needs one (arms are position-controlled, plugs ride springs, cables are pinned at both ends).
`auto_blueprint()` warns if a plane is present.

### Cable into a connector

Match Newton's RJ45 example: exit the **back of the plug (the boot)**, ~10mm segments, a short
kinematic prefix at the boot (threading it *inside* the plug body shoves the plug), and start near
the gravity rest shape with extra bend damping so a hanging cable doesn't swing.

## Friction grasp (in progress)

The arm grasp is currently a **force-spring**, which has known limits: it can't take cable-vs-arm
contact (the cable pushes the position-controlled gripper off the plug). The example we adapted
the arm from (`example_robot_panda_hydro`) grasps with **real finger friction**; Newton's
`example_cloth_franka` shows the workable pattern — a **kinematically driven arm** (so contact
can't shake it) co-simulated with a VBD deformable, gripping via finger contact.

`record_grasp_test.py` validates the first milestone: kinematic fingers holding a free VBD rigid
body by contact friction (no spring). Remaining milestones: conclusive grip + lift, insertion
while gripping (the grip must resist the insertion force — the genuinely hard part of robotic
connector insertion), a fingertip-driven latch press, and the cable draping on the now-stable arm.

## Running it

Requires [uv](https://docs.astral.sh/uv/). The pure-logic gate runs anywhere:

```bash
uv run ruff format . && uv run ruff check .
uv run ty check newton_cabling/__init__.py newton_cabling/timeline.py \
    newton_cabling/report.py newton_cabling/connector.py newton_cabling/cable.py tests
uv run pytest
```

The `record_*.py` demos need a **CUDA GPU** and the `sim` extra, which installs Newton from git
(the demos use main-branch APIs the released PyPI build lags on):

```bash
uv run --extra sim python record_panda_cycle.py
```

The demos import Newton/Warp and are excluded from the static gate above.

## License & attribution

Licensed under [Apache-2.0](LICENSE). This project adapts patterns and small helpers from
[Newton](https://github.com/newton-physics/newton) (Apache-2.0) — see [NOTICE](NOTICE). Robot and
connector assets (Franka URDF, RJ45 mesh) are downloaded at runtime from Newton's asset registry
and are not redistributed here.
