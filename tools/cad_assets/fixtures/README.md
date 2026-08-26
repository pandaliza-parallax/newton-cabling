# 1422N17 jack — single-body 3D-print fixture

A **single connected body** printed fixture that holds the **1422N17 Panel-Mount Data
Adapter** (RJ45 feed-through coupler) at a rigid, repeatable pose for robotic
plug-insertion data collection. Built by `../build_jack_fixture.py` straight from the
McMaster STEP.

**Convention: socket up.** The robot inserts the plug straight down (+Z) into the jack.

## Files
| file | what |
|---|---|
| `jack_fixture.stl` / `.obj` | the fixture — print this (104 × 124 × 120 mm, 1189 cm³ solid) |
| `jack_fixture_preview.png`  | top-down + XZ/YZ sections with the jack nested |

**Envelope = 4× the jack's bbox in every axis** (`JACK_MULT = 4.0` in the build
script): X/Y grow the base-plate ears, Z grows the base downward into a tall
pedestal. The jack pocket, flange recess, and all clearances are **unscaled** —
they stay carved to the real part.

## How it holds the jack
The jack **drops in from the top** and is located by two stacked features:
- a **tapered body pocket** carved to the jack's real *outer envelope* (+0.4 mm), so
  the body nests along its **full length** — not just at the top. (The body tapers
  from 23.6 mm just under the flange down to ~10 mm at the rear, so a plain
  rectangular pocket would grip only ~4 mm and let the jack rock.)
- a **flange recess** (26.4 × 31.4 mm) above it. The flange seats on the ledge at its
  underside — the **+Z-down stop that takes the whole insertion load** — and is
  laterally captured by the recess walls. The socket collar is left **proud** of the
  top face so the plug is fully accessible.

A **rear feed-through hole** (16 × 14 mm) through the base plate keeps the coupler's
back open (mating cable can pass) and lines up under the pocket.

**Positive retention: the jack bolts down.** The 1422N17's flange has two Ø3.5 mm
screw holes (diagonal corners, at ±9.5 / ∓12 mm); the ledge has matching **Ø2.5 ×
12 mm pilot holes** (auto-located from the STEP at build time). Drive **2 × M3
self-tapping screws** through the flange into the ledge and the jack cannot lift out.
Without the screws it still works as a gravity drop-in: plug removal releases via the
plug's own latch, and the insertion load pushes *down* into the seat.

## Verified against the real STEP
- **0/5000** jack points intrude the fixture → clean drop-in with real clearance
- **6/6** flange edges supported → positive seat, not a taper wedge
- Center column **fully open** top-to-bottom → clear plug path + rear feed-through
- **1 connected body**, 0 boundary edges (closed solid), correct volume. 6 of ~180k
  edges are non-manifold — a harmless marching-cubes artifact of the voxel pocket,
  not a hole; every slicer handles it.

## Hardware
**4 × M5** cap-screws, fixture → bench/optical table, on a **71.4 × 83.8 mm** rectangle
(hole centres at ±35.7 / ±41.9 mm). The Ø10 counterbores run **85.2 mm deep** so the
screw threads through only the last 8 mm of the base (`MOUNT_GRIP`) — standard-length
M5 screws work; drive them with a ≥100 mm 4 mm hex key.

## Print
- **Orientation:** base plate flat on the bed, pocket opening up. **No supports** —
  the pocket opens upward, its taper is only ~16° from vertical, and the counterbores
  open upward.
- **Settings:** 0.2 mm layers, ≥4 perimeters / ≥30 % infill (it's a load-bearing jig).
  PLA or PETG; the ~0.4 mm fit tolerance suits FDM. The body is a 93 mm-tall solid
  block — infill (not solid plastic) keeps the print manageable (~350–450 g at 30 %).
- **Fit too tight/loose?** Change `PITCH` / `DIL_ITERS` (body clearance ≈ their product)
  or `FLANGE_CL`, then re-run. Overall size is `JACK_MULT` (envelope multiple of the
  jack bbox per axis; 4.0 → 104 × 124 × 120 mm).

## Rebuild / tune
```
cd newton-cabling
.venv/bin/python tools/cad_assets/build_jack_fixture.py
```
All dims (clearances, wall, base plate, screw sizes) are tunables at the top of the
script; the jack's flange/body/collar geometry is measured from the STEP at build time.

## Sim-side twin (Newton)
`build_jack_fixture_usd.py` bakes `jack_fixture.stl` into the sim's SOCKET frame
(`newton_cabling/assets/jack_fixture_rj45.usd`), registering `jack_seated.stl`
against `cad_rj45.usd`'s `/World/Socket` (same STEP, exact rotation chain; only the
translation is solved — lateral residual ~0.01 mm). The env mounts it with
`RigidCableVecEnv(..., jack_fixture=True)` / `gen_trajectories --jack-fixture` as a
second shape on the kinematic jack body, so per-episode placement and `--jack-yaw`
DR carry it automatically. In that frame the sleeve's front face sits 3.2 mm behind
the jack mouth (collar proud), base pedestal extending +y (away from the arm).
Registration preview: `jack_fixture_usd_preview.png`.
```
.venv/bin/python tools/cad_assets/build_jack_fixture_usd.py
```
