# 1422N17 jack — single-body 3D-print fixture

A **single connected body** printed fixture that holds the **1422N17 Panel-Mount Data
Adapter** (RJ45 feed-through coupler) at a rigid, repeatable pose for robotic
plug-insertion data collection. Built by `../build_jack_fixture.py` straight from the
McMaster STEP.

**Convention: socket up.** The robot inserts the plug straight down (+Z) into the jack.

## Files
| file | what |
|---|---|
| `jack_fixture.stl` / `.obj` | the fixture — print this (58.8 × 63.8 × 32.8 mm, 55.6 cm³) |
| `jack_fixture_preview.png`  | top-down + XZ/YZ sections with the jack nested |

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

**Lift-out (+Z) is intentionally open.** A single *rigid* printed body cannot trap the
flange from above — a closed capture needs either a second part or a snap/flex feature.
In practice nothing lifts the jack: plug removal releases via the plug's own latch, and
the insertion load pushes *down* into the seat. If you later want positive retention,
the options are a bolt-on clamp cap (2 parts) or integrated snap fingers.

## Verified against the real STEP
- **0/5000** jack points intrude the fixture → clean drop-in with real clearance
- **6/6** flange edges supported → positive seat, not a taper wedge
- Center column **fully open** top-to-bottom → clear plug path + rear feed-through
- **1 connected body**, 0 boundary edges (closed solid), correct volume. 6 of ~180k
  edges are non-manifold — a harmless marching-cubes artifact of the voxel pocket,
  not a hole; every slicer handles it.

## Hardware
**4 × M5** cap-screws, fixture → bench/optical table, on a **68.8 × 73.8 mm** rectangle
(ear centres at ±34.4 / ±36.9 mm). Top counterbores (Ø10 × 4 mm) recess the heads.

## Print
- **Orientation:** base plate flat on the bed, pocket opening up. **No supports** —
  the pocket opens upward, its taper is only ~16° from vertical, and the counterbores
  open upward.
- **Settings:** 0.2 mm layers, ≥4 perimeters / ≥30 % infill (it's a load-bearing jig).
  PLA or PETG; the ~0.4 mm fit tolerance suits FDM.
- **Fit too tight/loose?** Change `PITCH` / `DIL_ITERS` (body clearance ≈ their product)
  or `FLANGE_CL`, then re-run.

## Rebuild / tune
```
cd newton-cabling
.venv/bin/python tools/cad_assets/build_jack_fixture.py
```
All dims (clearances, wall, base plate, screw sizes) are tunables at the top of the
script; the jack's flange/body/collar geometry is measured from the STEP at build time.
