# Ethernet plug + cable — asset inventory (old vs new)

Reference for the assets behind the RJ45 connector-insertion task and the cable demos.
Two connector assets exist side by side: the **old/bundled toy** `rj45` and the **new
real-CAD** `cad_rj45`. Pick one with the `asset=` arg (`ConnectorVecEnv(..., asset="rj45"|"cad_rj45")`,
or `--asset` on the train/record scripts).

---

## 1. PLUG + JACK — OLD assets

### a) Bundled "toy" RJ45 (the original, still the default)
| File | What |
|---|---|
| `.venv/.../newton/examples/assets/rj45_plug.usd` | bundled Newton example asset — prims `/World/Socket`, `/World/Plug`, `/World/Latch`, `/World/Cable`. Simplified/low-poly, **loose** clearances. |
| `newton_cabling/connector.py` → `rj45_connector()` | the spec that loads it (prim paths, latch hinge, contact gaps). |

Loaded via `newton.examples.get_asset("rj45_plug.usd")`. This is what all the original
RL runs + recordings were built on. **Untouched** by the CAD work.

### b) Original gs-sim-vla CAD source (the user's real McMaster parts)
Location: `gs-sim-vla/scene/assets/objects/ethernet/`
| File | What |
|---|---|
| `9953K216_Category 5E Ethernet Cord.STEP` | **real plug** (Cat5e cord, the RJ45 plug head). Source of truth. |
| `1422N17_Panel-Mount Data Adapter.STEP` | **real jack** (panel-mount coupler). Source of truth. |
| `build_collision.py` | tessellates the STEPs (cascadio) → **voxel-remeshes @0.10 mm** → `*_meters.obj`. |
| `plug_meters.obj`, `jack_meters.obj` | 0.10 mm voxel-remesh outer envelopes. **Lossy**: the 0.10 mm pitch ≈ the real 0.11 mm clearance, so it quantized the plug↔cavity gap to ~0 (the "conversion error"). |
| `jack_raw_meters.obj` | pre-remesh tessellation variant. |
| `plug_converted.usd`, `jack_converted.usd`, `config.yaml` | Isaac MeshConverter outputs (PhysX SDF collision) from the OBJs. Not used by Newton. |
| `splat-cord.ply`, `splat-mount.ply` | Gaussian-splat render assets (for the GS renderer, not physics). |

---

## 2. PLUG + JACK — NEW assets (`cad_rj45`)

> **Current build (2026-06-22): CLEAN IDEALIZED.** `build_cad_rj45_clean.py` authors
> `cad_rj45.usd` directly from clean parametric primitives sized to the real McMaster parts,
> booleaned (manifold3d) into watertight solids: `/World/Socket` = square D-flange panel jack +
> keyed RJ45 cavity + **catch ledge** + 2 mounting holes; `/World/Plug` = clean 8P8C body +
> molded boot + cable stub; `/World/Latch` = the **stepped-keyway tab** the user drew, on a
> revolute hinge. Insertion +Y, latch + keyway on −Z (matches the 1422N17 drawing), cavity floor
> +12 mm. The plug cross-section is the user's sketched shape (`KEYWAY_STEPS`).
>
> **Latch click:** the `CATCH_LEDGE` (`LEDGE_H`) makes the latch physically deflect ~1.3° + snap
> back on insertion — deliberately SMALL. A rigid VBD latch can't flex over a real-depth catch
> without jamming the plug: a visible/locking click drops seating to ~36%; this subtle one keeps
> ~86%; no ledge = 100%. (User chose the subtle physical click.) The STEP-tessellation + carve
> pipeline below (`build_cad_rj45_usd.py` / `build_jack_carved.py` / `raw_export.py`) is kept for
> reference but no longer shipped.

### Full-fidelity-from-STEP pipeline (the current shipped build)

### Output (what the sim loads)
| File | What |
|---|---|
| `newton_cabling/assets/cad_rj45.usd` | merged USD: prims `/World/Socket`, `/World/Plug`, `/World/Latch`. Built from the real STEP, **no lossy voxel pass** for the plug. ~97%-size plug, fills the carved cavity, seats 100%. **`/World/Latch` is the REAL latch cantilever** split off the plug (a separate body on a revolute hinge, like the original Newton example) — no longer a stand-in box. |

### Source meshes — `tools/cad_assets/src/`
| File | What |
|---|---|
| `plug_raw.obj` | **canonical** plug: raw cascadio tessellation of the STEP head (real geometry/clearance, non-watertight but consistently wound — Newton's SDF handles it). |
| `jack_raw.obj` | raw tessellation of the jack receptacle. |
| `jack_carved_meters.obj` | jack with a **smooth cavity carved** to admit the full-size plug (catch removed — current fill build uses this). |
| `plug_meters.obj`, `jack_meters.obj` | copies of the old 0.10 mm voxel OBJs (fallback). |

### Build scripts — `tools/cad_assets/`
| File | What |
|---|---|
| `raw_export.py` | STEP → `plug_raw.obj` / `jack_raw.obj` (cascadio tessellate + orient, **no voxel**). |
| `build_jack_carved.py` | voxel-carve a clean cavity into the jack → `jack_carved_meters.obj` (sized to the plug; catch on/off via the carve mask). |
| `build_cad_rj45_usd.py` | **main builder**: load src meshes → split latch (face-selection) → orient to Newton (+Y insertion) → fit/scale plug → author `cad_rj45.usd`. Key knobs: `FORCE_FIT_SCALE`, `PLUG_CLEARANCE_M`, `STANDIN_LATCH`, `split_latch(...)` box. |
| `probe_cad.py`, `debug_contacts.py` | diagnostics (cavity/plug fit, in-sim contacts). |

### Code integration
| File | What |
|---|---|
| `newton_cabling/connector.py` → `cad_rj45_connector(gap_meters, sdf_max_resolution)` | the spec (prim paths, latch hinge/spring, sub-mm contact gaps). |
| `newton_cabling/sim/scene.py` → `resolve_asset_path()` | resolves `cad_rj45.usd` from `newton_cabling/assets/` (bundled `rj45_plug.usd` falls through to Newton's cache). |
| `rl/connector_env.py` → `ASSET_PROFILES` + `asset=` arg | per-asset geometry (seat depth, clearance/rigid_gap). `"rj45"` reproduces the original constants exactly. |
| `rl/smoke_asset.py` | smoke test (`uv run --extra sim python rl/smoke_asset.py [rj45|cad_rj45]`). |
| `rl/train_ppo.py`, `rl/record_policy.py`, `rl/record_success_fail.py` | all take `--asset rj45|cad_rj45`. |

### Recordings
`cad_clean.rrd` / `cad_clean.rbl` — latest CAD rollout (view: `uvx --from rerun-sdk rerun cad_clean.rrd cad_clean.rbl`).

### Regenerate the new asset from scratch
```bash
cd newton-cabling
.venv/bin/python tools/cad_assets/raw_export.py          # STEP -> *_raw.obj
.venv/bin/python tools/cad_assets/build_jack_carved.py   # -> jack_carved_meters.obj
.venv/bin/python tools/cad_assets/build_cad_rj45_usd.py  # -> newton_cabling/assets/cad_rj45.usd
```

---

## 3. CABLE (the deformable cord) — unchanged

The cable is **procedural**, not a file asset:
| File | What |
|---|---|
| `newton_cabling/cable.py` → `route_cable_from_boot()` | generates a draped cable centerline; `builder.add_rod(...)` makes it a Cosserat rod (bend/twist springs). |
| `record_patch_panel.py`, `record_panda_cycle.py` | demos that attach the cable to the plug boot. |

Note: the **RL insertion env (`rl/connector_env.py`) has no cable** — it's socket + plug +
latch only. The cable appears only in the recording/demo scripts (and the bundled
`rj45_plug.usd` has a `/World/Cable` prim used by `example_contacts_rj45_plug.py`).

---

## Key facts (why the new asset is what it is)
- Newton can't read STEP directly → must tessellate. Import via USD / `newton.Mesh`.
- The old 0.10 mm voxel-remesh quantized away the real ~0.11 mm clearance (the size bug).
- The plug size is **geometry-limited, not conversion-limited**: with the real catch a rigid
  plug caps at ~7.8 mm; removing the catch (smooth carve) lets it fill at ~97%. Real life
  gets both (full-size + catch) only because the plastic **flexes** — which rigid VBD doesn't.
- **Latch (`STANDIN_LATCH=False`, current build):** the real latch tab is split off the plug
  by `split_latch(...)` (face-selection box tuned to the 9953K216 spring tab: wings at
  z[6,9]mm/x±3.05, tip to ~6.5mm, boot at z>15 kept on the body) into a separate articulated
  `/World/Latch`, matching the original Newton example. With the smooth carved bore it rides
  on top without a hard catch. To make it physically **click**, rebuild `jack_carved` with the
  catch preserved (`build_jack_carved.py` has the `in_catch` mask; the catch sits on the +Y
  top, same side as the plug latch) and tune the latch spring — best done with a rerun view.
