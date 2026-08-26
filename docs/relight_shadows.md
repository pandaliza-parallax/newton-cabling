# Adding ray-traced contact shadows to a Newton + Gaussian-splat render

How to reuse the `relight` shadow pass from `~/parallax/parallax-demo-newton` in
other Newton→GS pipelines (e.g. this repo's cable envs). Written for an agent or
developer wiring it into new code.

## What it does and why

A 3DGS render is a rasteriser: every gaussian emits its baked color, so a
simulated robot composited into a splat scene casts no shadow and looks pasted
on. The relight pass computes the shadow in Newton — where the geometry is
known — and multiplies it onto the GS frame as a per-pixel mask:

```
cast    = OVRTX path-trace of the Newton scene, as-is
nocast  = same trace, but DYNAMIC objects are invisible to shadow rays
mask    = clip(lum(cast) / lum(nocast), 0, 1)     # 1 = untouched
out     = gs_frame * mask
```

Because albedo, lights, geometry, and camera are identical between the two
passes, everything except occlusion divides out to exactly 1.0:

- Pixels covered by a moving object land on 1.0 by themselves — no
  segmentation pass needed.
- The splat scene's own baked-in shadows cancel instead of being darkened
  twice (only *dynamic* geometry is un-shadowed, via the per-prim primvar
  `primvars:doNotCastShadows`).
- The mask is clamped at 1.0: shadows only, never added light.

## Requirements

- The implementation: `~/parallax/parallax-demo-newton/src/relight.py`
  (self-contained module, ~420 lines; import it or copy it).
- The OVRTX renderer: `pip install -e '.[relight]'` in parallax-demo-newton,
  or depend on the same wheel — `newton[rtx]` from the Dalus-AI newton fork
  (see that repo's `pyproject.toml` `[project.optional-dependencies].relight`).
  Importing `relight.py` itself never needs ovrtx; only building the pass does.
- A CUDA GPU (verified on RTX 5090 / sm_120).
- A **finalized Newton model** whose shapes distinguish static from dynamic:
  the pass classifies every mesh prototype as "moves" or "doesn't" from
  `ShapeInstance.static`. Static background/table must actually be flagged
  static or their shadows will (wrongly) enter the mask.

## Integration (per the demo's `run_demo.py` usage)

```python
from relight import RelightPass   # parallax-demo-newton/src on sys.path

# Once, after the Newton model is finalized (the same model the sim steps):
relight = RelightPass(
    model,                 # finalized newton model
    cam_Ks=[K_front, K_wrist],  # one 3x3 intrinsics per GS camera, renderer order
    light_scale=0.15,      # scale the OVRTX studio rig; <1 avoids clipping
    accum=12,              # path-tracer accumulation frames per pose per pass
    max_gain=1.0,          # clamp: shadows only
    strength=1.0,          # 0..1 blends mask toward no-op
    up_axis="Z",
    light_angle=12.0,      # key-light angular diameter (deg) — penumbra width
    mask_scale=1.0,        # trace at a fraction of GS res and upsample (0.5 is cheap)
)

# Every frame, AFTER the GS renderer returns its frame(s):
gs = client.render(poses, ...)              # (C, H, W, 3) uint8 from the GS bridge
views = [(cam_pos, cam_quat_wxyz), ...]     # SAME poses handed to the GS renderer,
                                            # one per camera, in cam_Ks order
masks = relight.masks(state, views, gs.shape[1:3])   # (C, H, W) float, 1 = untouched
gs = relight.apply(gs, masks)               # uint8 in, uint8 out

# On shutdown:
relight.close()
```

### Camera conventions — the part that silently breaks

- `views` poses are in the **renderer's optical frame**: quaternion is wxyz,
  +z forward, +x right, **+y down**. Pass exactly what the GS bridge gets.
- One mask per camera per frame. A mask is only valid for the view it was
  traced from — never reuse the front camera's mask on a wrist camera.
- Intrinsics come from the same 3×3 K handed to the GS renderer. The pass
  bakes fx/fy/cx/cy into the USD camera (aperture + apertureOffset), so an
  off-center principal point is honored; a roll-carrying pose is honored
  (the eye-in-hand camera rolls with the gripper).
- The pass sizes itself from the **first GS frame it sees** and the passes are
  built once — a mid-clip resolution change raises.

### Tuning (demo defaults in `config/scene.json` → `relight`)

| knob | default | meaning |
|---|---|---|
| `light_scale` | 0.15 | dim the stock studio rig so bright surfaces don't clip (a ratio of two clipped values is 1 = no shadow) |
| `accum` | 12–20 | path-trace samples per pose per pass; raise with wider `light_angle` |
| `light_angle` | 12.0° | penumbra width; USD's 0.53° default looks synthetically sharp indoors |
| `strength` | 1.0 | blend toward no-op: `1 + strength*(mask-1)` |
| `mask_scale` | 1.0 | trace at reduced res and bilinearly upsample; shadow maps are low-frequency, 0.5 is nearly free |

## Pitfalls (all hit while standing this up, Aug 2026)

1. **Version lock.** parallax-demo-newton, DalusPySim, and DalusSimCore must be
   pulled together. A protocol mismatch does not error — the GS pane renders
   from a garbage camera or the first frame never arrives. Check
   `git log HEAD..origin/HEAD` in all three before debugging anything else.
2. **SHM permissions.** After a renderer restart, `/dev/shm/*dal_*` segments are
   root-owned 0600. Non-sudo runs fail with "No permission to access this
   segment" (sender) or time out on the first view (receiver). Either run with
   `sudo -E`, or: `docker exec parallax_sim_fp sh -c 'chmod 666 /dev/shm/*dal_*'`
   — the `send_dal_*` pair only exists after the client's SETUP, so chmod again
   once it appears.
3. **Editable installs don't pick up new extras.** After a pull that changes
   `pyproject.toml`, re-run `pip install -e '.[relight]'`; the symptom is
   `ModuleNotFoundError: ovrtx`.
4. **Root-owned output dirs** from old `sudo -E` runs make the final PNG/mp4
   write fail after an otherwise-successful run. Check ownership first.
5. **Static/dynamic classification is load-bearing.** A mesh prototype shared
   by both a static and a dynamic shape is treated static (warned); if nothing
   is dynamic the pass raises rather than compute an all-1.0 mask. `set_model()`
   and one `log_state()` must run before the stage is built.
6. **Cost.** Two path-traced passes per distinct lens per frame. Cameras with
   identical K share a (cast, nocast) viewer pair; a wrist cam with its own
   lens costs its own pair. Use `mask_scale=0.5` and `denoise` (on by default)
   to keep it cheap.

## Reference

- Implementation: `parallax-demo-newton/src/relight.py`
- Wiring example: `parallax-demo-newton/run_demo.py` (`--shadows`,
  `--shadow-strength`), `demos/demo_deformables.py` (always-on)
- Docs: parallax-demo-newton `README.md` § "Shadows (`--shadows`)"
- Prior art it follows: DalusPySim
  `parallax_sim/isaac_lab/vision_augmentation/vision_utils.py` (Isaac RTX
  shadow masks, global toggle) — relight scopes the toggle to dynamic geometry
  so baked static shadows aren't double-darkened.
