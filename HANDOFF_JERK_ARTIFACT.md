# Known issue: action spike ("jerk") at the grasp handoff in the sbot RJ45 training data

**Status (2026-07-08):** diagnosed, quantified, fix designed (data v2), not yet regenerated.
**Affects:** `parallax/rj45_sbot` LeRobot dataset (500 eps, 66.6k frames) and every policy
trained on it (`pi05_rj45_sbot_lora/rj45_sbot_v1`).

## Symptom

At closed-loop eval, immediately after the policy closes the gripper it commands a fast
wrist reorientation — a burst of ~13–23°/step rotations over ~4 steps (measured in
`evalrun/ep_0001`: 22.8°, 20.7°, 13.4°, 7.7° at steps 52–55, with |dpos| spiking to 12 mm).
Visually the hand "glitches" right after grasping, and the RIGID eye-in-hand wrist camera
swings into the tabletop, corrupting the policy's own next observations (all-tan wrist
frames).

This is NOT an eval-harness bug and NOT an IK-branch flip: the burst is present in the
policy's raw commanded actions (`trace.npz` `drot`), while surrounding steps are ~0.2°/0.1 mm.

## Root cause: the data taught it

The data generator (`scripts/record_sbot_scene_gs.py --plug-traj ... --dump`) builds each episode as
**hold (12) → approach (45) → insertion (~50–200)**. The approach eases the arm from home to
`q_grasp` — an IK solution for the plug's start pose solved ONCE (48 iters from the home
seed). The insertion then re-solves IK **per frame** against the trajectory targets. Two
independent solves of nominally the same pose disagree (different iteration counts/seeds,
plus the rotation-target quaternion-order quirk below), so the recorded EEF state JUMPS
between the last approach frame and the first insertion frame. `dump_episode()` faithfully
records that jump as a legitimate one-step action.

Measured across all 500 training episodes (`train_handoff_jerk.png`, window ±20 around the
first phase-2 frame):

| | at handoff (±2 steps) | everywhere else |
|---|---|---|
| max \|drot\| per step | median **12°**, p90 **22–24°** | median 7.5°, p90 7.6° |
| max \|dpos\| per step | median **12 mm**, p90 **22 mm** | smooth ramps ≤13 mm |

The spike appears in essentially every episode at the same task moment (gripper just
closed), so BC treats it as intended behavior: *"after closing the gripper, whip the wrist."*

### Contributing bug: wxyz vs xyzw rotation targets

`newton.ik.IKObjectiveRotation` expects quaternions in **(x, y, z, w)** order. The replay /
data-gen path passes **wxyz** (`rot_obj.set_target_rotations(wp.vec4(*cq))` with `cq` in
wxyz), garbling the intended orientation target. The dataset stayed *self-consistent*
(recorded states are the ACHIEVED FK poses), but the garbled targets are why the two IK
solves at the handoff land far apart, and why achieved hand orientations sit in odd
configurations generally. The closed-loop eval path was fixed on 2026-07-08
(probe-verified: correct xyzw reproduces the home pose to 0.000°; wxyz flings the arm
~130°); **the data-gen path still has the old behavior** — kept intentionally until the v2
regen so that eval-vs-data conventions stay coherent for the v1 policy.

## Evidence artifacts

- `train_handoff_jerk.png` — 500-episode aligned plot: smooth decel → one-step spike → crawl.
- `evalrun/ep_0001/trace.npz` + `.log` — policy reproducing the burst after `GRASP latched
  at step 43`.
- Probe result (xyzw vs wxyz IK targets): wxyz-as-xyzw deviates 130–188° from home; correct
  xyzw = 0.000°.

## Fix plan (data v2)

1. **Blend the handoff:** lerp the approach into the replay's actual frame-0 IK solution
   (solve replay frame 0 FIRST, use it as the approach's end target) instead of a separately
   solved `q_grasp`. Removes the state jump entirely.
2. **Fix the rotation-target order** (wxyz → xyzw) in the data-gen IK, matching the eval fix.
3. Regenerate 500 episodes (`tools/render_batch.sh`, ~50 min), reconvert to
   `parallax/rj45_sbot_v2`, retrain (`pi05_rj45_sbot_lora`, new exp-name), re-eval with
   `tools/eval_batch.sh` + `tools/plot_eval.py` and compare `jerk_mm_mean` / `CLEAN n/50`
   against v1.

## How to re-measure

```bash
# training-data spike plot (writes train_handoff_jerk.png)
.venv/bin/python - <<'EOF'   # see git history of this file / train_handoff_jerk.png provenance
EOF
# eval-side: per-step |drot| lives in evalrun/ep_XXXX/trace.npz ("drot"), plotted by
.venv/bin/python tools/plot_eval.py
```

Related: the eval harness's grasp is a proximity-gated (<3 cm) rigid LATCH (no snap), so
post-grasp motion in eval reflects the policy's commands, not harness artifacts — see
`--policy-server` in `scripts/record_sbot_scene_gs.py`.
