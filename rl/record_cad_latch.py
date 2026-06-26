"""Record a cad_rj45 insertion with the RJ45 latch visibly DEFLECTING + LATCHING, the
demo way (example_contacts_rj45_plug's recording scripts the latch via targets too).

Physical latch click is rigid-SDF-capped for a small clip + tight plug (see ASSETS.md),
so the latch flip is SCRIPTED here: the plug seats under the real base controller, and the
latch angle is driven on a timeline keyed to insertion depth -- it rests slanted up,
flattens as it enters the jack, then snaps back up to lock. With the slanted CAD-scaled
latch this looks like the demo. Seating physics is untouched (the override only sets the
latch's render pose each frame).

    .venv/bin/python rl/record_cad_latch.py
    uvx --from rerun-sdk rerun cad_friction.rrd cad_friction.rbl
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connector_env import ConnectorVecEnv  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: E402

DEV = "cuda:0"
FRAMES = 200
FLIP_MAX = 0.40      # rad, latch flattens (back press-tab swings DOWN toward the body)


def quat_mul(a, b):  # [x,y,z,w]
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_x(theta):  # rotation about world +X
    return np.array([np.sin(theta / 2), 0.0, 0.0, np.cos(theta / 2)])


INSERT_FRAMES = 95   # frames to drive the plug from out -> seated


def flip_profile(frac):
    """Latch angle vs insertion fraction [0,1]: rests slanted (0), FLATTENS over the entry,
    then SNAPS back up to latch near seat. Returning to 0 = sprung up = latched."""
    if frac < 0.35:
        return 0.0
    if frac < 0.78:
        return FLIP_MAX * (frac - 0.35) / 0.43          # flatten as the clip enters
    if frac < 0.88:
        return FLIP_MAX * (1.0 - (frac - 0.78) / 0.10)  # snap up to latch (the click)
    return 0.0


def smoothstep(x):
    return x * x * (3 - 2 * x)


def main():
    # KINEMATIC clean demo: the plug is driven dead-straight into the centred cavity (no random
    # offset, no tilt, no penetration) and the latch flips on its hinge -- a guaranteed-clean
    # visualization, like the demo's own scripted recording. (Friction's effect is the sweep
    # plot; physics insertion is rl/record_policy.py.)
    env = ConnectorVecEnv(1, seed=7, random_easy=False, asset="cad_rj45")
    env.set_stage(0)
    env.reset()
    pi, li = int(env.plug_idx.numpy()[0]), int(env.latch_idx.numpy()[0])

    q0 = env.state_0.body_q.numpy()
    seat = env.seated.numpy()[0].copy()              # centred seated plug pos (x, seat_y, z)
    seat_x, seat_y, seat_z = float(seat[0]), float(seat[1]), float(seat[2])
    y_start = seat_y - 0.026                          # start 26mm out of the mouth
    plug_quat0 = q0[pi][3:7].copy()                  # aligned (rotation-locked) orientation
    latch_off = (q0[li][:3] - q0[pi][:3]).copy()     # latch position relative to the plug
    latch_quat0 = q0[li][3:7].copy()

    viewer = open_rrd_recorder("cad_clean.rrd")
    viewer.set_model(env.model)
    dt, t = 1.0 / 60.0, 0.0
    for f in range(FRAMES):
        frac = min(f / INSERT_FRAMES, 1.0)
        y = y_start + (seat_y - y_start) * smoothstep(frac)
        plug_pos = np.array([seat_x, y, seat_z])
        theta = flip_profile(frac)
        q = env.state_0.body_q.numpy()
        q[pi][:3], q[pi][3:7] = plug_pos, plug_quat0                 # plug: straight, centred
        q[li][:3] = plug_pos + latch_off                            # latch: attached
        q[li][3:7] = quat_mul(quat_x(theta), latch_quat0)           # latch: flip on the hinge
        env.state_0.body_q.assign(q)
        viewer.begin_frame(t)
        viewer.log_state(env.state_0)
        viewer.end_frame()
        t += dt
        if f % 30 == 0 or f == FRAMES - 1:
            print(f"frame {f:3d}: plug_y={(y - (seat_y - 0.012)) * 1000:6.1f}mm(from mouth) "
                  f"latch_theta={np.degrees(theta):+5.1f} deg", flush=True)

    bp = auto_blueprint("cad_clean.rbl", env.model)
    print(f"\nrecording complete: cad_clean.rrd ; blueprint {bp}")
    print("view with:  uvx --from rerun-sdk rerun cad_clean.rrd cad_clean.rbl")


if __name__ == "__main__":
    main()
