"""Render the GROUND-TRUTH Newton physics of a cable PPO rollout to PNG frames (matplotlib,
headless). This is the reference the GS render must match: it draws the ACTUAL bodies --
gripper fingertips, the connector (oriented box), the flexible cable rods, and the jack (box +
mouth arrow) -- from a front and a side view, and prints per-frame diagnostics:

  * finger<->connector distance  (is the gripper holding the cable?)
  * connector-axis vs jack-mouth-axis angle  (is it aligned to insert?)
  * connector->seat distance  (does it insert?)

    .venv/bin/python rl/render_cable_newton.py --checkpoint rl/runs/cable_v3/best_model.pt \
        --stage 4 --cable-tilt 5 --steps 130 --out newton_ref
"""
import argparse
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from cable_env import CableInsertVecEnv  # noqa: E402
from train_ppo import ActorCritic  # noqa: E402

DEV = "cuda:0"


def box(ax, c, R, half, color, alpha=0.35):
    """Draw an oriented box (center c, rotation matrix R, half-extents half)."""
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    v = (R @ s.T).T + c
    faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection([v[list(f)] for f in faces], facecolor=color,
                                         edgecolor="k", alpha=alpha, linewidths=0.4))


def render_from_traj(ep_dir, out, frames_req):
    """Render the GROUND TRUTH of an ALREADY-SAVED episode (rl/gen_cable_traj.py output).

    This is the mode to use when debugging the GS render: it draws the exact same episode the
    GS renderer replays. Re-rolling the policy instead (the --checkpoint path below) gives a
    DIFFERENT rollout -- VBD is nondeterministic -- so the two views would not be comparable.
    Everything is in the saved SEAT frame (seat at the origin), which is also the frame the GS
    renderer anchors to the jack.
    """
    import json
    face = np.load(os.path.join(ep_dir, "face_traj.npy"))
    eef = np.load(os.path.join(ep_dir, "eef_traj.npy"))
    tips = np.load(os.path.join(ep_dir, "tips_traj.npy"))
    rods = np.load(os.path.join(ep_dir, "rods_traj.npy"))
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    jp = np.array(meta.get("jack_pos_seatrel", [0.0, 0.0, 0.0]))
    T = len(face)
    seat = np.zeros(3)                       # seat IS the origin in this frame
    ins = np.array([0.0, 1.0, 0.0])          # insertion axis in the seat frame (+y)
    frames = frames_req if frames_req else [0, T // 2, T - 1]
    print(f"[newton] {ep_dir}: {T} frames, success={meta.get('success')}, "
          f"tilt={meta.get('tilt_this_env_deg', 0):.1f}deg, ckpt={meta.get('checkpoint')}")
    print("  frame | finger<->conn (mm) | conn-vs-jack axis (deg) | conn->seat (mm)")
    for fr in frames:
        fr = min(fr, T - 1)
        fp, fq = face[fr, :3], face[fr, 3:]
        fqx = fq[[1, 2, 3, 0]]                                   # wxyz -> xyzw
        cax = Rot.from_quat(fqx).apply([0, 1.0, 0])
        fingmid = tips[fr].mean(axis=0)
        d_fing = np.linalg.norm(fingmid - fp) * 1000.0
        ang = math.degrees(math.acos(np.clip(abs(np.dot(cax, ins)), -1, 1)))
        d_seat = np.linalg.norm(fp - seat) * 1000.0
        print(f"   {fr:4d} | {d_fing:8.1f}          | {ang:8.1f}              | {d_seat:8.1f}")
        fig = plt.figure(figsize=(12, 6))
        for k, (elev, azim, name) in enumerate([(12, -60, "FRONT-ish"), (12, 30, "SIDE-ish")]):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            Rr = rods[fr]
            ax.plot(Rr[:, 0], Rr[:, 1], Rr[:, 2], "-o", color="tab:orange", ms=2, lw=1, label="cable")
            box(ax, fp, Rot.from_quat(fqx).as_matrix(), np.array([0.006, 0.011, 0.004]), "magenta")
            box(ax, jp, np.eye(3), np.array([0.009, 0.009, 0.009]), "black", alpha=0.25)
            ax.quiver(*seat, *(-ins * 0.03), color="green", lw=2)               # jack mouth
            ax.quiver(*fp, *(Rot.from_quat(fqx).apply([0, 0.03, 0])), color="red", lw=2)  # conn fwd
            ax.scatter(*tips[fr, 0], c="blue", s=30)
            ax.scatter(*tips[fr, 1], c="blue", s=30, label="fingertips")
            ax.scatter(*eef[fr, :3], c="gray", s=40, marker="s", label="wrist_3")
            ax.scatter(*seat, c="green", s=40, marker="*")
            ax.set_xlim(-0.12, 0.12); ax.set_ylim(-0.12, 0.12); ax.set_zlim(-0.12, 0.12)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{name}  frame {fr}\nfing<->conn {d_fing:.0f}mm  axis {ang:.0f}deg  seat {d_seat:.0f}mm")
            if k == 0:
                ax.legend(loc="upper left", fontsize=7)
        o = f"{out}_frame_{fr:04d}.png"
        fig.tight_layout(); fig.savefig(o, dpi=110); plt.close(fig)
        print(f"  wrote {o}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-traj", default=None,
                    help="render a SAVED episode dir (cable_traj/ep_XXXX) instead of re-rolling the "
                         "policy -- use this to debug the GS render, so both views show the SAME "
                         "episode (VBD is nondeterministic, so a re-roll would differ).")
    ap.add_argument("--checkpoint", default="rl/runs/cable_v3/best_model.pt")
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--cable-tilt", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=130)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="newton_ref")
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="explicit frame indices to render (default: start, mid, seated)")
    args = ap.parse_args()

    if args.from_traj:                       # ground truth of an already-saved episode
        render_from_traj(args.from_traj, args.out, args.frames)
        return

    env = CableInsertVecEnv(1, seed=args.seed, cable_tilt_deg=args.cable_tilt)
    env.set_stage(args.stage)
    ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
    ac.load_state_dict(torch.load(args.checkpoint, map_location=DEV))
    ac.eval()

    labels = list(env.model.body_label)
    _sfx = lambda s: next(i for i, l in enumerate(labels) if l.endswith(s))
    tip1 = _sfx("gripper_finger1_finger_tip_link")
    tip2 = _sfx("gripper_finger2_finger_tip_link")
    jb = int(env.jack_body[0])
    rods = [int(r) for r in env.rod_bodies_all[0]]
    rf = int(env.rod_front[0])
    wb = int(env.wrist_body[0])

    obs = env.reset()
    seat = env.seat_pos[0].copy()
    ins = env.ins[0].copy() if hasattr(env, "ins") else np.array([0, 1.0, 0])
    ins = ins / (np.linalg.norm(ins) + 1e-9)

    traj = []  # per-frame geometry + the diagnostics
    hold_hit = None
    for t in range(args.steps):
        bqn = env.state_0.body_q.numpy()
        fp, fq = env._face_pose(bqn)
        fp = fp[0]; fq = fq[0]
        t1 = bqn[tip1, :3]; t2 = bqn[tip2, :3]
        fingmid = 0.5 * (t1 + t2)
        conn_axis = Rot.from_quat(fq).apply([0, 1.0, 0])   # connector local +y = insertion fwd
        d_fing = np.linalg.norm(fingmid - fp) * 1000.0
        ang = math.degrees(math.acos(np.clip(abs(np.dot(conn_axis, ins)), -1, 1)))
        d_seat = np.linalg.norm(fp - seat) * 1000.0
        traj.append(dict(t=t, fp=fp, fq=fq, t1=t1, t2=t2, wp=bqn[wb, :3].copy(),
                         rods=bqn[rods, :3].copy(), jp=bqn[jb, :3].copy(), jq=bqn[jb, 3:7].copy(),
                         d_fing=d_fing, ang=ang, d_seat=d_seat))
        if hold_hit is None and env.hold[0] >= 20:
            hold_hit = t
        with torch.no_grad():
            a = ac.mean_action(obs)
        obs, _, _, _, _ = env.step(a)

    seated = hold_hit if hold_hit is not None else args.steps - 1
    frames = args.frames if args.frames else [0, max(0, seated // 2), seated]
    print(f"[newton] rollout {len(traj)} frames; hold@{hold_hit}; rendering {frames}")
    print(f"[newton] jack pos {np.round(traj[0]['jp'],3)}  seat {np.round(seat,3)}  ins {np.round(ins,2)}")
    print("  frame | finger<->conn (mm) | conn-vs-jack axis (deg) | conn->seat (mm)")
    for fr in frames:
        d = traj[fr]
        print(f"   {fr:4d} | {d['d_fing']:8.1f}          | {d['ang']:8.1f}              | {d['d_seat']:8.1f}")

    jmouth = Rot.from_quat(traj[0]["jq"]).apply([0, 1.0, 0])  # jack local +y (report only)
    for fr in frames:
        d = traj[fr]
        fig = plt.figure(figsize=(12, 6))
        for k, (elev, azim, name) in enumerate([(12, -60, "FRONT-ish"), (12, 30, "SIDE-ish")]):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            # cable rods as a polyline + capsule dots
            R = d["rods"]
            ax.plot(R[:, 0], R[:, 1], R[:, 2], "-o", color="tab:orange", ms=2, lw=1, label="cable")
            # connector box (~11.68 x 8 x 22 mm) at the face pose
            box(ax, d["fp"], Rot.from_quat(d["fq"]).as_matrix(),
                np.array([0.006, 0.011, 0.004]), "magenta")
            # jack box + mouth (mouth opens toward -ins: the connector approaches along +ins)
            box(ax, d["jp"], Rot.from_quat(d["jq"]).as_matrix(),
                np.array([0.009, 0.009, 0.009]), "black", alpha=0.25)
            ax.quiver(*seat, *(-ins * 0.03), color="green", lw=2)          # jack mouth (opens -ins)
            ax.quiver(*d["fp"], *(Rot.from_quat(d["fq"]).apply([0, 0.03, 0])), color="red", lw=2)  # conn fwd
            ax.scatter(*d["t1"], c="blue", s=30); ax.scatter(*d["t2"], c="blue", s=30, label="fingertips")
            ax.scatter(*d["wp"], c="gray", s=40, marker="s", label="wrist_3")
            ax.scatter(*seat, c="green", s=40, marker="*")
            c = seat
            ax.set_xlim(c[0] - 0.12, c[0] + 0.12); ax.set_ylim(c[1] - 0.12, c[1] + 0.12)
            ax.set_zlim(c[2] - 0.12, c[2] + 0.12)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{name}  frame {fr}\nfing<->conn {d['d_fing']:.0f}mm  axis {d['ang']:.0f}deg  seat {d['d_seat']:.0f}mm")
            if k == 0:
                ax.legend(loc="upper left", fontsize=7)
        out = f"{args.out}_frame_{fr:04d}.png"
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
