"""Side-by-side GS render vs Newton ground truth, frame-for-frame — the debugging view.

For each frame it composes one image:   [GS FRONT] [GS WRIST] [Newton 3D truth]
so a GS/physics disagreement (plug rotated, jack facing, connector detached) is visible
directly instead of being inferred from two separate files.

Frame indices align 1:1 because render_batch_v4.sh renders with --grasped-only: the GS dump
is insertion-only, exactly the frames in the saved trajectory. If you render WITHOUT
--grasped-only the GS dump gains a home+approach preamble and the mapping shifts by that
many frames — pass --gs-offset to correct it.

    .venv/bin/python tools/compare_gs_newton.py --gs datagen_v4/ep_0000 \
        --traj cable_traj/ep_0000 --out compare/ep_0000 --stride 4
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl"))
from render_cable_newton import box  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs", required=True, help="GS episode dir (datagen_v4/ep_XXXX)")
    ap.add_argument("--traj", required=True, help="trajectory dir (cable_traj/ep_XXXX)")
    ap.add_argument("--out", default="compare/ep", help="output prefix (dir is created)")
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="explicit frame indices")
    ap.add_argument("--gs-offset", type=int, default=0,
                    help="GS frame index = traj index + offset (nonzero only if the GS dump has a "
                         "home/approach preamble, i.e. rendered without --grasped-only)")
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    args = ap.parse_args()

    face = np.load(os.path.join(args.traj, "face_traj.npy"))
    eef = np.load(os.path.join(args.traj, "eef_traj.npy"))
    tips = np.load(os.path.join(args.traj, "tips_traj.npy"))
    rods = np.load(os.path.join(args.traj, "rods_traj.npy"))
    meta = json.load(open(os.path.join(args.traj, "meta.json")))
    jp = np.array(meta.get("jack_pos_seatrel", [0.0, 0.0, 0.0]))
    T = len(face)
    seat = np.zeros(3)
    ins = np.array([0.0, 1.0, 0.0])

    frames = args.frames if args.frames else list(range(0, T, max(1, args.stride)))
    outdir = os.path.dirname(args.out) or "."
    os.makedirs(outdir, exist_ok=True)
    print(f"[cmp] {args.traj}: {T} frames, success={meta.get('success')}, "
          f"tilt={meta.get('tilt_this_env_deg', 0):.1f}deg, ckpt={meta.get('checkpoint')}")
    print(f"[cmp] GS {args.gs} (offset {args.gs_offset}) | rendering {len(frames)} frames")
    print("  frame | finger<->conn (mm) | axis (deg) | conn->seat (mm) | GS frame")

    missing = 0
    for fr in frames:
        fr = min(fr, T - 1)
        gsi = fr + args.gs_offset
        fp, fq = face[fr, :3], face[fr, 3:]
        fqx = fq[[1, 2, 3, 0]]
        cax = Rot.from_quat(fqx).apply([0, 1.0, 0])
        d_fing = np.linalg.norm(tips[fr].mean(axis=0) - fp) * 1000.0
        ang = math.degrees(math.acos(np.clip(abs(np.dot(cax, ins)), -1, 1)))
        d_seat = np.linalg.norm(fp - seat) * 1000.0

        fig = plt.figure(figsize=(16, 5.4))
        for k, (sub, lbl) in enumerate((("image", "GS FRONT"), ("wrist_image", "GS WRIST"))):
            ax = fig.add_subplot(1, 3, k + 1)
            p = os.path.join(args.gs, sub, f"frame_{gsi:04d}.png")
            if os.path.exists(p):
                ax.imshow(plt.imread(p))
            else:
                ax.text(0.5, 0.5, f"missing\n{os.path.relpath(p)}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="crimson")
                missing += 1
            ax.set_title(f"{lbl}  (gs frame {gsi})", fontsize=10)
            ax.axis("off")

        ax = fig.add_subplot(1, 3, 3, projection="3d")
        Rr = rods[fr]
        ax.plot(Rr[:, 0], Rr[:, 1], Rr[:, 2], "-o", color="tab:orange", ms=2, lw=1, label="cable")
        box(ax, fp, Rot.from_quat(fqx).as_matrix(), np.array([0.006, 0.011, 0.004]), "magenta")
        box(ax, jp, np.eye(3), np.array([0.009, 0.009, 0.009]), "black", alpha=0.25)
        ax.quiver(*seat, *(-ins * 0.03), color="green", lw=2)
        ax.quiver(*fp, *(Rot.from_quat(fqx).apply([0, 0.03, 0])), color="red", lw=2)
        ax.scatter(*tips[fr, 0], c="blue", s=30)
        ax.scatter(*tips[fr, 1], c="blue", s=30, label="fingertips")
        ax.scatter(*eef[fr, :3], c="gray", s=40, marker="s", label="wrist_3")
        ax.scatter(*seat, c="green", s=40, marker="*")
        ax.set_xlim(-0.12, 0.12); ax.set_ylim(-0.12, 0.12); ax.set_zlim(-0.12, 0.12)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_title(f"NEWTON truth  (traj frame {fr})", fontsize=10)
        ax.legend(loc="upper left", fontsize=7)

        fig.suptitle(f"frame {fr}/{T - 1}   fing<->conn {d_fing:.0f}mm   "
                     f"conn-vs-jack axis {ang:.1f}deg   conn->seat {d_seat:.1f}mm", fontsize=11)
        o = f"{args.out}_cmp_{fr:04d}.png"
        fig.tight_layout(); fig.savefig(o, dpi=100); plt.close(fig)
        print(f"   {fr:4d} | {d_fing:8.1f}          | {ang:6.1f}     | {d_seat:8.1f}        | {gsi}")

    if missing:
        print(f"[cmp] WARNING: {missing} GS frames missing — is the GS dump rendered, and is "
              f"--gs-offset right? (offset 0 assumes --grasped-only, i.e. no preamble)")
    print(f"[cmp] wrote {len(frames)} composites -> {args.out}_cmp_*.png")


if __name__ == "__main__":
    main()
