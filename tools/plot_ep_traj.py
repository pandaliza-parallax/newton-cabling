#!/usr/bin/env python
"""Per-episode trajectory over time: GT (teacher) vs POLICY rollout, position AND rotation.

Both are plotted in the SOCKET frame (the frame plug_traj.npy uses), so they're directly
comparable. Rotation is shown as a rotation vector (rx, ry, rz, degrees) — the axis-angle
of the plug's orientation relative to the socket.

  .venv/bin/python tools/plot_ep_traj.py --ep 9 --run evalrun_v2
"""
import argparse, os
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
GT, POL = "#2a78d6", "#1baf7a"          # validated categorical slots 1 & 2
INK, MUTED, GRID = "#1a1a19", "#6b6b68", "#e5e5e2"


CONN_RPY = (-90.0, 0.0, 0.0)     # --conn-rpy: plug-splat calibration baked into the render


def _rot(q_wxyz):
    q = np.asarray(q_wxyz, float).reshape(-1, 4)
    return R.from_quat(np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=1))


def rotvec_deg(q_wxyz):
    """(N,4) wxyz -> (N,3) rotation vector in degrees."""
    return _rot(q_wxyz).as_rotvec(degrees=True)


def gt_rotvec_deg(q_wxyz):
    """GT plug_traj quat -> the SAME convention the trace records.

    conn_world_q = jack_q ⊗ s_q ⊗ conn_align, and the trace stores jinv ⊗ cq_p = s_q ⊗ conn_align.
    So the GT socket quat must be post-multiplied by conn_align to be comparable.
    """
    align = R.from_euler("xyz", CONN_RPY, degrees=True)
    return (_rot(q_wxyz) * align).as_rotvec(degrees=True)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=9)
    ap.add_argument("--run", default="evalrun_v2")
    ap.add_argument("--traj-dir", default="eval_traj_v2")
    args = ap.parse_args()
    name = f"ep_{args.ep:04d}"

    gt = np.load(f"{args.traj_dir}/{name}/plug_traj.npy")        # (F,7) socket frame
    gt_p, gt_r = gt[:, :3] * 1000.0, gt_rotvec_deg(gt[:, 3:7])   # same convention as the trace
    tr = np.load(f"{args.run}/{name}/trace.npz")
    if "plug_s" not in tr.files:
        raise SystemExit(f"{args.run}/{name}/trace.npz has no plug_s/plug_sq — re-run this episode "
                         f"with the updated scripts/record_sbot_scene_gs.py to record orientation.")
    po_p, po_r = np.asarray(tr["plug_s"]) * 1000.0, rotvec_deg(tr["plug_sq"])

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), facecolor=SURFACE)
    fig.suptitle(f"{name} — plug trajectory in socket frame:  GT (teacher)  vs  policy rollout",
                 color=INK, fontsize=13, fontweight="bold", y=0.98)

    chans = [("x", gt_p[:, 0], po_p[:, 0], "mm"), ("y  (insertion)", gt_p[:, 1], po_p[:, 1], "mm"),
             ("z", gt_p[:, 2], po_p[:, 2], "mm"),
             ("rx", gt_r[:, 0], po_r[:, 0], "deg"), ("ry", gt_r[:, 1], po_r[:, 1], "deg"),
             ("rz", gt_r[:, 2], po_r[:, 2], "deg")]
    for ax, (lab, g, p, unit) in zip(axes.ravel(), chans):
        style(ax)
        ax.plot(g, color=GT, lw=2, label="GT (teacher)")
        ax.plot(p, color=POL, lw=2, label="policy")
        ax.set_title(f"{lab}  ({unit})", color=INK, fontsize=10, pad=6)
        ax.set_xlabel("frame / step", color=MUTED, fontsize=8.5)
    axes[0, 1].axhline(11.8, color=MUTED, lw=1, ls=":", alpha=0.8)   # seat on the y panel
    axes[0, 1].axhline(0, color=MUTED, lw=1, ls="--", alpha=0.6)

    leg = axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
    for t in leg.get_texts():
        t.set_color(INK)

    fig.text(0.5, 0.005,
             f"GT {len(gt_p)} frames  |  policy {len(po_p)} steps   "
             f"(different lengths: the policy solves it itself, it does not replay the teacher)",
             color=MUTED, fontsize=8.5, ha="center")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = f"/home/pandaliza/parallax/Difix3D/outputs_test/{name}_traj_gt_vs_policy.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")
    print(f"  GT     : {len(gt_p)} frames  start rot=({gt_r[0,0]:+.1f},{gt_r[0,1]:+.1f},{gt_r[0,2]:+.1f})deg "
          f"end rot=({gt_r[-1,0]:+.1f},{gt_r[-1,1]:+.1f},{gt_r[-1,2]:+.1f})deg")
    print(f"  policy : {len(po_p)} steps   start rot=({po_r[0,0]:+.1f},{po_r[0,1]:+.1f},{po_r[0,2]:+.1f})deg "
          f"end rot=({po_r[-1,0]:+.1f},{po_r[-1,1]:+.1f},{po_r[-1,2]:+.1f})deg")


if __name__ == "__main__":
    main()
