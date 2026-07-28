#!/usr/bin/env python
"""Overlay the training vs held-out eval plug TRAJECTORIES (socket frame).

Left  : insertion funnel — lateral offset vs depth. Shows the spatial path into the socket.
Right : insertion profile — depth vs time. Shows the temporal march from outside to seated.

Thin line per trajectory (500 train + 100 eval) forms a density cloud; the bold line is the
elementwise median path of each group.

  .venv/bin/python tools/plot_traj_overlay.py
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TRAIN, EVAL = "#2a78d6", "#1baf7a"        # validated categorical slots 1 & 2
INK, MUTED, GRID = "#1a1a19", "#6b6b68", "#e5e5e2"
SEAT_MM = 11.8


def load(pattern):
    T = [np.load(f)[:, :3] * 1000.0 for f in sorted(glob.glob(pattern))]  # (F,3) mm, socket frame
    return np.stack(T)                                                    # (N,F,3)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    tr = load("seated_traj/ep_*/plug_traj.npy")
    ev = load("eval_traj_v2/ep_*/plug_traj.npy")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    fig.suptitle("Plug trajectories — training vs held-out eval  (socket frame)",
                 color=INK, fontsize=13, fontweight="bold", y=0.98)

    # ── LEFT: insertion funnel (lateral offset vs depth) ─────────────────────
    style(ax1)
    for D, c, a in ((tr, TRAIN, 0.05), (ev, EVAL, 0.16)):
        lat = np.linalg.norm(D[:, :, [0, 2]], axis=2)          # (N,F) lateral magnitude
        for i in range(len(D)):
            ax1.plot(D[i, :, 1], lat[i], color=c, lw=0.6, alpha=a, solid_capstyle="round")
    for D, c, lab in ((tr, TRAIN, "train"), (ev, EVAL, "eval")):
        lat = np.linalg.norm(D[:, :, [0, 2]], axis=2)
        ax1.plot(np.median(D[:, :, 1], axis=0), np.median(lat, axis=0),
                 color=c, lw=2.5, label=f"{lab}  (n={len(D)})", zorder=5,
                 path_effects=None, solid_capstyle="round")
    ax1.axvline(0, color=MUTED, lw=1.2, ls="--", alpha=0.8)
    ax1.axvline(SEAT_MM, color=MUTED, lw=1.2, ls=":", alpha=0.8)
    ax1.set_xlim(-32, 14)          # focus the funnel; a few over-insertion outliers fall outside
    ax1.set_ylim(0, 12)
    ax1.set_xlabel("insertion depth  y  (mm)      mouth = 0,  seat = +11.8",
                   color=MUTED, fontsize=10)
    ax1.set_ylabel("lateral offset  (mm)", color=MUTED, fontsize=10)
    ax1.set_title("Insertion funnel — lateral offset vs depth", color=INK, fontsize=11, pad=8)
    leg = ax1.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)

    # ── RIGHT: depth vs time ────────────────────────────────────────────────
    style(ax2)
    for D, c, a in ((tr, TRAIN, 0.05), (ev, EVAL, 0.16)):
        for i in range(len(D)):
            ax2.plot(D[i, :, 1], color=c, lw=0.6, alpha=a, solid_capstyle="round")
    for D, c, lab, yoff in ((tr, TRAIN, "train", -3.5), (ev, EVAL, "eval", 2.5)):
        med = np.median(D[:, :, 1], axis=0)
        ax2.plot(med, color=c, lw=2.5, zorder=5, solid_capstyle="round")
        ax2.text(len(med) * 0.72, med[int(len(med) * 0.72)] + yoff, lab,
                 color=INK, fontsize=10, fontweight="bold")
    ax2.axhline(0, color=MUTED, lw=1.2, ls="--", alpha=0.8)
    ax2.axhline(SEAT_MM, color=MUTED, lw=1.2, ls=":", alpha=0.8)
    ax2.text(len(tr[0]) * 0.99, SEAT_MM, " seat", color=MUTED, fontsize=8.5, va="bottom", ha="right")
    ax2.text(len(tr[0]) * 0.99, 0, " mouth", color=MUTED, fontsize=8.5, va="bottom", ha="right")
    ax2.set_ylim(-34, 16)          # same focus; outliers above the seat clipped
    ax2.set_xlabel("trajectory frame", color=MUTED, fontsize=10)
    ax2.set_ylabel("insertion depth  y  (mm)", color=MUTED, fontsize=10)
    ax2.set_title("Insertion profile — depth vs time", color=INK, fontsize=11, pad=8)

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    out = "/home/pandaliza/parallax/Difix3D/outputs_test/traj_overlay_train_vs_eval.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")
    for D, lab in ((tr, "train"), (ev, "eval")):
        lat0 = np.linalg.norm(D[:, 0, [0, 2]], axis=1)
        print(f"{lab:6s} n={len(D):4d}  start y={D[:,0,1].mean():+.1f}±{D[:,0,1].std():.1f}mm  "
              f"start lat={lat0.mean():.1f}mm  final y={D[:,-1,1].mean():+.1f}mm")


if __name__ == "__main__":
    main()
