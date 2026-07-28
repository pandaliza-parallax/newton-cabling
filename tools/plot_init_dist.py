#!/usr/bin/env python
"""Compare the INITIAL (frame-0) plug positions of the training vs held-out eval trajectories.

Left  : lateral start footprint (socket-frame x-z plane) — spatial coverage.
Right : distribution of start insertion depth (socket-frame y; mouth=0, seat=+11.8mm).

Answers "does the held-out eval start distribution match training?" — which matters because the
closed-loop eval turned out to be extremely sensitive to the start pose.

  .venv/bin/python tools/plot_init_dist.py
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"          # chart surface
TRAIN = "#2a78d6"            # categorical slot 1 (blue)
EVAL = "#1baf7a"             # categorical slot 2 (aqua)
INK, MUTED, GRID = "#1a1a19", "#6b6b68", "#e5e5e2"
SEAT_MM = 11.8               # full seat depth


def frame0(pattern):
    pts = [np.load(f)[0, :3] for f in sorted(glob.glob(pattern))]
    return np.asarray(pts) * 1000.0          # m -> mm, socket frame [x, y(insert), z]


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
    tr = frame0("seated_traj/ep_*/plug_traj.npy")
    ev = frame0("eval_traj_v2/ep_*/plug_traj.npy")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE)
    fig.suptitle("Initial plug position — training vs held-out eval",
                 color=INK, fontsize=13, fontweight="bold", x=0.5, y=0.98)

    # ── LEFT: lateral start footprint ────────────────────────────────────────
    style(ax1)
    ax1.scatter(tr[:, 0], tr[:, 2], s=26, c=TRAIN, alpha=0.45,
                edgecolors=SURFACE, linewidths=0.5, label=f"train  (n={len(tr)})")
    ax1.scatter(ev[:, 0], ev[:, 2], s=34, c=EVAL, alpha=0.95,
                edgecolors=SURFACE, linewidths=0.8, label=f"eval  (n={len(ev)})")
    ax1.axhline(0, color=MUTED, lw=1, alpha=0.5, zorder=0)
    ax1.axvline(0, color=MUTED, lw=1, alpha=0.5, zorder=0)
    ax1.set_xlabel("lateral x  (mm)", color=MUTED, fontsize=10)
    ax1.set_ylabel("lateral z  (mm)", color=MUTED, fontsize=10)
    ax1.set_title("Lateral start offset (socket frame)", color=INK, fontsize=11, pad=8)
    ax1.set_aspect("equal", adjustable="datalim")
    leg = ax1.legend(frameon=False, fontsize=9.5, loc="upper right",
                     markerscale=1.5, handletextpad=0.5, borderpad=0.2)
    for t in leg.get_texts():
        t.set_color(INK)

    # ── RIGHT: distribution of start depth ───────────────────────────────────
    style(ax2)
    lo = min(tr[:, 1].min(), ev[:, 1].min()); hi = max(tr[:, 1].max(), ev[:, 1].max())
    bins = np.linspace(lo, hi, 17)          # n=100 eval: fewer bins => signal, not noise
    for d, c in ((tr, TRAIN), (ev, EVAL)):
        ax2.hist(d[:, 1], bins=bins, density=True, color=c, alpha=0.28)
        ax2.hist(d[:, 1], bins=bins, density=True, histtype="step", color=c, lw=2)
    ax2.set_xlim(lo - 2, 2.5)               # keep the mouth in view without dead space
    ax2.axvline(0, color=MUTED, lw=1.2, ls="--", alpha=0.8)
    top = ax2.get_ylim()[1]
    ax2.text(-0.6, top * 0.98, "socket mouth ", color=MUTED, fontsize=8.5,
             ha="right", va="top")
    ax2.set_xlabel("start insertion depth  y  (mm)   — negative = outside the socket",
                   color=MUTED, fontsize=10)
    ax2.set_ylabel("density", color=MUTED, fontsize=10)
    ax2.set_title("Start-depth distribution", color=INK, fontsize=11, pad=8)
    # direct labels w/ leader dots (relief rule: aqua is < 3:1 on this surface)
    for d, c, lab, yf in ((tr, TRAIN, "train", 0.50), (ev, EVAL, "eval", 0.88)):
        x = float(np.median(d[:, 1]))
        ax2.plot([x], [top * yf], "o", color=c, ms=7, mec=SURFACE, mew=1.2, zorder=5)
        ax2.text(x + 0.7, top * yf, f" {lab}", color=INK, fontsize=10,
                 fontweight="bold", ha="left", va="center")

    sub = (f"train  depth {tr[:,1].mean():+.1f}±{tr[:,1].std():.1f}mm | "
           f"lateral {np.linalg.norm(tr[:,[0,2]],axis=1).mean():.1f}mm      "
           f"eval  depth {ev[:,1].mean():+.1f}±{ev[:,1].std():.1f}mm | "
           f"lateral {np.linalg.norm(ev[:,[0,2]],axis=1).mean():.1f}mm")
    fig.text(0.5, 0.005, sub, color=MUTED, fontsize=8.5, ha="center")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = "/home/pandaliza/parallax/Difix3D/outputs_test/init_pos_train_vs_eval.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")
    print(f"train n={len(tr)}  depth mean={tr[:,1].mean():+.2f} std={tr[:,1].std():.2f} "
          f"range=[{tr[:,1].min():+.1f},{tr[:,1].max():+.1f}]")
    print(f"eval  n={len(ev)}  depth mean={ev[:,1].mean():+.2f} std={ev[:,1].std():.2f} "
          f"range=[{ev[:,1].min():+.1f},{ev[:,1].max():+.1f}]")


if __name__ == "__main__":
    main()
