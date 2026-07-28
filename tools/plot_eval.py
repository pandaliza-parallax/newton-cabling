"""Plot eval rollout motion from evalrun/ep_*/trace.npz (+ result.json).

Per episode -> evalrun/ep_XXXX/motion.png:
  top view (x-y) of eef + plug paths over the table | side view (y-z) vs the tabletop line
  socket depth y_sock vs step (seat threshold)      | per-step |dpos| + gripper command
Aggregate -> evalrun/summary.png + a text table (seat / clean-seat / penetration / jerk).

    .venv/bin/python tools/plot_eval.py [--run evalrun]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_episode(ep_dir):
    t = np.load(os.path.join(ep_dir, "trace.npz"))
    res = json.load(open(os.path.join(ep_dir, "result.json")))
    eef, plug = t["eef_w"], t["plug_w"]
    lo, hi, top, jack = t["tbl_lo"], t["tbl_hi"], float(t["tbl_top"]), t["jack"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    a = ax[0, 0]  # top view
    a.add_patch(plt.Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                              fc="tan", alpha=0.4, ec="k"))
    a.plot(eef[:, 0], eef[:, 1], "b.-", ms=2, lw=0.8, label="eef")
    a.plot(plug[:, 0], plug[:, 1], "r.-", ms=2, lw=0.8, label="plug")
    a.plot(*jack[:2], "k*", ms=14, label="jack")
    a.plot(eef[0, 0], eef[0, 1], "bo", ms=8, mfc="none")
    a.set_title("top view (x-y)"); a.set_aspect("equal"); a.legend(); a.grid(alpha=0.3)

    a = ax[0, 1]  # side view
    a.axhline(top, color="k", lw=2, label=f"tabletop z={top:.3f}")
    a.axhspan(top - 0.1, top, color="tan", alpha=0.4)
    a.plot(eef[:, 1], eef[:, 2], "b.-", ms=2, lw=0.8, label="eef")
    a.plot(plug[:, 1], plug[:, 2], "r.-", ms=2, lw=0.8, label="plug")
    a.plot(jack[1], jack[2], "k*", ms=14)
    a.set_title("side view (y-z) — below the line = INSIDE the table")
    a.legend(); a.grid(alpha=0.3)

    a = ax[1, 0]  # socket depth
    ys = t["y_sock"] * 1000
    a.plot(ys, "r-")
    a.axhline(11.0, color="g", ls="--", label="seat (+11mm)")
    a.axhline(0.0, color="k", lw=0.5)
    a.set_title(f"socket depth y_sock (mm) — final {res['final_y_sock_mm']}mm")
    a.set_xlabel("step"); a.legend(); a.grid(alpha=0.3)

    a = ax[1, 1]  # action magnitude + grip
    step_mm = np.linalg.norm(t["dpos"], axis=1) * 1000
    a.plot(step_mm, "b-", lw=0.8, label="|dpos| mm/step")
    a2 = a.twinx()
    a2.plot(t["grip"], "g-", lw=1.2, label="grip")
    a2.set_ylim(-0.05, 1.05)
    a.set_title(f"actions — mean {step_mm.mean():.1f}mm max {step_mm.max():.1f}mm "
                f"jerk {res['jerk_mm_mean']}mm")
    a.set_xlabel("step"); a.grid(alpha=0.3); a.legend(loc="upper left"); a2.legend(loc="upper right")

    verdict = "CLEAN SEAT" if res.get("clean_seat") else ("seated (DIRTY)" if res["seated"] else "FAILED")
    fig.suptitle(f"{os.path.basename(ep_dir)} — {verdict} | plug pen {res['plug_pen_mm']}mm "
                 f"({res['plug_pen_frames']} frames) | path {res['path_len_mm']}mm")
    fig.tight_layout()
    out = os.path.join(ep_dir, "motion.png")
    fig.savefig(out, dpi=100); plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="evalrun")
    args = ap.parse_args()
    eps = sorted(d for d in glob.glob(os.path.join(args.run, "ep_*")) if os.path.isdir(d)
                 and os.path.isfile(os.path.join(d, "trace.npz")))
    if not eps:
        raise SystemExit(f"no traces under {args.run}")
    results = []
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for d in eps:
        r = plot_episode(d)
        results.append((os.path.basename(d), r))
        t = np.load(os.path.join(d, "trace.npz"))
        c = "g" if r.get("clean_seat") else ("orange" if r["seated"] else "r")
        ax[0].plot(t["y_sock"] * 1000, color=c, lw=0.8, alpha=0.7)
        ax[1].plot(t["plug_w"][:, 1], t["plug_w"][:, 2], color=c, lw=0.8, alpha=0.7)
    ax[0].axhline(11, color="g", ls="--"); ax[0].set_title("y_sock (mm) all episodes"); ax[0].grid(alpha=0.3)
    tt = float(np.load(os.path.join(eps[0], "trace.npz"))["tbl_top"])
    ax[1].axhline(tt, color="k", lw=2); ax[1].set_title("plug side view (y-z) all episodes"); ax[1].grid(alpha=0.3)
    fig.suptitle("green=clean seat  orange=seated but dirty  red=failed")
    fig.tight_layout(); fig.savefig(os.path.join(args.run, "summary.png"), dpi=110)

    print(f"{'episode':10s} {'seated':7s} {'clean':6s} {'seat@':6s} {'y_end':>8s} {'pen_mm':>7s} "
          f"{'path':>7s} {'jerk':>6s}")
    ns = nc = 0
    for name, r in results:
        ns += r["seated"]; nc += r.get("clean_seat", False)
        print(f"{name:10s} {str(r['seated']):7s} {str(r.get('clean_seat', False)):6s} "
              f"{r['seat_step']:6d} {r['final_y_sock_mm']:8.1f} {r['plug_pen_mm']:7.1f} "
              f"{r['path_len_mm']:7.0f} {r['jerk_mm_mean']:6.2f}")
    print(f"\nSEAT {ns}/{len(results)}  CLEAN {nc}/{len(results)}  "
          f"-> per-episode motion.png + {args.run}/summary.png")


if __name__ == "__main__":
    main()
