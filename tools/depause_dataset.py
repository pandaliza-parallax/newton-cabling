#!/usr/bin/env python3
"""Dataset surgery: cut the expert's motionless pause frames from rendered episodes.

The drmix_1000 expert parks motionless at the pre-dock standoff (stable_steps +
servo settle) before pushing. BC learns "output ~zero" at exactly the state the
policy must push through -> closed-loop stall/bounce. This tool removes interior
low-motion runs (keeping short transitions) and RECOMPUTES action labels across
the cuts with the exact dump_episode() convention:
    action[i] = [p[k+1]-p[k] (base frame), rotvec(q[k]^-1 * q[k+1]), grip[k+1]]
so no zero-action pause labels survive. Retreat/recovery frames are never cut.

Measured signature (2026-08-19): there are no flat parks — the servo decelerates
smoothly through the standoff (down to ~0.02mm/frame) and back up, so a fixed
"cut runs below eps" rule barely fires. The surgery is ARC-LENGTH RESAMPLING:
keep a frame only once accumulated motion since the last kept frame reaches
--floor-mm (or --floor-deg of rotation). Slow valleys/tails compress, cruise
segments pass through untouched, and every recomputed action magnitude lands in
[floor, floor + one original step] — in-distribution by construction.
Protected (always kept): frame 0, RETREAT(4) frames, the last --term-pad frames
(the terminal "stop at seat" signal), and any gripper-value change.

Usage:
    depause_dataset.py --analyze [--verbose] [--eps chunk_00/ep_0000 ...]
    depause_dataset.py --apply   [--eps ...]     # writes OUT_ROOT, symlinks frames
"""
import argparse
import glob
import json
import math
import os

import numpy as np

DIFIX = os.path.expanduser("~/parallax/data/vla_train/drmix_1000_final_difix")
RAW = os.path.expanduser("~/parallax/data/vla_train/drmix_1000")
OUT = os.path.expanduser("~/parallax/data/vla_train/drmix_1000_depaused")

PH_NAMES = {0: "SETTLE", 1: "ALIGN", 2: "PUSH", 3: "HOLD", 4: "RETREAT"}


def rot6d_to_R(r6):
    c0, c1 = np.asarray(r6[:3], float), np.asarray(r6[3:6], float)
    c0 = c0 / np.linalg.norm(c0)
    c1 = c1 - c0 * np.dot(c0, c1)
    c1 = c1 / np.linalg.norm(c1)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=1)


def R_to_quat_wxyz(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w)])
    # w ~ 0 never happens here (wrist stays far from 180deg from base); fall back via largest diag
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4.0
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_to_rotvec(q):
    w = float(max(-1.0, min(1.0, q[0])))
    ang = 2.0 * math.acos(w)
    if ang > math.pi:
        ang -= 2.0 * math.pi
    s = math.sqrt(max(1e-12, 1.0 - w * w))
    if s < 1e-8:
        return np.zeros(3)
    return (np.asarray(q[1:4], float) / s) * ang


def keep_mask(act, ph, floor_mm, floor_deg, term_pad):
    """Arc-length resample: keep frame t once accumulated |dpos|/|drot| since the
    last kept frame reaches the floor. RETREAT frames, the terminal pad, frame 0,
    and gripper changes are always kept."""
    T = len(act)
    keep = np.zeros(T, bool)
    keep[0] = True
    acc_p = np.zeros(3)
    acc_r = 0.0
    for t in range(1, T):
        acc_p += act[t - 1, :3]                    # delta from frame t-1 to t
        acc_r += float(np.linalg.norm(act[t - 1, 3:6]))
        protected = (ph[t] == 4) or (t >= T - term_pad) or (act[t - 1, 6] != act[t, 6])
        if protected or np.linalg.norm(acc_p) * 1000.0 >= floor_mm \
                or math.degrees(acc_r) >= floor_deg:
            keep[t] = True
            acc_p[:] = 0.0
            acc_r = 0.0
    return keep


ROT_DEADBAND = 2e-3  # rad. The original dump's w-clamp floored sub-~0.11deg/frame
# rotations (non-unit sim quats pushed qrel w past 1) to exact zero; reproduce that
# convention so recomputed labels keep the same near-dead drot distribution.


def recompute(states, keep):
    """Slice states by keep mask and rebuild actions with dump_episode()'s exact convention."""
    idx = np.flatnonzero(keep)
    st = states[idx]
    P = st[:, :3].astype(float)
    G = st[:, 9].astype(float)
    quats = [R_to_quat_wxyz(rot6d_to_R(s[3:9])) for s in st]
    acts = np.zeros((len(idx), 7), np.float32)
    for k in range(len(idx) - 1):
        q, q1 = quats[k], quats[k + 1]
        qrel = quat_mul_wxyz([q[0], -q[1], -q[2], -q[3]], q1)
        rv = quat_to_rotvec(qrel)
        acts[k, :3] = P[k + 1] - P[k]
        acts[k, 3:6] = rv if np.linalg.norm(rv) >= ROT_DEADBAND else 0.0
        acts[k, 6] = G[k + 1]
    acts[-1, 6] = G[-1]
    return idx, st.astype(np.float32), acts


def process(rel, floor_mm, floor_deg, term_pad, apply_out=None):
    ep = os.path.join(DIFIX, rel)
    states = np.load(os.path.join(ep, "state.npy"))
    act = np.load(os.path.join(ep, "action.npy"))
    ph = np.load(os.path.join(RAW, rel, "phase.npy"))
    keep = keep_mask(act, ph, floor_mm, floor_deg, term_pad)
    idx, st2, act2 = recompute(states, keep)
    stats = {"rel": rel, "T": len(states), "kept": int(keep.sum()),
             "cut_by_phase": {PH_NAMES[p]: int(((~keep) & (ph == p)).sum()) for p in np.unique(ph)},
             "max_new_dpos_mm": float(np.linalg.norm(act2[:-1, :3], axis=1).max() * 1000.0) if len(act2) > 1 else 0.0,
             "retreat": bool((ph == 4).any())}
    if apply_out:
        od = os.path.join(apply_out, rel)
        os.makedirs(os.path.join(od, "image"), exist_ok=True)
        os.makedirs(os.path.join(od, "wrist_image"), exist_ok=True)
        np.save(os.path.join(od, "state.npy"), st2)
        np.save(os.path.join(od, "action.npy"), act2)
        rph = np.load(os.path.join(ep, "phase.npy"))
        np.save(os.path.join(od, "phase.npy"), rph[idx])
        meta = json.load(open(os.path.join(ep, "meta.json")))
        meta.update({"T": int(len(idx)), "depaused": {
            "from": ep, "orig_T": int(len(states)), "floor_mm": floor_mm,
            "floor_deg": floor_deg, "term_pad": term_pad,
            "note": "arc-length resampled; actions recomputed across cuts"}})
        json.dump(meta, open(os.path.join(od, "meta.json"), "w"), indent=2)
        for sub in ("image", "wrist_image"):
            for k, t in enumerate(idx):
                dst = os.path.join(od, sub, f"frame_{k:04d}.png")
                if not os.path.lexists(dst):
                    os.symlink(os.path.join(ep, sub, f"frame_{t:04d}.png"), dst)
    return stats, keep, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--eps", nargs="*", default=None, help="episode rel paths (default: all)")
    ap.add_argument("--floor-mm", type=float, default=0.2)
    ap.add_argument("--floor-deg", type=float, default=0.2)
    ap.add_argument("--term-pad", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="print per-episode timelines")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--difix-root", default=None, help="override the difixed-episode root")
    ap.add_argument("--raw-root", default=None,
                    help="override the Stage-A root (per-episode phase.npy, frame-aligned)")
    args = ap.parse_args()
    global DIFIX, RAW
    if args.difix_root:
        DIFIX = os.path.expanduser(args.difix_root)
    if args.raw_root:
        RAW = os.path.expanduser(args.raw_root)

    eps = args.eps or sorted(os.path.relpath(p, DIFIX)
                             for p in glob.glob(os.path.join(DIFIX, "chunk_*", "ep_*")))
    tot_T = tot_kept = 0
    cut_by_phase = {}
    max_dpos = 0.0
    for rel in eps:
        stats, keep, ph = process(rel, args.floor_mm, args.floor_deg,
                                  args.term_pad, apply_out=(args.out if args.apply else None))
        tot_T += stats["T"]
        tot_kept += stats["kept"]
        max_dpos = max(max_dpos, stats["max_new_dpos_mm"])
        for k, v in stats["cut_by_phase"].items():
            cut_by_phase[k] = cut_by_phase.get(k, 0) + v
        if args.verbose:
            line = "".join(("." if keep[t] else "x") if ph[t] != 4 else "R" for t in range(len(keep)))
            print(f"{rel}  T={stats['T']} kept={stats['kept']} "
                  f"({100 * (1 - stats['kept'] / stats['T']):.0f}% cut) maxnew={stats['max_new_dpos_mm']:.2f}mm"
                  f"{' RETREAT' if stats['retreat'] else ''}")
            print("   " + line)
    print(f"\n[depause] {len(eps)} eps: {tot_T} -> {tot_kept} frames "
          f"({100 * (1 - tot_kept / tot_T):.1f}% cut), cut by phase: {cut_by_phase}, "
          f"max recomputed |dpos| {max_dpos:.2f}mm")
    if args.apply:
        print(f"[depause] written under {args.out}")


if __name__ == "__main__":
    main()
