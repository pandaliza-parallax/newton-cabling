#!/usr/bin/env python3
"""Split Stage-A episodes at their perturbation-kick windows.

gen_trajectories --kick-prob episodes contain windows where the executed action
was an injected off-path kick (marked in kick.npy). Those frames must never
become training labels — Stage B recomputes actions from pose deltas, so a kick
frame would teach "drift off the path". This tool splits each episode into
kick-free segments (the kick frames themselves are dropped):

    [nominal approach][KICK][recovery + insert]  ->  ep_A (pre_kick) + ep_B (post_kick)

Every segment is a self-consistent episode: within it all pose deltas are the
controller's own (corrective) motion, and the post-kick segment starts at the
displaced pose — an off-nominal-start recovery demonstration with real cable
deformation. Segments shorter than --min-frames are dropped. Episodes without
kick.npy (or with no kicks fired) pass through as directory symlinks.

Output dirs are renumbered ep_0000.. sequentially and feed Stage B unchanged.

Usage:
    split_kick_episodes.py --in <chunk_dir> --out <chunk_dir_split> [--min-frames 25]
"""
import argparse
import glob
import json
import os

import numpy as np

FRAME_ARRAYS = ("eef_traj", "conn_traj", "face_traj", "tips_traj", "rods_traj",
                "actions_policy", "state_sim", "phase")


def segments_from_kick(kick, min_frames):
    """Kick-free [start, end) spans, longest-first order preserved (chronological)."""
    T = len(kick)
    segs, s = [], None
    for t in range(T):
        if not kick[t] and s is None:
            s = t
        elif kick[t] and s is not None:
            segs.append((s, t))
            s = None
    if s is not None:
        segs.append((s, T))
    return [(a, b) for a, b in segs if b - a >= min_frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Stage-A chunk dir with ep_*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-frames", type=int, default=25)
    args = ap.parse_args()

    eps = sorted(glob.glob(os.path.join(args.inp, "ep_*")))
    os.makedirs(args.out, exist_ok=True)
    n_out = n_split = n_pass = n_dropped_segs = 0
    for ep in eps:
        kick_path = os.path.join(ep, "kick.npy")
        kick = np.load(kick_path) if os.path.exists(kick_path) else None
        if kick is None or not kick.any():
            dst = os.path.join(args.out, f"ep_{n_out:04d}")
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(ep), dst)
            n_out += 1
            n_pass += 1
            continue
        segs = segments_from_kick(kick, args.min_frames)
        raw_segs = segments_from_kick(kick, 1)
        n_dropped_segs += len(raw_segs) - len(segs)
        meta = json.load(open(os.path.join(ep, "meta.json")))
        n_split += 1
        for si, (a, b) in enumerate(segs):
            d = os.path.join(args.out, f"ep_{n_out:04d}")
            os.makedirs(d, exist_ok=True)
            for name in FRAME_ARRAYS:
                p = os.path.join(ep, f"{name}.npy")
                if os.path.exists(p):
                    np.save(os.path.join(d, f"{name}.npy"), np.load(p)[a:b])
            m = dict(meta)
            last = si == len(segs) - 1
            m.update({
                "frames": int(b - a),
                "success": bool(meta.get("success", False)) and last,
                "segment": {"parent": os.path.basename(ep), "index": si,
                            "of": len(segs), "span": [int(a), int(b)],
                            "kind": "pre_kick" if a == 0 else "post_kick"},
            })
            json.dump(m, open(os.path.join(d, "meta.json"), "w"), indent=2)
            n_out += 1
    print(f"[split] {len(eps)} eps in -> {n_out} out "
          f"({n_pass} passed through, {n_split} split, {n_dropped_segs} short segments dropped)")


if __name__ == "__main__":
    main()
