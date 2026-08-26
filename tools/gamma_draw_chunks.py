#!/usr/bin/env python3
"""Per-episode gamma draw over rendered drmix chunks (exact post-hoc derivation).

Draws gamma from {1.0, 1.2, 1.4, 1.8} per episode (deterministic in episode path),
converts stored 1.8-baked PNGs via out = png**(1.8/g), copies npys/meta, and records
the draw in meta.json. Skips episodes already converted; safe to re-run as chunks land.
"""
import argparse
import hashlib
import json
import os
import shutil
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

GAMMAS = [1.0, 1.2, 1.4, 1.8]


def convert_ep(job):
    src, dst, seed = job
    key = int(hashlib.sha1((str(seed) + src).encode()).hexdigest()[:8], 16)
    g = GAMMAS[key % len(GAMMAS)]
    exp = 1.8 / g
    os.makedirs(dst, exist_ok=True)
    for cam in ("image", "wrist_image"):
        od = os.path.join(dst, cam)
        os.makedirs(od, exist_ok=True)
        for f in os.listdir(os.path.join(src, cam)):
            if g == 1.8:
                shutil.copy2(os.path.join(src, cam, f), os.path.join(od, f))
            else:
                a = np.asarray(Image.open(os.path.join(src, cam, f)), np.float32) / 255.0
                Image.fromarray((np.clip(a, 0, 1) ** exp * 255.0 + 0.5).astype(np.uint8)
                                ).save(os.path.join(od, f))
    for f in ("state.npy", "action.npy", "phase.npy"):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    m = json.load(open(os.path.join(src, "meta.json")))
    m["render_gamma_effective"] = g
    json.dump(m, open(os.path.join(dst, "meta.json"), "w"), indent=1)
    return os.path.basename(dst), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    jobs = []
    for ck in sorted(os.listdir(args.src)):
        cs = os.path.join(args.src, ck)
        if not (ck.startswith("chunk_") and os.path.isdir(cs)):
            continue
        for ep in sorted(os.listdir(cs)):
            s = os.path.join(cs, ep)
            d = os.path.join(args.dst, ck, ep)
            if not (ep.startswith("ep_") and os.path.isfile(os.path.join(s, "state.npy"))):
                continue
            if os.path.isfile(os.path.join(d, "meta.json")):
                continue
            jobs.append((s, d, args.seed))
    print(f"[gamma] {len(jobs)} episodes to convert")
    from collections import Counter
    dist = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for ep, g in ex.map(convert_ep, jobs):
            dist[g] += 1
    print(f"[gamma] done; draw distribution: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
