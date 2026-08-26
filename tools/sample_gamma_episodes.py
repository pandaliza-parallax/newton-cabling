#!/usr/bin/env python3
"""Randomly sample FULL rendered episodes and re-emit them at a different RENDER_GAMMA.

Episode-level counterpart of sample_gamma_frames.py, same math: the renderer
applies gamma POST-render, so  png ** (baked/target)  is pixel-identical to a
re-render at RENDER_GAMMA=target (modulo 8-bit rounding). Every frame of image/
and wrist_image/ is converted; state/action/phase npy files and meta.json are
copied through untouched (gamma changes pixels only).

Output is a flat openpi-layout tree (ep_0000..ep_NNNN, renumbered) usable by
difix_datagen.py / datagen_to_lerobot.py. Each meta.json gains "gamma_resample"
provenance, and _MANIFEST.txt maps output eps to their sources.

    .venv/bin/python tools/sample_gamma_episodes.py \
        --sources /home/pandaliza/parallax/data/vla_train/yaw3_sample \
                  /home/pandaliza/parallax/data/vla_train/servoD_roll150_500 \
        --n 50 --gamma 1.2 --out /home/pandaliza/parallax/data/vla_train/gamma1p2_eps
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image


def find_episodes(raw_root: str) -> list[tuple[str, str]]:
    """(label, rendered_ep_dir) for every complete episode under <root>_gs."""
    gs_root = raw_root.rstrip("/") + "_gs"
    if not os.path.isdir(gs_root):
        sys.exit(f"[sample] no rendered sibling for {raw_root} (expected {gs_root})")
    root_name = os.path.basename(raw_root.rstrip("/"))
    subsets = [d for d in sorted(os.listdir(gs_root))
               if os.path.isdir(os.path.join(gs_root, d)) and not d.startswith(("_", "ep_"))]
    layouts = [(s, os.path.join(gs_root, s)) for s in subsets] or [("", gs_root)]
    eps = []
    for subset, sdir in layouts:
        for ep in sorted(d for d in os.listdir(sdir)
                         if d.startswith("ep_") and os.path.isdir(os.path.join(sdir, d))):
            d = os.path.join(sdir, ep)
            if os.path.isfile(os.path.join(d, "state.npy")):
                eps.append((f"{root_name}/{subset}/{ep}" if subset else f"{root_name}/{ep}", d))
    return eps


def convert_episode(job: tuple[str, str, str, float, float, float]) -> str:
    src, dst, label, exp, baked, target = job
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if f in ("image", "wrist_image"):
            od = os.path.join(dst, f)
            os.makedirs(od, exist_ok=True)
            for fr in os.listdir(p):
                a = np.asarray(Image.open(os.path.join(p, fr)).convert("RGB"),
                               np.float32) / 255.0
                a = np.clip(a, 0.0, 1.0) ** exp
                Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(os.path.join(od, fr))
        elif f == "meta.json":
            meta = json.load(open(p))
            meta["gamma_resample"] = {"source": label, "baked_gamma": baked,
                                      "target_gamma": target, "exp": exp}
            json.dump(meta, open(os.path.join(dst, f), "w"), indent=1)
        elif os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, f))
    return label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sources", nargs="+", required=True, help="RAW trajectory roots")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=1.2, help="target RENDER_GAMMA")
    ap.add_argument("--baked-gamma", type=float, default=1.8,
                    help="gamma the source PNGs were rendered with (1.8 = servoD era)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pool = []
    for src in args.sources:
        eps = find_episodes(src)
        print(f"[sample] {src}: {len(eps)} rendered episodes")
        pool += eps
    if len(pool) < args.n:
        sys.exit(f"[sample] pool ({len(pool)}) smaller than --n {args.n}")

    picks = sorted(random.Random(args.seed).sample(pool, args.n))
    os.makedirs(args.out, exist_ok=True)
    exp = args.baked_gamma / args.gamma
    jobs = [(src, os.path.join(args.out, f"ep_{i:04d}"), label, exp,
             args.baked_gamma, args.gamma)
            for i, (label, src) in enumerate(picks)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, label in enumerate(ex.map(convert_episode, jobs)):
            print(f"[sample] ep_{i:04d} <- {label}")

    with open(os.path.join(args.out, "_MANIFEST.txt"), "w") as fh:
        fh.write(f"gamma_resample: target {args.gamma:g} from baked {args.baked_gamma:g} "
                 f"(exp {exp:.6f}), seed {args.seed}\n")
        for i, (label, _) in enumerate(picks):
            fh.write(f"ep_{i:04d}  {label}\n")
    print(f"[sample] {args.n} full episodes at gamma {args.gamma:g} -> {args.out}")


if __name__ == "__main__":
    main()
