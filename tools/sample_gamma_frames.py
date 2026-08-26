#!/usr/bin/env python3
"""Randomly sample GS-rendered frames and produce them at a different RENDER_GAMMA.

No re-render needed: the datagen renderer applies gamma POST-render
(rgb ** (1/gamma), scripts/record_sbot_scene_gs_cable.py), and the stored PNGs
carry --baked-gamma (1.8 for every servoD-era set). A frame at target gamma g is
therefore exactly  png ** (baked/g)  -- identical to re-rendering with
RENDER_GAMMA=g, modulo 8-bit quantization.

Sources are the RAW trajectory roots; each is mapped to its rendered ``<root>_gs``
sibling (subset layouts like clean/grip45/... are handled). Output filenames keep
full provenance: <root>__<subset>__<ep>__<cam>__<frame>__g<gamma>.png

    .venv/bin/python tools/sample_gamma_frames.py \
        --sources /home/pandaliza/parallax/data/vla_train/yaw3_sample \
                  /home/pandaliza/parallax/data/vla_train/servoD_roll150_500 \
        --n 50 --gamma 1.2 --out /home/pandaliza/parallax/data/vla_train/gamma1p2_frames
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
from PIL import Image


def find_frames(raw_root: str, cams: list[str]) -> list[tuple[str, str, str, str, str]]:
    """(root_name, subset, ep, cam, png_path) for every rendered frame under <root>_gs."""
    gs_root = raw_root.rstrip("/") + "_gs"
    if not os.path.isdir(gs_root):
        sys.exit(f"[sample] no rendered sibling for {raw_root} (expected {gs_root})")
    root_name = os.path.basename(raw_root.rstrip("/"))
    # subset dirs (clean/grip45/...) or flat ep_XXXX layout
    subsets = [d for d in sorted(os.listdir(gs_root))
               if os.path.isdir(os.path.join(gs_root, d)) and not d.startswith(("_", "ep_"))]
    layouts = [(s, os.path.join(gs_root, s)) for s in subsets] or [("", gs_root)]
    out = []
    for subset, sdir in layouts:
        for ep in sorted(d for d in os.listdir(sdir)
                         if d.startswith("ep_") and os.path.isdir(os.path.join(sdir, d))):
            for cam in cams:
                cdir = os.path.join(sdir, ep, cam)
                if not os.path.isdir(cdir):
                    continue
                out += [(root_name, subset, ep, cam, os.path.join(cdir, f))
                        for f in sorted(os.listdir(cdir)) if f.endswith(".png")]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sources", nargs="+", required=True, help="RAW trajectory roots")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=1.2, help="target RENDER_GAMMA")
    ap.add_argument("--baked-gamma", type=float, default=1.8,
                    help="gamma the source PNGs were rendered with (1.8 = servoD era)")
    ap.add_argument("--cams", nargs="+", default=["image"],
                    choices=["image", "wrist_image"], help="camera dirs to sample from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = []
    for src in args.sources:
        frames = find_frames(src, args.cams)
        print(f"[sample] {src}: {len(frames)} rendered frames")
        pool += frames
    if len(pool) < args.n:
        sys.exit(f"[sample] pool ({len(pool)}) smaller than --n {args.n}")

    picks = random.Random(args.seed).sample(pool, args.n)
    os.makedirs(args.out, exist_ok=True)
    exp = args.baked_gamma / args.gamma
    for root_name, subset, ep, cam, path in sorted(picks):
        a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
        a = np.clip(a, 0.0, 1.0) ** exp
        frame = os.path.splitext(os.path.basename(path))[0]
        parts = [root_name] + ([subset] if subset else []) + [ep, cam, frame, f"g{args.gamma:g}"]
        name = "__".join(parts) + ".png"
        Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(os.path.join(args.out, name))
    print(f"[sample] wrote {args.n} frames at gamma {args.gamma:g} "
          f"(from baked {args.baked_gamma:g}) -> {args.out}")


if __name__ == "__main__":
    main()
