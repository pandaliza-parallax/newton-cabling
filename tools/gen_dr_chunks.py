#!/usr/bin/env python3
"""Chunked DR generation driver (Stage A of the DR campaign).

Runs gen_trajectories once per CHUNK with per-chunk draws (grip-from-head,
controller jitter, seed) on top of the per-episode/per-env DR the generator
samples itself. Servo plant + fixture everywhere; every Nth chunk is a
HARD/recovery chunk (jam->retreat knobs). Chunk manifest -> _CHUNKS.json.

    pilot:    .venv/bin/python tools/gen_dr_chunks.py --out .../drmix_pilot \
                  --chunks 5 --eps 10 --envs 16 --base-seed 3000
    campaign: .venv/bin/python tools/gen_dr_chunks.py --out .../drmix_1000 \
                  --chunks 20 --eps 50 --envs 32 --base-seed 4000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = [os.path.join(REPO, ".venv", "bin", "python"),
       "-m", "newton_cabling.scripted_controller.gen_trajectories"]

COMMON = [
    "--servo-plant", "--jack-fixture",
    "--connector-usd", "cad_rj45.usd", "--grasp-roll-180",
    "--grasp-roll-deg", "150",
    "--jack-yaw", "5",
    "--cable-tilt", "-5", "8",
    "--grasp-roll-jitter", "3",
    "--offset-z", "3",
    "--approach-jitter", "25", "35",
    "--offset-uniform-disk",
    "--stage", "4",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, required=True)
    ap.add_argument("--eps", type=int, required=True, help="episodes saved per chunk")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--base-seed", type=int, required=True)
    ap.add_argument("--hard-every", type=int, default=5,
                    help="every Nth chunk (0-indexed: c %% N == N-1) is a hard/recovery chunk")
    ap.add_argument("--start-chunk", type=int, default=0, help="resume from this chunk index")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "_CHUNKS.json")
    manifest = json.load(open(manifest_path)) if os.path.isfile(manifest_path) else {}

    for c in range(args.start_chunk, args.chunks):
        rng = np.random.default_rng(args.base_seed + c)
        hard = (c % args.hard_every) == args.hard_every - 1
        grip = int(rng.choice([45, 50, 55]))
        seed = args.base_seed * 10 + c
        if hard:
            # retuned after the pilot: at the 150-deg roll base the plant tracks too well
            # for noise-induced jams (0.10->1/10, 0.18->1/10 retreats). The clean lever is
            # the jam DETECTOR: jam_window 10 fires retreats on the servo plant's slow
            # pushes -> 5/10 recovery episodes at LOWER noise (measured, seed 3300).
            knobs = dict(action_noise=0.08, standoff=3.0, push_corr=0.15,
                         grip_z=0.0, steps=600, jam_window=10)
        else:
            knobs = dict(action_noise=0.0,
                         standoff=round(float(rng.uniform(4.0, 7.0)), 2),
                         push_corr=round(float(rng.uniform(0.2, 0.4)), 3),
                         grip_z=round(float(rng.uniform(-2.0, 2.0)), 2),
                         steps=450)
        cdir = os.path.join(args.out, f"chunk_{c:02d}")
        rounds = max(3, (2 * args.eps) // args.envs + 2)
        cmd = GEN + COMMON + [
            "--out", cdir,
            "--envs", str(args.envs), "--rounds", str(rounds),
            "--steps", str(knobs["steps"]), "--max-keep", str(knobs["steps"]),
            "--grip-from-head", str(grip),
            "--standoff-mm", str(knobs["standoff"]),
            "--push-correction", str(knobs["push_corr"]),
            "--grip-z-off-mm", str(knobs["grip_z"]),
            "--action-noise", str(knobs["action_noise"]),
            *(["--jam-window", str(knobs["jam_window"])] if "jam_window" in knobs else []),
            "--max-save", str(args.eps),
            "--seed", str(seed),
            "--keep-existing",
        ]
        manifest[f"chunk_{c:02d}"] = dict(hard=hard, grip_from_head=grip, seed=seed, **knobs)
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        print(f"[chunks] === chunk {c:02d}/{args.chunks - 1} "
              f"({'HARD' if hard else 'normal'}, grip {grip}, seed {seed}) ===", flush=True)
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            print(f"[chunks] chunk {c:02d} FAILED (rc {r.returncode}) — continuing", flush=True)
    open(os.path.join(args.out, "_GEN_DONE"), "w").write("done\n")
    print("[chunks] all chunks done", flush=True)


if __name__ == "__main__":
    main()
