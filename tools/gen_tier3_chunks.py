#!/usr/bin/env python3
"""Tier-3 recovery campaign, Stage A driver — data targeted at the two failure
modes measured in the 2026-08-21 eval matrices:

  A (chunks 0-3): TERMINAL DENSITY + IN-BORE KICKS — start 14-20mm out; small
    1-3mm kicks may fire in ALIGN or PUSH (--kick-phases 1,2), so the expert
    demonstrates correcting alignment while pushing in the bore — the regime
    where every current model either slips its grasp or scrubs off.
  B (chunks 4-7): FAR-RECOVERY STARTS — lateral offsets up to 30mm (curriculum
    override), matching the 10-35mm drift observed in failed rollouts; the
    expert pulls back, re-centers, and inserts. No kicks.
  C (chunks 8-9): BIG KICKS — tier-2 mechanism at 5-15mm magnitude.

Kick chunks are split at the kick windows as in tier-2. Manifest + _CHUNK_DONE
markers as in gen_tier2_chunks (render_tier2_campaign.sh consumes it via SRC).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = [os.path.join(REPO, ".venv", "bin", "python"),
       "-m", "newton_cabling.scripted_controller.gen_trajectories"]
SPLIT = [os.path.join(REPO, ".venv", "bin", "python"),
         os.path.join(REPO, "tools", "split_kick_episodes.py")]

COMMON = [
    "--servo-plant", "--jack-fixture",
    "--connector-usd", "cad_rj45.usd", "--grasp-roll-180",
    "--grasp-roll-deg", "150",
    "--jack-yaw", "5",
    "--cable-tilt", "-5", "8",
    "--grasp-roll-jitter", "3",
    "--offset-z", "3",
    "--offset-uniform-disk",
    "--stage", "4",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--eps", type=int, default=50)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--base-seed", type=int, required=True)
    ap.add_argument("--start-chunk", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "_CHUNKS.json")
    manifest = json.load(open(manifest_path)) if os.path.isfile(manifest_path) else {}

    for c in range(args.start_chunk, args.chunks):
        rng = np.random.default_rng(args.base_seed + c)
        mode = "A" if c < 4 else ("B" if c < 8 else "C")
        grip = int(rng.choice([45, 50, 55]))
        seed = args.base_seed * 10 + c
        knobs = dict(standoff=round(float(rng.uniform(4.0, 7.0)), 2),
                     push_corr=round(float(rng.uniform(0.2, 0.4)), 3),
                     grip_z=round(float(rng.uniform(-2.0, 2.0)), 2))
        cdir = os.path.join(args.out, f"chunk_{c:02d}")
        cmd = GEN + COMMON + [
            "--envs", str(args.envs),
            "--grip-from-head", str(grip),
            "--standoff-mm", str(knobs["standoff"]),
            "--push-correction", str(knobs["push_corr"]),
            "--grip-z-off-mm", str(knobs["grip_z"]),
            "--action-noise", "0.0",
            "--max-save", str(args.eps),
            "--seed", str(seed),
            "--keep-existing",
        ]
        split_after = False
        if mode == "A":
            split_after = True
            cmd += ["--out", cdir + "_raw", "--rounds", "6",
                    "--approach-jitter", "14", "20",
                    "--steps", "450", "--max-keep", "450",
                    "--kick-prob", "0.03", "--kick-max", "2", "--kick-phases", "1,2",
                    "--kick-mag-mm", "1", "3", "--kick-frames", "3", "6"]
        elif mode == "B":
            cmd += ["--out", cdir, "--rounds", "8",
                    "--approach-jitter", "14", "25",
                    "--offset-mag-mm", "30",
                    "--steps", "600", "--max-keep", "600"]
        else:
            split_after = True
            cmd += ["--out", cdir + "_raw", "--rounds", "6",
                    "--approach-jitter", "25", "35",
                    "--steps", "600", "--max-keep", "600",
                    "--kick-prob", "0.02", "--kick-max", "2", "--kick-phases", "1",
                    "--kick-mag-mm", "5", "15", "--kick-frames", "6", "10"]
        manifest[f"chunk_{c:02d}"] = dict(mode=mode, grip_from_head=grip, seed=seed, **knobs)
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        print(f"[tier3] === chunk {c:02d}/{args.chunks - 1} ({mode}, grip {grip}, "
              f"seed {seed}) ===", flush=True)
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            print(f"[tier3] chunk {c:02d} gen FAILED (rc {r.returncode}) — continuing", flush=True)
            continue
        if split_after:
            shutil.rmtree(cdir, ignore_errors=True)
            r = subprocess.run(SPLIT + ["--in", cdir + "_raw", "--out", cdir,
                                        "--min-frames", "25"], cwd=REPO)
            if r.returncode != 0:
                print(f"[tier3] chunk {c:02d} SPLIT FAILED", flush=True)
                continue
        open(os.path.join(cdir, "_CHUNK_DONE"), "w").write("done\n")
    open(os.path.join(args.out, "_GEN_DONE"), "w").write("done\n")
    print("[tier3] all chunks done", flush=True)


if __name__ == "__main__":
    main()
