#!/usr/bin/env python3
"""Tier-2 recovery campaign, Stage A driver.

8 chunks x 50 episodes = 400 raw episodes of perturbation-recovery data
(validated by the 2026-08-20 pilot), alternating two modes:

  R (even chunks): NEAR-DOCK OFF-NOMINAL STARTS — approach 15-22mm (pre-dock
    sits at 16-19mm) + the stage-4 8mm lateral disk: starts 4-16mm off-path in
    the diagnosed failure zone; every frame is a corrective label.
  K (odd chunks):  ALIGN-PHASE KICKS — the executed action is replaced by a
    2-6mm lateral kick for 4-8 frames (marked in kick.npy), then the controller
    recovers. Split at the kick windows (tools/split_kick_episodes.py) right
    here in the driver, so Stage B renders kick-free segments and the off-path
    kick motion never becomes a label. The last K chunk adds jam-window 10
    (retreat recovery mixed with kicks).

Same servo plant + fixture + COMMON DR as drmix_1000. Manifest -> _CHUNKS.json
(entry written at chunk start, same contract render_*_campaign.sh waits on).

    .venv/bin/python tools/gen_tier2_chunks.py \
        --out ~/parallax/data/vla_train/tier2_recovery --base-seed 6000
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
    ap.add_argument("--chunks", type=int, default=8)
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
        mode = "R" if c % 2 == 0 else "K"
        grip = int(rng.choice([45, 50, 55]))
        seed = args.base_seed * 10 + c
        knobs = dict(standoff=round(float(rng.uniform(4.0, 7.0)), 2),
                     push_corr=round(float(rng.uniform(0.2, 0.4)), 3),
                     grip_z=round(float(rng.uniform(-2.0, 2.0)), 2))
        cdir = os.path.join(args.out, f"chunk_{c:02d}")
        # R-mode holds ~52% (hard starts) -> extra rounds; gen stops at --max-save anyway
        rounds = 7 if mode == "R" else 5
        cmd = GEN + COMMON + [
            "--envs", str(args.envs), "--rounds", str(rounds),
            "--grip-from-head", str(grip),
            "--standoff-mm", str(knobs["standoff"]),
            "--push-correction", str(knobs["push_corr"]),
            "--grip-z-off-mm", str(knobs["grip_z"]),
            "--max-save", str(args.eps),
            "--seed", str(seed),
            "--keep-existing",
        ]
        if mode == "R":
            cmd += ["--out", cdir, "--approach-jitter", "15", "22",
                    "--steps", "450", "--max-keep", "450", "--action-noise", "0.0"]
        else:
            raw = cdir + "_raw"
            jam = (c == args.chunks - 1)   # last K chunk: retreats mixed with kicks
            cmd += ["--out", raw, "--approach-jitter", "25", "35",
                    "--steps", "600", "--max-keep", "600",
                    "--kick-prob", "0.02", "--kick-max", "2",
                    "--kick-mag-mm", "2", "6", "--kick-frames", "4", "8",
                    "--action-noise", "0.08" if jam else "0.0"]
            if jam:
                cmd += ["--jam-window", "10"]
            knobs["jam_window"] = 10 if jam else None
        manifest[f"chunk_{c:02d}"] = dict(mode=mode, grip_from_head=grip, seed=seed, **knobs)
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        print(f"[tier2] === chunk {c:02d}/{args.chunks - 1} ({mode}, grip {grip}, "
              f"seed {seed}) ===", flush=True)
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            print(f"[tier2] chunk {c:02d} gen FAILED (rc {r.returncode}) — continuing", flush=True)
            continue
        if mode == "K":
            shutil.rmtree(cdir, ignore_errors=True)
            r = subprocess.run(SPLIT + ["--in", cdir + "_raw", "--out", cdir,
                                        "--min-frames", "25"], cwd=REPO)
            if r.returncode != 0:
                print(f"[tier2] chunk {c:02d} SPLIT FAILED", flush=True)
                continue
        # per-chunk gate for the render driver: the chunk dir is complete (K chunks
        # only exist post-split, so the manifest-entry gate alone races)
        open(os.path.join(cdir, "_CHUNK_DONE"), "w").write("done\n")
    open(os.path.join(args.out, "_GEN_DONE"), "w").write("done\n")
    print("[tier2] all chunks done", flush=True)


if __name__ == "__main__":
    main()
