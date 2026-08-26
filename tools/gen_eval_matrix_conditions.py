#!/usr/bin/env python3
"""Emit the 20-condition eval matrix (10 random TRAIN episodes with their recorded
conditions + 10 held-out EVAL draws) as JSON for eval_matrix_depaused.sh.

TRAIN conditions reproduce each sampled episode's recorded tilt / grasp-roll / grip /
jack position / gamma / bg-yaw / shadow. Known accepted deviations (same as the
trainep0000 evals): per-episode jack-yaw and approach are drawn by the live env from
the seed rather than forced to the recorded values; grip_z is not plumbed.
"""
import glob
import json
import os
import re

import numpy as np

ROOT = "/home/pandaliza/parallax/data/vla_train"
RAW, GS, DIFIX = f"{ROOT}/drmix_1000", f"{ROOT}/drmix_1000_gs", f"{ROOT}/drmix_1000_final_difix"

rng = np.random.default_rng(20260821)
conds = []

# ---- 10 random TRAIN episodes ----
eps = sorted(glob.glob(os.path.join(DIFIX, "chunk_*", "ep_*")))
manifest = json.load(open(os.path.join(RAW, "_CHUNKS.json")))
for rel in [os.path.relpath(p, DIFIX) for p in rng.choice(eps, 10, replace=False)]:
    chunk, ep = rel.split("/")
    a = json.load(open(os.path.join(RAW, rel, "meta.json")))
    d = json.load(open(os.path.join(DIFIX, rel, "meta.json")))
    cfg = open(os.path.join(GS, chunk, "_CONFIG.txt")).read()
    bg_yaw = int(re.search(r"BG_YAW_DEG=(\d+)", cfg).group(1))
    shadow = float(re.search(r"strength=([\d.]+)", cfg).group(1))
    log = open(os.path.join(GS, chunk, ep + ".log")).read()
    jp = re.search(r"jack @ \[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", log)
    conds.append(dict(
        name=f"train_{chunk}_{ep}", split="train",
        tilt=round(float(a["tilt_this_env_deg"]), 3),
        roll=round(float(a["grasp_roll_this_env_deg"]), 3),
        grip=int(manifest[chunk]["grip_from_head"]),
        jack_pos=f"{jp.group(1)} {jp.group(2)} {jp.group(3)}",
        gamma=float(d["render_gamma_effective"]),
        bg_yaw=bg_yaw, shadow=shadow,
        seed=41000 + len(conds),
        live_tilt_flag="fixed",     # exact tilt via single-value --live-tilt
    ))

# ---- 10 held-out EVAL draws ----
for i in range(10):
    conds.append(dict(
        name=f"eval_{i:02d}", split="eval",
        tilt=None,                   # env draws from U(-5, 8) via the seed
        roll=150.0,
        grip=[45, 50, 55][i % 3],
        jack_pos=f"{0.235 + rng.uniform(-0.05, 0.05):.4f} "
                 f"{-0.835 + rng.uniform(-0.05, 0.05):.4f} 0.835",
        gamma=[1.0, 1.2, 1.4, 1.8][i % 4],
        bg_yaw=[85, 265][i % 2],     # two SETUPs only (renderer restart economy)
        shadow=round(0.5 + 0.03 * i, 2),
        seed=91000 + i,
        live_tilt_flag="range",
    ))

# group by bg_yaw to minimise renderer restarts, stable within groups
conds.sort(key=lambda c: (c["bg_yaw"], c["name"]))
out = f"{ROOT}/evalmatrix_conditions.json"
json.dump(conds, open(out, "w"), indent=1)
print(f"wrote {out}: {len(conds)} conditions, "
      f"{len(set(c['bg_yaw'] for c in conds))} distinct bg-yaws (renderer restarts)")
