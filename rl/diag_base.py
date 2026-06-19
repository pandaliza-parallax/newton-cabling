"""Ground-truth measurement of the base controller (no RL), to resolve the
training-SR (100%) vs eval-SR (9%) discrepancy. Runs the base controller for a
fixed horizon at a chosen curriculum stage and reports, from the raw plug state:
final insertion depth, final lateral offset, and the fraction meeting each
success criterion (repo travel>=20mm vs strict offset<=1mm)."""

import os
import sys

import numpy as np
import torch
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connector_env import ConnectorVecEnv  # noqa: E402

DEV = "cuda:0"
N = 2000
STAGE = 3      # 5mm max offset
STEPS = 80


def main():
    env = ConnectorVecEnv(N, seed=1)
    env.residual_scale = 0.0          # base controller only
    env.set_stage(STAGE)
    zero = torch.zeros(N, env.act_dim, device=DEV)
    env.reset()
    seated_frac = []
    for t in range(STEPS):
        _, _, _, succ, _ = env.step(zero)
        seated_frac.append(succ.mean().item())

    # raw final state
    q = env.state_0.body_q.numpy()
    seat = env.seated.numpy()
    plug = np.array([q[i][:3] for i in [r for r in env.plug_idx.numpy()]])
    e = plug - seat
    offset = np.sqrt(e[:, 0] ** 2 + e[:, 2] ** 2)   # lateral (x,z), m
    # actual inserted travel = plug_y - start_y
    travel_mm = (plug[:, 1] - env.start_y.numpy()) * 1000

    print(f"=== base controller, stage {STAGE} (<= {env._mag*1000:.1f}mm offset), {N} envs ===")
    print(f"instantaneous seated fraction: first step {seated_frac[0]*100:.1f}% -> "
          f"last step {seated_frac[-1]*100:.1f}%  (peak {max(seated_frac)*100:.1f}%)")
    print(f"final insertion travel (mm):  median {np.median(travel_mm):.1f}  "
          f"mean {travel_mm.mean():.1f}  p10 {np.percentile(travel_mm,10):.1f}")
    print(f"final lateral offset (mm):    median {np.median(offset)*1000:.2f}  "
          f"mean {offset.mean()*1000:.2f}  p90 {np.percentile(offset,90)*1000:.2f}")
    print(f"frac inserted >= 20mm (repo def):     {(travel_mm >= 20).mean()*100:.1f}%")
    print(f"frac offset <= 1mm (strict centering): {(offset <= 0.001).mean()*100:.1f}%")
    print(f"frac BOTH (strict seated):            "
          f"{((travel_mm >= 20) & (offset <= 0.001)).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
