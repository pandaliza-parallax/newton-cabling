"""Smoke-test a connector asset end-to-end: build the vec env, run the BASE
controller (zero residual) on near-aligned starts, and check it inserts without
NaNs or VBD ejection.

    uv run --extra sim python rl/smoke_asset.py            # both rj45 + cad_rj45
    uv run --extra sim python rl/smoke_asset.py cad_rj45   # one asset
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connector_env import ConnectorVecEnv  # noqa: E402

DEV = "cuda:0"


def run(asset, n=64, steps=120):
    env = ConnectorVecEnv(n, seed=0, random_easy=True, asset=asset)
    env.set_stage(0)            # near-aligned (curriculum scale 0.1) so the base seats
    env.residual_scale = 0.0    # pure scripted base controller, no policy
    obs = env.reset()
    assert torch.isfinite(obs).all(), f"{asset}: non-finite obs after reset"
    seated_any = torch.zeros(n, device=DEV)
    depth_tr = []
    for t in range(steps):
        a = torch.zeros(n, env.act_dim, device=DEV)   # zero residual = base only
        obs, rew, done, succ, depth = env.step(a)
        bad = (~torch.isfinite(obs).all()) or (~torch.isfinite(rew).all())
        assert not bad, f"{asset}: non-finite obs/rew at step {t}"
        seated_any = torch.maximum(seated_any, succ)
        depth_tr.append(depth.mean().item())
    print(f"[{asset}] OK  bodies={env.model.body_count}  obs_dim={env.obs_dim} act_dim={env.act_dim}")
    print(f"    seat target +{env.seat_aim_dy*1000:.0f}mm | held(any) {seated_any.mean()*100:.0f}%")
    print(f"    mean depth mm: {[f'{d:.0f}' for d in depth_tr[::15]]}  -> {depth_tr[-1]:.0f}")
    return depth_tr[-1], seated_any.mean().item()


if __name__ == "__main__":
    assets = sys.argv[1:] or ["rj45", "cad_rj45"]
    for a in assets:
        run(a)
