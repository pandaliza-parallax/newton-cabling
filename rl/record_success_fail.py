"""Record separate SUCCESS and FAIL rollouts of a trained policy on random_easy_subset.

Scouts a batch to find which start poses the policy seats vs misses, then replays a
grid of clean successes (re_success.rrd) and a grid of failures (re_fail.rrd) so you
can watch what the policy does right and where it breaks.

    uv run --extra sim python rl/record_success_fail.py --checkpoint rl/runs/re_6dof_5k/best_model.pt
    uvx --from rerun-sdk rerun re_success.rrd re_success.rbl
    uvx --from rerun-sdk rerun re_fail.rrd re_fail.rbl
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connector_env import ConnectorVecEnv  # noqa: E402
from train_ppo import ActorCritic  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: E402

DEV = "cuda:0"


def load_policy(env, ckpt):
    ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
    ac.load_state_dict(torch.load(ckpt, map_location=DEV))
    ac.eval()
    return ac


def record_grid(ckpt, starts, idx, out, frames, asset="rj45"):
    """Build a small env, inject the selected start poses, roll out the policy, record."""
    n = len(idx)
    env = ConnectorVecEnv(n, seed=0, random_easy=True, asset=asset)
    env.set_stage(env.num_stages - 1)
    env.set_fixed_starts(starts[0][idx], starts[1][idx], starts[2][idx], starts[3][idx])
    ac = load_policy(env, ckpt)
    viewer = open_rrd_recorder(f"{out}.rrd")
    viewer.set_model(env.model)
    obs = env.reset()
    sim_time = 0.0
    ever = torch.zeros(n, device=DEV)
    for f in range(frames):
        with torch.no_grad():
            a = ac.mean_action(obs)
        obs, _, _, succ, depth = env.step(a)
        ever = torch.maximum(ever, succ)
        viewer.begin_frame(sim_time)
        viewer.log_state(env.state_0)
        viewer.end_frame()
        sim_time += 1.0 / 60.0
    blueprint = auto_blueprint(f"{out}.rbl", env.model)
    print(f"  {out}.rrd ({n} envs) — seated {ever.mean().item()*100:.0f}% of this grid; "
          f"blueprint {blueprint}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="rl/runs/re_6dof_5k/best_model.pt")
    ap.add_argument("--scout", type=int, default=256, help="envs to scout for success/fail starts")
    ap.add_argument("--grid", type=int, default=9, help="envs to record per rollout")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--asset", default="rj45", choices=["rj45", "cad_rj45"])
    args = ap.parse_args()

    # ── scout: find which random starts the policy seats vs misses ───────────────
    env = ConnectorVecEnv(args.scout, seed=20, random_easy=True, asset=args.asset)
    env.set_stage(env.num_stages - 1)  # full subset
    ac = load_policy(env, args.checkpoint)
    obs = env.reset()
    starts = (env._latx_keep.clone(), env._latz_keep.clone(),
              env._ins_keep.clone(), env._rot_keep.clone())  # the start poses used this episode
    ever = torch.zeros(args.scout, device=DEV)
    for _ in range(args.frames):
        with torch.no_grad():
            a = ac.mean_action(obs)
        obs, _, _, succ, _ = env.step(a)
        ever = torch.maximum(ever, succ)
    succ_idx = torch.nonzero(ever > 0.5).flatten()
    fail_idx = torch.nonzero(ever < 0.5).flatten()
    print(f"scout: {len(succ_idx)}/{args.scout} seated "
          f"({len(succ_idx)/args.scout*100:.0f}%); recording grids of {args.grid}")

    g = args.grid
    if len(succ_idx) >= 1:
        record_grid(args.checkpoint, starts, succ_idx[:g], "re_success", args.frames, args.asset)
    if len(fail_idx) >= 1:
        record_grid(args.checkpoint, starts, fail_idx[:g], "re_fail", args.frames, args.asset)
    print("view: uvx --from rerun-sdk rerun re_success.rrd re_success.rbl   (and re_fail.*)")


if __name__ == "__main__":
    main()
