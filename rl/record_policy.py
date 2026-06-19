"""Record a rollout of the trained insertion policy (or the base controller) to a
rerun .rrd, so you can watch the plugs seat from misaligned starts.

    uv run --extra sim python rl/record_policy.py                 # trained RL policy
    uv run --extra sim python rl/record_policy.py --base          # scripted base controller
    uv run --extra sim python rl/record_policy.py --stage 6       # harder (15mm) offset
    uvx --from rerun-sdk rerun rl_rollout.rrd rl_rollout.rbl      # view it
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="rl/runs/extend/best_model.pt")
    ap.add_argument("--envs", type=int, default=9)       # 3x3 grid of connectors
    ap.add_argument("--stage", type=int, default=5)      # curriculum stage (5 = 11mm offset)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--out", default="rl_rollout")
    ap.add_argument("--base", action="store_true", help="record the base controller (no policy)")
    ap.add_argument("--random-easy", action="store_true",
                    help="random_easy_subset starts (lateral + approach + <=15deg rotation)")
    ap.add_argument("--asset", default="rj45", choices=["rj45", "cad_rj45"])
    args = ap.parse_args()

    env = ConnectorVecEnv(args.envs, seed=7, random_easy=args.random_easy, asset=args.asset)
    env.set_stage(args.stage)
    # difficulty label differs by mode: random_easy ramps a scale, else a lateral offset
    diff = (f"re_scale {env.re_scale:.2f}" if env.random_easy
            else f"<= {env._mag*1000:.1f}mm offset")
    if args.base:
        env.residual_scale = 0.0
        ac = None
        print(f"recording BASE controller | {args.asset} | stage {args.stage} ({diff})")
    else:
        ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
        ac.load_state_dict(torch.load(args.checkpoint, map_location=DEV))
        ac.eval()
        print(f"recording POLICY {args.checkpoint} | {args.asset} | stage {args.stage} ({diff})")

    viewer = open_rrd_recorder(f"{args.out}.rrd")
    viewer.set_model(env.model)
    obs = env.reset()
    zero = torch.zeros(args.envs, env.act_dim, device=DEV)
    dt = 1.0 / 60.0
    sim_time = 0.0
    for f in range(args.frames):
        with torch.no_grad():
            a = zero if ac is None else ac.mean_action(obs)
        obs, _, _, succ, depth = env.step(a)
        viewer.begin_frame(sim_time)
        viewer.log_state(env.state_0)
        viewer.end_frame()
        sim_time += dt
        if f % 30 == 0 or f == args.frames - 1:
            print(f"frame {f:3d}: seated {succ.mean().item()*100:5.1f}% | "
                  f"depth {depth.mean().item():5.1f}mm", flush=True)

    blueprint = auto_blueprint(f"{args.out}.rbl", env.model)
    print(f"\nrecording complete: {args.out}.rrd ; blueprint {blueprint}")
    print(f"view with:  uvx --from rerun-sdk rerun {args.out}.rrd {args.out}.rbl")


if __name__ == "__main__":
    main()
