"""Record an ArmConnectorVecEnv rollout under a trained policy to a rerun .rrd (physics, for
viewing the RO1 insert the plug). Mirrors record_sbot_grasp_test.py's rerun logging.

    .venv/bin/python rl/record_arm_policy.py --checkpoint rl/runs/arm_v2/best_model.pt \
        --envs 4 --stage 0 --steps 200 --out arm_rollout

View:  uvx --from rerun-sdk rerun arm_rollout.rrd arm_rollout.rbl
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from arm_connector_env import ArmConnectorVecEnv  # noqa: E402
from train_ppo import ActorCritic  # noqa: E402
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: E402

DEV = "cuda:0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="policy .pt; omit for the base controller (zero residual)")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--residual-scale", type=float, default=0.2)
    ap.add_argument("--jack-drop", type=float, default=0.03,
                    help="0.03 = jack at the hand (coupled); 0.25 = jack on a table ~25cm below (decoupled)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="arm_rollout")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    env = ArmConnectorVecEnv(args.envs, seed=args.seed, residual_scale=args.residual_scale,
                             jack_drop=args.jack_drop)
    env.set_stage(args.stage)
    ac = None
    if args.checkpoint:
        ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
        ac.load_state_dict(torch.load(args.checkpoint, map_location=DEV))
        ac.eval()
        print(f"loaded {args.checkpoint}")
    else:
        print("no checkpoint -> base controller (zero residual)")

    rrd = f"{args.out}.rrd"
    viewer = open_rrd_recorder(rrd)
    viewer.set_model(env.model)

    obs = env.reset()
    dt = 1.0 / args.fps
    seated_any = torch.zeros(env.n, device=DEV)
    for t in range(args.steps):
        with torch.no_grad():
            a = ac.mean_action(obs) if ac is not None else torch.zeros(env.n, env.act_dim, device=DEV)
        obs, rew, done, succ, depth = env.step(a)
        seated_any = torch.maximum(seated_any, succ)
        viewer.begin_frame(t * dt)
        viewer.log_state(env.state_0)
        viewer.end_frame()
        if t % 40 == 0 or t == args.steps - 1:
            print(f"  t={t:3d}  seated_now={succ.mean().item():.2f}  depth={depth.mean().item():.1f}mm", flush=True)

    auto_blueprint(f"{args.out}.rbl", env.model)
    print(f"\nrecorded {rrd} (+ .rbl)  |  ever-seated {seated_any.mean().item():.0%} of {env.n} envs")
    print(f"view:  uvx --from rerun-sdk rerun {rrd} {args.out}.rbl")


if __name__ == "__main__":
    main()
