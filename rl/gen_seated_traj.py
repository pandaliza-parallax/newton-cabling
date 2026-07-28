"""Generate SEATED plug trajectories with the trained PPO policy (no renderer needed).

Rolls out the learned ActorCritic in ConnectorVecEnv (pure physics), records each env's
plug pose in ITS socket frame per step -- the same (T,7) [pos3, quat4 wxyz] format as
record_policy.py --eval-vla's plug_traj.npy -- and saves every episode that seats. Feed
a saved file straight into scripts/record_sbot_scene_gs.py --plug-traj.

    uv run --extra sim python rl/gen_seated_traj.py --out seated_traj --envs 16 --rounds 6

Envs are a grid (SPACING=0.4), so each env's socket world pos is _socket_world(asset)+shift[e];
the per-env socket is subtracted so the saved trajectory is socket-relative (jack-anchorable).
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connector_env import SPACING, ConnectorVecEnv  # noqa: E402
from record_policy import _socket_world  # noqa: E402
from train_ppo import ActorCritic  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from newton_cabling.render.gs_bridge import newton_pose  # noqa: E402

DEV = "cuda:0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="rl/runs/extend/best_model.pt")
    ap.add_argument("--out", default="../data/vla_train/seated_traj")
    ap.add_argument("--asset", default="rj45", choices=["rj45", "cad_rj45", "cad_rj45_real"])
    ap.add_argument("--random-easy", action="store_true",
                    help="6-DOF random_easy task (obs=12, act=6) — REQUIRED for the cad_* checkpoints")
    ap.add_argument("--stage", type=int, default=5, help="curriculum stage (lower = easier = seats more)")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=6, help="rollout rounds; episodes = envs*rounds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-save", type=int, default=5, help="stop after saving this many seated eps")
    args = ap.parse_args()

    env = ConnectorVecEnv(args.envs, seed=args.seed, asset=args.asset, random_easy=args.random_easy)
    env.set_stage(args.stage)
    ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
    ac.load_state_dict(torch.load(args.checkpoint, map_location=DEV))
    ac.eval()

    # per-env socket world pos = template socket (_socket_world) + grid shift[e]
    sock0 = _socket_world(args.asset)
    cols = max(1, int(np.ceil(np.sqrt(args.envs))))
    shift = np.array([[(e % cols) * SPACING, 0.0, (e // cols) * SPACING] for e in range(args.envs)])
    sock = sock0[None, :] + shift                                   # (envs, 3)
    pi_np = env.plug_idx.numpy()

    os.makedirs(args.out, exist_ok=True)
    print(f"recording PPO {args.checkpoint} | {args.asset} | stage {args.stage} | "
          f"{args.envs} envs x {args.rounds} rounds")
    saved = tot = seat_tot = 0
    for r in range(args.rounds):
        obs = env.reset()
        T = np.zeros((args.frames, args.envs, 7), np.float32)
        seated = np.zeros(args.envs, bool)
        for f in range(args.frames):
            bq = env.state_0.body_q.numpy()
            for e in range(args.envs):
                pp, pq = newton_pose(bq, int(pi_np[e]))
                T[f, e, :3] = np.asarray(pp) - sock[e]
                T[f, e, 3:] = pq                                    # wxyz
            with torch.no_grad():
                a = ac.mean_action(obs)
            obs, _, _, succ, _ = env.step(a)
            seated |= (succ.detach().cpu().numpy() > 0.5)
        tot += args.envs
        seat_tot += int(seated.sum())
        for e in np.where(seated)[0]:
            if saved >= args.max_save:
                break
            d = os.path.join(args.out, f"ep_{saved:04d}")
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "plug_traj.npy"), T[:, e, :])
            print(f"[gen] round {r} env {e}: SEATED -> {d}/plug_traj.npy "
                  f"(y {T[0, e, 1]*1000:.1f}->{T[-1, e, 1]*1000:.1f}mm)")
            saved += 1
        print(f"[gen] round {r + 1}/{args.rounds}: seated so far {seat_tot}/{tot}, saved {saved}")
        if saved >= args.max_save:
            break
    print(f"[gen] done: {seat_tot}/{tot} seated ({100*seat_tot/max(tot,1):.0f}%); "
          f"saved {saved} trajectories to {args.out}/")


if __name__ == "__main__":
    main()
