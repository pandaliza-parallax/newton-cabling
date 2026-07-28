"""Record a CableInsertVecEnv rollout to a rerun .rrd (base servo or a checkpoint policy).

    .venv/bin/python rl/record_cable_env.py --envs 1 --steps 260 --out cable_env_state
    view:  uvx --from rerun-sdk rerun cable_env_state.rrd cable_env_state.rbl
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from cable_env import CableInsertVecEnv  # noqa: E402
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: E402

DEV = "cuda:0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help=".pt policy; omit for the base servo")
    ap.add_argument("--envs", type=int, default=1)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cable-tilt", type=float, default=0.0,
                    help="deg the hanging connector droops below horizontal at episode start")
    ap.add_argument("--rigid", action="store_true",
                    help="record the RIGID solid-cable env instead of the deformable one")
    ap.add_argument("--out", default="cable_rollout")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    if args.rigid:
        from rigid_cable_env import RigidCableVecEnv as EnvCls
    else:
        EnvCls = CableInsertVecEnv
    env = EnvCls(args.envs, seed=args.seed, cable_tilt_deg=args.cable_tilt)
    env.set_stage(args.stage)
    ac = None
    if args.checkpoint:
        from train_ppo import ActorCritic
        ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
        ac.load_state_dict(torch.load(args.checkpoint, map_location=DEV))
        ac.eval()
        print(f"loaded {args.checkpoint}")
    else:
        print("base servo (zero residual)")

    viewer = open_rrd_recorder(f"{args.out}.rrd")
    viewer.set_model(env.model)
    obs = env.reset()
    ever = torch.zeros(env.n, device=DEV)
    # trajectory dump (pi05 rj45_sbot layout): obs[:, :10] = state [eef_pos(3), eef_rot6d(6),
    # gripper(1)]; action in [-1,1]^7 = [dpos(3), drotvec(3), gripper(1)] (scales in meta)
    traj = {k: [] for k in ("obs", "action", "face_pos", "face_quat",
                            "along", "latn", "ang", "viol", "held", "depth_mm")}
    for t in range(args.steps):
        with torch.no_grad():
            a = ac.mean_action(obs) if ac is not None else env.servo_action()
        traj["obs"].append(obs.cpu().numpy())
        traj["action"].append(a.cpu().numpy())
        obs, rew, done, succ, depth = env.step(a)
        ever = torch.maximum(ever, succ)
        bqn = env.state_0.body_q.numpy()
        face, faceq = env._face_pose(bqn)
        along, lat, latn, ang = env._terms(face, faceq)
        for k, v in (("face_pos", face), ("face_quat", faceq), ("along", along),
                     ("latn", latn), ("ang", ang), ("viol", env.viol_last.copy()),
                     ("held", succ.cpu().numpy()), ("depth_mm", depth.cpu().numpy())):
            traj[k].append(v)
        viewer.begin_frame(t / args.fps)
        viewer.log_state(env.state_0)
        viewer.end_frame()
        if t % 40 == 0 or t == args.steps - 1:
            print(f"  t={t:3d} along {along[0]*1000:6.1f}mm lat {latn[0]*1000:4.1f}mm "
                  f"ang {np.degrees(ang[0]):5.1f}deg viol {env.viol_last.sum():.0f}", flush=True)
    auto_blueprint(f"{args.out}.rbl", env.model)
    from cable_env import MAX_DPOS, MAX_DROT
    np.savez_compressed(
        f"{args.out}_traj.npz",
        **{k: np.asarray(v) for k, v in traj.items()},        # each (T, n, ...)
        seat_pos=env.seat_pos, seat_quat=env.seat_q, ins_axis=env.ins,
        meta=np.array([f"policy={args.checkpoint or 'servo-teacher'}",
                       f"stage={args.stage}", f"cable_tilt_deg={args.cable_tilt}",
                       f"seed={args.seed}", f"hz=60", f"max_dpos_m={MAX_DPOS}",
                       f"max_drot_rad={MAX_DROT}",
                       "state=obs[:,:10]=[eef_pos(3),eef_rot6d(6),gripper(1)]",
                       "action=[dpos(3),drotvec(3),gripper(1)] in [-1,1], x scales"]))
    print(f"\nrecorded {args.out}.rrd | ever-held {ever.mean().item():.0%} of {env.n}")
    print(f"trajectory -> {args.out}_traj.npz")
    print(f"view:  uvx --from rerun-sdk rerun {args.out}.rrd {args.out}.rbl")


if __name__ == "__main__":
    main()
