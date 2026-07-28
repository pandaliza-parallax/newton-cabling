"""PPO trainer for the arm-in-the-loop RJ45 insertion (ArmConnectorVecEnv).

Same CleanRL-style GPU PPO as rl/train_ppo.py, reusing its ActorCritic, but the policy now
predicts GRIPPER (end-effector) motion — a 7-D [dpos(3), drotvec(3), gripper(1)] residual on a
scripted base drive, realized through DLS IK on the RO1 (see arm_connector_env.py). The plug
moves only via the physical AG-145 grasp.

    uv run --extra sim python rl/train_arm_ppo.py --envs 128 --iters 300 --run-name arm
    uv run --extra sim python rl/train_arm_ppo.py --envs 32 --iters 15 --rollout 16 --run-name smoke

Logs to runs/<name>/log.csv; saves best_model.pt (by eval seated) and final_model.pt.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque

import torch
import torch.nn as nn

from arm_connector_env import ArmConnectorVecEnv
from train_ppo import ActorCritic  # reuse the exact network

DEV = "cuda:0"


@torch.no_grad()
def evaluate_policy(ac: ActorCritic, env: ArmConnectorVecEnv, steps: int = 200, settle_window: int = 80):
    obs = env.reset()
    seated, depth = [], []
    for t in range(steps):
        a = ac.mean_action(obs)
        obs, _, _, succ, depth_mm = env.step(a)
        if t >= steps - settle_window:
            seated.append(succ.mean().item()); depth.append(depth_mm.mean().item())
    return {"success_rate": sum(seated) / max(1, len(seated)),
            "mean_depth_mm": sum(depth) / max(1, len(depth)), "episodes": env.n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--ent-coef-final", type=float, default=None)
    ap.add_argument("--ent-anneal-frac", type=float, default=1.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--no-anneal-lr", action="store_true")
    ap.add_argument("--residual-scale", type=float, default=0.2)
    ap.add_argument("--ik-iters", type=int, default=2)
    ap.add_argument("--socket-mu", type=float, default=0.5)
    ap.add_argument("--advance-seated", type=float, default=0.55,
                    help="advance the curriculum when the rolling seated fraction exceeds this")
    ap.add_argument("--advance-window", type=int, default=10)
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="arm")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--eval-every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    n, T = args.envs, args.rollout
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    csv_f = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["iter", "stage", "env_steps", "mean_rew", "seated", "mean_depth_mm",
                     "approx_kl", "entropy", "pol_loss", "val_loss", "steps_per_s"])

    print(f"Building {n} arm envs ...", flush=True)
    t0 = time.perf_counter()
    env = ArmConnectorVecEnv(n, seed=args.seed, residual_scale=args.residual_scale,
                             ik_iters=args.ik_iters, socket_mu=args.socket_mu)
    print(f"  built in {time.perf_counter()-t0:.1f}s | obs {env.obs_dim} act {env.act_dim}", flush=True)
    cc, cbuf = env.contact_count()
    print(f"  contact buffer: {cc}/{cbuf}{'  OVERFLOW' if cc >= cbuf else ''}", flush=True)

    ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
    if args.init_from:
        ac.load_state_dict(torch.load(args.init_from, map_location=DEV))
        print(f"  warm-started from {args.init_from}")
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    print(f"  policy params: {sum(p.numel() for p in ac.parameters()):,}")

    env.set_stage(env.num_stages - 1 if args.no_curriculum else 0)
    sr_window = deque(maxlen=args.advance_window)
    print(f"  curriculum: {'OFF' if args.no_curriculum else f'{env.num_stages} stages'} | "
          f"start stage {env.stage} (re_scale {env.re_scale:.2f})", flush=True)

    batch = T * n
    mb = batch // args.minibatches
    O = torch.zeros(T, n, env.obs_dim, device=DEV); A = torch.zeros(T, n, env.act_dim, device=DEV)
    LP = torch.zeros(T, n, device=DEV); V = torch.zeros(T, n, device=DEV)
    R = torch.zeros(T, n, device=DEV); D = torch.zeros(T, n, device=DEV)
    best_sr, total_steps = -1.0, 0

    for it in range(args.iters):
        if not args.no_anneal_lr:
            for g in opt.param_groups:
                g["lr"] = args.lr * (1.0 - it / args.iters)
        if args.ent_coef_final is not None:
            p = min(1.0, (it / max(1, args.iters - 1)) / max(1e-6, args.ent_anneal_frac))
            ent_coef = args.ent_coef + (args.ent_coef_final - args.ent_coef) * p
        else:
            ent_coef = args.ent_coef
        t_iter = time.perf_counter()
        obs = env.reset()
        seated_acc, depth_acc = 0.0, 0.0
        for t in range(T):
            with torch.no_grad():
                a, lp, v = ac.act(obs)
            nobs, rew, done, succ, depth_mm = env.step(a)
            step_r = torch.nan_to_num(rew).clamp(-20.0, 150.0)
            O[t], A[t], LP[t], V[t], R[t], D[t] = obs, a, lp, v, step_r, done
            obs = nobs
            # honest instantaneous-seated fraction (not the horizon-diluted held-success mean)
            seated_acc += env.seated_inst_frac; depth_acc += depth_mm.mean().item(); total_steps += n

        with torch.no_grad():
            last_v = ac.value(obs)
        adv = torch.zeros(T, n, device=DEV); gae = torch.zeros(n, device=DEV)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            delta = R[t] + args.gamma * nv * (1 - D[t]) - V[t]
            gae = delta + args.gamma * args.lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        bo = O.reshape(batch, env.obs_dim); ba = A.reshape(batch, env.act_dim)
        blp = LP.reshape(batch); badv = adv.reshape(batch); bret = ret.reshape(batch)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)

        kl_acc, ent_acc, pl_acc, vl_acc, nupd = 0.0, 0.0, 0.0, 0.0, 0
        for _ in range(args.epochs):
            idx = torch.randperm(batch, device=DEV)
            epoch_kl, epoch_nb = 0.0, 0
            for s in range(0, batch, mb):
                j = idx[s:s + mb]
                nlp, ent, nv = ac.evaluate(bo[j], ba[j])
                logratio = nlp - blp[j]; ratio = logratio.exp()
                pl = -torch.min(ratio * badv[j],
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * badv[j]).mean()
                vl = (nv - bret[j]).pow(2).mean()
                loss = pl + args.vf_coef * vl - ent_coef * ent.mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), args.max_grad); opt.step()
                with torch.no_grad():
                    approx_kl = (ratio - 1 - logratio).mean().item()
                kl_acc += approx_kl; epoch_kl += approx_kl; epoch_nb += 1
                ent_acc += ent.mean().item(); pl_acc += pl.item(); vl_acc += vl.item(); nupd += 1
            if epoch_kl / max(1, epoch_nb) > args.target_kl:
                break

        sps = (T * n) / (time.perf_counter() - t_iter)
        sr = seated_acc / T; md = depth_acc / T
        kl, ent_m, pl_m, vl_m = kl_acc / nupd, ent_acc / nupd, pl_acc / nupd, vl_acc / nupd
        writer.writerow([it, env.stage, total_steps, R.mean().item(), sr, md, kl, ent_m, pl_m, vl_m, sps])
        csv_f.flush()
        if it % 2 == 0 or it == args.iters - 1:
            print(f"it {it:4d} | stg {env.stage} | steps {total_steps:>9,} | R {R.mean().item():+.3f} | "
                  f"seated {sr:6.2%} | depth {md:5.1f}mm | KL {kl:+.4f} | ent {ent_m:+.2f} | "
                  f"{sps:,.0f} st/s", flush=True)

        sr_window.append(sr)
        if (not args.no_curriculum and env.stage < env.num_stages - 1
                and len(sr_window) == sr_window.maxlen
                and sum(sr_window) / len(sr_window) >= args.advance_seated):
            env.set_stage(env.stage + 1); sr_window.clear()
            print(f"   [curriculum] -> stage {env.stage} (re_scale {env.re_scale:.2f})", flush=True)

        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            r = evaluate_policy(ac, env)
            print(f"   [eval] seated {r['success_rate']:6.2%} | depth {r['mean_depth_mm']:5.1f}mm", flush=True)
            if r["success_rate"] > best_sr:
                best_sr = r["success_rate"]
                torch.save(ac.state_dict(), os.path.join(run_dir, "best_model.pt"))
                print(f"   [eval] new best {best_sr:.2%} -> best_model.pt", flush=True)

    torch.save(ac.state_dict(), os.path.join(run_dir, "final_model.pt"))
    csv_f.close()
    print(f"\nDone. best eval seated {best_sr:.2%} | {run_dir}", flush=True)


if __name__ == "__main__":
    main()
