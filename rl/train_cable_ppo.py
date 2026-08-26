"""PPO trainer for the dangling-cable RJ45 insertion (CableInsertVecEnv) — Stage D.

PURE RL (user spec): the 7-D action [dpos(3), drotvec(3), gripper(1)] IS the gripper motion,
executed verbatim through DLS IK — no scripted base in the loop. The scripted funnel-servo
teacher exists only as `env.servo_action()`; `--bc-iters` uses it for an optional behavior-
cloning warm-start of the policy mean before PPO takes over (log_std stays explorative).

    .venv/bin/python rl/train_cable_ppo.py --envs 16 --iters 400 --tilt 8 --run-name cable
    smoke: .venv/bin/python rl/train_cable_ppo.py --envs 4 --iters 3 --rollout 16 --bc-iters 1

Logs to runs/<name>/log.csv; saves best_model.pt (by eval held-success) and final_model.pt.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque

import torch
import torch.nn as nn

from cable_env import CableInsertVecEnv
from train_ppo import ActorCritic  # reuse the exact network

DEV = "cuda:0"


@torch.no_grad()
def evaluate_policy(ac: ActorCritic, env: CableInsertVecEnv, steps: int = 200,
                    settle_window: int = 80):
    """Deterministic rollout; tail-window mean of HELD-success (seated + held, not inserting)."""
    obs = env.reset()
    held, depth = [], []
    for t in range(steps):
        a = ac.mean_action(obs)
        obs, _, _, succ, depth_mm = env.step(a)
        if t >= steps - settle_window:
            held.append(succ.mean().item()); depth.append(depth_mm.mean().item())
    return {"success_rate": sum(held) / max(1, len(held)),
            "mean_depth_mm": sum(depth) / max(1, len(depth))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--rollout", type=int, default=96,
                    help="steps per iteration; must cover level+approach+HOLD_STEPS")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--no-anneal-lr", action="store_true")
    ap.add_argument("--ik-iters", type=int, default=2)
    ap.add_argument("--socket-mu", type=float, default=0.5)
    ap.add_argument("--tilt", type=float, default=8.0,
                    help="max droop deg at the TOP stage; per-env uniform in [0, tilt], scaled "
                         "by stage/(num_stages-1) (tilt curriculum; 0 = level only)")
    ap.add_argument("--tilt-lo", type=float, default=0.0,
                    help="LOWER edge of the per-env droop DR (deg). Raise (e.g. 4) to bias "
                         "training into the hard full-droop corner instead of uniform 0..tilt")
    ap.add_argument("--bc-iters", type=int, default=15,
                    help="teacher rollouts for BC warm-start (0 = pure PPO from scratch)")
    ap.add_argument("--bc-epochs", type=int, default=50)
    ap.add_argument("--vf-warmup", type=int, default=8,
                    help="iterations training ONLY the value net (protects the BC mean from "
                         "garbage advantages out of a fresh critic)")
    ap.add_argument("--init-logstd", type=float, default=-1.25)
    ap.add_argument("--logstd-final", type=float, default=-2.25,
                    help="anneal exploration noise to this log_std (cable_v2 lesson: PPO never "
                         "shrinks it on its own — entropy pinned at init for 100 iters — and "
                         "sigma~0.29 random-walks ~2.6mm over a 20-step hold vs the 3mm seat "
                         "tolerance, capping held ~30%%). None disables the anneal")
    ap.add_argument("--logstd-anneal-frac", type=float, default=0.8,
                    help="fraction of the run over which log_std reaches its final value")
    ap.add_argument("--advance-held", type=float, default=0.55,
                    help="advance the curriculum when the rolling held fraction exceeds this")
    ap.add_argument("--advance-window", type=int, default=10)
    ap.add_argument("--connector-usd", default="cad_rj45.usd",
                    help="connector asset for the RIGID env (e.g. scan_rj45.usd)")
    ap.add_argument("--rigid", action="store_true",
                    help="use RigidCableVecEnv (solid cable, physical friction grasp) "
                         "instead of the deformable CableInsertVecEnv")
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="cable")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--eval-every", type=int, default=25)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    n, T = args.envs, args.rollout
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    csv_f = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["iter", "stage", "env_steps", "mean_rew", "held", "seated_inst",
                     "mean_depth_mm", "viol_frac", "approx_kl", "entropy", "pol_loss",
                     "val_loss", "steps_per_s"])

    tilt = (args.tilt_lo, args.tilt) if args.tilt > 0 else 0.0
    if args.rigid:
        from rigid_cable_env import RigidCableVecEnv as EnvCls
    else:
        EnvCls = CableInsertVecEnv
    print(f"Building {n} {'RIGID' if args.rigid else 'deformable'} cable envs "
          f"(tilt DR {tilt}) ...", flush=True)
    t0 = time.perf_counter()
    env_kw = dict(seed=args.seed, ik_iters=args.ik_iters,
                  socket_mu=args.socket_mu, cable_tilt_deg=tilt)
    if args.rigid:
        env_kw["connector_usd"] = args.connector_usd
    env = EnvCls(n, **env_kw)
    print(f"  built in {time.perf_counter()-t0:.1f}s | obs {env.obs_dim} act {env.act_dim} "
          f"| front_room {env.front_room*1000:.1f}mm", flush=True)

    ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
    ac.log_std.data.fill_(args.init_logstd)
    if args.init_from:
        ac.load_state_dict(torch.load(args.init_from, map_location=DEV))
        print(f"  warm-started from {args.init_from}")
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    print(f"  policy params: {sum(p.numel() for p in ac.parameters()):,}")

    # ── optional BC warm-start on the funnel-servo teacher ────────────────────
    if args.bc_iters > 0:
        print(f"BC warm-start: {args.bc_iters} teacher rollouts x {T} steps ...", flush=True)
        bco, bca = [], []
        for b in range(args.bc_iters):
            # consecutive same-stage blocks (round-robin would trigger a tilt re-align on
            # every single rollout)
            env.set_stage((b * env.num_stages) // args.bc_iters)
            obs = env.reset()
            for _ in range(T):
                a = env.servo_action()
                bco.append(obs); bca.append(a)
                obs, _, _, _, _ = env.step(a)
        bo = torch.cat(bco); ba = torch.cat(bca)
        bc_opt = torch.optim.Adam(ac.pi.parameters(), lr=1e-3)
        nb = bo.shape[0]
        for ep in range(args.bc_epochs):
            idx = torch.randperm(nb, device=DEV)
            tot = 0.0
            for s in range(0, nb, 1024):
                j = idx[s:s + 1024]
                loss = (ac.pi(bo[j]) - ba[j]).pow(2).mean()
                bc_opt.zero_grad(); loss.backward(); bc_opt.step()
                tot += loss.item() * len(j)
            if ep % 5 == 0 or ep == args.bc_epochs - 1:
                print(f"  bc epoch {ep:2d} mse {tot/nb:.5f}", flush=True)
        del bo, ba, bco, bca

    env.set_stage(env.num_stages - 1 if args.no_curriculum else 0)
    sr_window = deque(maxlen=args.advance_window)
    print(f"  curriculum: {'OFF' if args.no_curriculum else f'{env.num_stages} stages'} | "
          f"start stage {env.stage}", flush=True)

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
        if args.logstd_final is not None:
            p = min(1.0, (it / max(1, args.iters - 1)) / max(1e-6, args.logstd_anneal_frac))
            ac.log_std.data.fill_(args.init_logstd + (args.logstd_final - args.init_logstd) * p)
        t_iter = time.perf_counter()
        obs = env.reset()
        held_acc, seated_acc, depth_acc, viol_acc = 0.0, 0.0, 0.0, 0.0
        for t in range(T):
            with torch.no_grad():
                a, lp, v = ac.act(obs)
            nobs, rew, done, succ, depth_mm = env.step(a)
            step_r = torch.nan_to_num(rew).clamp(-20.0, 150.0)
            O[t], A[t], LP[t], V[t], R[t], D[t] = obs, a, lp, v, step_r, done
            obs = nobs
            held_acc += succ.mean().item()
            seated_acc += env.seated_inst_frac
            depth_acc += depth_mm.mean().item()
            viol_acc += float((env.viol_last > 0).mean())
            total_steps += n

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
                if it < args.vf_warmup:
                    loss = args.vf_coef * vl        # critic-only warmup: BC mean untouched
                else:
                    loss = pl + args.vf_coef * vl - args.ent_coef * ent.mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), args.max_grad); opt.step()
                with torch.no_grad():
                    approx_kl = (ratio - 1 - logratio).mean().item()
                kl_acc += approx_kl; epoch_kl += approx_kl; epoch_nb += 1
                ent_acc += ent.mean().item(); pl_acc += pl.item(); vl_acc += vl.item(); nupd += 1
            if epoch_kl / max(1, epoch_nb) > args.target_kl:
                break

        sps = (T * n) / (time.perf_counter() - t_iter)
        held = held_acc / T; sr = seated_acc / T; md = depth_acc / T; vf = viol_acc / T
        kl, ent_m, pl_m, vl_m = kl_acc / nupd, ent_acc / nupd, pl_acc / nupd, vl_acc / nupd
        writer.writerow([it, env.stage, total_steps, R.mean().item(), held, sr, md, vf,
                         kl, ent_m, pl_m, vl_m, sps])
        csv_f.flush()
        if it % 2 == 0 or it == args.iters - 1:
            print(f"it {it:4d} | stg {env.stage} | steps {total_steps:>9,} | R {R.mean().item():+.3f} | "
                  f"held {held:6.2%} | inst {sr:6.2%} | viol {vf:5.2%} | KL {kl:+.4f} | "
                  f"{sps:,.0f} st/s", flush=True)

        sr_window.append(held)
        if (not args.no_curriculum and env.stage < env.num_stages - 1
                and len(sr_window) == sr_window.maxlen
                and sum(sr_window) / len(sr_window) >= args.advance_held):
            env.set_stage(env.stage + 1); sr_window.clear()
            print(f"   [curriculum] -> stage {env.stage}", flush=True)

        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            r = evaluate_policy(ac, env)
            print(f"   [eval] held {r['success_rate']:6.2%} | depth {r['mean_depth_mm']:5.1f}mm",
                  flush=True)
            if r["success_rate"] > best_sr:
                best_sr = r["success_rate"]
                torch.save(ac.state_dict(), os.path.join(run_dir, "best_model.pt"))
                print(f"   [eval] new best {best_sr:.2%} -> best_model.pt", flush=True)

    torch.save(ac.state_dict(), os.path.join(run_dir, "final_model.pt"))
    csv_f.close()
    print(f"\nDone. best eval held {best_sr:.2%} | {run_dir}", flush=True)


if __name__ == "__main__":
    main()
