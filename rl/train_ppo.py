"""PPO trainer for the Newton RJ45-insertion task (ConnectorVecEnv).

Hand-rolled CleanRL-style PPO that runs entirely on the GPU against the batched
Newton env — actor proposes a target nudge, critic estimates value, advantages
are GAE, and each update is clipped to a trust region. Built for thousands of
parallel envs feeding one shared on-policy buffer.

Run (needs a CUDA GPU + the `sim` extra):
    uv run --extra sim python rl/train_ppo.py --envs 1000 --iters 400
    uv run --extra sim python rl/train_ppo.py --envs 5000 --iters 300 --run-name big

Logs per-iter metrics to stdout and runs/<name>/log.csv; saves best_model.pt
(by success rate) and final_model.pt.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque

import torch
import torch.nn as nn

from connector_env import ConnectorVecEnv

DEV = "cuda:0"
W_ACTION = 0.0      # OFF: a per-step action penalty rewards staying still (do-little optimum).
#                     The potential-based reward already discourages unhelpful actions.
LOG_STD_MIN = -2.0  # floor on policy log-std (sigma >= ~0.135). Without it the policy
LOG_STD_MAX = 1.0   # std collapses to ~0 (entropy -> -18) and the policy locks onto a
#                     single deterministic action that rams + diverges (the `big` run).


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, act_dim))
        self.vf = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        # start with small residual std so the base controller dominates early
        self.log_std = nn.Parameter(-1.0 * torch.ones(act_dim))

    def _dist(self, obs):
        mean = self.pi(obs)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs):
        dist = self._dist(obs)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), self.vf(obs).squeeze(-1)

    def mean_action(self, obs):
        return self.pi(obs)

    def value(self, obs):
        return self.vf(obs).squeeze(-1)

    def evaluate(self, obs, act):
        dist = self._dist(obs)
        return dist.log_prob(act).sum(-1), dist.entropy().sum(-1), self.vf(obs).squeeze(-1)


@torch.no_grad()
def evaluate_policy(ac: ActorCritic, env: ConnectorVecEnv, steps: int = 200):
    """Deterministic (mean-action) rollout reporting the SETTLED steady-state seated
    fraction: average over the last `settle_window` steps, after insertion has fully
    completed. NOTE: horizon raised 80 -> 200 and the window moved to the tail because
    the old 80-step / second-half metric averaged over the insertion ramp (the plug is
    still travelling in from the far stage-5 start, longer at high friction), which
    under-reported true success by ~40 points (e.g. mu=0.5 read 55% but is ~95% settled).
    The tail window measures whether the plug is seated AND held, not still inserting."""
    settle_window = 80
    obs = env.reset()
    seated, depth = [], []
    for t in range(steps):
        a = ac.mean_action(obs)
        obs, _, _, succ, depth_mm = env.step(a)
        if t >= steps - settle_window:
            seated.append(succ.mean().item())
            depth.append(depth_mm.mean().item())
    sr = sum(seated) / max(1, len(seated))
    md = sum(depth) / max(1, len(depth))
    return {"success_rate": sr, "mean_depth_mm": md, "episodes": env.n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--ent-coef-final", type=float, default=None,
                    help="anneal ent-coef to this (e.g. 0.0) to converge higher")
    ap.add_argument("--ent-anneal-frac", type=float, default=1.0,
                    help="reach ent-coef-final by this fraction of iters, then hold (steep decay = small)")
    ap.add_argument("--init-from", default=None, help="warm-start the policy from a checkpoint .pt")
    ap.add_argument("--reset-log-std", type=float, default=None,
                    help="after warm-start, reset log_std to this (re-inflate exploration to learn "
                         "a NEW objective from a converged/deterministic policy)")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.02,
                    help="stop the epoch loop early once an epoch's mean KL exceeds this")
    ap.add_argument("--no-anneal-lr", action="store_true", help="disable linear LR decay")
    ap.add_argument("--advance-depth", type=float, default=18.0,
                    help="advance the curriculum stage when window-mean insertion depth (mm) "
                         "exceeds this (depth is honest/unbiased, unlike the seated ratio)")
    ap.add_argument("--advance-window", type=int, default=10,
                    help="iters of sustained progress required before advancing a stage")
    ap.add_argument("--advance-seated", type=float, default=0.12,
                    help="random_easy: advance when rolling instantaneous seated-fraction exceeds this")
    ap.add_argument("--no-curriculum", action="store_true",
                    help="pin the start distance to the final (hardest) stage")
    ap.add_argument("--residual-scale", type=float, default=None,
                    help="override env.residual_scale; 0.0 = base controller alone (no RL)")
    ap.add_argument("--random-easy", action="store_true",
                    help="random_easy_subset starts (lateral + approach + <=15deg rotation); "
                         "uses the 6-DOF driven-d6 rig, no curriculum")
    ap.add_argument("--contact-buffer", type=int, default=64)
    ap.add_argument("--asset", default="rj45", choices=["rj45", "cad_rj45", "cad_rj45_real"],
                    help="connector: toy 'rj45', idealized 'cad_rj45', or real-mesh 'cad_rj45_real'")
    ap.add_argument("--friction", type=float, default=None,
                    help="override contact friction μ (cad_rj45 only; default = spec value)")
    ap.add_argument("--friction-dr", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="domain-randomize friction per-env over [LO,HI] -> one policy across μ")
    ap.add_argument("--plug-scale-dr", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="domain-randomize plug SIZE per-env over [LO,HI] (varies fit clearance)")
    ap.add_argument("--connector-scale-dr", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="domain-randomize whole-connector SIZE per-env over [LO,HI] (socket+plug+latch)")
    ap.add_argument("--obs-contact", action="store_true",
                    help="add the plug-frame net contact force (3-dim) to the observation")
    ap.add_argument("--angular-kd", type=float, default=None,
                    help="override the d6 angular drive damping (default 6.0; higher damps seat wobble)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="ppo")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--eval-every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    n, T = args.envs, args.rollout
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "log.csv")
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["iter", "stage", "env_steps", "mean_rew", "success_rate", "mean_depth_mm",
                     "ep_done", "approx_kl", "entropy", "pol_loss", "val_loss", "steps_per_s"])

    print(f"Building {n} envs ...", flush=True)
    t0 = time.perf_counter()
    env = ConnectorVecEnv(n, contact_buffer=args.contact_buffer, seed=args.seed,
                          random_easy=args.random_easy, asset=args.asset, friction=args.friction,
                          obs_contact=args.obs_contact, residual_scale=args.residual_scale,
                          angular_kd_override=args.angular_kd, friction_dr=args.friction_dr,
                          plug_scale_dr=args.plug_scale_dr, connector_scale_dr=args.connector_scale_dr)
    if args.residual_scale is not None:
        env.residual_scale = args.residual_scale
    print(f"  built in {time.perf_counter()-t0:.1f}s | obs_dim {env.obs_dim} act_dim {env.act_dim}")
    cc, cbuf = env.contact_count()
    print(f"  contact buffer: {cc} active / {cbuf} capacity"
          f"{'  ⚠ OVERFLOW — raise --contact-buffer' if cc >= cbuf else ''}")

    ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
    if args.init_from:
        ac.load_state_dict(torch.load(args.init_from, map_location=DEV))
        print(f"  warm-started policy from {args.init_from}")
        if args.reset_log_std is not None:
            with torch.no_grad():
                ac.log_std.fill_(args.reset_log_std)
            print(f"  reset log_std -> {args.reset_log_std} (re-inflated exploration)")
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    print(f"  policy params: {sum(p.numel() for p in ac.parameters()):,}")

    env.set_stage(env.num_stages - 1 if args.no_curriculum else 0)
    sr_window = deque(maxlen=args.advance_window)
    _lvl = f"re_scale {env.re_scale:.2f}" if args.random_easy else f"max offset {env._mag*1000:.1f}mm"
    print(f"  curriculum: {'OFF (final stage)' if args.no_curriculum else f'{env.num_stages} stages'}"
          f" | start stage {env.stage} ({_lvl})")
    obs = env.reset()
    batch = T * n
    mb = batch // args.minibatches
    best_sr = -1.0
    total_steps = 0
    O = torch.zeros(T, n, env.obs_dim, device=DEV)
    A = torch.zeros(T, n, env.act_dim, device=DEV)
    LP = torch.zeros(T, n, device=DEV)
    V = torch.zeros(T, n, device=DEV)
    R = torch.zeros(T, n, device=DEV)
    D = torch.zeros(T, n, device=DEV)

    for it in range(args.iters):
        if not args.no_anneal_lr:
            lr_now = args.lr * (1.0 - it / args.iters)
            for g in opt.param_groups:
                g["lr"] = lr_now
        # entropy-coef annealing: high early (explore), low late (commit/converge higher).
        # Reaches ent_coef_final by ent_anneal_frac of training, then holds (steep = small frac).
        if args.ent_coef_final is not None:
            p = min(1.0, (it / max(1, args.iters - 1)) / max(1e-6, args.ent_anneal_frac))
            ent_coef = args.ent_coef + (args.ent_coef_final - args.ent_coef) * p
        else:
            ent_coef = args.ent_coef
        t_iter = time.perf_counter()
        obs = env.reset()  # fixed-horizon episode per rollout (VBD-safe: reset only at boundary)
        seated_acc, depth_acc = 0.0, 0.0  # HONEST: instantaneous seated fraction / mean depth
        for t in range(T):
            with torch.no_grad():
                a, lp, v = ac.act(obs)
            nobs, rew, done, succ, depth_mm = env.step(a)
            act_pen = W_ACTION * (a.clamp(-1, 1).pow(2).mean(-1))  # ||action||^2 normalized
            step_r = torch.nan_to_num(rew - act_pen).clamp(-20.0, 150.0)  # safety net vs blow-up
            O[t], A[t], LP[t], V[t], R[t], D[t] = obs, a, lp, v, step_r, done
            obs = nobs
            seated_acc += succ.mean().item()
            depth_acc += depth_mm.mean().item()
            total_steps += n

        with torch.no_grad():
            last_v = ac.value(obs)
        adv = torch.zeros(T, n, device=DEV)
        gae = torch.zeros(n, device=DEV)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            delta = R[t] + args.gamma * nv * (1 - D[t]) - V[t]
            gae = delta + args.gamma * args.lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V

        bo = O.reshape(batch, env.obs_dim)
        ba = A.reshape(batch, env.act_dim)
        blp = LP.reshape(batch)
        badv = adv.reshape(batch)
        bret = ret.reshape(batch)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)

        kl_acc, ent_acc, pl_acc, vl_acc, nupd = 0.0, 0.0, 0.0, 0.0, 0
        for _ in range(args.epochs):
            idx = torch.randperm(batch, device=DEV)
            epoch_kl, epoch_nb = 0.0, 0
            for s in range(0, batch, mb):
                j = idx[s:s + mb]
                nlp, ent, nv = ac.evaluate(bo[j], ba[j])
                logratio = nlp - blp[j]
                ratio = logratio.exp()
                pl = -torch.min(ratio * badv[j],
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * badv[j]).mean()
                vl = (nv - bret[j]).pow(2).mean()
                loss = pl + args.vf_coef * vl - ent_coef * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), args.max_grad)
                opt.step()
                with torch.no_grad():
                    approx_kl = (ratio - 1 - logratio).mean().item()  # non-negative estimator
                kl_acc += approx_kl
                epoch_kl += approx_kl
                epoch_nb += 1
                ent_acc += ent.mean().item()
                pl_acc += pl.item()
                vl_acc += vl.item()
                nupd += 1
            if epoch_kl / max(1, epoch_nb) > args.target_kl:  # trust-region early stop
                break

        sps = (T * n) / (time.perf_counter() - t_iter)
        sr = seated_acc / T   # honest: mean instantaneous seated fraction over the rollout
        md = depth_acc / T
        kl, ent_m, pl_m, vl_m = (kl_acc / nupd, ent_acc / nupd, pl_acc / nupd, vl_acc / nupd)
        writer.writerow([it, env.stage, total_steps, R.mean().item(), sr, md, n,
                         kl, ent_m, pl_m, vl_m, sps])
        csv_f.flush()
        if it % 5 == 0 or it == args.iters - 1:
            print(f"it {it:4d} | stg {env.stage} | steps {total_steps:>10,} | "
                  f"R {R.mean().item():+.3f} | seated {sr:6.2%} | depth {md:5.1f}mm | "
                  f"KL {kl:+.4f} | ent {ent_m:+.2f} | {sps:,.0f} st/s", flush=True)

        # curriculum advancement. translation env: advance on insertion depth. random_easy
        # subset: advance on seated% (depth is a poor signal when rotation/lateral are the hard
        # part). Both ramp the start difficulty once the policy is solid at the current level.
        adv_metric = sr if args.random_easy else md
        adv_thresh = args.advance_seated if args.random_easy else args.advance_depth
        sr_window.append(adv_metric)
        if (not args.no_curriculum and env.stage < env.num_stages - 1
                and len(sr_window) == sr_window.maxlen
                and sum(sr_window) / len(sr_window) >= adv_thresh):
            env.set_stage(env.stage + 1)
            sr_window.clear()
            lvl = f"re_scale {env.re_scale:.2f}" if args.random_easy else f"offset {env._mag*1000:.1f}mm"
            print(f"   [curriculum] advance -> stage {env.stage} ({lvl})", flush=True)

        if (it + 1) % args.eval_every == 0 or it == args.iters - 1:
            r = evaluate_policy(ac, env)
            print(f"   [eval] SR {r['success_rate']:6.2%} | depth {r['mean_depth_mm']:5.1f}mm "
                  f"| {r['episodes']} eps", flush=True)
            if r["success_rate"] > best_sr:
                best_sr = r["success_rate"]
                torch.save(ac.state_dict(), os.path.join(run_dir, "best_model.pt"))
                print(f"   [eval] new best SR {best_sr:.2%} -> best_model.pt", flush=True)
            obs = env.reset()  # eval perturbed the env state

    torch.save(ac.state_dict(), os.path.join(run_dir, "final_model.pt"))
    csv_f.close()
    print(f"\nDone. best eval SR {best_sr:.2%} | logs+checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
