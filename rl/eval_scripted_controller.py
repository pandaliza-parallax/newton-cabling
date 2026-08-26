"""A/B the align-then-insert scripted controller against the servo teacher and a PPO policy.

All three drive the SAME RigidCableVecEnv through the same 7-D action, so the numbers are
directly comparable.

    .venv/bin/python rl/eval_scripted_controller.py --envs 8 --stage 4
    .venv/bin/python rl/eval_scripted_controller.py --envs 8 --all-stages
    .venv/bin/python rl/eval_scripted_controller.py --mode teacher
    .venv/bin/python rl/eval_scripted_controller.py --mode policy \\
        --checkpoint rl/runs/rigid_v9/final_model.pt

`--mode all` (the default) runs every available controller in one build.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import rigid_cable_env  # noqa: E402
from newton_cabling.scripted_controller import (  # noqa: E402
    AlignInsertController,
    config_for_rigid_cable_env,
    observe_rigid_cable_env,
)

DEV = "cuda:0"


def run(env, act_fn, steps: int, settle: int, ctrl: AlignInsertController | None = None) -> dict:
    """One deterministic rollout; report the tail-window held rate and the seat quality."""
    obs = env.reset()
    if ctrl is not None:
        ctrl.reset()
    held, viol_any, depth = [], np.zeros(env.n), []
    along_t, latn_t, ang_t = [], [], []
    ever = np.zeros(env.n)
    # An env whose plug leaves the EJECT_DIST ball has had its grasp destroyed by the
    # solver; its pose error is then metres wide and would swamp the seat-quality means,
    # so track it separately rather than averaging garbage into the report.
    ejected = np.zeros(env.n, dtype=bool)
    eject_step = np.full(env.n, -1)
    eject_phase = np.full(env.n, -1)
    for t in range(steps):
        with torch.no_grad():
            action = act_fn(obs)
        phase_now = ctrl.phase.copy() if ctrl is not None else None
        obs, _, _, succ, depth_mm = env.step(action)
        s = succ.cpu().numpy()
        ever = np.maximum(ever, s)
        viol_any += env.viol_last > 0
        bqn = env.state_0.body_q.numpy()
        face, faceq = env._face_pose(bqn)
        along, _, latn, ang = env._terms(face, faceq)
        gone = (~np.isfinite(face).all(axis=1)) | (
            np.linalg.norm(face - env.seat_pos, axis=1) > rigid_cable_env.EJECT_DIST)
        first = gone & ~ejected
        eject_step[first] = t
        if phase_now is not None:
            eject_phase[first] = phase_now[first]
        ejected |= gone
        if t >= steps - settle:
            held.append(s)
            depth.append(depth_mm.mean().item())
            along_t.append(along)
            latn_t.append(latn)
            ang_t.append(ang)
    H = np.asarray(held)
    ok = ~ejected
    m = ok if ok.any() else np.ones(env.n, dtype=bool)
    out = {
        "held": float(H.mean()),
        # the number that reflects the CONTROLLER: envs the solver destroyed at reset are
        # unrecoverable by any policy, so score them out rather than blaming the controller
        "held_ok": float(H[:, m].mean()),
        "held_per_env": H.mean(axis=0),
        "ever": float(ever.mean()),
        "depth_mm": float(np.mean(depth)),
        "viol_envs": int((viol_any > 0).sum()),
        "viol_step_frac": float((viol_any.sum()) / (steps * env.n)),
        # seat quality over the SURVIVING envs only
        "along_mm": float(np.mean(np.asarray(along_t)[:, m]) * 1000.0),
        "latn_mm": float(np.mean(np.asarray(latn_t)[:, m]) * 1000.0),
        "ang_deg": float(np.degrees(np.mean(np.asarray(ang_t)[:, m]))),
        "clean": int(((H.mean(axis=0) > 0.5) & (viol_any == 0) & ok).sum()),
        "ejected": int(ejected.sum()),
        "eject_steps": eject_step[ejected].tolist(),
        "eject_phases": eject_phase[ejected].tolist(),
    }
    if ctrl is not None:
        out["phases"] = ctrl.phase_counts()
        out["attempts"] = int(ctrl.attempts.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "scripted", "teacher", "policy"])
    ap.add_argument("--checkpoint", default=os.path.join(_HERE, "runs/rigid_v9/final_model.pt"))
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--all-stages", action="store_true")
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--settle", type=int, default=80)
    ap.add_argument("--tilt", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--standoff-mm", type=float, default=5.0,
                    help="pre-dock parking distance OUTSIDE the jack mouth")
    ap.add_argument("--push-correction", type=float, default=0.3,
                    help="0.0 = strictly open-loop straight push")
    args = ap.parse_args()

    tilt = (0.0, args.tilt) if args.tilt > 0 else 0.0
    t0 = time.perf_counter()
    env = rigid_cable_env.RigidCableVecEnv(args.envs, seed=args.seed, cable_tilt_deg=tilt)
    print(f"[build] {time.perf_counter() - t0:.1f}s | front_room {env.front_room * 1000:.1f}mm",
          flush=True)

    cfg = config_for_rigid_cable_env(
        rigid_cable_env,
        align_standoff_m=args.standoff_mm / 1000.0,
        push_correction=args.push_correction,
    )
    print(f"[config] mouth at along {cfg.mouth_along_m * 1000:+.1f}mm | pre-dock "
          f"{cfg.align_along_m * 1000:+.1f}mm | push_correction {cfg.push_correction}", flush=True)
    ctrl = AlignInsertController(args.envs, cfg)

    ac = None
    if args.mode in ("all", "policy") and os.path.exists(args.checkpoint):
        from train_ppo import ActorCritic
        ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
        ac.load_state_dict(torch.load(args.checkpoint, map_location=DEV))
        ac.eval()
    elif args.mode == "policy":
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    runners: list[tuple[str, object]] = []
    if args.mode in ("all", "scripted"):
        runners.append(("scripted", ctrl))
    if args.mode in ("all", "teacher"):
        runners.append(("teacher", None))
    if ac is not None and args.mode in ("all", "policy"):
        runners.append(("policy", ac))

    stages = range(env.num_stages) if args.all_stages else [args.stage]
    rows = []
    for stage in stages:
        env.set_stage(stage)
        for name, obj in runners:
            if name == "scripted":
                def act_fn(_obs):
                    return ctrl.act(observe_rigid_cable_env(env))
                res = run(env, act_fn, args.steps, args.settle, ctrl=ctrl)
            elif name == "teacher":
                res = run(env, lambda _o: env.servo_action(), args.steps, args.settle)
            else:
                res = run(env, obj.mean_action, args.steps, args.settle)
            res["stage"] = stage
            res["name"] = name
            rows.append(res)
            extra = ""
            if res["ejected"]:
                extra += (f" | EJECTED {res['ejected']} at steps {res['eject_steps']}"
                          f" phases {res['eject_phases']}")
            if "phases" in res:
                live = {k: v for k, v in res["phases"].items() if v}
                extra += f" | phases {live} retries {res['attempts']}"
            print(f"  stg {stage} {name:9s} held {res['held']:6.2%} (survivors {res['held_ok']:6.2%}) "
                  f"ever {res['ever']:5.0%} "
                  f"clean {res['clean']}/{env.n} | along {res['along_mm']:+6.2f}mm "
                  f"lat {res['latn_mm']:5.2f}mm ang {res['ang_deg']:5.2f}deg | "
                  f"viol {res['viol_envs']}/{env.n}{extra}", flush=True)

    print(f"\n{'stage':>5} {'controller':>10} {'held':>8} {'clean':>7} {'along':>8} "
          f"{'lat':>7} {'ang':>7} {'viol':>6}")
    for r in rows:
        print(f"{r['stage']:>5} {r['name']:>10} {r['held']:>7.1%} {r['clean']:>4}/{env.n} "
              f"{r['along_mm']:>7.2f} {r['latn_mm']:>6.2f} {r['ang_deg']:>6.2f} "
              f"{r['viol_envs']:>3}/{env.n}")
    for name, _ in runners:
        sel = [r["held_ok"] for r in rows if r["name"] == name]
        print(f"  mean held (survivors) over {len(sel)} stage(s), {name}: {np.mean(sel):.2%}")


if __name__ == "__main__":
    main()
