"""Generate SEATED cable-insertion trajectories with the trained cable PPO policy.

The cable track runs the OPPOSITE direction from the rigid track (gen_seated_traj.py):

    rigid:  plug pose -> (heuristic cord offset) -> gripper pose -> IK -> arm
    cable:  gripper pose (policy) -> cable physics -> plug pose

So this exporter records the **wrist/EEF** pose as the primary channel -- it is what the
policy commands and what the renderer must replay -- plus the Newton-tracked connector
pose as a secondary channel, so the plug splat can be drawn at its TRUE physics pose
instead of being inferred from a cord-axis estimate. Feed the result into
``scripts/record_sbot_scene_gs.py --eef-traj`` (see tools/render_batch_v4.sh).

    .venv/bin/python rl/gen_cable_traj.py --checkpoint rl/runs/cable_v3/best_model.pt \
        --out cable_traj --envs 64 --rounds 8 --stage 4 --cable-tilt 8

Conventions: poses are stored FULLY SEAT-RELATIVE -- position AND orientation re-based on
that episode's seat frame -- scalar-first **wxyz** (Newton's body_q is scalar-last xyzw, see
newton_pose). This differs from plug_traj.npy, which stored absolute quats; that was only safe
because the rigid replay drove the arm from the home-relative grasp hack and used the quat
just to pose the plug splat. Here the recorded orientation IS the arm's command, so leaving it
in sim-world frame flips the arm ~236 deg off home in the renderer's scene.

Success is read from the env's hold counter (>= HOLD_STEPS consecutive seated steps), never
from a fixed time window: CAD_RJ45_PPO.md §8b documents a horizon artifact where a
fixed-window metric understated success by ~40 points by measuring insertion-in-progress.
Episodes are cut at the success frame (+ --pad) and capped at --max-keep, per CABLE_PPO.md's
ejection caveat (some seeds twist violently past t~150).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cable_env import HOLD_STEPS, CableInsertVecEnv  # noqa: E402
from train_ppo import ActorCritic  # noqa: E402

DEV = "cuda:0"


def _pose_wxyz(bqn, idx):
    """(n,7) [pos3, quat4 wxyz] for a body-index array from a body_q array (xyzw rows)."""
    p = bqn[idx, :3]
    q = bqn[idx, 3:7]                                   # qx qy qz qw
    return np.concatenate([p, q[:, [3, 0, 1, 2]]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="rl/runs/cable_v3/best_model.pt",
                    help="trained cable PPO checkpoint; omit to use the scripted servo teacher")
    ap.add_argument("--out", default="../data/vla_train/cable_traj")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=8, help="episodes = envs * rounds (before filtering)")
    ap.add_argument("--stage", type=int, default=4, help="curriculum stage (4 = hardest)")
    ap.add_argument("--cable-tilt", type=float, nargs="+", default=[8.0], metavar="DEG",
                    help="cable droop (deg). ONE value = every episode gets exactly that droop "
                         "(cable_env fills n with it); TWO values = per-env uniform DR in [lo,hi]. "
                         "Datagen wants the range -- a single value makes every episode start "
                         "from an identical hang, which is the axis the task is about.")
    ap.add_argument("--steps", type=int, default=130,
                    help="rollout horizon; keep <= ~130 (CABLE_PPO.md ejection caveat)")
    ap.add_argument("--max-keep", type=int, default=100,
                    help="hard CEILING on saved episode length. NOTE: this is not a target -- the "
                         "success cut (first_hold + 1 + --pad, ~42 frames) normally binds first, so "
                         "raising this alone changes nothing. Use --no-cut-at-success to make it bind.")
    ap.add_argument("--pad", type=int, default=5, help="frames kept after the success frame")
    ap.add_argument("--no-cut-at-success", action="store_true",
                    help="do NOT truncate at success+pad; keep the whole rollout up to --max-keep. "
                         "Success is still REQUIRED for the episode to be saved (hold>=HOLD_STEPS), "
                         "it just no longer sets the length. WARNING: the policy seats by ~frame 18, "
                         "so every extra frame is the plug sitting STILL -- at 100 frames the episode "
                         "is ~82% static vs ~58% at the default cut.")
    ap.add_argument("--keep-existing", action="store_true",
                    help="append to the out dir instead of clearing stale ep_* dirs first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-save", type=int, default=500)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample from the policy instead of the mean -- widens the behaviour "
                         "distribution for BC at the cost of some success rate")
    ap.add_argument("--max-viol-frac", type=float, default=0.05,
                    help="reject episodes whose finger<->jack violation fraction exceeds this")
    ap.add_argument("--keep-failures", action="store_true",
                    help="also save episodes that never held (default: successes only)")
    args = ap.parse_args()

    # scalar -> np.full(n, tilt) (identical hang every env); tuple -> per-env uniform DR
    tilt = tuple(args.cable_tilt) if len(args.cable_tilt) > 1 else args.cable_tilt[0]
    env = CableInsertVecEnv(args.envs, seed=args.seed, cable_tilt_deg=tilt)
    env.set_stage(args.stage)
    ac = None
    if args.checkpoint:
        ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
        ac.load_state_dict(torch.load(args.checkpoint, map_location=DEV))
        ac.eval()
        print(f"[gen] policy {args.checkpoint} ({'sampled' if args.stochastic else 'mean'})")
    else:
        print("[gen] scripted servo teacher")

    # Ground-truth geometry indices, so the saved episode can be re-rendered EXACTLY by
    # rl/render_cable_newton.py --from-traj. Without these the Newton view has to re-roll the
    # policy, and VBD nondeterminism makes it a DIFFERENT episode than the one GS replays.
    _labels = list(env.model.body_label)
    _sfx = lambda s: [i for i, l in enumerate(_labels) if l.endswith(s)]
    tip1_i = _sfx("gripper_finger1_finger_tip_link")
    tip2_i = _sfx("gripper_finger2_finger_tip_link")
    rods_i = [[int(b) for b in env.rod_bodies_all[i]] for i in range(args.envs)]
    n_rod = len(rods_i[0])

    os.makedirs(args.out, exist_ok=True)
    # Clear stale ep_* dirs: a shorter/rejected run leaves higher-numbered episodes from a previous
    # run behind, silently mixing two configs (e.g. 42-frame and 100-frame episodes) in one dataset.
    _stale = sorted(glob.glob(os.path.join(args.out, "ep_*")))
    if _stale and not args.keep_existing:
        import shutil
        for p in _stale:
            shutil.rmtree(p, ignore_errors=True)
        print(f"[gen] cleared {len(_stale)} existing ep_* dirs in {args.out} "
              f"(pass --keep-existing to append instead)")
    saved = tot = succ_tot = 0
    T = args.steps

    for r in range(args.rounds):
        obs = env.reset()
        eef = np.zeros((T, args.envs, 7), np.float32)     # wrist pose  (the POLICY's channel)
        conn = np.zeros((T, args.envs, 7), np.float32)    # front-rod body (carries the connector)
        face = np.zeros((T, args.envs, 7), np.float32)    # plug FACE pose (the seat metric's frame)
        acts = np.zeros((T, args.envs, 7), np.float32)    # raw policy output, pre-clamp
        state = np.zeros((T, args.envs, 10), np.float32)  # pi05 state layout = obs[:10]
        tips = np.zeros((T, args.envs, 2, 3), np.float32)      # fingertip positions (grasp truth)
        rods = np.zeros((T, args.envs, n_rod, 3), np.float32)  # cable rod chain (the cord)
        hold = np.zeros((T, args.envs), np.int32)
        viol = np.zeros((T, args.envs), np.float32)
        seat_pos = env.seat_pos.copy()                    # (n,3) this episode's seat
        seat_q = env.seat_q.copy()
        jack_p = env.state_0.body_q.numpy()[[int(j) for j in env.jack_body], :3].copy()

        for t in range(T):
            bqn = env.state_0.body_q.numpy()
            eef[t] = _pose_wxyz(bqn, env.wrist_body)
            conn[t] = _pose_wxyz(bqn, env.rod_front)
            fp, fq = env._face_pose(bqn)
            face[t] = np.concatenate([fp, fq[:, [3, 0, 1, 2]]], axis=1)
            tips[t, :, 0] = bqn[tip1_i, :3]
            tips[t, :, 1] = bqn[tip2_i, :3]
            for _e in range(args.envs):
                rods[t, _e] = bqn[rods_i[_e], :3]
            state[t] = obs[:, :10].detach().cpu().numpy()
            with torch.no_grad():
                if ac is None:
                    a = env.servo_action()
                elif args.stochastic:
                    a, _, _ = ac.act(obs)
                else:
                    a = ac.mean_action(obs)
            acts[t] = np.clip(np.asarray(a.detach().cpu(), np.float32), -1.0, 1.0)
            obs, _, _, _, _ = env.step(a)
            hold[t] = env.hold
            viol[t] = (env.viol_last > 0).astype(np.float32)

        # per-env success = the first frame the hold counter reaches HOLD_STEPS
        tot += args.envs
        for e in range(args.envs):
            if saved >= args.max_save:
                break
            hit = np.where(hold[:, e] >= HOLD_STEPS)[0]
            ok = len(hit) > 0
            succ_tot += int(ok)
            # success still GATES inclusion; --no-cut-at-success only stops it setting the LENGTH
            cut = T if (args.no_cut_at_success or not ok) else min(int(hit[0]) + 1 + args.pad, T)
            cut = min(cut, args.max_keep)
            vf = float(viol[:cut, e].mean())
            if not ok and not args.keep_failures:
                continue
            if vf > args.max_viol_frac:
                print(f"[gen] round {r} env {e}: rejected, viol_frac {vf:.3f}")
                continue
            d = os.path.join(args.out, f"ep_{saved:04d}")
            os.makedirs(d, exist_ok=True)
            # Fully SEAT-RELATIVE: position AND orientation. plug_traj.npy stored absolute quats,
            # which was safe there only because the arm was driven by the home-relative grasp hack
            # and the absolute quat merely posed the plug splat. Here the recorded orientation IS
            # the arm's command, so a sim-world -> scene frame mismatch flips the arm (measured
            # 236 deg off home). Re-basing on the seat makes the transfer jack-anchored, exactly
            # like the positions: q_world = jack_q (x) q_saved.
            qs_inv = Rot.from_quat(seat_q[e]).inv()
            for name, arr in (("eef_traj", eef), ("conn_traj", conn), ("face_traj", face)):
                a7 = arr[:cut, e].copy()
                a7[:, :3] = qs_inv.apply(a7[:, :3] - seat_pos[e])
                q_xyzw = a7[:, [4, 5, 6, 3]]                            # stored wxyz -> xyzw
                q_rel = (qs_inv * Rot.from_quat(q_xyzw)).as_quat()      # xyzw
                a7[:, 3:] = q_rel[:, [3, 0, 1, 2]]                      # back to wxyz
                np.save(os.path.join(d, f"{name}.npy"), a7)
            # ground-truth geometry (positions only), same seat-relative frame as the poses above
            for name, arr in (("tips_traj", tips), ("rods_traj", rods)):
                pts = arr[:cut, e].copy()                               # (T,k,3)
                sh = pts.shape
                pts = qs_inv.apply((pts.reshape(-1, 3) - seat_pos[e])).reshape(sh)
                np.save(os.path.join(d, f"{name}.npy"), pts.astype(np.float32))
            np.save(os.path.join(d, "actions_policy.npy"), acts[:cut, e])
            np.save(os.path.join(d, "state_sim.npy"), state[:cut, e])
            json.dump({
                "success": bool(ok), "frames": int(cut), "viol_frac": vf,
                "seat_pos": seat_pos[e].tolist(), "seat_quat_xyzw": seat_q[e].tolist(),
                "jack_pos_seatrel": qs_inv.apply(jack_p[e] - seat_pos[e]).tolist(),
                "stage": args.stage, "cable_tilt_deg": args.cable_tilt,
                "tilt_this_env_deg": float(np.degrees(env.cable_tilt[e])),
                "seed": args.seed, "round": r, "env": e,
                "checkpoint": args.checkpoint, "stochastic": bool(args.stochastic),
            }, open(os.path.join(d, "meta.json"), "w"), indent=2)
            saved += 1
        print(f"[gen] round {r + 1}/{args.rounds}: held {succ_tot}/{tot} "
              f"({100 * succ_tot / max(tot, 1):.0f}%), saved {saved}", flush=True)
        if saved >= args.max_save:
            break

    print(f"[gen] done: {succ_tot}/{tot} held ({100 * succ_tot / max(tot, 1):.0f}%); "
          f"saved {saved} episodes to {args.out}/")


if __name__ == "__main__":
    main()
