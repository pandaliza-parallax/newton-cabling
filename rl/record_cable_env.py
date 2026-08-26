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
    ap.add_argument("--scripted", action="store_true",
                    help="drive with the align-then-insert scripted controller "
                         "(newton_cabling.scripted_controller) instead of the servo teacher")
    ap.add_argument("--standoff-mm", type=float, default=5.0,
                    help="--scripted: pre-dock parking distance OUTSIDE the jack mouth")
    ap.add_argument("--push-correction", type=float, default=0.3,
                    help="--scripted: 0.0 = strictly open-loop straight push")
    ap.add_argument("--disturb-yaw", type=float, default=0.0,
                    help="before the controller starts, yaw the WRIST IN PLACE by this many "
                         "degrees about world +z. Unlike --cable-tilt (which servos nose droop "
                         "about the tilted jaw axis and mixes in yaw+roll) this is a PURE yaw. "
                         "Rotating in place commands dpos=0, so it costs no standoff-clamp "
                         "budget -- the failure mode that makes --cable-tilt 90 unrecoverable.")
    ap.add_argument("--disturb-rpy", default=None,
                    help="'R,P,Y' in degrees, applied in the SEAT frame: R = roll about the "
                         "insertion axis (the bore), P = pitch (nose up/down), Y = yaw. "
                         "Generalises --disturb-yaw to any axis; the reported decomposition "
                         "matches these numbers by construction.")
    ap.add_argument("--disturb-rate", type=float, default=0.3,
                    help="deg/step for the disturbance (the rate the friction grasp follows)")
    ap.add_argument("--grip-from-head", type=float, default=None,
                    help="mm from the plug FACE back to the grip point (default 68 = GRIP_BACK "
                         "50 + BOOT 18). Walking the jaws toward the head costs `front_room` "
                         "1:1, and seating needs front_room >= APPROACH + MARGIN = 33mm, so "
                         "~40mm is the hard floor (measured) -- below that the fingertips "
                         "enter the jack cavity before the plug seats.")
    ap.add_argument("--grasp-roll-180", action="store_true",
                    help="roll the grasp 180deg about the tool axis so the plug sits on the "
                         "SAME side as the wrist camera (matches gen_trajectories --grasp-roll-180)")
    ap.add_argument("--connector-usd", default="cad_rj45.usd",
                    help="connector asset for the RIGID env (e.g. scan_rj45.usd)")
    ap.add_argument("--jack-fixture", action="store_true",
                    help="mount the jack in the 3D-print bench fixture (rigid env only; "
                         "matches gen_trajectories --jack-fixture)")
    ap.add_argument("--out", default="cable_rollout")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    if args.rigid:
        import rigid_cable_env as env_mod
        EnvCls = env_mod.RigidCableVecEnv
    else:
        import cable_env as env_mod
        EnvCls = CableInsertVecEnv
    if args.grip_from_head is not None:
        if not args.rigid:
            raise SystemExit("--grip-from-head applies to the rigid env only")
        gb = args.grip_from_head / 1000.0 - env_mod.BOOT
        if gb <= 0.0:
            raise SystemExit(f"--grip-from-head must exceed BOOT ({env_mod.BOOT*1000:.1f}mm)")
        env_mod.GRIP_BACK = gb   # read as a module global inside __init__
        need = (env_mod.APPROACH + env_mod.MARGIN) * 1000
        print(f"grip {args.grip_from_head:.0f}mm from the plug face "
              f"(GRIP_BACK {gb*1000:.1f}mm); needs front_room >= {need:.0f}mm to seat")
    env_kw = dict(seed=args.seed, cable_tilt_deg=args.cable_tilt)
    if args.rigid:
        env_kw["connector_usd"] = args.connector_usd
        env_kw["grasp_roll_180"] = args.grasp_roll_180
        env_kw["jack_fixture"] = args.jack_fixture
    elif args.jack_fixture:
        raise SystemExit("--jack-fixture applies to the rigid env only")
    env = EnvCls(args.envs, **env_kw)
    env.set_stage(args.stage)
    ctrl = observe = None
    if args.scripted:
        from newton_cabling.scripted_controller import (
            AlignInsertController,
            config_for_rigid_cable_env,
            observe_rigid_cable_env,
        )
        cfg = config_for_rigid_cable_env(
            env_mod, align_standoff_m=args.standoff_mm / 1000.0,
            push_correction=args.push_correction)
        ctrl, observe = AlignInsertController(args.envs, cfg), observe_rigid_cable_env
        print(f"scripted align-then-insert | mouth {cfg.mouth_along_m * 1000:+.1f}mm | "
              f"pre-dock {cfg.align_along_m * 1000:+.1f}mm | "
              f"push_correction {cfg.push_correction}")
    ac = None
    if args.checkpoint:
        from train_ppo import ActorCritic
        ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
        ac.load_state_dict(torch.load(args.checkpoint, map_location=DEV))
        ac.eval()
        print(f"loaded {args.checkpoint}")
    elif ctrl is None:
        print("base servo (zero residual)")

    viewer = open_rrd_recorder(f"{args.out}.rrd")
    viewer.set_model(env.model)
    obs = env.reset()
    ever = torch.zeros(env.n, device=DEV)
    # trajectory dump (pi05 rj45_sbot layout): obs[:, :10] = state [eef_pos(3), eef_rot6d(6),
    # gripper(1)]; action in [-1,1]^7 = [dpos(3), drotvec(3), gripper(1)] (scales in meta)
    traj = {k: [] for k in ("obs", "action", "face_pos", "face_quat",
                            "along", "latn", "ang", "viol", "held", "depth_mm", "phase")}
    # ── PURE-YAW disturbance, applied BEFORE recording starts ────────────────
    # This is SCENE SETUP, not behaviour: it establishes the mismatched initial pose the
    # episode starts from, so it is stepped here, outside the recording loop -- no frames
    # logged, no trajectory samples. t=0 of the rollout is the controller's first action.
    # Mechanism: env.step applies a[:,3:6] as a WORLD-frame pre-multiply on the wrist
    # quaternion, so a[:,5] alone is a clean yaw; with a[:,0:3]=0 the wrist ORIGIN does not
    # translate at all (measured 0.0 mm in 3-D), `advanced` never moves, and the standoff
    # clamp stays wide open. The plug swings on the ~0.26 m lever, which is what actually
    # produces the pose offset.
    if args.disturb_yaw or args.disturb_rpy:
        from scipy.spatial.transform import Rotation as _Rot
        rpy = ([float(v) for v in args.disturb_rpy.split(",")] if args.disturb_rpy
               else [0.0, 0.0, args.disturb_yaw])
        rate = np.radians(args.disturb_rate)
        # Build the world-frame rotation axis per env. The requested R/P/Y are expressed in
        # the SEAT frame (x = insertion axis, y = horizontal-perpendicular, z = world up),
        # conjugated into world coords by M. Ramping about this ONE fixed axis composes
        # correctly under repeated world pre-multiplies (a rotation commutes with itself),
        # so the achieved pose matches the request rather than drifting.
        da = np.zeros((args.envs, 7))
        n_dist = 0
        for i in range(args.envs):
            ins_i = np.asarray(env.ins[i]); up_i = np.array([0.0, 0.0, 1.0])
            side_i = np.cross(up_i, ins_i)
            M = np.column_stack([ins_i, side_i, up_i])
            Rl = _Rot.from_euler("zyx", [rpy[2], rpy[1], rpy[0]], degrees=True).as_matrix()
            rv = _Rot.from_matrix(M @ Rl @ M.T).as_rotvec()
            tot = float(np.linalg.norm(rv))
            if tot < 1e-9:
                continue
            n_dist = max(n_dist, int(np.ceil(tot / rate)))
            da[i, 3:6] = (rv / tot) * rate / env_mod.MAX_DROT
        for _ in range(n_dist):
            obs, *_ = env.step(da)
        # re-baseline the episode-relative bookkeeping so the recorded rollout measures
        # itself, not the setup: depth_mm is |face - face_start| and would otherwise carry
        # the whole swing as a constant offset.
        bqn = env.state_0.body_q.numpy()
        face_d, faceq_d = env._face_pose(bqn)
        env.face_start = face_d.copy()
        env.prev_dist = env._dist(face_d, faceq_d)
        _, _, latn_d, ang_d = env._terms(face_d, faceq_d)
        print(f"setup: seat-frame roll/pitch/yaw = {rpy[0]:+.1f}/{rpy[1]:+.1f}/{rpy[2]:+.1f} deg "
              f"over {n_dist} pre-roll steps (NOT recorded) -> episode starts at "
              f"lat {latn_d[0]*1000:.1f}mm ang {np.degrees(ang_d[0]):.1f}deg")
    for t in range(args.steps):
        with torch.no_grad():
            if ctrl is not None:
                a = ctrl.act(observe(env))
            elif ac is not None:
                a = ac.mean_action(obs)
            else:
                a = env.servo_action()
        traj["obs"].append(obs.cpu().numpy())
        traj["action"].append(a if isinstance(a, np.ndarray) else a.cpu().numpy())
        # the scripted phase makes each episode trivially segmentable for datagen
        traj["phase"].append(ctrl.phase.copy() if ctrl is not None
                             else np.full(env.n, -1, dtype=np.int64))
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
    # read the scales from the env ACTUALLY recorded, not always cable_env
    MAX_DPOS, MAX_DROT = env_mod.MAX_DPOS, env_mod.MAX_DROT
    driver = ("scripted-align-insert" if ctrl is not None
              else args.checkpoint or "servo-teacher")
    np.savez_compressed(
        f"{args.out}_traj.npz",
        **{k: np.asarray(v) for k, v in traj.items()},        # each (T, n, ...)
        seat_pos=env.seat_pos, seat_quat=env.seat_q, ins_axis=env.ins,
        meta=np.array([f"policy={driver}",
                       f"stage={args.stage}", f"cable_tilt_deg={args.cable_tilt}",
                       f"seed={args.seed}", f"hz=60", f"max_dpos_m={MAX_DPOS}",
                       f"max_drot_rad={MAX_DROT}",
                       "state=obs[:,:10]=[eef_pos(3),eef_rot6d(6),gripper(1)]",
                       "action=[dpos(3),drotvec(3),gripper(1)] in [-1,1], x scales",
                       "phase=InsertPhase (SETTLE0 ALIGN1 PUSH2 HOLD3 RETREAT4), -1 if unscripted"]))
    print(f"\nrecorded {args.out}.rrd | ever-held {ever.mean().item():.0%} of {env.n}")
    print(f"trajectory -> {args.out}_traj.npz")
    print(f"view:  uvx --from rerun-sdk rerun {args.out}.rrd {args.out}.rbl")


if __name__ == "__main__":
    main()
