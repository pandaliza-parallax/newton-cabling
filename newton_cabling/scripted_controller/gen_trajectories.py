"""Export VLA training trajectories driven by the scripted align-then-insert controller.

Drop-in replacement for the PPO-checkpoint exporter (``rl/gen_cable_traj.py``): it writes the
identical ``ep_XXXX/`` layout, so ``tools/render_batch_v4.sh`` and ``tools/datagen_to_lerobot.py``
consume it with no changes.

    .venv/bin/python -m newton_cabling.scripted_controller.gen_trajectories \\
        --out ../data/vla_train/cable_traj --envs 32 --rounds 4 --stage 4

Why this instead of the policy: the controller seats 100% of surviving envs across all five
curriculum stages with ZERO finger<->jack violations, and parks at the design depth
(``along`` -0.04 mm) rather than ~8 mm past it. So the success gate, the ``--max-viol-frac``
filter and the ``viol == 0`` datagen filter all become no-ops, and the demonstrated end-state
is the correct one.

Two differences from the policy data, both deliberate:
  * Episodes are ~2.3x longer (~110 vs ~46 frames): the controller rate-limits rotation to the
    0.3 deg/step the friction grasp actually follows, so ALIGN takes real time. Its length
    scales with the start error, which is a *feature* for BC -- the dataset covers the whole
    approach, not just the last few centimetres.
  * The actions are smooth and deterministic, peaking near 40% of the per-step caps. That is
    well-conditioned but narrow, so ``--action-noise`` optionally perturbs the EXECUTED action
    (and records what was executed, never the clean command -- recording the clean one would
    teach the VLA an action it never saw the consequences of).

Module-level imports are pure NumPy; the Newton environment is imported lazily inside
``main()`` so this file can still be imported, linted and inspected without a GPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import sys

import numpy as np

from newton_cabling.scripted_controller.align_insert import (
    AlignInsertConfig,
    AlignInsertController,
)
from newton_cabling.scripted_controller.quaternion import (
    quat_conjugate,
    quat_multiply,
    quat_rotate,
)

__all__ = ["main"]


def _repo_root() -> str:
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def _load_env_module():
    """Import ``rl/rigid_cable_env.py`` (pulls Newton/Warp/Torch -- GPU only)."""
    root = _repo_root()
    for p in (os.path.join(root, "rl"), root):
        if p not in sys.path:
            sys.path.insert(0, p)
    import rigid_cable_env

    return rigid_cable_env


def _pose_wxyz(bqn: np.ndarray, idx) -> np.ndarray:
    """(n,7) [pos3, quat4 WXYZ] for a body-index array. Newton's body_q rows are XYZW."""
    return np.concatenate([bqn[idx, :3], bqn[idx, 3:7][:, [3, 0, 1, 2]]], axis=1)


def to_seat_frame(poses: np.ndarray, seat_pos: np.ndarray, seat_quat: np.ndarray) -> np.ndarray:
    """Re-base ``(T,7)`` [pos, WXYZ] world poses onto the episode's seat frame.

    Position AND orientation, unlike the older ``plug_traj.npy`` which stored absolute quats.
    That was only safe because the rigid replay drove the arm from a home-relative grasp hack
    and used the quat merely to pose the plug splat. Here the recorded orientation IS the arm's
    command, so leaving it in sim-world frame flips the arm ~236 deg off home in the renderer.
    """
    q_inv = quat_conjugate(np.asarray(seat_quat, dtype=np.float64))
    out = np.asarray(poses, dtype=np.float64).copy()
    out[:, :3] = quat_rotate(q_inv, out[:, :3] - np.asarray(seat_pos, dtype=np.float64))
    q_rel = quat_multiply(q_inv, out[:, [4, 5, 6, 3]])  # stored WXYZ -> XYZW -> relative
    out[:, 3:] = q_rel[:, [3, 0, 1, 2]]  # back to WXYZ
    return out.astype(np.float32)


def to_seat_frame_points(
    pts: np.ndarray, seat_pos: np.ndarray, seat_quat: np.ndarray
) -> np.ndarray:
    """Re-base ``(T,k,3)`` world points onto the seat frame."""
    q_inv = quat_conjugate(np.asarray(seat_quat, dtype=np.float64))
    p = np.asarray(pts, dtype=np.float64)
    flat = quat_rotate(q_inv, p.reshape(-1, 3) - np.asarray(seat_pos, dtype=np.float64))
    return flat.reshape(p.shape).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="../data/vla_train/cable_traj")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument(
        "--rounds", type=int, default=4, help="episodes = envs * rounds (before filtering)"
    )
    ap.add_argument("--stage", type=int, default=4, help="curriculum stage (4 = hardest)")
    ap.add_argument(
        "--cable-tilt",
        type=float,
        nargs="+",
        default=[0.0, 8.0],
        metavar="DEG",
        help="ONE value = identical droop in every env; TWO = per-env uniform DR. "
        "Defaults to DR: a scalar goes to np.full(n, tilt) and every episode "
        "gets the same hang, which silently collapses the dataset's variety.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=160,
        help="rollout horizon. The controller needs SETTLE + ALIGN + PUSH + "
        "HOLD_STEPS ~= 110 frames at stage 4; the policy's old 130 default "
        "truncates it. No ejection cap is needed -- unlike the policy, this "
        "controller is rate-limited and closed-loop (verified clean to 3000).",
    )
    ap.add_argument("--max-keep", type=int, default=160, help="hard cap on saved frames")
    ap.add_argument("--pad", type=int, default=5, help="frames kept after the success frame")
    ap.add_argument(
        "--no-cut-at-success",
        action="store_true",
        help="keep the whole rollout instead of truncating at success + --pad",
    )
    ap.add_argument(
        "--action-noise",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Gaussian sigma added to the dpos/drot action channels, in [-1,1] "
        "action units, to widen the BC action distribution. The controller is "
        "closed-loop so it corrects the perturbation, which is what makes this "
        "useful rather than merely noisy. Commands peak near 0.4, so sigma "
        "<=0.2 stays inside the caps. 0 = deterministic.",
    )
    ap.add_argument(
        "--standoff-mm",
        type=float,
        default=5.0,
        help="pre-dock parking distance OUTSIDE the jack mouth",
    )
    ap.add_argument(
        "--push-correction", type=float, default=0.3, help="0.0 = strictly open-loop straight push"
    )
    ap.add_argument(
        "--grip-from-head",
        type=float,
        default=None,
        help="mm from the plug FACE back to the grip point (default 68). Below "
        "~40mm the fingertips enter the jack cavity before the plug seats.",
    )
    ap.add_argument(
        "--boot-mm",
        type=float,
        default=None,
        help="override the env's BOOT length in mm (default 18): the rigid boot gap "
        "between the plug FACE and the front of the bare-cable capsule. Without "
        "--grip-from-head this also moves the grip to (BOOT + 50mm) from the face.",
    )
    ap.add_argument(
        "--grip-z-off-mm",
        type=float,
        default=None,
        help="WORLD-z offset in mm of the cable centerline from the measured pad-face "
        "midpoint (negative = held lower between the pads; default 0). Keep within "
        "the pad flats, ~ a few mm.",
    )
    ap.add_argument(
        "--max-viol-frac",
        type=float,
        default=0.05,
        help="reject an episode if this fraction of frames has finger<->jack "
        "contact (expected to be a no-op here; kept as a guard)",
    )
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument(
        "--keep-existing",
        action="store_true",
        help="append to the output dir instead of clearing stale ep_* dirs",
    )
    ap.add_argument(
        "--connector-usd",
        default="cad_rj45.usd",
        help="connector asset: 'cad_rj45.usd' = clean parametric primitives (222-face plug); "
        "'scan_rj45.usd' = the headA SCAN plug with the socket re-carved to fit it "
        "(30k-face plug, 0.66mm seat slop). The scan is ~135x denser, so expect lower "
        "throughput and watch for contact-buffer pressure.",
    )
    ap.add_argument(
        "--grasp-roll-180",
        action="store_true",
        help="roll the grasp 180 deg about the tool axis. The AG-145 is symmetric under this "
        "so the grip is unchanged, but the plug swings to the same side as the eye-in-hand "
        "camera (wrist_3 +z). Without it the camera and plug sit on opposite sides of the "
        "hand and the closed jaws occlude the plug/jack interface in EVERY wrist frame.",
    )
    ap.add_argument(
        "--jack-fixture",
        action="store_true",
        help="mount the jack in the 3D-print bench fixture (jack_fixture_rj45.usd, baked "
        "into the socket frame by build_jack_fixture_usd.py): one extra static-on-jack "
        "collision shape riding the kinematic jack body, so placement + yaw DR carry it. "
        "The sleeve's front face sits 3.2mm BEHIND the mouth plane (collar proud), so the "
        "approach and seat are geometrically unchanged; only stray contacts differ.",
    )
    ap.add_argument(
        "--jack-yaw",
        type=float,
        default=0.0,
        metavar="DEG",
        help="per-EPISODE jack yaw DR: |yaw| <= DEG about world-z, uniform, drawn every "
        "reset. Yaw is about the gravity axis so the drape needs no re-settle; in the "
        "seat-relative episode it reads as a yawed gripper start. Keep the combined "
        "start cone (yaw + droop) well inside SEAT_ANGLE (8 deg). 0 = off.",
    )
    ap.add_argument(
        "--grasp-roll-jitter",
        type=float,
        default=0.0,
        metavar="DEG",
        help="per-ENV grasp-roll DR: GRASP_ROLL_DEG +- DEG about the tool axis, uniform, "
        "one draw per env (baked into the settled snapshot -- more envs = more distinct "
        "rolls; rounds do NOT resample it). Rolls the whole hand+cable; the grip itself "
        "is unchanged and the drape re-settles. 0 = off.",
    )
    ap.add_argument(
        "--jam-window",
        type=int,
        default=None,
        metavar="STEPS",
        help="controller jam detector: declare a jam after this many PUSH steps without "
        "progress (default 20). Tighter values fire clean retreats on the servo plant's "
        "slow pushes — the recovery-demo knob that does NOT dirty the actions.",
    )
    ap.add_argument(
        "--jam-progress-m",
        type=float,
        default=None,
        metavar="M",
        help="progress threshold per step for the jam detector (default 0.0005)",
    )
    ap.add_argument(
        "--kick-prob",
        type=float,
        default=0.0,
        metavar="P",
        help="perturbation-recovery kicks: per-step per-env probability of starting a "
        "lateral kick while the controller is in ALIGN. During a kick the executed action "
        "is REPLACED by the kick vector (the controller keeps running on true state and "
        "corrects afterwards). Kick frames are marked in kick.npy; split episodes at the "
        "kick windows before rendering (tools/split_kick_episodes.py) so the off-path kick "
        "motion never becomes a training label — only the recovery does.",
    )
    ap.add_argument(
        "--kick-mag-mm", type=float, nargs=2, default=[2.0, 6.0], metavar=("LO", "HI"),
        help="total commanded kick displacement, U(LO, HI) mm (actual is less: servo lag)",
    )
    ap.add_argument(
        "--kick-frames", type=int, nargs=2, default=[4, 8], metavar=("LO", "HI"),
        help="kick window length in frames, uniform integer in [LO, HI]",
    )
    ap.add_argument(
        "--kick-max", type=int, default=2, metavar="N",
        help="max kicks per env per round",
    )
    ap.add_argument(
        "--kick-phases", type=str, default="1", metavar="CSV",
        help="controller phases in which kicks may START (default ALIGN only; "
        "'1,2' adds PUSH — use small --kick-mag-mm there, the plug may be in the bore)",
    )
    ap.add_argument(
        "--offset-mag-mm", type=float, default=None, metavar="MM",
        help="override the curriculum lateral offset magnitude (stage 4 default 8mm). "
        "Larger values start the plug far off-axis for far-recovery demonstrations.",
    )
    ap.add_argument(
        "--grasp-roll-deg",
        type=float,
        default=None,
        metavar="DEG",
        help="override the module GRASP_ROLL_DEG base (default 180; production servoD used "
        "150, which also keeps servo joint 5 off the bank's branch boundary)",
    )
    ap.add_argument(
        "--offset-z",
        type=float,
        default=0.0,
        metavar="MM",
        help="per-episode VERTICAL jack-offset component, U(-MM, +MM) along the grasp "
        "frame's nfw axis (the built-in offset disk is horizontal-only). 0 = off.",
    )
    ap.add_argument(
        "--approach-jitter",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO_MM", "HI_MM"),
        help="per-episode approach distance U(LO, HI) mm instead of the curriculum "
        "constant (30 at stage 4)",
    )
    ap.add_argument(
        "--offset-uniform-disk",
        action="store_true",
        help="sample the jack offset uniformly over the disk (mag = R*sqrt(U)) instead of "
        "U(0, R), which over-represents small offsets",
    )
    ap.add_argument(
        "--servo-plant",
        action="store_true",
        help="drive the CALIBRATED servo arm (ServoCableVecEnv + servo_source='commanded', "
        "the Wave-4 gated shipping preset) instead of the kinematic arm. Motion carries the "
        "identified plant's jerk-limited profiles + transport delay; seat takes ~2x longer, "
        "so budget --steps >= 450 (>= 600 with jam/retreat knobs).",
    )
    ap.add_argument("--max-save", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    env_mod = _load_env_module()

    if args.boot_mm is not None:
        env_mod.BOOT = args.boot_mm / 1000.0
    if args.grip_z_off_mm is not None:
        env_mod.GRIP_Z_OFF = args.grip_z_off_mm / 1000.0
    if args.grasp_roll_deg is not None:
        env_mod.GRASP_ROLL_DEG = args.grasp_roll_deg
    if args.grip_from_head is not None:
        gb = args.grip_from_head / 1000.0 - env_mod.BOOT
        if gb <= 0.0:
            raise SystemExit(f"--grip-from-head must exceed BOOT ({env_mod.BOOT * 1000:.1f}mm)")
        env_mod.GRIP_BACK = gb

    tilt = tuple(args.cable_tilt) if len(args.cable_tilt) > 1 else args.cable_tilt[0]
    if args.servo_plant:
        # calibrated servo plant (Wave 4 shipping preset): ServoCableVecEnv subclasses
        # RigidCableVecEnv, so every DR knob passes through; module-level overrides
        # (BOOT/GRIP_BACK/GRASP_ROLL_DEG) still act via the parent module's globals.
        import servo_cable_env  # noqa: PLC0415  (rl/ already on sys.path)
        env_cls = servo_cable_env.ServoCableVecEnv
    else:
        env_cls = env_mod.RigidCableVecEnv
    env = env_cls(
        args.envs,
        seed=args.seed,
        cable_tilt_deg=tilt,
        connector_usd=args.connector_usd,
        grasp_roll_180=args.grasp_roll_180,
        jack_yaw_deg=args.jack_yaw,
        grasp_roll_jitter_deg=args.grasp_roll_jitter,
        jack_fixture=args.jack_fixture,
        offset_z_mm=args.offset_z,
        approach_jitter_mm=args.approach_jitter,
        offset_uniform_disk=args.offset_uniform_disk,
        offset_mag_mm=args.offset_mag_mm,
    )
    env.set_stage(args.stage)

    cfg = AlignInsertConfig(
        mouth_depth_m=float(env_mod.SEAT_AIM_DY),
        max_dpos_m=float(env_mod.MAX_DPOS),
        max_drot_rad=float(env_mod.MAX_DROT),
        align_standoff_m=args.standoff_mm / 1000.0,
        push_correction=args.push_correction,
        # servo plant: servo the COMMANDED pose (the loop the plant cannot lag);
        # phase gates stay on measured truth. Wave 4 gate: 100% held vs 0% measured-mode.
        servo_source="commanded" if args.servo_plant else "measured",
        **({"jam_window": args.jam_window} if args.jam_window is not None else {}),
        **({"jam_progress_m": args.jam_progress_m} if args.jam_progress_m is not None else {}),
    )
    ctrl = AlignInsertController(args.envs, cfg)
    from newton_cabling.scripted_controller.rigid_cable_adapter import observe_rigid_cable_env

    rng = np.random.default_rng(args.seed + 9973)
    kick_phases = {int(x) for x in args.kick_phases.split(",")}
    print(
        f"[gen] scripted align-then-insert | pre-dock {cfg.align_along_m * 1000:+.1f}mm | "
        f"push_correction {cfg.push_correction} | action-noise {args.action_noise}",
        flush=True,
    )

    labels = list(env.model.body_label)
    sfx = lambda s: [i for i, lb in enumerate(labels) if lb.endswith(s)]  # noqa: E731
    tip1_i, tip2_i = sfx("gripper_finger1_finger_tip_link"), sfx("gripper_finger2_finger_tip_link")
    rods_i = [[int(b) for b in env.rod_bodies_all[i]] for i in range(args.envs)]
    n_rod = len(rods_i[0])

    os.makedirs(args.out, exist_ok=True)
    stale = sorted(glob.glob(os.path.join(args.out, "ep_*")))
    if stale and not args.keep_existing:
        for p in stale:
            shutil.rmtree(p, ignore_errors=True)
        print(f"[gen] cleared {len(stale)} existing ep_* dirs in {args.out}")

    saved = tot = succ_tot = 0
    T, N = args.steps, args.envs
    quarantine: set[int] = set()   # envs that hit the servo branch guard: actions zeroed

    for r in range(args.rounds):
        obs = env.reset()
        ctrl.reset()
        eef = np.zeros((T, N, 7), np.float32)
        conn = np.zeros((T, N, 7), np.float32)
        face = np.zeros((T, N, 7), np.float32)
        acts = np.zeros((T, N, 7), np.float32)
        state = np.zeros((T, N, 10), np.float32)
        tips = np.zeros((T, N, 2, 3), np.float32)
        rods = np.zeros((T, N, n_rod, 3), np.float32)
        phase = np.zeros((T, N), np.int32)
        hold = np.zeros((T, N), np.int32)
        viol = np.zeros((T, N), np.float32)
        kick_on = np.zeros((T, N), bool)
        kick_left = np.zeros(N, np.int64)
        kick_used = np.zeros(N, np.int64)
        kick_vec = np.zeros((N, 3))
        if args.kick_prob > 0.0:
            # last known-finite servo targets: a kicked env whose sim blows up (NaN body
            # state) gets frozen at these instead of poisoning the batched quat math
            last_p = np.array(env.wrist_tgt_p, dtype=float, copy=True)
            last_q = np.array(env.wrist_tgt_q, dtype=float, copy=True)
        seat_pos, seat_q = env.seat_pos.copy(), env.seat_q.copy()
        jack_p = env.state_0.body_q.numpy()[[int(j) for j in env.jack_body], :3].copy()

        round_fault = None
        for t in range(T):
            bqn = env.state_0.body_q.numpy()
            eef[t] = _pose_wxyz(bqn, env.wrist_body)
            conn[t] = _pose_wxyz(bqn, env.rod_front)
            fp, fq = env._face_pose(bqn)
            face[t] = np.concatenate([fp, fq[:, [3, 0, 1, 2]]], axis=1)
            tips[t, :, 0], tips[t, :, 1] = bqn[tip1_i, :3], bqn[tip2_i, :3]
            for e in range(N):
                rods[t, e] = bqn[rods_i[e], :3]
            state[t] = obs[:, :10].detach().cpu().numpy()
            phase[t] = ctrl.phase
            a = ctrl.act(observe_rigid_cable_env(env))
            if args.kick_prob > 0.0:
                a = a.copy()
                for e in range(N):
                    if (kick_left[e] == 0 and kick_used[e] < args.kick_max
                            and e not in quarantine and phase[t, e] in kick_phases
                            and rng.random() < args.kick_prob):
                        w = int(rng.integers(args.kick_frames[0], args.kick_frames[1] + 1))
                        mag = rng.uniform(args.kick_mag_mm[0], args.kick_mag_mm[1]) * 1e-3
                        ang = rng.uniform(0.0, 2.0 * np.pi)
                        lat = (math.cos(ang) * env.frame[e]["jaw"]
                               + math.sin(ang) * env.frame[e]["tool"])
                        kick_vec[e] = np.clip(lat * (mag / w) / env_mod.MAX_DPOS, -1.0, 1.0)
                        kick_left[e] = w
                        kick_used[e] += 1
                    if kick_left[e] > 0:
                        a[e, 0:3] = kick_vec[e]
                        a[e, 3:6] = 0.0
                        kick_on[t, e] = True
                        kick_left[e] -= 1
            if args.action_noise > 0.0:
                a = a.copy()
                a[:, 0:6] += rng.normal(0.0, args.action_noise, (N, 6))
                a = np.clip(a, -1.0, 1.0)
            if args.kick_prob > 0.0:
                # nonfinite guard: a kick can shove the plug into the fixture hard enough
                # to blow up that env's solver state; NaN then reaches the controller
                # action and the servo targets, and one poisoned row crashes the BATCHED
                # Rot.from_quat for every env. Quarantine + freeze the bad rows instead.
                wp = np.asarray(env.wrist_tgt_p)
                wq = np.asarray(env.wrist_tgt_q)
                badr = (~np.isfinite(a).all(axis=1) | ~np.isfinite(wp).all(axis=1)
                        | ~np.isfinite(wq).all(axis=1)
                        | (np.linalg.norm(wq, axis=1) < 1e-6))
                if badr.any():
                    rows = [int(i) for i in np.flatnonzero(badr)]
                    quarantine.update(rows)
                    a[rows] = 0.0
                    wp[badr] = last_p[badr]
                    wq[badr] = last_q[badr]
                    print(f"[gen] t={t}: quarantined env(s) {rows} "
                          f"(nonfinite state/action after kick)", flush=True)
                ok = ~badr
                last_p[ok] = wp[ok]
                last_q[ok] = wq[ok]
            if quarantine:
                a[list(quarantine)] = 0.0   # frozen: no goal motion -> no branch fault
            acts[t] = a.astype(np.float32)  # what was EXECUTED, noise included
            try:
                obs, _, _, _, _ = env.step(a)
            except ValueError as exc:
                # servo-plant revolution-branch guard: an env's wrist crossed the bank's
                # branch boundary. Contracted to raise (seam Addendum 3); for DATAGEN we
                # QUARANTINE the env (its actions are zeroed from the next round on, so it
                # simply never seats/saves) and drop the current round.
                if "revolution branch" in str(exc):
                    bad = [int(m) for m in re.findall(r"\((\d+), \d+\)", str(exc))]
                    quarantine.update(bad)
                    round_fault = f"env(s) {bad} quarantined; " + str(exc).splitlines()[0][:110]
                    break
                if args.kick_prob > 0.0:
                    # kick-related solver fault the nonfinite guard didn't catch: drop the
                    # round, keep the process (rounds have margin; the chunk survives)
                    round_fault = "kick-related fault: " + str(exc).splitlines()[0][:110]
                    break
                raise
            hold[t] = env.hold
            viol[t] = (env.viol_last > 0).astype(np.float32)

        if round_fault is not None:
            tot += N
            print(f"[gen] round {r + 1}/{args.rounds}: ABANDONED at t={t} "
                  f"(servo branch fault: {round_fault})", flush=True)
            continue
        tot += N
        for e in range(N):
            if saved >= args.max_save:
                break
            hit = np.where(hold[:, e] >= env_mod.HOLD_STEPS)[0]
            ok = len(hit) > 0
            succ_tot += int(ok)
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
            for name, arr in (("eef_traj", eef), ("conn_traj", conn), ("face_traj", face)):
                np.save(
                    os.path.join(d, f"{name}.npy"),
                    to_seat_frame(arr[:cut, e], seat_pos[e], seat_q[e]),
                )
            for name, arr in (("tips_traj", tips), ("rods_traj", rods)):
                np.save(
                    os.path.join(d, f"{name}.npy"),
                    to_seat_frame_points(arr[:cut, e], seat_pos[e], seat_q[e]),
                )
            np.save(os.path.join(d, "actions_policy.npy"), acts[:cut, e])
            np.save(os.path.join(d, "state_sim.npy"), state[:cut, e])
            np.save(os.path.join(d, "phase.npy"), phase[:cut, e])
            if args.kick_prob > 0.0:
                np.save(os.path.join(d, "kick.npy"), kick_on[:cut, e])
            q_inv = quat_conjugate(seat_q[e])
            meta = {
                "success": bool(ok),
                "frames": int(cut),
                "viol_frac": vf,
                "seat_pos": seat_pos[e].tolist(),
                "seat_quat_xyzw": seat_q[e].tolist(),
                "jack_pos_seatrel": quat_rotate(q_inv, jack_p[e] - seat_pos[e]).tolist(),
                "stage": args.stage,
                "cable_tilt_deg": args.cable_tilt,
                "tilt_this_env_deg": float(np.degrees(env.cable_tilt[e])),
                "jack_yaw_max_deg": args.jack_yaw,
                "jack_yaw_this_ep_deg": float(env.jack_yaw_ep[e]),
                "offset_z_mm_max": args.offset_z,
                "approach_jitter_mm": args.approach_jitter,
                "approach_this_ep_mm": float(env.approach_ep[e] * 1000.0),
                "offset_uniform_disk": bool(args.offset_uniform_disk),
                "seed": args.seed,
                "round": r,
                "env": e,
                "driver": "scripted-align-insert",
                "plant": "servo-commanded" if args.servo_plant else "kinematic",
                "connector_usd": args.connector_usd,
                "jack_fixture": bool(args.jack_fixture),
                "grasp_roll_180": bool(args.grasp_roll_180),
                # render_batch_v4.sh's staleness guard reads this to refuse blind-side
                # data, so it must track the AUTHORING flip (the only thing that moves the
                # plug to the camera side). GRASP_ROLL_DEG rolls the whole hand, which
                # carries the wrist-mounted camera with it -> no change in occlusion.
                "grasp_roll_deg": 180.0 if args.grasp_roll_180 else 0.0,
                "env_grasp_roll_deg": float(env_mod.GRASP_ROLL_DEG),
                "grasp_roll_jitter_deg": args.grasp_roll_jitter,
                "grasp_roll_this_env_deg": float(env.grasp_roll_ep[e]),
                "env_grasp_yaw_deg": float(getattr(env_mod, "GRASP_YAW_DEG", 0.0)),
                "env_arm_pitch_deg": float(env_mod.ARM_PITCH_DEG),
                "align_standoff_mm": args.standoff_mm,
                "push_correction": args.push_correction,
                "action_noise": args.action_noise,
                "phase_codes": "SETTLE0 ALIGN1 PUSH2 HOLD3 RETREAT4",
            }
            if args.kick_prob > 0.0:
                meta.update({
                    "kick_prob": args.kick_prob,
                    "kick_mag_mm": list(args.kick_mag_mm),
                    "kick_frames": list(args.kick_frames),
                    "kicks_this_env": int(kick_used[e]),
                    "kick_frames_saved": int(kick_on[:cut, e].sum()),
                })
            with open(os.path.join(d, "meta.json"), "w") as fh:
                json.dump(meta, fh, indent=2)
            saved += 1
        print(
            f"[gen] round {r + 1}/{args.rounds}: held {succ_tot}/{tot} "
            f"({100 * succ_tot / max(tot, 1):.0f}%), saved {saved}",
            flush=True,
        )
        if saved >= args.max_save:
            break

    print(
        f"[gen] done: {succ_tot}/{tot} held ({100 * succ_tot / max(tot, 1):.0f}%); "
        f"saved {saved} episodes to {args.out}/"
    )


if __name__ == "__main__":
    main()
