"""Convert a record_cable_env.py `<out>_traj.npz` rollout into the cable_traj/ep_XXXX layout,
so it can be GS-rendered by the existing v4 path with no other changes.

    .venv/bin/python tools/npz_to_cable_traj.py --npz cable_ppo_rollout_traj.npz \
        --out cable_traj_npz --ep 0
    sudo TRAJROOT=$PWD/cable_traj_npz bash tools/render_batch_v4.sh 0 1

The npz carries everything the --eef-traj renderer needs:
  * wrist pose  <- obs[:, :10] = [eef_pos(3), eef_rot6d(6), gripper(1)]   (WORLD frame)
  * plug face   <- face_pos / face_quat                                    (WORLD frame)
  * seat frame  <- seat_pos / seat_quat
Conventions (verified against the file, not assumed): face_quat and seat_quat are **XYZW**
(scipy `.as_quat()`); rot6d is the first two COLUMNS of the rotation matrix, row-major, so
col2 = cross(col0, col1). Everything is re-based into the SEAT frame (position AND orientation)
and written scalar-first **wxyz**, matching rl/gen_cable_traj.py exactly.

NOT in the npz: the front-rod body pose, fingertips, and cable-rod chain. So the GS render works
(it only needs eef + face), but the NEWTON side-by-side (`--from-traj`, tools/compare_gs_newton.py)
does not -- those need tips_traj/rods_traj from gen_cable_traj.py. Rendering falls back cleanly;
the Newton panel is simply unavailable for converted episodes.
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def rot6d_to_quat_xyzw(r6):
    """(T,6) first-two-COLUMNS rot6d -> (T,4) xyzw. Gram-Schmidt for numerical safety."""
    c = np.asarray(r6, float).reshape(-1, 3, 2)
    a, b = c[:, :, 0], c[:, :, 1]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b - (np.sum(a * b, axis=1, keepdims=True)) * a
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    c2 = np.cross(a, b)
    M = np.stack([a, b, c2], axis=2)                      # columns
    return Rot.from_matrix(M).as_quat()                   # xyzw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="../data/vla_train/cable_traj_npz", help="root dir for the ep_XXXX folder")
    ap.add_argument("--ep", type=int, default=0, help="episode index -> ep_XXXX")
    ap.add_argument("--env", type=int, default=0, help="which env column of the npz")
    ap.add_argument("--cut-at-hold", action="store_true",
                    help="truncate at the first held frame + --pad (default: keep the whole episode)")
    ap.add_argument("--pad", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    e = args.env
    obs = d["obs"][:, e, :]
    act = d["action"][:, e, :]
    fp = d["face_pos"][:, e, :]
    fq = d["face_quat"][:, e, :]                          # xyzw
    sp = d["seat_pos"][0]
    sq = d["seat_quat"][0]                                # xyzw
    held = d["held"][:, e] if "held" in d else np.zeros(len(obs))
    viol = d["viol"][:, e] if "viol" in d else np.zeros(len(obs))
    meta_kv = dict(s.split("=", 1) for s in [str(x) for x in d["meta"]] if "=" in s)

    T = len(obs)
    cut = T
    if args.cut_at_hold:
        hit = np.where(held > 0.5)[0]
        if len(hit):
            cut = min(int(hit[0]) + 1 + args.pad, T)

    eef_p = obs[:cut, 0:3].astype(float)
    eef_q = rot6d_to_quat_xyzw(obs[:cut, 3:9])            # xyzw
    face_p = fp[:cut].astype(float)
    face_q = fq[:cut].astype(float)

    # ---- re-base into the SEAT frame (position AND orientation), then store wxyz ----
    qs_inv = Rot.from_quat(sq).inv()

    def rebase(p, q_xyzw):
        pr = qs_inv.apply(p - sp)
        qr = (qs_inv * Rot.from_quat(q_xyzw)).as_quat()   # xyzw
        return np.concatenate([pr, qr[:, [3, 0, 1, 2]]], axis=1).astype(np.float32)

    eef_t = rebase(eef_p, eef_q)
    face_t = rebase(face_p, face_q)

    d_out = os.path.join(args.out, f"ep_{args.ep:04d}")
    os.makedirs(d_out, exist_ok=True)
    np.save(os.path.join(d_out, "eef_traj.npy"), eef_t)
    np.save(os.path.join(d_out, "face_traj.npy"), face_t)
    np.save(os.path.join(d_out, "actions_policy.npy"), act[:cut].astype(np.float32))
    np.save(os.path.join(d_out, "state_sim.npy"), obs[:cut, :10].astype(np.float32))
    json.dump({
        "success": bool(held.max() > 0.5), "frames": int(cut),
        "viol_frac": float((viol[:cut] > 0).mean()),
        "seat_pos": [float(v) for v in sp], "seat_quat_xyzw": [float(v) for v in sq],
        # the npz has no jack body pose; the Newton panel draws the jack box at the seat.
        "jack_pos_seatrel": [0.0, 0.0, 0.0],
        "checkpoint": meta_kv.get("policy"), "stage": meta_kv.get("stage"),
        "cable_tilt_deg": meta_kv.get("cable_tilt_deg"), "seed": meta_kv.get("seed"),
        "hz": meta_kv.get("hz"), "source_npz": os.path.abspath(args.npz),
        "note": "converted from record_cable_env npz; no conn/tips/rods channels "
                "(GS render works, Newton side-by-side does not)",
    }, open(os.path.join(d_out, "meta.json"), "w"), indent=2)

    seat_mm = np.linalg.norm(face_t[-1, :3]) * 1000.0
    print(f"[npz->traj] {args.npz} -> {d_out}  ({cut}/{T} frames, "
          f"held={bool(held.max() > 0.5)}, ckpt={meta_kv.get('policy')})")
    print(f"[npz->traj] final face->seat {seat_mm:.2f}mm; eef start (seat frame) "
          f"{np.round(eef_t[0, :3], 4)}")
    print(f"[npz->traj] render:  sudo TRAJROOT=$PWD/{args.out} bash tools/render_batch_v4.sh "
          f"{args.ep} {args.ep + 1}")


if __name__ == "__main__":
    main()
