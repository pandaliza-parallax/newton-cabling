"""Convert a rl/record_cable_env.py rollout npz into a render-ready ep_XXXX dir.

The npz stores WORLD-frame poses with scipy xyzw quats; the GS render pipeline
(scripts/record_sbot_scene_gs_cable.py --eef-traj) wants SEAT-relative (T,7)
[pos3, quat4 wxyz] in eef_traj.npy, with the plug splat riding face_traj.npy.
The eef orientation comes from obs[:, 3:9] -- the pi05 6D rotation (first two
COLUMNS of R, per the rj45_sbot state layout) -- Gram-Schmidt back to R.

    .venv/bin/python tools/servo_npz_to_ep.py servo_D_jam_demo_traj.npz \
        /home/pandaliza/parallax/data/vla_train/servo_D_jam_demo/ep_0000
"""

import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def rot6d_to_mat(d6):
    a, b = d6[:, :3], d6[:, 3:]
    x = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b - (x * b).sum(1, keepdims=True) * x
    y = b / np.linalg.norm(b, axis=1, keepdims=True)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=2)          # columns = x, y, z


def main():
    src, dst = sys.argv[1], sys.argv[2]
    d = np.load(src, allow_pickle=True)
    obs = d["obs"][:, 0]
    T = len(obs)
    seat_p = np.asarray(d["seat_pos"][0], float)
    R_seat = Rot.from_quat(np.asarray(d["seat_quat"][0], float))     # xyzw

    def to_seat(p_w, R_w):
        p = R_seat.inv().apply(p_w - seat_p)
        q = (R_seat.inv() * R_w).as_quat()                            # xyzw
        return p, np.concatenate([q[:, 3:4], q[:, :3]], axis=1)       # -> wxyz

    eef_p, eef_q = to_seat(obs[:, :3], Rot.from_matrix(rot6d_to_mat(obs[:, 3:9])))
    face_p, face_q = to_seat(d["face_pos"][:, 0], Rot.from_quat(d["face_quat"][:, 0]))

    os.makedirs(dst, exist_ok=True)
    np.save(os.path.join(dst, "eef_traj.npy"), np.concatenate([eef_p, eef_q], 1).astype(np.float64))
    np.save(os.path.join(dst, "face_traj.npy"), np.concatenate([face_p, face_q], 1).astype(np.float64))
    meta = {"frames": T, "source_npz": os.path.abspath(src),
            "jack_pos_seatrel": [0.0, -0.012, 0.0],
            "note": "converted by tools/servo_npz_to_ep.py (world->seat, xyzw->wxyz)"}
    for m in np.asarray(d["meta"]).tolist():
        k, _, v = str(m).partition("=")
        meta[f"src_{k}"] = v
    with open(os.path.join(dst, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {dst}: {T} frames | seat-rel eef[0]={np.round(np.concatenate([eef_p, eef_q],1)[0],3).tolist()}")


if __name__ == "__main__":
    main()
