"""Side-by-side GS-vs-Newton frames, rendered from the SAME two cameras as the pipeline.

Unlike tools/compare_gs_newton.py (matplotlib 3D truth panel), this poses the Newton
scene per frame exactly like tools/render_wrist_cam_newton.py (arm relocated so wrist_3
is bit-exact on eef_traj; plug/cable/jack from the same trajectory) and renders it with
Newton's headless GL viewer from the two cameras the GS pipeline uses:

  * FRONT: the GS scene's fixed camera (FRONT_EYE -> FRONT_TARGET from
    tools/datagen_v2_config.sh), mapped into the seat frame through the inverse of the
    renderer's seat->world transform (world = JACK_POS + Rz(EEF_RPY_Z) @ p_seat; the
    batch script runs jack_rpy = 0 0 0, so seat_R is just the --eef-rpy yaw).
  * WRIST: the USD eye-in-hand camera at its EXACT 6-DOF pose. Newton's Camera API
    only takes pitch/yaw (no roll), but the GL renderer consumes get_front()/get_up()
    alone, so rw.render(up=...) pins the true USD basis on the camera instance and the
    wrist pair matches natively (--no-roll-fix falls back to the stock no-roll camera).

Output: one PNG per frame laid out
        [ GS front | NEWTON front ]
        [ GS wrist | NEWTON wrist ]
with the Newton 640x480 frames resized to the GS dump size (512x512, the same squash
the pipeline applies) so features correspond.

    .venv/bin/python tools/compare_gs_newton_cams.py \
        --traj  /home/pandaliza/parallax/data/vla_train/cable_traj_y45p45/ep_0000 \
        --gs    /home/pandaliza/parallax/data/vla_train/datagen_y45p45/ep_0000 \
        --frames 0,40,80,120,159 --out out_compare/ep_0000

    # every 4th frame (feed the folder to ffmpeg for a video):
    ... --step 4 --out out_compare/ep_0000_seq
"""

import argparse
import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import render_wrist_cam_newton as rw  # noqa: E402  (reuses build/pose/render machinery)
from newton_cabling.render.scene_gs_common import load_usd_wrist_cam  # noqa: E402

# GS scene constants -- tools/datagen_v2_config.sh + render_batch_v4.sh defaults.
JACK_POS = np.array([0.295, -0.876, 0.835])
FRONT_EYE = np.array([0.53, -0.681, 0.984])
FRONT_TARGET = np.array([0.29, -0.87, 0.90])
EEF_RPY_Z = 180.0            # --eef-rpy "0 0 180" (rigid env); jack_rpy is 0 0 0
FY_D415 = 605.2867431640625  # front cam intrinsics (configs/cameras.yaml)
DUMP = 512                   # pipeline dump size (square; 640x480 squashed)


def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def front_cam_seat():
    """(eye, fwd) of the GS front camera expressed in the trajectory's seat frame."""
    Rinv = _rz(EEF_RPY_Z).T
    eye = Rinv @ (FRONT_EYE - JACK_POS)
    fwd = Rinv @ (FRONT_TARGET - FRONT_EYE)
    return eye, fwd / np.linalg.norm(fwd)


def cam_roll_up(model, state, t_cam, q_local):
    """(roll_deg, up_world) of the TRUE USD camera. roll_deg is the angle about the
    optical axis between Newton's no-roll GL camera (up = world +Z projected off the
    view axis) and the real camera; up_world is the real camera's up vector, which,
    fed to rw.render(up=...), makes the GL camera assume the exact 6-DOF USD pose."""
    from newton_cabling.render.gs_bridge import newton_pose, quat_mul_wxyz, quat_rotate_wxyz

    pos, quat = newton_pose(state.body_q.numpy(), rw._wrist_index(model))
    q_cam = quat_mul_wxyz(list(quat), q_local)
    fwd = np.asarray(quat_rotate_wxyz(q_cam, [0.0, 0.0, 1.0]), float)
    up_true = np.asarray(quat_rotate_wxyz(q_cam, [0.0, -1.0, 0.0]), float)  # ROS optical: +y down
    z = np.array([0.0, 0.0, 1.0])
    up_gl = z - (z @ fwd) * fwd
    up_gl /= np.linalg.norm(up_gl) + 1e-9
    s = float(np.dot(np.cross(up_gl, up_true), fwd))
    c = float(np.dot(up_gl, up_true))
    return math.degrees(math.atan2(s, c)), up_true


def gs_frame(gs_dir, sub, frame):
    from PIL import Image

    p = os.path.join(gs_dir, sub, f"frame_{frame:04d}.png")
    if not os.path.isfile(p):
        return None
    return Image.open(p).convert("RGB")


def label(im, text):
    from PIL import ImageDraw

    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 220, 0))
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True, help="trajectory episode dir (eef_traj.npy...)")
    ap.add_argument("--gs", required=True, help="GS-rendered episode dir (image/, wrist_image/)")
    ap.add_argument("--out", default="out_compare", help="output dir for the stitched PNGs")
    ap.add_argument("--frames", default=None, help="comma list of frame indices")
    ap.add_argument("--step", type=int, default=None,
                    help="instead of --frames: every Nth frame of the episode")
    ap.add_argument("--grip-theta", type=float, default=rw.THETA_CABLE)
    ap.add_argument("--usd", default=str(rw.SBOT_USD))
    ap.add_argument("--no-roll-fix", action="store_true",
                    help="render the wrist with Newton's stock no-roll camera instead "
                         "of the exact USD pose")
    ap.add_argument("--plug-mesh", choices=("scan", "headA", "cad"), default="cad",
                    help="'cad' = cad_rj45.usd's parametric plug + socket (what servoD/roll3-era "
                         "episodes simulate against; default); 'scan' = the scan_rj45 /World/Plug "
                         "crop + orange cable capsule; 'headA' = the raw scan mesh with the real "
                         "boot + ~60mm of curved cord (bend frozen in the scanned pose)")
    args = ap.parse_args()

    from PIL import Image

    n_frames = len(np.load(os.path.join(args.traj, "eef_traj.npy")))
    if args.frames:
        frames = [int(x) for x in args.frames.split(",")]
    elif args.step:
        frames = list(range(0, n_frames, args.step))
    else:
        frames = sorted({0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1})

    t_cam, q_local, _ = load_usd_wrist_cam(args.usd)
    vfov_wrist = rw.vfov_from_usd(args.usd, 640, 480)
    vfov_front = math.degrees(2.0 * math.atan(240.0 / FY_D415))
    eye_f, fwd_f = front_cam_seat()
    print(f"front cam (seat frame): eye={np.round(eye_f, 3)} fwd={np.round(fwd_f, 3)} "
          f"vfov={vfov_front:.1f}  | wrist vfov={vfov_wrist:.1f}")

    os.makedirs(args.out, exist_ok=True)
    for frame in frames:
        print(f"[frame {frame}]")
        poses = rw.load_frame(args.traj, frame)
        base = rw.base_for_wrist(args.usd, args.grip_theta, poses["eef"])
        model, state = rw.build(args.usd, args.grip_theta, base_pose=base, poses=poses,
                                plug_mesh=args.plug_mesh)
        eye_w, fwd_w = rw.wrist_cam_pose(model, state, t_cam, q_local)
        roll, up_w = cam_roll_up(model, state, t_cam, q_local)
        nf = rw.render(model, state, eye_f, fwd_f, vfov_front, 640, 480)
        nw = rw.render(model, state, eye_w, fwd_w, vfov_wrist, 640, 480,
                       up=None if args.no_roll_fix else up_w)

        cells = []
        for sub, newton_u8, name in (("image", nf, "front"), ("wrist_image", nw, "wrist")):
            g = gs_frame(args.gs, sub, frame)
            g = label(g.resize((DUMP, DUMP), Image.LANCZOS) if g.size != (DUMP, DUMP) else g,
                      f"GS {name} f{frame}") if g else \
                label(Image.new("RGB", (DUMP, DUMP), (40, 40, 40)), f"GS {name} MISSING")
            n = Image.fromarray(newton_u8).resize((DUMP, DUMP), Image.LANCZOS)
            tag = f"NEWTON {name} f{frame}"
            if name == "wrist":
                tag += "  (no roll)" if args.no_roll_fix else \
                    f"  (exact USD pose, roll {roll:+.1f} deg)"
            cells.append((g, label(n, tag)))
        grid = Image.new("RGB", (2 * DUMP, 2 * DUMP))
        for r, (g, n) in enumerate(cells):
            grid.paste(g, (0, r * DUMP))
            grid.paste(n, (DUMP, r * DUMP))
        out = os.path.join(args.out, f"compare_{frame:04d}.png")
        grid.save(out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
