"""Headless StandardBots RO1 arm rendered through DalusPySim's Gaussian-Splat renderer.

Newton drives the 6-DOF arm; each frame we read the seven arm-link world poses out of
``state.body_q`` and ship them + a camera to the parallax_sim renderer over POSIX SHM,
saving the returned RGB. This is the RJ45 GS pipeline (examples/record_rj45_insert_gs.py) with
seven moving objects instead of one.

Registration is trivial -- and already proven on the Isaac side. Each ``sbot_gs/*.ply``
is authored *link-local* (its gaussian means are centred on the link's joint frame, in
metres, Z-up), exactly the convention parallax-demo-isaac-lab/run_envs.py renders them
with: the link's *world* transform IS its splat transform, no align / centroid / scale.
So per frame we just hand the renderer ``newton_pose(body_q, link_body_idx)`` for each
link, in a fixed order. Newton imports the same cm USD and scales it to metres, so the
splats (metres) and the bodies line up 1:1.

GRIPPER: the AG-145 has no splat yet, so it renders invisible here (it still exists in
the physics -- driven through the coupling, just not drawn). Only the seven arm links
are sent: base, shoulder, upper_arm, forearm, wrist_1/2/3.

The one thing this script *verifies* rather than assumes: that Newton's per-link
``body_q`` frame coincides with the USD link frame the splats were authored in. Run
``--smoke`` first -- a single static home-pose frame. A coherent robot confirms it; if
links look detached, capture each link's constant offset once and pass it through
gs_bridge.place_on_body(..., centroid=offset) (the bridge already supports exactly that).

Run (renderer container up with ipc: host; /dev/shm/dal_buffer* are root-owned -> sudo):
    cd newton-cabling
    uv pip install posix_ipc                                            # one-time
    sudo PYTHONPATH=/home/pandaliza/parallax/data-generator/sim_engine/DalusPySim \
        .venv/bin/python scripts/record_sbot_gs.py --out out_sbot_gs

    --smoke    render one static home-pose frame and exit (validate registration)
    --dry-run  skip the renderer/SHM entirely (still runs Newton, returns black frames)
    --still    hold the home pose instead of sweeping the arm
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

import newton
import numpy as np
import warp as wp

newton.use_coord_layout_targets = True

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root
from newton.solvers import SolverVBD  # noqa: E402

from newton_cabling.render.gs_bridge import (  # noqa: E402
    NewtonGSClient,
    look_at_quat,
    make_intrinsics,
    newton_pose,
)
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: E402
from newton_cabling.sim.safe_vbd import finalize_for_vbd, new_vbd_builder  # noqa: E402
from newton_cabling.sim.sbot import (  # noqa: E402
    GRIPPER_THETA_CLOSED,
    GRIPPER_THETA_OPEN,
    SBOT_HOME,
    add_sbot,
    set_arm_home,
    set_gripper,
    set_pd_gains,
)

FPS = 30
DURATION_SECONDS = 5.0
SIM_SUBSTEPS = 8

# The seven arm links, in the order Isaac's run_envs.py registers them. The render()
# pose list and the SETUP ply_paths must share this order.
LINK_SPLATS: tuple[str, ...] = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    # AG-145 gripper: the 8 finger links are dynamic bodies, so they pose from
    # body_q and articulate with the gripper. Splats synthesized from the CAD via
    # tools/sbot/synth_splats_from_meshes.py (the TRELLIS finger splat was too
    # inflated to cut per sub-link). resolve_body suffix-matches these to the
    # gripper_* bodies; the .ply filenames drop the gripper_ prefix to match.
    "finger1_knuckle_link",
    "finger1_inner_knuckle_link",
    "finger1_finger_link",
    "finger1_finger_tip_link",
    "finger2_knuckle_link",
    "finger2_inner_knuckle_link",
    "finger2_finger_link",
    "finger2_finger_tip_link",
)

# Host location of the per-link splats and the host->container path rewrite (the renderer
# opens the plys from inside its container, where host ~/parallax is bind-mounted).
HOST_PARALLAX = "/home/pandaliza/parallax"
CONTAINER_PARALLAX = "/root/parallax"
# flat/ = SH stripped to deg-0 (tools/strip_arm_sh.py): the renderer mis-reads the arm
# splats' f_rest layout -> rainbow; the flat path (0.5+SH_C0*f_dc) gives correct matte colour.
HOST_SBOT_GS = f"{HOST_PARALLAX}/parallax-demo-isaac-lab/assets/sbot_gs/flat"


def host_to_container(p: str) -> str:
    return p.replace(HOST_PARALLAX, CONTAINER_PARALLAX)


def xform_pose(xf: wp.transform):
    """``(pos[xyz], quat[wxyz])`` for a wp.transform (Warp stores quats scalar-last)."""
    p = wp.transform_get_translation(xf)
    q = wp.transform_get_rotation(xf)  # [x, y, z, w]
    return [float(p[0]), float(p[1]), float(p[2])], [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def resolve_body(model: newton.Model, name: str) -> int:
    """Index of the link body named ``name`` (exact label, else unique suffix match)."""
    labels = list(model.body_label)
    exact = [i for i, lbl in enumerate(labels) if lbl == name]
    if len(exact) == 1:
        return exact[0]
    suffix = [i for i, lbl in enumerate(labels) if lbl.endswith(name)]
    if len(suffix) == 1:
        return suffix[0]
    raise RuntimeError(
        f"could not uniquely resolve body {name!r}: exact={exact} suffix={suffix}. "
        f"available body labels: {labels}"
    )


def arm_target(t: float, duration: float, *, still: bool) -> list[float]:
    """Home pose, plus a gentle forward-reach + wrist sweep so the GS sync is visible.

    One smooth sine period across the whole clip, so the arm returns to home at the end.
    """
    q = list(SBOT_HOME)
    if not still:
        phase = math.sin(2.0 * math.pi * t / max(duration, 1e-6))
        q[1] += 0.4 * phase  # shoulder pitch: reach forward (+y) and back
        q[4] += 0.3 * phase  # wrist_2
    return q


def gripper_theta(t: float, duration: float, *, still: bool) -> float:
    """Drive the AG-145 open -> closed -> open over the clip so the jaws articulate.

    theta = GRIPPER_THETA_OPEN at the ends, GRIPPER_THETA_CLOSED at the midpoint
    (one cosine cycle). ``still`` holds the jaws open.
    """
    if still:
        return GRIPPER_THETA_OPEN
    s = 0.5 - 0.5 * math.cos(2.0 * math.pi * t / max(duration, 1e-6))  # 0 -> 1 -> 0
    return GRIPPER_THETA_OPEN + (GRIPPER_THETA_CLOSED - GRIPPER_THETA_OPEN) * s


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="out_sbot_gs", help="output dir for frames + gif")
    ap.add_argument("--bg-ply", default=None, help="optional background splat (host path)")
    ap.add_argument("--rrd", default=None, help="also record physics to this .rrd (optional)")
    ap.add_argument("--width", type=int, default=852)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--fov-deg", type=float, default=45.0)
    ap.add_argument("--eye", type=float, nargs=3, default=None, help="camera eye xyz (world)")
    ap.add_argument("--target", type=float, nargs=3, default=None, help="camera target xyz (world)")
    ap.add_argument("--azim", type=float, default=-90.0, help="azimuth deg (0=+x, -90=from -y)")
    ap.add_argument("--elev", type=float, default=20.0, help="elevation deg above horizon")
    ap.add_argument("--dist-scale", type=float, default=1.6, help="eye dist = scale x bbox diagonal")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--duration", type=float, default=DURATION_SECONDS)
    ap.add_argument("--still", action="store_true", help="hold the home pose (no arm sweep)")
    ap.add_argument("--show-viewer", action="store_true", help="also open the renderer's pygame viewer")
    ap.add_argument("--smoke", action="store_true", help="render a single static home-pose frame and exit")
    ap.add_argument("--dry-run", action="store_true", help="skip the renderer/SHM (still runs Newton)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── build the robot (mirrors scripts/record_sbot_smoke.py; visual-only, no contact) ──────
    builder = new_vbd_builder(gravity=-9.81)
    base_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    handles = add_sbot(builder, base_xform, with_gripper=True)
    set_pd_gains(builder, handles)
    set_arm_home(builder, handles, SBOT_HOME)
    set_gripper(builder, handles, GRIPPER_THETA_OPEN)  # jaws open (invisible, but posed)

    # Visual render: the meshes are not drawn here, so drop all collisions to isolate
    # "does the arm hold its pose?" and keep stepping cheap.
    for shape_idx in range(len(builder.shape_body)):
        builder.shape_flags[shape_idx] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

    model = finalize_for_vbd(builder)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()
    solver = SolverVBD(model, iterations=12)

    # base_link is the fixed root: collapse_fixed_joints folds it into the world anchor,
    # so it has no body_q row -- it rides base_xform (static). The other six links are
    # dynamic bodies read from body_q each frame.
    base_pose = xform_pose(base_xform)
    static_links = {"base_link": base_pose}
    dyn_bodies = {n: resolve_body(model, n) for n in LINK_SPLATS if n not in static_links}
    obj_plys = [host_to_container(f"{HOST_SBOT_GS}/{name}.ply") for name in LINK_SPLATS]

    def link_poses(bq):
        return [
            static_links[n] if n in static_links else newton_pose(bq, dyn_bodies[n])
            for n in LINK_SPLATS
        ]

    # ── camera: auto-frame the arm at its home pose ─────────────────────────────────
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    bq0 = state_0.body_q.numpy()
    pts = np.array([p for p, _ in link_poses(bq0)], dtype=np.float64)
    pad = 0.2  # half a link, to include the splat extent beyond the joint origins
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    center = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo))
    if args.target is not None:
        center = np.asarray(args.target, np.float64)
    if args.eye is not None:
        eye = np.asarray(args.eye, np.float64)
    else:
        a, e = math.radians(args.azim), math.radians(args.elev)
        direction = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        eye = center + direction * args.dist_scale * diag
    cam_K = make_intrinsics(args.width, args.height, args.fov_deg)
    cam_quat = look_at_quat(eye.tolist(), center.tolist(), up=(0.0, 0.0, 1.0), convention="ros")
    print(f"[sbot-gs] bbox center={np.round(center, 4)} diag={diag:.3f}")
    print(f"[sbot-gs] cam eye={np.round(eye, 4)} target={np.round(center, 4)}")
    print(f"[sbot-gs] base_link: static @ {np.round(base_pose[0], 4)}; dyn {dyn_bodies}")

    client = NewtonGSClient(
        ply_paths=obj_plys,
        cam_K=cam_K,
        cam_pos=eye.tolist(),
        cam_quat=cam_quat,
        bg_ply=host_to_container(args.bg_ply) if args.bg_ply else None,
        show_viewer=args.show_viewer,
        dry_run=args.dry_run,
    )

    # ── smoke: one static home-pose frame, no stepping (cleanest registration check) ─
    if args.smoke:
        rgb = client.render(link_poses(bq0))
        _save([(rgb * 255.0).clip(0, 255).astype("uint8")], args.out, smoke=True)
        print(f"[sbot-gs] smoke frame -> {args.out}/frame_0000.png")
        return

    # ── rollout ─────────────────────────────────────────────────────────────────────
    viewer = open_rrd_recorder(args.rrd) if args.rrd else None
    if viewer is not None:
        viewer.set_model(model)

    frame_dt = 1.0 / args.fps
    sim_dt = frame_dt / SIM_SUBSTEPS
    num_frames = int(args.duration * args.fps)
    target_q = model.joint_target_q.numpy().copy()
    frames = []
    for frame in range(num_frames):
        t = frame * frame_dt
        for j, qj in zip(handles.arm_joints, arm_target(t, args.duration, still=args.still)):
            target_q[j] = qj
        theta = gripper_theta(t, args.duration, still=args.still)
        for idx, ratio in handles.gripper_coupling:
            target_q[idx] = ratio * theta
        control.joint_target_q.assign(target_q)

        for _ in range(SIM_SUBSTEPS):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

        bq = state_0.body_q.numpy()
        rgb = client.render(link_poses(bq))
        frames.append((rgb * 255.0).clip(0, 255).astype("uint8"))

        if viewer is not None:
            viewer.begin_frame(t)
            viewer.log_state(state_0)
            viewer.end_frame()

        if frame % args.fps == 0:
            print(f"t={t:4.1f}s  shoulder_target={target_q[handles.arm_joints[1]]:+.3f}", flush=True)

    _save(frames, args.out, smoke=False, fps=args.fps)
    if viewer is not None:
        auto_blueprint(args.rrd.replace(".rrd", ".rbl"), model)
    print(f"[sbot-gs] done: {len(frames)} frame(s) -> {args.out}")


def _save(frames, out_dir: str, *, smoke: bool, fps: int = 30) -> None:
    from PIL import Image

    if smoke or len(frames) == 1:
        Image.fromarray(frames[0]).save(os.path.join(out_dir, "frame_0000.png"))
        return
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(out_dir, f"frame_{i:04d}.png"))
    path = os.path.join(out_dir, "sbot.mp4")
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1, pixelformat="yuv420p") as w:
            for f in frames:
                h, ww = f.shape[:2]
                w.append_data(np.ascontiguousarray(f[: h - h % 2, : ww - ww % 2]))
        print(f"[sbot-gs] wrote {path} ({len(frames)} frames @ {fps} fps)")
    except Exception as e:
        print(f"[sbot-gs] mp4 skipped: {e}")


if __name__ == "__main__":
    main()
