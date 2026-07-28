"""Record a rollout of the trained insertion policy (or the base controller) to a
rerun .rrd, so you can watch the plugs seat from misaligned starts.

    uv run --extra sim python rl/record_policy.py                 # trained RL policy
    uv run --extra sim python rl/record_policy.py --base          # scripted base controller
    uv run --extra sim python rl/record_policy.py --stage 6       # harder (15mm) offset
    uvx --from rerun-sdk rerun rl_rollout.rrd rl_rollout.rbl      # view it
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connector_env import ConnectorVecEnv  # noqa: E402
from train_ppo import ActorCritic  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder  # noqa: E402

DEV = "cuda:0"
HOST_GS = "/home/pandaliza/parallax/gs-sim-vla/scene/assets"
HOST_ETH = f"{HOST_GS}/objects/ethernet"
HOST_TABLE = f"{HOST_GS}/objects/table/splat.ply"
HOST_BG = f"{HOST_GS}/background/splat.ply"


def _h2c(p):
    """Host path -> renderer-container path (host ~/parallax is bind-mounted to /root)."""
    return p.replace("/home/pandaliza/parallax", "/root/parallax") if p else None


def _socket_world(asset):
    """World pos of the static socket for env 0 = /World/Socket prim translation + Z_LIFT."""
    import newton  # noqa: PLC0415
    import newton.usd  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415
    from pxr import Usd  # noqa: PLC0415

    from connector_env import ASSET_PROFILES, Z_LIFT  # noqa: PLC0415
    from newton_cabling.sim.scene import resolve_asset_path  # noqa: PLC0415

    spec = ASSET_PROFILES[asset]["spec"]()
    stage = Usd.Stage.Open(resolve_asset_path(spec.usd_asset_name))
    sb = wp.transform_get_translation(
        newton.usd.get_transform(stage.GetPrimAtPath(spec.socket_prim_path), local=False)
    )
    return np.array([sb[0], sb[1], sb[2]]) + Z_LIFT

# --- scripted latch flip (render-only overlay) -------------------------------
# A real physics latch click is rigid-VBD-capped (a box catch jams the plug; see ASSETS.md),
# so for the VISUAL the latch flip is scripted on top of the real physics rollout: keyed to
# the plug's actual insertion fraction, the clip flattens over the entry then snaps back up to
# latch near the seat. It is applied to the logged pose only and restored each frame, so the
# policy/physics are untouched. Toggle with --no-latch-anim; flip FLIP_MAX sign if it inverts.
FLIP_MAX = 0.40   # rad, latch flatten amplitude


def _latch_theta(frac: float) -> float:
    """Latch flip angle vs insertion fraction [0,1]: rest (0) -> flatten -> snap back to latch."""
    if frac < 0.35:
        return 0.0
    if frac < 0.78:
        return FLIP_MAX * (frac - 0.35) / 0.43          # flatten as the clip enters
    if frac < 0.88:
        return FLIP_MAX * (1.0 - (frac - 0.78) / 0.10)  # snap up to latch (the click)
    return 0.0


def _quat_mul(a, b):  # hamilton product, (x,y,z,w)
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], dtype=np.float32)


def _quat_x(th):  # rotation about local x by th (the hinge axis)
    return np.array([np.sin(th / 2.0), 0.0, 0.0, np.cos(th / 2.0)], dtype=np.float32)


def _save_gs(frames, out_dir):
    """Save GS frames as PNGs + a VLC-playable mp4.

    imageio.mimsave's h264 (even with +faststart) wouldn't open in VLC, so we encode the PNG
    sequence with a direct ffmpeg call using the settings that play in VLC/QuickTime
    (libx264 + yuv420p + high profile + crf18 + faststart). Falls back to a gif if ffmpeg fails.
    """
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image

    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(out_dir, f"frame_{i:04d}.png"))
    try:
        import subprocess  # noqa: PLC0415

        import imageio_ffmpeg  # noqa: PLC0415
        mp4 = os.path.join(out_dir, "rollout.mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", "30",
                        "-i", os.path.join(out_dir, "frame_%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
                        "-crf", "18", "-movflags", "+faststart", "-an", mp4],
                       check=True, capture_output=True)
        vid = "rollout.mp4"
    except Exception as e:  # noqa: BLE001 — ffmpeg may fail; gif always works
        print(f"[gs] ffmpeg mp4 failed ({e}); writing gif")
        import imageio.v2 as imageio  # noqa: PLC0415
        imageio.mimsave(os.path.join(out_dir, "rollout.gif"), frames[::2], fps=20)
        vid = "rollout.gif"
    print(f"[gs] {len(frames)} frames -> {out_dir}/ ({vid})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="rl/runs/extend/best_model.pt")
    ap.add_argument("--seed", type=int, default=7, help="env seed (start poses); use a different "
                    "value for a held-out eval set disjoint from the training dump")
    ap.add_argument("--envs", type=int, default=9)       # 3x3 grid of connectors
    ap.add_argument("--stage", type=int, default=5)      # curriculum stage (5 = 11mm offset)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--out", default="rl_rollout")
    ap.add_argument("--base", action="store_true", help="record the base controller (no policy)")
    ap.add_argument("--random-easy", action="store_true",
                    help="random_easy_subset starts (lateral + approach + <=15deg rotation)")
    ap.add_argument("--asset", default="rj45", choices=["rj45", "cad_rj45", "cad_rj45_real"])
    ap.add_argument("--latch-anim", action="store_true",
                    help="add a scripted latch-flip overlay (render-only, physics untouched; "
                         "OFF by default — its timing is approximate)")
    ap.add_argument("--residual-scale", type=float, default=None,
                    help="match the policy's training action authority (e.g. 1.0); default 0.15")
    ap.add_argument("--re-scale", type=float, default=None,
                    help="override the start-misalignment scale (>1.0 = harder than stage 5)")
    # GS video options (render env 0 of the rollout through the parallax_sim splat renderer)
    ap.add_argument("--gs", action="store_true",
                    help="also render env 0 through the GS renderer to a video (needs the "
                         "parallax_sim renderer up; run under sudo + PYTHONPATH=...DalusPySim)")
    ap.add_argument("--gs-out", default=None, help="GS frame/gif dir (default <out>_gs)")
    ap.add_argument("--gs-mount", default=f"{HOST_ETH}/cad_jack_registered.ply", help="JACK splat (host path)")
    ap.add_argument("--gs-plug", default=f"{HOST_ETH}/cad_plug_registered.ply", help="PLUG splat (host path)")
    # obj->Newton calibration for the registered cad splats (registered to *_meters.obj, +Z insertion).
    # rotation R = R180Y . R_USER_TO_NEWTON = XYZ-intrinsic euler (90,0,-180): obj -Z (leading face) -> Newton +Y.
    # anchors pin the splat's MATING FACE (plug, obj z=0) / cavity MOUTH (jack, obj z=MOUTH_Z=30mm) to the
    # Newton body origin -- NOT the splat's geometric centroid, which would offset insertion depth.
    ap.add_argument("--mount-rot", type=float, nargs=3, default=[90.0, 0.0, -180.0],
                    help="JACK splat euler deg (obj->Newton); default (90,0,-180)")
    ap.add_argument("--plug-rot", type=float, nargs=3, default=[-90.0, 0.0, 0.0],
                    help="PLUG splat euler deg (obj->Newton); (-90,0,0) maps obj +Z (gold-contact/mating "
                         "end, measured at z~18mm) -> Newton +Y so the plug inserts contacts-first.")
    ap.add_argument("--mount-anchor", type=float, nargs=3, default=[0.0, 0.0, 0.030],
                    help="JACK obj-frame point pinned to the Newton socket origin (m); cavity mouth z=MOUTH_Z")
    ap.add_argument("--mount-pos", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="extra translation (m) to MOVE THE JACK only (with --mount-rot to rotate it). "
                         "NOTE: moves the jack but not the plug -> they decouple (framing/prop use).")
    ap.add_argument("--plug-anchor", type=float, nargs=3, default=[-0.0015, -0.0015, 0.0172],
                    help="PLUG obj-frame point pinned to the Newton plug origin (m); mating-face "
                         "centroid (gold contacts at z~17mm), measured from cad_plug_registered.ply")
    ap.add_argument("--verify-overlay", action="store_true",
                    help="render ONE frame with the plug at a fixed pose (overlay_*.png) and exit")
    ap.add_argument("--unseated", action="store_true",
                    help="--verify-overlay: render the plug at its START/approach pose (NOT seated)")
    # openpi/pi0.5 dataset dump (raw per-step image+state+action; convert with raw_to_lerobot.py)
    ap.add_argument("--dump", default=None,
                    help="dump a raw VLA dataset to this dir (needs --gs): per step image + plug pose + "
                         "executed action; loops --episodes. Convert to LeRobot with raw_to_lerobot.py.")
    ap.add_argument("--episodes", type=int, default=30, help="episodes to record for --dump")
    ap.add_argument("--prompt", default="insert the ethernet plug into the jack",
                    help="language instruction stored in the dump meta.json")
    ap.add_argument("--seated-only", action="store_true", help="--dump: drop episodes that never seat")
    # closed-loop VLA eval (drive the plug with a pi0.5 policy server, NO base controller)
    ap.add_argument("--eval-vla", action="store_true",
                    help="eval a pi0.5 policy (served via openpi serve_policy) closed-loop over --episodes; "
                         "needs --gs + openpi_client in this venv; reports seated %")
    ap.add_argument("--vla-host", default="0.0.0.0", help="VLA policy-server host")
    ap.add_argument("--vla-port", type=int, default=8000, help="VLA policy-server port")
    # composite scene (table + room) — ON by default; pass "" to either to disable
    ap.add_argument("--gs-bg", default=HOST_BG, help="room background splat ('' = none)")
    ap.add_argument("--gs-table", default=HOST_TABLE, help="table splat ('' = none)")
    ap.add_argument("--table-pos", type=float, nargs=3, default=[0.0, 0.0, 0.091418],
                    help="table splat position in the room frame (gs-sim-vla value)")
    ap.add_argument("--table-place", type=float, nargs=3, default=[-0.205, 0.125, 0.83],
                    help="where the socket sits on the table (room frame); the whole physics "
                         "connector is shifted here so it rests on the table")
    ap.add_argument("--gs-cam-dist", type=float, default=0.4,
                    help="camera distance from the connector in METRES (0.4 shows the table; "
                         "drop to ~0.12 for a tight connector close-up)")
    ap.add_argument("--gs-cam-dir", type=float, nargs=3, default=[0.5, -0.6, 0.5],
                    help="camera direction (unit-ish offset from the connector); orbit this to "
                         "frame the insertion. e.g. side-on '1 -0.2 0.2', down-the-mouth '0 -1 0.25'")
    ap.add_argument("--gs-gamma", type=float, default=1.0,
                    help="post-render gamma lift to brighten the dark jack cavity (gs splats can't "
                         "be relit). 1.0=off; try 1.8-2.4. Lifts shadows much more than highlights.")
    args = ap.parse_args()

    env = ConnectorVecEnv(args.envs, seed=args.seed, random_easy=args.random_easy, asset=args.asset)
    env.set_stage(args.stage)
    if args.residual_scale is not None:
        env.residual_scale = args.residual_scale
    if args.re_scale is not None:
        env.re_scale = args.re_scale   # push past the stage-5 distribution (extrapolation)
    # difficulty label differs by mode: random_easy ramps a scale, else a lateral offset
    diff = (f"re_scale {env.re_scale:.2f}" if env.random_easy
            else f"<= {env._mag*1000:.1f}mm offset")
    if args.base or args.eval_vla or args.verify_overlay:
        # No RL ActorCritic needed: --eval-vla uses the pi0.5 server (step_direct);
        # --verify-overlay forces the seated pose; --base is the scripted controller.
        env.residual_scale = 0.0
        ac = None
        _mode = "VLA eval (no RL policy)" if args.eval_vla else (
            "verify-overlay" if args.verify_overlay else "BASE controller")
        print(f"recording {_mode} | {args.asset} | stage {args.stage} ({diff})")
    else:
        ac = ActorCritic(env.obs_dim, env.act_dim).to(DEV)
        ac.load_state_dict(torch.load(args.checkpoint, map_location=DEV))
        ac.eval()
        print(f"recording POLICY {args.checkpoint} | {args.asset} | stage {args.stage} ({diff})")

    viewer = open_rrd_recorder(f"{args.out}.rrd")
    viewer.set_model(env.model)
    obs = env.reset()
    zero = torch.zeros(args.envs, env.act_dim, device=DEV)
    dt = 1.0 / 60.0
    sim_time = 0.0
    # latch-flip overlay state: key the flip to each plug's insertion fraction (0 at the
    # start pose -> 1 at the seat). Render-only: override the latch quat for logging, restore.
    animate = args.latch_anim
    pi_np, li_np = env.plug_idx.numpy(), env.latch_idx.numpy()
    seat_y = env.seated.numpy()[:, 1]
    start_y = env.state_0.body_q.numpy()[pi_np, 1].copy()

    # GS video setup (renders env 0 of this exact rollout through the splat renderer)
    gs_frames = None
    if args.gs:
        from newton_cabling.render.gs_bridge import (  # noqa: PLC0415
            NewtonGSClient, euler_deg_to_quat_wxyz, look_at_quat, make_intrinsics,
            newton_pose, place_on_body, ply_centroid,
        )
        socket_pos = _socket_world(args.asset)
        # Shift the whole physics connector (socket + plug) from its physics frame onto the
        # table in the ROOM frame, so it composites with the table/room splats.
        scene_off = np.array(args.table_place) - socket_pos
        mount_off = np.array(args.mount_pos, dtype=float)   # extra jack-only translation
        # Anchors (obj-frame mating face / mouth) pin each splat to the Newton body origin,
        # NOT the splat's geometric centroid (ply_centroid) which would offset insertion depth.
        mount_c, plug_c = np.array(args.mount_anchor), np.array(args.plug_anchor)
        mount_align = euler_deg_to_quat_wxyz(*args.mount_rot)
        plug_align = euler_deg_to_quat_wxyz(*args.plug_rot)
        # camera frames the connector (on the table); 0.4 m back shows the table around it
        target = np.array(args.table_place)
        cam_dir = np.array(args.gs_cam_dir, dtype=float)
        cam_dir /= np.linalg.norm(cam_dir)
        eye = (target + cam_dir * args.gs_cam_dist).tolist()
        # object order [table?, mount, plug]; room is the client's separate bg slot
        has_table = bool(args.gs_table)
        objs = ([_h2c(args.gs_table)] if has_table else []) + [_h2c(args.gs_mount), _h2c(args.gs_plug)]
        table_pose = (list(args.table_pos), [1.0, 0.0, 0.0, 0.0])
        gs_client = NewtonGSClient(
            objs, make_intrinsics(852, 640, 45.0), eye,
            look_at_quat(eye, target.tolist(), up=(0, 0, 1), convention="ros"),
            bg_ply=_h2c(args.gs_bg) if args.gs_bg else None,
        )
        # post-render gamma lift: gs splats are baked (no scene light possible), so this is the
        # only "brighten". gamma>1 lifts shadows (the dark jack cavity) far more than highlights.
        _gamma = float(args.gs_gamma)
        def _post(rgb):  # noqa: E306
            return rgb if _gamma == 1.0 else np.clip(rgb, 0.0, 1.0) ** (1.0 / _gamma)
        gs_pi0, gs_frames = int(pi_np[0]), []
        print(f"[gs] composite | socket->{np.round(args.table_place, 3)} | "
              f"bg={'on' if args.gs_bg else 'off'} table={'on' if has_table else 'off'} | "
              f"mount_rot={args.mount_rot} plug_rot={args.plug_rot}")

    if args.gs and args.verify_overlay:
        # Render ONE frame at a fixed plug pose. SEATED by default (checks the obj->Newton
        # calibration); --unseated uses the env reset/start pose (plug not inserted).
        from PIL import Image  # noqa: PLC0415
        env.reset()                                   # deterministic start pose (per --seed)
        bq = env.state_0.body_q.numpy()
        if not args.unseated:
            bq[gs_pi0, 0:3] = env.seated.numpy()[0]   # force seated position (env 0)
            bq[gs_pi0, 3:7] = env.plug_rot.numpy()[0]  # seated orientation (xyzw, scalar-last)
            env.state_0.body_q.assign(bq)
        pp, pq = newton_pose(env.state_0.body_q.numpy(), gs_pi0)
        mount_pose = place_on_body(socket_pos + scene_off + mount_off, [1.0, 0.0, 0.0, 0.0],
                                   align_quat_wxyz=mount_align, centroid=mount_c)
        plug_pose = place_on_body(np.array(pp) + scene_off, pq,
                                  align_quat_wxyz=plug_align, centroid=plug_c)
        poses = ([table_pose] if has_table else []) + [mount_pose, plug_pose]
        rgb = _post(gs_client.render(poses))
        outd = args.gs_out or f"{args.out}_gs"
        os.makedirs(outd, exist_ok=True)
        name = "overlay_unseated.png" if args.unseated else "overlay_seated.png"
        Image.fromarray((rgb * 255.0).clip(0, 255).astype("uint8")).save(os.path.join(outd, name))
        print(f"[gs] verify-overlay: plug at {'START (unseated)' if args.unseated else 'SEATED'} pose -> {outd}/{name}")
        return

    if args.gs and args.dump:
        # openpi/pi0.5 raw dump: one episode folder per rollout, per-step image + state + action.
        import json  # noqa: PLC0415
        import shutil  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415
        os.makedirs(args.dump, exist_ok=True)
        H = W = None
        n_saved = n_seated = n_frames = 0
        for ep in range(args.episodes):
            obs = env.reset()
            ep_dir = os.path.join(args.dump, f"ep_{ep:04d}")
            os.makedirs(ep_dir, exist_ok=True)
            states, actions, seated = [], [], False
            for f in range(args.frames):
                bq = env.state_0.body_q.numpy()
                pp, pq = newton_pose(bq, gs_pi0)               # plug pose (world)
                mount_pose = place_on_body(socket_pos + scene_off + mount_off, [1.0, 0.0, 0.0, 0.0],
                                           align_quat_wxyz=mount_align, centroid=mount_c)
                plug_pose = place_on_body(np.array(pp) + scene_off, pq,
                                          align_quat_wxyz=plug_align, centroid=plug_c)
                poses = ([table_pose] if has_table else []) + [mount_pose, plug_pose]
                rgb = (_post(gs_client.render(poses)) * 255.0).clip(0, 255).astype("uint8")
                if H is None:
                    H, W = rgb.shape[:2]
                Image.fromarray(rgb).save(os.path.join(ep_dir, f"frame_{f:06d}.png"))
                # state = plug pose in the socket frame [pos3, quat4]
                states.append(np.concatenate([np.array(pp) - socket_pos, np.array(pq)]).astype(np.float32))
                with torch.no_grad():
                    a = ac.mean_action(obs)
                obs, _, _, succ, _ = env.step(a)
                # executed normalized command: [pos total (±1·MAX_DELTA), rot cmd (±1·ROT_CMD_RANGE)]
                act = torch.cat([env._act_keep[0], env._rotcmd_keep[0].clamp(-1.0, 1.0)])
                actions.append(act.detach().cpu().numpy().astype(np.float32))
                seated = seated or bool(succ[0].item() > 0.5)
            if args.seated_only and not seated:
                shutil.rmtree(ep_dir)
                print(f"[dump] ep {ep + 1}/{args.episodes}: not seated -> dropped")
                continue
            np.save(os.path.join(ep_dir, "state.npy"), np.stack(states))
            np.save(os.path.join(ep_dir, "actions.npy"), np.stack(actions))
            n_saved += 1; n_seated += int(seated); n_frames += len(states)
            print(f"[dump] ep {ep + 1}/{args.episodes}: {len(states)} frames seated={seated}")
        with open(os.path.join(args.dump, "meta.json"), "w") as fj:
            json.dump({
                "prompt": args.prompt, "fps": 30, "robot_type": f"newton_{args.asset}",
                "task": args.asset, "num_episodes": n_saved, "image_hw": [H, W],
                "state_dim": 7, "action_dim": 6,
                "state_layout": "plug pose in socket frame [x,y,z, qw,qx,qy,qz] (m, unit quat)",
                "action_layout": "executed cmd [dpos x,y,z (norm, ±1=2mm), rot x,y,z (norm, ±1=ROT_CMD_RANGE)]",
            }, fj, indent=2)
        print(f"[dump] done: {n_saved} episodes ({n_frames} frames, {n_seated} seated) -> {args.dump}")
        return

    if args.gs and args.eval_vla:
        # Closed-loop eval: a pi0.5 policy server drives the plug DIRECTLY (step_direct, no base
        # controller), so seating reflects the policy alone. Reports seated % over --episodes.
        from openpi_client import websocket_client_policy as _wcp  # noqa: PLC0415
        client = _wcp.WebsocketClientPolicy(host=args.vla_host, port=args.vla_port)
        print(f"[eval] VLA server {args.vla_host}:{args.vla_port} | meta={client.get_server_metadata()}")
        outd = args.gs_out or f"{args.out}_gs"   # rollout videos saved here, one folder per episode
        n_seat = 0
        for ep in range(args.episodes):
            obs = env.reset()
            seated, depth, frames, traj = False, None, [], []
            for _ in range(args.frames):
                bq = env.state_0.body_q.numpy()
                pp, pq = newton_pose(bq, gs_pi0)
                mount_pose = place_on_body(socket_pos + scene_off + mount_off, [1.0, 0.0, 0.0, 0.0],
                                           align_quat_wxyz=mount_align, centroid=mount_c)
                plug_pose = place_on_body(np.array(pp) + scene_off, pq,
                                          align_quat_wxyz=plug_align, centroid=plug_c)
                poses = ([table_pose] if has_table else []) + [mount_pose, plug_pose]
                rgb = (_post(gs_client.render(poses)) * 255.0).clip(0, 255).astype("uint8")
                frames.append(rgb)
                state = np.concatenate([np.array(pp) - socket_pos, np.array(pq)]).astype(np.float32)
                traj.append(state)                                    # plug pose in socket frame [pos3,quat4] (wxyz)
                result = client.infer({"observation/image": rgb, "observation/state": state, "prompt": args.prompt})
                act = np.asarray(result["actions"])[0]                # first action of the chunk (6,)
                a = torch.as_tensor(act, dtype=torch.float32, device=DEV).unsqueeze(0)
                _, succ, depth = env.step_direct(a)
                seated = seated or bool(succ[0].item() > 0.5)
            _save_gs(frames, os.path.join(outd, f"ep_{ep:04d}"))      # PNGs + rollout.mp4 per episode
            np.save(os.path.join(outd, f"ep_{ep:04d}", "plug_traj.npy"), np.stack(traj))  # (T,7) for arm replay
            n_seat += int(seated)
            print(f"[eval] ep {ep + 1}/{args.episodes}: seated={seated}  depth={float(depth.mean()):.1f}mm")
        print(f"[eval] VLA seated {n_seat}/{args.episodes} -> videos in {outd}/ep_*/  (seed {args.seed})")
        return

    for f in range(args.frames):
        with torch.no_grad():
            a = zero if ac is None else ac.mean_action(obs)
        obs, _, _, succ, depth = env.step(a)
        viewer.begin_frame(sim_time)
        if animate:
            bq = env.state_0.body_q
            arr = bq.numpy()
            frac = np.clip((arr[pi_np, 1] - start_y) / (seat_y - start_y + 1e-9), 0.0, 1.0)
            mod = arr.copy()
            for k in range(env.n):
                th = _latch_theta(float(frac[k]))
                if th != 0.0:
                    mod[li_np[k], 3:7] = _quat_mul(arr[li_np[k], 3:7], _quat_x(th))
            bq.assign(mod)
            viewer.log_state(env.state_0)
            bq.assign(arr)          # restore so the next physics step is unperturbed
        else:
            viewer.log_state(env.state_0)
        viewer.end_frame()
        if gs_frames is not None:
            bq = env.state_0.body_q.numpy()
            pp, pq = newton_pose(bq, gs_pi0)
            # shift socket + plug onto the table (room frame), keeping their relative motion
            mount_pose = place_on_body(socket_pos + scene_off + mount_off, [1.0, 0.0, 0.0, 0.0],
                                       align_quat_wxyz=mount_align, centroid=mount_c)
            plug_pose = place_on_body(np.array(pp) + scene_off, pq,
                                      align_quat_wxyz=plug_align, centroid=plug_c)
            poses = ([table_pose] if has_table else []) + [mount_pose, plug_pose]
            rgb = _post(gs_client.render(poses))
            gs_frames.append((rgb * 255.0).clip(0, 255).astype("uint8"))
        sim_time += dt
        if f % 30 == 0 or f == args.frames - 1:
            print(f"frame {f:3d}: seated {succ.mean().item()*100:5.1f}% | "
                  f"depth {depth.mean().item():5.1f}mm", flush=True)

    if gs_frames is not None:
        _save_gs(gs_frames, args.gs_out or f"{args.out}_gs")

    blueprint = auto_blueprint(f"{args.out}.rbl", env.model)
    print(f"\nrecording complete: {args.out}.rrd ; blueprint {blueprint}")
    print(f"view with:  uvx --from rerun-sdk rerun {args.out}.rrd {args.out}.rbl")


if __name__ == "__main__":
    main()
