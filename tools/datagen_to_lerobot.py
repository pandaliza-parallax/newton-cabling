"""Convert the sbot GS datagen dump (tools/render_batch.sh) into a LeRobot v2.1 dataset.

Input layout (per episode, from scripts/record_sbot_scene_gs.py --dump):
  <raw>/ep_XXXX/{image/frame_*.png, wrist_image/frame_*.png,
                 state.npy (T,10), action.npy (T,7), phase.npy (T,), meta.json}

  state  (10-D): [eef_pos(3), eef_rot6d(6), gripper(1)]   absolute, robot base frame
  action  (7-D): [dpos(3), drotvec(3), gripper(1)]        base-frame delta to next frame

Run in the OPENPI venv (its pinned lerobot writes the v2.1 format openpi trains on):
  cd ~/parallax/openpi && uv run python ~/parallax/newton-cabling/tools/datagen_to_lerobot.py \
      --raw ~/parallax/newton-cabling/datagen --repo_id parallax/rj45_sbot

Output goes to $HF_LEROBOT_HOME/<repo_id> (default ~/.cache/huggingface/lerobot).
"""
import argparse
import glob
import os
import shutil

import imageio.v2 as imageio
import numpy as np

try:  # lerobot >= ~0.4 dropped the `common` subpackage; openpi's pin still has it
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

PROMPT = "pick up the ethernet cable and plug it into the jack"
FPS = 30
STATE_DIM, ACTION_DIM = 10, 7


def main(raw: str, repo_id: str, push_to_hub: bool, limit: int | None,
         traj_dir: str = "", truncate_after_seat: int = 20) -> None:
    ep_dirs = sorted(d for d in glob.glob(os.path.join(raw, "ep_*")) if os.path.isdir(d))
    # only complete episodes (state + both cams with matching frame counts)
    complete = []
    for d in ep_dirs:
        if not os.path.isfile(os.path.join(d, "state.npy")):
            continue
        n = len(np.load(os.path.join(d, "state.npy")))
        if (len(glob.glob(os.path.join(d, "image", "frame_*.png"))) == n
                and len(glob.glob(os.path.join(d, "wrist_image", "frame_*.png"))) == n):
            complete.append(d)
    if limit:
        complete = complete[:limit]
    if not complete:
        raise SystemExit(f"no complete episodes under {raw}")
    probe = imageio.imread(sorted(glob.glob(os.path.join(complete[0], "image", "frame_*.png")))[0])
    H, W = probe.shape[:2]
    print(f"[convert] {len(complete)}/{len(ep_dirs)} complete episodes, image {H}x{W}, "
          f"state {STATE_DIM}, action {ACTION_DIM}")

    out_path = HF_LEROBOT_HOME / repo_id
    if out_path.exists():
        print(f"[convert] removing existing dataset at {out_path}")
        shutil.rmtree(out_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="standardbots_ro1_gs",
        fps=FPS,
        features={
            "image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_frames = 0
    for ep_dir in complete:
        states = np.load(os.path.join(ep_dir, "state.npy"))
        actions = np.load(os.path.join(ep_dir, "action.npy"))
        imgs = sorted(glob.glob(os.path.join(ep_dir, "image", "frame_*.png")))
        wrists = sorted(glob.glob(os.path.join(ep_dir, "wrist_image", "frame_*.png")))
        assert len(imgs) == len(wrists) == len(states) == len(actions), ep_dir
        total = len(imgs)
        # Truncate the dead post-seat tail: the PPO seats at median traj frame ~52/200, after
        # which ~150 frames are the plug just sitting there (near-zero actions -> stillness bias).
        # Keep everything up to seat + `truncate_after_seat` frames of seated hold.
        if truncate_after_seat >= 0:
            ep = os.path.basename(ep_dir)
            tp = os.path.join(traj_dir, ep, "plug_traj.npy")
            ph = os.path.join(ep_dir, "phase.npy")
            if os.path.isfile(tp) and os.path.isfile(ph):
                y = np.load(tp)[:, 1] * 1000.0
                seated = y >= 11.0                                 # within 0.8mm of full depth
                phase = np.load(ph)
                if seated.any() and (phase == 2).any():
                    seat = int(np.argmax(seated))                  # first seated traj frame
                    insert_start = int(np.argmax(phase == 2))      # dataset frame where replay begins
                    cut = min(len(states), insert_start + seat + truncate_after_seat + 1)
                    states, actions = states[:cut], actions[:cut]
                    imgs, wrists = imgs[:cut], wrists[:cut]
        for i in range(len(imgs)):
            dataset.add_frame({
                "image": imageio.imread(imgs[i]),
                "wrist_image": imageio.imread(wrists[i]),
                "state": states[i].astype(np.float32),
                "actions": actions[i].astype(np.float32),
                "task": PROMPT,
            })
        dataset.save_episode()
        n_frames += len(imgs)
        print(f"[convert] {os.path.basename(ep_dir)}: {len(imgs)}/{total} frames"
              + ("" if len(imgs) == total else " (post-seat tail dropped)"))

    print(f"[convert] done. {len(complete)} episodes, {n_frames} frames -> {out_path}")
    if push_to_hub:
        dataset.push_to_hub(tags=["standardbots", "rj45", "parallax"], private=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.path.expanduser("~/parallax/newton-cabling/datagen"))
    ap.add_argument("--repo_id", default="parallax/rj45_sbot")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N episodes")
    ap.add_argument("--traj_dir", default=os.path.expanduser("~/parallax/newton-cabling/seated_traj"),
                    help="source trajectories (for the seat-frame lookup)")
    ap.add_argument("--truncate_after_seat", type=int, default=20,
                    help="keep this many frames after the plug seats, drop the static tail "
                         "(-1 = keep full episodes)")
    ap.add_argument("--push_to_hub", action="store_true")
    args = ap.parse_args()
    main(args.raw, args.repo_id, args.push_to_hub, args.limit,
         traj_dir=args.traj_dir, truncate_after_seat=args.truncate_after_seat)
