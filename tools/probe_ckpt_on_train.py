#!/usr/bin/env python3
"""Feed TRAINING-SET frames to a trained checkpoint and score predicted vs expert actions.

Open-loop, on-distribution probe: for sampled frames of a difixed training episode,
run policy.infer({front, wrist, state, prompt}) and compare the first predicted action
against the dataset's ground-truth action at that frame.

Run under the openpi venv, one checkpoint per invocation (JAX memory):
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 uv run python .../probe_ckpt_on_train.py \
        --ckpt .../checkpoints/pi05_drmix1000_lora/19999 --ep <difixed ep dir> --stride 10
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--server", default=None, help="host:port of a running policy server "
                    "(reuses its GPU instead of loading a second model)")
    ap.add_argument("--config", default="pi05_drmix1000_lora")
    ap.add_argument("--ep", required=True, help="difixed training episode dir")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--prompt", default="pick up the ethernet cable and plug it into the jack")
    ap.add_argument("--dump", default=None, help="save predictions/GT to this .npz for plotting")
    args = ap.parse_args()

    from PIL import Image

    if args.server:
        from openpi_client import websocket_client_policy
        host, port = args.server.split(":")
        policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=int(port))
    else:
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
        policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.ckpt)

    st = np.load(os.path.join(args.ep, "state.npy"))
    gt = np.load(os.path.join(args.ep, "action.npy"))
    T = len(st)
    frames = list(range(0, T - 1, args.stride))
    pred = []
    for t in frames:
        obs = {
            "observation/image": np.asarray(Image.open(f"{args.ep}/image/frame_{t:04d}.png")),
            "observation/wrist_image": np.asarray(Image.open(f"{args.ep}/wrist_image/frame_{t:04d}.png")),
            "observation/state": st[t].astype(np.float32),
            "prompt": args.prompt,
        }
        pred.append(np.asarray(policy.infer(obs)["actions"])[0])   # first action of the chunk
    P = np.stack(pred)
    G = gt[frames]
    dp_err = np.linalg.norm(P[:, :3] - G[:, :3], axis=1) * 1000
    gt_mag = np.linalg.norm(G[:, :3], axis=1) * 1000
    # direction agreement where the expert actually moves
    mask = gt_mag > 0.05
    cos = np.array([float(np.dot(P[i, :3], G[i, :3]) /
                    (np.linalg.norm(P[i, :3]) * np.linalg.norm(G[i, :3]) + 1e-12)) for i in range(len(P))])
    if args.dump:
        np.savez(args.dump, pred=P, gt=G, frames=np.asarray(frames), state=st[frames])
    step = args.server or os.path.basename(args.ckpt.rstrip("/"))
    print(f"[probe] ckpt {step}  ep {os.path.basename(args.ep)}  frames {len(frames)}")
    print(f"[probe]   dpos error: mean {dp_err.mean():.3f}mm  median {np.median(dp_err):.3f}mm  "
          f"(expert |dpos| mean {gt_mag.mean():.3f}mm)")
    print(f"[probe]   direction cos (moving frames): mean {cos[mask].mean():.3f}  "
          f"frac>0.5: {(cos[mask] > 0.5).mean():.2f}")
    print(f"[probe]   grip pred mean {P[:, 6].mean():.2f} (gt {G[:, 6].mean():.2f})")


if __name__ == "__main__":
    main()
