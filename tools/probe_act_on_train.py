#!/usr/bin/env python3
"""probe_ckpt_on_train.py, ACT edition: feed training-set frames to a lerobot ACT
checkpoint and score the FIRST predicted action of a fresh chunk vs the dataset GT.

Run under the openpi venv (it has lerobot + torch):
    uv run python tools/probe_act_on_train.py --ckpt .../checkpoints/act_c10 \
        --ep .../drmix_1000_final_difix/chunk_00/ep_0000 --stride 10
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ep", required=True, help="difixed training episode dir")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
    except ImportError:
        from lerobot.policies.act.modeling_act import ACTPolicy

    # CPU on purpose: the openpi venv's torch has no kernels for this GPU (JAX env),
    # and a 51M ResNet18 ACT over a handful of frames is fast enough without one.
    dev = "cpu"
    policy = ACTPolicy.from_pretrained(args.ckpt)
    policy.config.device = dev
    policy.eval().to(dev)

    st = np.load(os.path.join(args.ep, "state.npy"))
    gt = np.load(os.path.join(args.ep, "action.npy"))
    frames = list(range(0, len(st) - 1, args.stride))
    pred = []
    with torch.no_grad():
        for t in frames:
            obs = {}
            for key, sub in (("observation.images.front", "image"),
                             ("observation.images.wrist", "wrist_image")):
                im = np.asarray(Image.open(f"{args.ep}/{sub}/frame_{t:04d}.png"), np.float32) / 255.0
                obs[key] = torch.from_numpy(im).permute(2, 0, 1)[None].to("cpu")
            obs["observation.state"] = torch.from_numpy(st[t].astype(np.float32))[None]
            policy.reset()                               # fresh chunk per probed frame
            a = policy.select_action(obs)                # first action of the new chunk
            pred.append(a[0].float().cpu().numpy())
    P, G = np.stack(pred), gt[frames]
    dp_err = np.linalg.norm(P[:, :3] - G[:, :3], axis=1) * 1000
    gt_mag = np.linalg.norm(G[:, :3], axis=1) * 1000
    mask = gt_mag > 0.05
    cos = np.array([float(np.dot(P[i, :3], G[i, :3]) /
                    (np.linalg.norm(P[i, :3]) * np.linalg.norm(G[i, :3]) + 1e-12))
                    for i in range(len(P))])
    if args.dump:
        np.savez(args.dump, pred=P, gt=G, frames=np.asarray(frames), state=st[frames])
    name = os.path.basename(args.ckpt.rstrip("/"))
    print(f"[act-probe] {name}  ep {os.path.basename(args.ep)}  frames {len(frames)}")
    print(f"[act-probe]   dpos error: mean {dp_err.mean():.3f}mm  median {np.median(dp_err):.3f}mm  "
          f"(expert |dpos| mean {gt_mag.mean():.3f}mm)")
    print(f"[act-probe]   direction cos (moving frames): mean {cos[mask].mean():.3f}  "
          f"frac>0.5: {(cos[mask] > 0.5).mean():.2f}")
    print(f"[act-probe]   pred |dpos| min {np.linalg.norm(P[:, :3], axis=1).min() * 1000:.3f}mm  "
          f"grip pred mean {P[:, 6].mean():.2f}")


if __name__ == "__main__":
    main()
