#!/usr/bin/env python3
"""Decoder cross-attention maps for a lerobot ACT checkpoint.

Hooks the (single) transformer decoder layer's cross-attention and visualizes,
per frame, where the action queries look in the two camera views. Encoder token
layout (lerobot ACT): [latent, robot_state, front 16x16, wrist 16x16].

Run in the cu128 venv:
    ~/parallax/act_venv/bin/python tools/act_attention_maps.py \
        --ckpt .../checkpoints/act_c10_combined \
        --ep .../drmix_1000_depaused/chunk_11/ep_0006 --stride 25 --out maps.png
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ep", required=True, help="episode dir with image/ wrist_image/ state.npy")
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from lerobot.common.policies.act.modeling_act import ACTPolicy

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(args.ckpt)
    policy.config.device = dev
    policy.eval().to(dev)

    captured = {}
    def hook(_mod, _inp, out):
        # nn.MultiheadAttention output: (attn_out, attn_weights (B, tgt, src) head-avg)
        captured["w"] = out[1].detach().float().cpu().numpy()
    policy.model.decoder.layers[0].multihead_attn.register_forward_hook(hook)

    st = np.load(os.path.join(args.ep, "state.npy"))
    T = len(st)
    frames = args.frames or list(range(0, T - 1, args.stride))
    G = 512 // 32  # ResNet18 stride-32 feature grid

    rows = []
    for t in frames:
        ims = {}
        batch = {}
        for key, sub in (("observation.images.front", "image"),
                         ("observation.images.wrist", "wrist_image")):
            im = Image.open(os.path.join(args.ep, sub, f"frame_{t:04d}.png")).convert("RGB")
            ims[sub] = np.asarray(im)
            arr = np.asarray(im, np.float32) / 255.0
            batch[key] = torch.from_numpy(arr).permute(2, 0, 1)[None].to(dev)
        batch["observation.state"] = torch.from_numpy(st[t].astype(np.float32))[None].to(dev)
        with torch.no_grad():
            policy.reset()
            policy.select_action(batch)
        w = captured["w"][0]                 # (n_queries, n_src)
        att = w.mean(axis=0)                 # avg over the 10 action queries
        n_img = G * G
        parts = {"latent": att[0], "state": att[1],
                 "front": att[2:2 + n_img], "wrist": att[2 + n_img:2 + 2 * n_img]}
        maps = {}
        for cam in ("front", "wrist"):
            m = parts[cam].reshape(G, G)
            m = (m - m.min()) / (m.max() - m.min() + 1e-9)
            maps[cam] = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                                   .resize((512, 512), Image.BICUBIC), np.float32) / 255.0
        rows.append((t, ims, maps,
                     {k: (float(v.sum()) if hasattr(v, "sum") else float(v))
                      for k, v in parts.items()}))

    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.3 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    jet = matplotlib.colormaps["jet"]
    for r, (t, ims, maps, mass) in enumerate(rows):
        for c, (sub, cam) in enumerate((("image", "front"), ("wrist_image", "wrist"))):
            axes[r, 2 * c].imshow(ims[sub])
            axes[r, 2 * c].set_title(f"t={t} {cam}", fontsize=9)
            heat = (jet(maps[cam])[:, :, :3] * 255).astype(np.uint8)
            blend = (0.55 * ims[sub] + 0.45 * heat).astype(np.uint8)
            axes[r, 2 * c + 1].imshow(blend)
            axes[r, 2 * c + 1].set_title(
                f"attn ({cam} {100 * mass[cam] / sum(mass.values()):.0f}% of mass)", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle(f"ACT decoder cross-attention — {os.path.basename(args.ckpt)} on "
                 f"{'/'.join(args.ep.rstrip('/').split('/')[-2:])}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
