#!/usr/bin/env python3
"""Serve a lerobot ACT checkpoint over the openpi websocket protocol, so the
closed-loop eval harness (record_sbot_scene_gs_cable.py --policy-server) can drive
it exactly like a pi05 checkpoint.

Obs mapping (openpi client -> ACT): observation/image -> observation.images.front,
observation/wrist_image -> observation.images.wrist (uint8 HWC 512 -> float CHW /255),
observation/state -> observation.state; prompt ignored (ACT is not language-conditioned).
Returns the policy's FULL action chunk per infer, so the client's execute-whole-chunk
loop reproduces each model's native recipe (c10 -> 10 steps/obs, c100 -> 100).

CPU on purpose: the openpi venv's torch has no CUDA kernels for this GPU, and a 51M
ACT is fast enough (one forward per chunk).

    cd ~/parallax/openpi && uv run python .../serve_act_policy.py \
        --ckpt .../checkpoints/act_c10 --port 8000
"""
import argparse
import logging
import sys

import numpy as np

# Runs under either the openpi venv (CPU torch) or ~/parallax/act_venv (cu128 torch
# for the RTX 5090). The path inserts make openpi.serving + openpi_client importable
# from the bare act_venv (they only need websockets/msgpack/numpy).
sys.path.insert(0, "/home/pandaliza/parallax/openpi/src")
sys.path.insert(0, "/home/pandaliza/parallax/openpi/packages/openpi-client/src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    args = ap.parse_args()

    import torch
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
    except ImportError:
        from lerobot.policies.act.modeling_act import ACTPolicy
    from openpi.serving import websocket_policy_server

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:  # Blackwell needs cu128 kernels; fall back rather than die
        torch.zeros(1, device=dev) + 1
    except RuntimeError:
        print(f"[act-server] {dev} unusable in this venv, falling back to cpu", flush=True)
        dev = "cpu"
    policy = ACTPolicy.from_pretrained(args.ckpt)
    policy.config.device = dev
    policy.eval().to(dev)
    n_act = int(policy.config.n_action_steps)

    class ActAdapter:
        def infer(self, obs):
            batch = {}
            for src, dst in (("observation/image", "observation.images.front"),
                             ("observation/wrist_image", "observation.images.wrist")):
                im = np.asarray(obs[src], np.float32) / 255.0
                batch[dst] = torch.from_numpy(im).permute(2, 0, 1)[None].to(dev)
            batch["observation.state"] = torch.from_numpy(
                np.asarray(obs["observation/state"], np.float32))[None].to(dev)
            with torch.no_grad():
                policy.reset()
                acts = [policy.select_action(batch)[0].float().cpu().numpy()
                        for _ in range(n_act)]
            return {"actions": np.stack(acts).astype(np.float64)}

    logging.basicConfig(level=logging.INFO)
    print(f"[act-server] {args.ckpt} chunk {n_act} on :{args.port}", flush=True)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=ActAdapter(), host="0.0.0.0", port=args.port,
        metadata={"model": "act", "ckpt": args.ckpt, "chunk": n_act})
    print("listening", flush=True)   # the eval scripts' readiness gate greps for this
    server.serve_forever()


if __name__ == "__main__":
    main()
