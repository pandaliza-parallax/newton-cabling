"""Run the cad_rj45 env a few steps under the base controller and dump where the
blocking contacts are (world y) and between which shapes (socket/plug/latch)."""
import os
import sys

import numpy as np
import torch
import warp as wp

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "rl"))
from connector_env import ConnectorVecEnv  # noqa: E402

env = ConnectorVecEnv(2, seed=0, random_easy=True, asset="cad_rj45")
env.set_stage(0)
env.residual_scale = 0.0
obs = env.reset()

# shape role lookup for env 0
rig0 = None
role = {}
# rebuild role map from shape_to_world + the rig shapes is internal; instead infer:
# socket shapes are the only ones attached to body -1. We can read model arrays.
shape_body = env.model.shape_body.numpy()
plug0 = int(env.plug_idx.numpy()[0]); latch0 = int(env.latch_idx.numpy()[0])
for s in range(env.model.shape_count):
    b = int(shape_body[s])
    if b == -1: role[s] = "socket"
    elif b == plug0: role[s] = "plug0"
    elif b == latch0: role[s] = "latch0"

for step in range(40):
    a = torch.zeros(2, env.act_dim, device="cuda:0")
    obs, rew, done, succ, depth = env.step(a)
    if step in (5, 15, 30, 39):
        env.model.collide(env.state_0, env.contacts)
        wp.synchronize()
        cnt = int(env.contacts.rigid_contact_count.numpy()[0])
        s0 = env.contacts.rigid_contact_shape0.numpy()[:cnt]
        s1 = env.contacts.rigid_contact_shape1.numpy()[:cnt]
        pq = env.state_0.body_q.numpy()
        py = pq[plug0][1]
        # contact points
        try:
            cp = env.contacts.rigid_contact_point0.numpy()[:cnt]
        except Exception:
            cp = None
        pairs = {}
        ys = []
        for i in range(cnt):
            r0 = role.get(int(s0[i]), "?"); r1 = role.get(int(s1[i]), "?")
            if "plug0" in (r0, r1) or "socket" in (r0, r1):
                key = tuple(sorted((r0, r1)))
                pairs[key] = pairs.get(key, 0) + 1
                if cp is not None:
                    ys.append(cp[i][1])
        ydesc = ""
        if ys:
            ys = np.array(ys)
            ydesc = f" contactY mm=[{ys.min()*1000:.1f},{ys.max()*1000:.1f}]"
        print(f"step {step:2d} plug.y={py*1000:6.1f}mm depth={depth[0].item():5.1f}  "
              f"contacts={cnt} pairs={pairs}{ydesc}")
print(f"\nseated y (env0) = {env.seated.numpy()[0][1]*1000:.1f}mm  (plug should reach here)")
