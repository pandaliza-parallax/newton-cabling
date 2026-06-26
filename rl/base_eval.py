import torch
from connector_env import ConnectorVecEnv

env = ConnectorVecEnv(2000, random_easy=True, asset="cad_rj45", seed=0)
print(f"asset=cad_rj45 random_easy 6-DOF  (friction=0.5, current default)")
print(f"success = plug within 5mm of the 12mm seat (>=7mm in) + lat<=3mm + ang<=seat_angle, held 20 steps\n")
for stage in range(env.num_stages):
    env.set_stage(stage)
    obs = env.reset()
    seated, depth = [], []
    for t in range(80):
        a = torch.zeros(env.n, env.act_dim, device=obs.device)
        obs, _, _, succ, depth_mm = env.step(a)
        if t >= 40:
            seated.append(succ.mean().item()); depth.append(depth_mm.mean().item())
    sr = sum(seated)/len(seated); md = sum(depth)/len(depth)
    print(f"stage {stage} (re_scale={env.re_scale:.2f}): BASE-controller SR {sr*100:5.1f}% | depth {md:5.1f}mm")
