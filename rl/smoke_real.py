import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from connector_env import ConnectorVecEnv
env=ConnectorVecEnv(512,random_easy=True,seed=0,asset="cad_rj45_real",residual_scale=1.0)
print("loaded cad_rj45_real OK; obs_dim", env.obs_dim)
for stage in [0,3,5]:
    env.set_stage(stage); obs=env.reset(); s=[]
    for t in range(100):
        obs,_,_,succ,d=env.step(torch.zeros(env.n,env.act_dim,device=obs.device))
        if t>=60: s.append(succ.mean().item())
    print(f"stage {stage}: BASE seated {sum(s)/len(s)*100:.1f}%")
