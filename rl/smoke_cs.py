import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from connector_env import ConnectorVecEnv
env=ConnectorVecEnv(512,random_easy=True,seed=0,asset="cad_rj45",residual_scale=1.0,connector_scale_dr=(0.85,1.2))
print("conn scales (first 6):", [round(float(x),3) for x in env.env_conn_scale[:6]])
for stage in [0,3,5]:
    env.set_stage(stage); obs=env.reset(); s=[]
    for t in range(100):
        obs,_,_,succ,d=env.step(torch.zeros(env.n,env.act_dim,device=obs.device))
        if t>=60: s.append(succ.mean().item())
    print(f"stage {stage}: BASE seated {sum(s)/len(s)*100:.1f}%")
