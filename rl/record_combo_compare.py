import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connector_env import ConnectorVecEnv
from train_ppo import ActorCritic
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder
DEV="cuda:0"
# 3 conditions, left->right: easy / nominal / hard  (friction, plug-size)
FRIC =[0.3, 1.0, 2.5]
PLUG =[0.92, 1.0, 1.04]
LABEL=["mu0.3 plug0.92 (loose+slick)","mu1.0 plug1.0 (nominal)","mu2.5 plug1.04 (tight+sticky)"]
N=3; CKPT="rl/runs/cad_dr_fric_plug_2k/best_model.pt"
env=ConnectorVecEnv(N, random_easy=True, seed=0, asset="cad_rj45", residual_scale=1.0, plug_scales=PLUG)
env.set_stage(5)
# per-env friction
s2w=env.shape_to_world.numpy(); mu=env.model.shape_material_mu.numpy().copy()
for i in range(N):
    for s in np.where(s2w==i)[0]: mu[s]=FRIC[i]
env.model.shape_material_mu.assign(mu)
# ONE shared hard start
latx=torch.full((N,),0.0035,device=DEV); latz=torch.full((N,),-0.0035,device=DEV)
insd=torch.full((N,),env.seat_aim_dy+0.030,device=DEV)
ax=np.array([1.,0.,1.]); ax/=np.linalg.norm(ax); h=np.deg2rad(15)/2.0
q=torch.tensor([ax[0]*np.sin(h),ax[1]*np.sin(h),ax[2]*np.sin(h),np.cos(h)],device=DEV).float()
env.set_fixed_starts(latx,latz,insd,q.unsqueeze(0).repeat(N,1).contiguous())
ac=ActorCritic(env.obs_dim,env.act_dim).to(DEV); ac.load_state_dict(torch.load(CKPT,map_location=DEV)); ac.eval()
for i,l in enumerate(LABEL): print(f"  env{i}: {l}")
viewer=open_rrd_recorder("cad_combo_compare.rrd"); viewer.set_model(env.model)
obs=env.reset(); t=0.0
for f in range(220):
    with torch.no_grad(): a=ac.mean_action(obs)
    obs,_,_,succ,depth=env.step(a); viewer.begin_frame(t); viewer.log_state(env.state_0); viewer.end_frame(); t+=1/60.
    if f%40==0 or f==219: print(f"  frame {f:3d}: seated {[int(x) for x in succ.cpu().numpy()]}")
auto_blueprint("cad_combo_compare.rbl", env.model)
print("view: uvx --from rerun-sdk rerun cad_combo_compare.rrd cad_combo_compare.rbl")
