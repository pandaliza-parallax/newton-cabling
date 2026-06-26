import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connector_env import ConnectorVecEnv
from train_ppo import ActorCritic
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder
DEV="cuda:0"
FRIC=[0.3,0.8,1.5,2.5]; N=len(FRIC)
CKPT="rl/runs/cad_fricDR_2k/best_model.pt"
env=ConnectorVecEnv(N, random_easy=True, seed=0, asset="cad_rj45", residual_scale=1.0)
env.set_stage(5)
# per-env friction: set each env's connector shapes to its mu
s2w=env.shape_to_world.numpy(); mu=env.model.shape_material_mu.numpy().copy()
for i in range(N):
    for s in np.where(s2w==i)[0]: mu[s]=FRIC[i]
env.model.shape_material_mu.assign(mu)
# ONE hard shared start for all envs: ~5mm diagonal offset, 15deg tilt, 30mm out
latx=torch.full((N,),0.0035,device=DEV); latz=torch.full((N,),-0.0035,device=DEV)
insd=torch.full((N,),env.seat_aim_dy+0.030,device=DEV)
ax=np.array([1.,0.,1.]); ax/=np.linalg.norm(ax); h=np.deg2rad(15)/2.0
q=torch.tensor([ax[0]*np.sin(h),ax[1]*np.sin(h),ax[2]*np.sin(h),np.cos(h)],device=DEV).float()
env.set_fixed_starts(latx,latz,insd,q.unsqueeze(0).repeat(N,1).contiguous())
ac=ActorCritic(env.obs_dim,env.act_dim).to(DEV); ac.load_state_dict(torch.load(CKPT,map_location=DEV)); ac.eval()
print(f"frictions per env (left->right): {FRIC} | same hard start (~5mm offset, 15deg tilt, 30mm out)")
viewer=open_rrd_recorder("cad_friction_compare.rrd"); viewer.set_model(env.model)
obs=env.reset(); t=0.0
for f in range(220):
    with torch.no_grad(): a=ac.mean_action(obs)
    obs,_,_,succ,depth=env.step(a); viewer.begin_frame(t); viewer.log_state(env.state_0); viewer.end_frame(); t+=1/60.
    if f%40==0 or f==219: print(f"  frame {f:3d}: seated per env {[int(x) for x in succ.cpu().numpy()]}")
auto_blueprint("cad_friction_compare.rbl", env.model)
print("view: uvx --from rerun-sdk rerun cad_friction_compare.rrd cad_friction_compare.rbl")
