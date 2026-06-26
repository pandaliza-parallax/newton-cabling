import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connector_env import ConnectorVecEnv
from train_ppo import ActorCritic
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from newton_cabling.sim.recording import auto_blueprint, open_rrd_recorder
DEV="cuda:0"
# left->right, easy->hard: (friction, fit, size)
FRIC=[0.3, 1.0, 1.5, 2.5]; FIT=[0.92, 1.0, 1.0, 1.04]; SIZE=[0.85, 1.0, 1.2, 1.2]
LABEL=["slick/loose/small","nominal","big/medium-fric","sticky/tight/big"]
N=4; CKPT="rl/runs/cad_dr_all3_2k/best_model.pt"
env=ConnectorVecEnv(N, random_easy=True, seed=0, asset="cad_rj45", residual_scale=1.0,
                    plug_scales=FIT, connector_scales=SIZE)
env.set_stage(5)
s2w=env.shape_to_world.numpy(); mu=env.model.shape_material_mu.numpy().copy()
for i in range(N):
    for s in np.where(s2w==i)[0]: mu[s]=FRIC[i]
env.model.shape_material_mu.assign(mu)
# same RELATIVE start: lateral offset scaled by size, seat depth per-env, same 15deg tilt
sz=torch.tensor(SIZE,device=DEV)
latx=(0.003*sz).contiguous(); latz=(-0.003*sz).contiguous()
insd=(env.env_seat_dy_t + 0.030).contiguous()
ax=np.array([1.,0.,1.]); ax/=np.linalg.norm(ax); h=np.deg2rad(15)/2.0
q=torch.tensor([ax[0]*np.sin(h),ax[1]*np.sin(h),ax[2]*np.sin(h),np.cos(h)],device=DEV).float()
env.set_fixed_starts(latx,latz,insd,q.unsqueeze(0).repeat(N,1).contiguous())
ac=ActorCritic(env.obs_dim,env.act_dim).to(DEV); ac.load_state_dict(torch.load(CKPT,map_location=DEV)); ac.eval()
for i,l in enumerate(LABEL): print(f"  env{i}: mu={FRIC[i]} fit={FIT[i]} size={SIZE[i]}x -> {l}")
viewer=open_rrd_recorder("cad_3axis_compare.rrd"); viewer.set_model(env.model)
obs=env.reset(); t=0.0
for f in range(240):
    with torch.no_grad(): a=ac.mean_action(obs)
    obs,_,_,succ,depth=env.step(a); viewer.begin_frame(t); viewer.log_state(env.state_0); viewer.end_frame(); t+=1/60.
    if f%40==0 or f==239: print(f"  frame {f:3d}: seated {[int(x) for x in succ.cpu().numpy()]}")
auto_blueprint("cad_3axis_compare.rbl", env.model)
print("view: uvx --from rerun-sdk rerun cad_3axis_compare.rrd cad_3axis_compare.rbl")
