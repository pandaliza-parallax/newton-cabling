import argparse, numpy as np, torch
from connector_env import ConnectorVecEnv
from train_ppo import ActorCritic
ap=argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--friction", type=float, default=0.5)
ap.add_argument("--residual-scale", type=float, default=1.0)
ap.add_argument("--stage", type=int, default=5)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--envs", type=int, default=2000)
a=ap.parse_args()
env=ConnectorVecEnv(a.envs,random_easy=True,seed=0,asset="cad_rj45",friction=a.friction,residual_scale=a.residual_scale)
ac=ActorCritic(env.obs_dim,env.act_dim).cuda(); ac.load_state_dict(torch.load(a.checkpoint,map_location="cuda")); ac.eval()
env.set_stage(a.stage); obs=env.reset()
occ=[]; ever=np.zeros(env.n)
for t in range(a.steps):
    with torch.no_grad(): act=ac.mean_action(obs)
    obs,_,_,succ,depth=env.step(act); s=succ.cpu().numpy(); occ.append(s.mean()); ever=np.maximum(ever,s)
occ=np.array(occ)
print(f"RESULT | mu={a.friction} stage{a.stage} | 80-step-eval(t40-80)={occ[40:80].mean()*100:.1f}% | "
      f"steady(t120-199)={occ[120:200].mean()*100:.1f}% | occ@199={occ[199]*100:.0f}% | ever-seated={(ever>0).mean()*100:.1f}%")
