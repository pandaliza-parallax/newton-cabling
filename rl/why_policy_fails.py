import numpy as np, torch
from connector_env import ConnectorVecEnv
from train_ppo import ActorCritic

def qmul(a,b):
    ax,ay,az,aw=a.T;bx,by,bz,bw=b.T
    return np.stack([aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw,aw*bw-ax*bx-ay*by-az*bz],1)
def qconj(a): return a*np.array([-1.,-1.,-1.,1.])
def qang(a,b): d=np.abs(np.sum(a*b,1)).clip(-1,1); return np.degrees(2*np.arccos(d))

CKPT="rl/runs/cad_res1p0_mu05_to5k/best_model.pt"
env=ConnectorVecEnv(2000,random_easy=True,seed=0,asset="cad_rj45",friction=0.5,residual_scale=1.0)
ac=ActorCritic(env.obs_dim,env.act_dim).cuda(); ac.load_state_dict(torch.load(CKPT,map_location="cuda")); ac.eval()
pi=env.plug_idx.numpy(); seat=env.seated.numpy(); seatq=env.plug_rot.numpy()
print(f"policy={CKPT}  thresholds depth<=5mm lat<=3mm ang<=3deg, held=20 steps\n")
for stage in [3,4,5]:
    env.set_stage(stage); obs=env.reset()
    held=np.zeros(env.n); 
    for t in range(120):
        with torch.no_grad(): a=ac.mean_action(obs)
        obs,_,_,succ,depth=env.step(a)
        if t>=60: held+=succ.cpu().numpy()
    held_sr=(held>0).mean()  # ever-seated in 2nd half ~ eval
    bq=env.state_0.body_q.numpy()[pi]; pos=bq[:,:3]; quat=bq[:,3:7]
    lat=np.sqrt((seat[:,0]-pos[:,0])**2+(seat[:,2]-pos[:,2])**2); gap=seat[:,1]-pos[:,1]; ang=qang(quat,seatq)
    d=gap<=.005; l=lat<=.003; an=ang<=3.0; allok=d&l&an
    print(f"=== stage {stage} (re_scale {env.re_scale:.2f}) ===")
    print(f"  held-SR~{held_sr*100:4.1f}% | final-instant: depth_ok {d.mean()*100:4.0f}% lat_ok {l.mean()*100:4.0f}% ang_ok {an.mean()*100:4.0f}% ALL {allok.mean()*100:4.0f}%")
    f=~allok
    print(f"  among the {f.sum()} not-seated-at-end: miss DEPTH {(~d&f).sum()/max(f.sum(),1)*100:3.0f}% | miss LAT {(~l&f).sum()/max(f.sum(),1)*100:3.0f}% | miss ANG {(~an&f).sum()/max(f.sum(),1)*100:3.0f}% | median ang {np.median(ang[f]):.1f}deg gap {np.median(gap[f])*1000:.1f}mm\n")
