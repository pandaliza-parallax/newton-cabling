import numpy as np, torch
from connector_env import ConnectorVecEnv

def qmul(a,b):
    ax,ay,az,aw=a.T; bx,by,bz,bw=b.T
    return np.stack([aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz],1)
def qconj(a): return a*np.array([-1.,-1.,-1.,1.])
def qang(a):  return np.degrees(2*np.arccos(np.clip(np.abs(a[:,3]),-1,1)))

env = ConnectorVecEnv(64, random_easy=True, asset="cad_rj45", seed=0)
pi, li = env.plug_idx.numpy(), env.latch_idx.numpy()
mouth_y = env.seated.numpy()[:,1] - 0.012     # jack mouth world-y (seat is +12mm in)
mouth_z = None
env.set_stage(0); obs = env.reset()
bq = env.state_0.body_q.numpy()
rel0 = qmul(qconj(bq[pi][:,3:7]), bq[li][:,3:7])
print("LEDGE is at jack-frame y in [6.0, 7.6]mm, z just above the bore ceiling (+4.4mm).")
print("step | plug_y_jack | latch_y_jack | latch_z_jack | defl_deg")
for t in range(140):
    obs,_,_,succ,depth = env.step(torch.zeros(env.n, env.act_dim, device=obs.device))
    bq = env.state_0.body_q.numpy()
    if t % 14 == 0 or t == 139:
        py = np.median(bq[pi][:,1]-mouth_y)*1000
        ly = np.median(bq[li][:,1]-mouth_y)*1000
        lz = np.median(bq[li][:,2]-(env.seated.numpy()[:,2]))*1000   # latch z rel seat-z
        rel = qmul(qconj(bq[pi][:,3:7]), bq[li][:,3:7])
        defl = np.median(qang(qmul(qconj(rel0), rel)))
        print(f" {t:3d} | {py:8.1f} | {ly:8.1f} | {lz:8.1f} | {defl:6.2f}")
