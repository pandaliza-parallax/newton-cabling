import numpy as np, torch
from connector_env import ConnectorVecEnv

env = ConnectorVecEnv(2000, random_easy=True, asset="cad_rj45", seed=0)
plug_idx = env.plug_idx.numpy()
seat = env.seated.numpy()        # (n,3) seated position ref
seat_q = env.plug_rot.numpy()    # (n,4) aligned rotation ref

def quat_angle(q1, q2):
    d = np.abs(np.sum(q1 * q2, axis=1)).clip(-1, 1)
    return np.degrees(2 * np.arccos(d))

print("base controller (zero residual). thresholds: depth<=5mm, lateral<=3mm, angle<=3deg\n")
for stage in [2, 4, 5]:
    env.set_stage(stage)
    obs = env.reset()
    # capture START misalignment (first obs) for context
    start_lat = np.sqrt((obs[:,0].cpu().numpy()/50)**2 + (obs[:,2].cpu().numpy()/50)**2)
    start_ang = np.sqrt(((obs[:,6:9].cpu().numpy()/3)**2).sum(1))
    for t in range(80):
        a = torch.zeros(env.n, env.act_dim, device=obs.device)
        obs, _, _, succ, depth_mm = env.step(a)
    bq = env.state_0.body_q.numpy()[plug_idx]
    pos, quat = bq[:, :3], bq[:, 3:7]
    lat = np.sqrt((seat[:,0]-pos[:,0])**2 + (seat[:,2]-pos[:,2])**2)
    gap = seat[:,1] - pos[:,1]
    ang = quat_angle(quat, seat_q)
    d_ok, l_ok, a_ok = gap<=0.005, lat<=0.003, ang<=3.0
    allok = d_ok & l_ok & a_ok
    print(f"=== stage {stage} (re_scale {env.re_scale:.2f}) | start: lat~{np.median(start_lat)*1000:.1f}mm ang~{np.degrees(np.median(start_ang)):.1f}deg ===")
    print(f"  depth_ok(<=5mm) {d_ok.mean()*100:5.1f}% | lat_ok(<=3mm) {l_ok.mean()*100:5.1f}% | ang_ok(<=3deg) {a_ok.mean()*100:5.1f}% | ALL {allok.mean()*100:5.1f}%")
    f = ~allok
    print(f"  failed (n={f.sum()}): median gap {np.median(gap[f])*1000:5.1f}mm | lat {np.median(lat[f])*1000:.1f}mm | ang {np.median(ang[f]):.1f}deg")
    # of the failures, how many would pass depth alone vs lateral alone vs angle alone
    print(f"  among failures: miss DEPTH {(~d_ok).mean()*100:.0f}% | miss LAT {(~l_ok).mean()*100:.0f}% | miss ANG {(~a_ok).mean()*100:.0f}%\n")
