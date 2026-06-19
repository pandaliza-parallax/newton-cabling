"""Test the DRIVEN d6 angular (the README's documented path) for stable orientation
control: place plugs rotated 20deg, let the d6's PD drive (target=aligned) straighten
them, sweep the drive stiffness. A free angular axis tumbles; a driven one should hold."""

import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import newton  # noqa: E402

newton.use_coord_layout_targets = True
from newton.solvers import SolverVBD  # noqa: E402

from newton_cabling.connector import rj45_connector  # noqa: E402
from newton_cabling.sim.safe_vbd import finalize_for_vbd, new_vbd_builder  # noqa: E402
from newton_cabling.sim.scene import add_connector_rig, load_connector_meshes  # noqa: E402

N = 16
Z = np.array([0.0, 0.0, 0.35])
PY = np.array([0.0, -0.025, 0.0])


@wp.kernel
def hold_pos(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
            body_f: wp.array(dtype=wp.spatial_vector), body_mass: wp.array(dtype=float),
            plug_idx: wp.array(dtype=int), target: wp.array(dtype=wp.vec3), grav: wp.vec3):
    w = wp.tid()
    p = plug_idx[w]
    wp.atomic_add(body_f, p, wp.spatial_vector(-grav * body_mass[p], wp.vec3(0.0)))
    pos = wp.transform_get_translation(body_q[p])
    vel = wp.spatial_top(body_qd[p])
    f = (10.0 + body_mass[p]) * (50.0 * (target[w] - pos) - 10.0 * vel)
    wp.atomic_add(body_f, p, wp.spatial_vector(f, wp.vec3(0.0)))


def main():
    spec = rj45_connector()
    meshes = load_connector_meshes(spec)
    sb, pb, lb = meshes.socket.base_position, meshes.plug.base_position, meshes.latch.base_position
    print("driven-d6 angular orientation test (place 20deg, drive target=aligned):")
    for ke in (0.0, 0.001, 0.01, 0.1, 1.0, 10.0):
        b = new_vbd_builder(gravity=-9.81)
        b.rigid_gap = 0.005
        rigs = []
        for i in range(N):
            sh = np.array([i * 0.4, 0.0, 0.0])
            mp = pb + PY + Z + sh
            rigs.append(add_connector_rig(b, spec, meshes, socket_pos=sb + Z + sh, plug_pos=mp,
                        latch_pos=lb + PY + Z + sh, plug_anchor_pos=mp,
                        lock_rotation=False, angular_ke=ke, angular_kd=ke * 0.3))
        m = finalize_for_vbd(b)
        s0, s1 = m.state(), m.state()
        ctrl, con = m.control(), m.contacts()
        solver = SolverVBD(m, iterations=8, rigid_contact_hard=False,
                           rigid_body_contact_buffer_size=64)
        pidx = wp.array([r.plug_body for r in rigs], dtype=int, device=m.device)
        tgt = wp.array([wp.vec3(*(pb + PY + Z + np.array([i * 0.4, 0.0, 0.0]))) for i in range(N)],
                       dtype=wp.vec3, device=m.device)
        grav = wp.vec3(0.0, 0.0, -9.81)
        q = s0.body_q.numpy()
        a = np.deg2rad(20.0)
        for r in rigs:
            q[r.plug_body][3:7] = [np.sin(a / 2), 0.0, 0.0, np.cos(a / 2)]  # 20deg about x
        s0.body_q.assign(q)
        newton.eval_ik(m, s0, m.joint_q, m.joint_qd)

        def err():
            qq = s0.body_q.numpy()
            return float(np.median([np.degrees(2 * np.arccos(min(1.0, abs(float(qq[r.plug_body][6])))))
                                    for r in rigs]))

        e0 = err()
        for _ in range(300):
            s0.clear_forces()
            wp.launch(hold_pos, dim=N, inputs=(s0.body_q, s0.body_qd, s0.body_f, m.body_mass,
                      pidx, tgt, grav), device=m.device)
            m.collide(s0, con)
            solver.step(s0, s1, ctrl, con, 1.0 / 240.0)
            s0, s1 = s1, s0
        ef = err()
        tag = "<-- STABLE + ALIGNS" if ef < 8 else ("(holds, slow)" if ef < e0 + 3 else "TUMBLES")
        print(f"  angular_ke={ke:7.3f}: start {e0:.0f}deg -> final {ef:6.1f}deg  {tag}")


if __name__ == "__main__":
    main()
