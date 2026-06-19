"""De-risk the 6-DOF mechanic: can a torque spring rotate the d6 plug back to aligned
under VBD? Place plugs rotated 45deg and drive a torque spring toward identity; the
orientation error should shrink to ~0."""

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
Z_LIFT = np.array([0.0, 0.0, 0.35])


@wp.func
def quat_to_rotvec(q: wp.quat) -> wp.vec3:
    w = q[3]
    v = wp.vec3(q[0], q[1], q[2])
    if w < 0.0:
        w = -w
        v = -v
    s = wp.length(v)
    if s < 1.0e-6:
        return v * 2.0
    angle = 2.0 * wp.atan2(s, w)
    return v * (angle / s)


@wp.kernel
def torque_spring(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                  body_f: wp.array(dtype=wp.spatial_vector), body_mass: wp.array(dtype=float),
                  plug_idx: wp.array(dtype=int), target_quat: wp.array(dtype=wp.quat),
                  gravity: wp.vec3, ke_rot: float, kd_rot: float,
                  body_frame: int, torque_top: int):
    w = wp.tid()
    p = plug_idx[w]
    wp.atomic_add(body_f, p, wp.spatial_vector(-gravity * body_mass[p], wp.vec3(0.0)))
    q = wp.transform_get_rotation(body_q[p])
    if body_frame == 1:
        q_err = wp.mul(wp.quat_inverse(q), target_quat[w])  # body-frame error
    else:
        q_err = wp.mul(target_quat[w], wp.quat_inverse(q))  # world-frame error
    rotvec = quat_to_rotvec(q_err)
    omega_top = wp.spatial_top(body_qd[p])
    omega_bot = wp.spatial_bottom(body_qd[p])
    if torque_top == 1:
        tau = ke_rot * rotvec - kd_rot * omega_top
        wp.atomic_add(body_f, p, wp.spatial_vector(tau, wp.vec3(0.0)))
    else:
        tau = ke_rot * rotvec - kd_rot * omega_bot
        wp.atomic_add(body_f, p, wp.spatial_vector(wp.vec3(0.0), tau))


def main():
    import dataclasses
    spec0 = rj45_connector()
    # lower the plug density: 1e6 makes it near-infinite-inertia (uncontrollable rotation)
    spec = dataclasses.replace(spec0, contact=dataclasses.replace(spec0.contact, density=1.0e3))
    meshes = load_connector_meshes(spec)
    sb, pb, lb = meshes.socket.base_position, meshes.plug.base_position, meshes.latch.base_position
    plug_y = np.array([0.0, -0.025, 0.0])
    builder = new_vbd_builder(gravity=-9.81)
    builder.rigid_gap = 0.005
    rigs = []
    for i in range(N):
        shift = np.array([i * 0.4, 0.0, 0.0])
        mp = pb + plug_y + Z_LIFT + shift
        rigs.append(add_connector_rig(builder, spec, meshes, socket_pos=sb + Z_LIFT + shift,
                                      plug_pos=mp, latch_pos=lb + plug_y + Z_LIFT + shift,
                                      plug_anchor_pos=mp, lock_rotation=False))
    model = finalize_for_vbd(builder)
    dev = model.device
    s0, s1 = model.state(), model.state()
    control, contacts = model.control(), model.contacts()
    solver = SolverVBD(model, iterations=8, rigid_contact_hard=False,
                       rigid_body_contact_buffer_size=64)
    plug_idx = wp.array([r.plug_body for r in rigs], dtype=int, device=dev)
    plug_ids = [r.plug_body for r in rigs]

    # rotate every plug 45deg about x, then drive a torque spring toward identity (aligned)
    q0 = s0.body_q.numpy()
    ang = np.deg2rad(45.0)
    rq = np.array([np.sin(ang / 2), 0.0, 0.0, np.cos(ang / 2)])  # 45deg about x (xyzw)
    for pid in plug_ids:
        q0[pid][3:7] = rq
    s0.body_q.assign(q0)
    newton.eval_ik(model, s0, model.joint_q, model.joint_qd)
    target_quat = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)] * N, dtype=wp.quat, device=dev)
    grav = wp.vec3(0.0, 0.0, -9.81)
    dt = 1.0 / 60.0 / 4.0

    def ang_err_deg():
        q = s0.body_q.numpy()
        errs = []
        for pid in plug_ids:
            w = abs(float(q[pid][6]))  # target is identity, so err angle = 2*acos(|w|)
            errs.append(np.degrees(2 * np.arccos(min(1.0, w))))
        return float(np.median(errs))

    print(f"start orientation error: {ang_err_deg():.1f} deg (placed at 45)")
    for body_frame in (0,):
        for torque_top in (0,):
            for ke_rot in (0.005, 0.02, 0.05, 0.1, 0.3):
                s0.body_q.assign(q0)
                s0.body_qd.zero_()
                newton.eval_ik(model, s0, model.joint_q, model.joint_qd)
                for _ in range(400):
                    s0.clear_forces()
                    wp.launch(torque_spring, dim=N, inputs=(s0.body_q, s0.body_qd, s0.body_f,
                              model.body_mass, plug_idx, target_quat, grav, ke_rot, ke_rot * 0.3,
                              body_frame, torque_top), device=dev)
                    model.collide(s0, contacts)
                    solver.step(s0, s1, control, contacts, dt)
                    s0, s1 = s1, s0
                tag = f"{'body' if body_frame else 'world'}/{'top' if torque_top else 'bot'}"
                print(f"  {tag:10s} ke={ke_rot:6.0f} -> err {ang_err_deg():6.1f} deg "
                      f"{'<-- WORKS' if ang_err_deg() < 20 else ''}")


if __name__ == "__main__":
    main()
