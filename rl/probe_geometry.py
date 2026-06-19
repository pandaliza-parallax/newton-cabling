"""Measure connector insertion geometry: where does the plug physically seat, and
where does socket contact begin? Drives a few plugs fully in and logs plug_y +
contact force vs the dy=0 plug origin, so the RL env's seat reference and
curriculum bands can be set from real numbers instead of guessed."""

import os
import sys

import newton
import numpy as np
import warp as wp

newton.use_coord_layout_targets = True
from newton.solvers import SolverVBD  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from newton_cabling.connector import rj45_connector  # noqa: E402
from newton_cabling.sim.safe_vbd import finalize_for_vbd, new_vbd_builder  # noqa: E402
from newton_cabling.sim.scene import add_connector_rig, load_connector_meshes  # noqa: E402

Z_LIFT = np.array([0.0, 0.0, 0.35])
N = 4


@wp.kernel
def drive(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
          body_f: wp.array(dtype=wp.spatial_vector), body_mass: wp.array(dtype=float),
          plug_idx: wp.array(dtype=int), latch_idx: wp.array(dtype=int),
          target_y: float, seat_xz: wp.array(dtype=wp.vec3), grav: wp.vec3):
    w = wp.tid()
    p = plug_idx[w]
    la = latch_idx[w]
    wp.atomic_add(body_f, p, wp.spatial_vector(-grav * body_mass[p], wp.vec3(0.0)))
    wp.atomic_add(body_f, la, wp.spatial_vector(-grav * body_mass[la], wp.vec3(0.0)))
    pos = wp.transform_get_translation(body_q[p])
    s = seat_xz[w]
    tgt = wp.vec3(s[0], target_y, s[2])
    vel = wp.spatial_top(body_qd[p])
    f = (10.0 + body_mass[p]) * (50.0 * (tgt - pos) - 10.0 * vel)
    wp.atomic_add(body_f, p, wp.spatial_vector(f, wp.vec3(0.0)))


def main():
    spec = rj45_connector()
    meshes = load_connector_meshes(spec)
    sb, pb, lb = meshes.socket.base_position, meshes.plug.base_position, meshes.latch.base_position
    plug_y = np.array([0.0, -0.025, 0.0])
    builder = new_vbd_builder(gravity=-9.81)
    builder.rigid_gap = 0.005
    rigs, seat = [], []
    for i in range(N):
        shift = np.array([i * 0.4, 0.0, 0.0])
        sp = pb + plug_y + Z_LIFT + shift                 # plug origin, dy=0
        start = sp + np.array([0.0, -0.060, 0.0])         # start 60mm out
        rigs.append(add_connector_rig(builder, spec, meshes, socket_pos=sb + Z_LIFT + shift,
                                      plug_pos=start, latch_pos=lb + plug_y + Z_LIFT + shift
                                      + np.array([0.0, -0.060, 0.0]), plug_anchor_pos=start))
        seat.append(sp)
    model = finalize_for_vbd(builder)
    dev = model.device
    s0, s1 = model.state(), model.state()
    control, contacts = model.control(), model.contacts()
    solver = SolverVBD(model, iterations=8, rigid_contact_hard=False,
                       rigid_body_contact_buffer_size=64)
    plug_idx = wp.array([r.plug_body for r in rigs], dtype=int, device=dev)
    latch_idx = wp.array([r.latch_body for r in rigs], dtype=int, device=dev)
    seat_xz = wp.array([wp.vec3(*s) for s in seat], dtype=wp.vec3, device=dev)
    grav = wp.vec3(0.0, 0.0, -9.81)
    dt = 1.0 / 60.0 / 4.0
    seat_y0 = float(seat[0][1])           # dy = 0 origin y
    plug_ids = [r.plug_body for r in rigs]

    print(f"dy=0 plug-origin y = {seat_y0:+.4f}  (start 60mm out at {seat_y0-0.060:+.4f})")
    print("ramping the commanded target DEEP (to dy=+60mm past origin) to find the socket")
    print(f"{'dy_cmd_mm':>10}{'plug_dy_mm':>11}{'contactN':>10}")
    NSTEPS = 1200
    for step in range(NSTEPS):
        dy_cmd = -0.060 + (0.120) * (step / NSTEPS)   # ramp target dy from -60mm to +60mm
        target_y = seat_y0 + dy_cmd
        s0.clear_forces()
        wp.launch(drive, dim=N, inputs=(s0.body_q, s0.body_qd, s0.body_f, model.body_mass,
                                        plug_idx, latch_idx, target_y, seat_xz, grav), device=dev)
        model.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
        if step % 100 == 0 or step == NSTEPS - 1:
            q = s0.body_q.numpy()
            py = float(np.median([q[i][1] for i in plug_ids]))
            cf = float(np.sum(np.linalg.norm(contacts.rigid_contact_force.numpy()[
                :int(contacts.rigid_contact_count.numpy()[0])], axis=1))) / N
            print(f"{dy_cmd*1000:>10.1f}{(py-seat_y0)*1000:>11.1f}{cf:>10.1f}")


if __name__ == "__main__":
    main()
