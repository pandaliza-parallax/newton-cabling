"""Spin-up scaling benchmark: how many connector envs fit/step on this GPU.

Builds N RJ45 connector worlds (socket+plug+latch, no arm) in ONE Newton model
using the repo's own batching blocks, finalizes, and steps a few frames with the
batched force-spring. Reports build time, GPU memory, and per-step time.

Usage:  python bench_scale.py 1000 [--steps 20]
This is a throwaway probe, not part of the package.
"""

import argparse
import subprocess
import time

import newton
import numpy as np
import warp as wp

newton.use_coord_layout_targets = True
from newton.solvers import SolverVBD  # noqa: E402

from newton_cabling.connector import rj45_connector  # noqa: E402
from newton_cabling.sim.safe_vbd import finalize_for_vbd, new_vbd_builder  # noqa: E402
from newton_cabling.sim.scene import add_connector_rig, load_connector_meshes  # noqa: E402

Z_LIFT = np.array([0.0, 0.0, 0.35])
SPACING = 0.4  # metres between worlds; far enough that worlds never touch


def gpu_mem_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        return float(out.decode().splitlines()[0])
    except Exception:
        return float("nan")


@wp.kernel
def _hold(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    plug_idx: wp.array(dtype=int),
    latch_idx: wp.array(dtype=int),
    targets: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
):
    w = wp.tid()
    p = plug_idx[w]
    la = latch_idx[w]
    wp.atomic_add(body_f, p, wp.spatial_vector(-gravity * body_mass[p], wp.vec3(0.0)))
    wp.atomic_add(body_f, la, wp.spatial_vector(-gravity * body_mass[la], wp.vec3(0.0)))
    pos = wp.transform_get_translation(body_q[p])
    vel = wp.spatial_top(body_qd[p])
    f = (10.0 + body_mass[p]) * (50.0 * (targets[w] - pos) - 10.0 * vel)
    wp.atomic_add(body_f, p, wp.spatial_vector(f, wp.vec3(0.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    n = args.n

    mem0 = gpu_mem_mb()
    spec = rj45_connector()
    meshes = load_connector_meshes(spec)  # shared across all worlds

    socket_base = meshes.socket.base_position
    plug_base = meshes.plug.base_position
    latch_base = meshes.latch.base_position
    plug_y = np.array([0.0, -0.025, 0.0])
    start = np.array([0.0, -0.050, 0.0])

    t0 = time.perf_counter()
    builder = new_vbd_builder(gravity=-9.81)
    builder.rigid_gap = 0.005
    rigs = []
    # grid layout so the row of N worlds stays compact
    cols = max(1, int(np.ceil(np.sqrt(n))))
    for i in range(n):
        shift = np.array([(i % cols) * SPACING, 0.0, (i // cols) * SPACING])
        socket_pos = socket_base + Z_LIFT + shift
        plug_pos = plug_base + plug_y + Z_LIFT + shift + start
        latch_pos = latch_base + plug_y + Z_LIFT + shift + start
        rigs.append(
            add_connector_rig(
                builder, spec, meshes,
                socket_pos=socket_pos, plug_pos=plug_pos,
                latch_pos=latch_pos, plug_anchor_pos=plug_pos,
            )
        )
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = finalize_for_vbd(builder)
    wp.synchronize()
    t_final = time.perf_counter() - t0
    mem_built = gpu_mem_mb()

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()
    solver = SolverVBD(
        model, iterations=8, rigid_contact_hard=False, rigid_body_contact_buffer_size=256
    )
    device = model.device
    plug_arr = wp.array([r.plug_body for r in rigs], dtype=int, device=device)
    latch_arr = wp.array([r.latch_body for r in rigs], dtype=int, device=device)
    body_q0 = state_0.body_q.numpy()
    targets = wp.array(
        [wp.vec3(*body_q0[r.plug_body][:3]) for r in rigs], dtype=wp.vec3, device=device
    )
    grav = wp.vec3(0.0, 0.0, -9.81)
    dt = 1.0 / 60.0 / 8.0

    def substep():
        nonlocal state_0, state_1
        state_0.clear_forces()
        wp.launch(_hold, dim=n,
                  inputs=(state_0.body_q, state_0.body_qd, state_0.body_f,
                          model.body_mass, plug_arr, latch_arr, targets, grav),
                  device=device)
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    substep(); wp.synchronize()  # warm up / compile
    mem_run = gpu_mem_mb()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        substep()
    wp.synchronize()
    t_step = (time.perf_counter() - t0) / args.steps

    print(f"\n==== {n} connector worlds ====")
    print(f"  bodies={model.body_count}  shapes={model.shape_count}  "
          f"joints={model.joint_count}  contacts(rigid_count buf)")
    print(f"  build (python loop) : {t_build:7.2f} s")
    print(f"  finalize_for_vbd    : {t_final:7.2f} s")
    print(f"  per substep         : {t_step*1000:7.2f} ms  ({1.0/(t_step*8):.0f} env-frames/s aggregate)")
    print(f"  GPU mem: baseline={mem0:.0f}  after build={mem_built:.0f}  "
          f"after step={mem_run:.0f} MiB  (delta={mem_run-mem0:.0f})")


if __name__ == "__main__":
    main()
