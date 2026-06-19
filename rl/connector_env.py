"""Vectorized Newton RJ45-connector insertion env for PPO, with reverse curriculum.

N connector worlds (socket + plug + latch, no arm) in ONE Newton model, stepped
together on the GPU. The plug rides a world-anchored d6 (translation only; the
rig soft-locks rotation) driven by a force-spring whose target the policy nudges
by +-2 mm/step. The latch is passive (return spring + travel limits) and catches
on the socket lip via SDF contact during insertion.

Reverse curriculum: each episode the plug starts a distance OUT of the socket
sampled from the current stage's band. Stage 0 starts near-seated (easy to
discover seating), later stages start further out. `set_stage()` widens the band;
the trainer advances it as the rolling success rate rises.

Interface (all torch tensors on cuda:0; obs/reward computed by Warp kernels and
handed to torch zero-copy):

    env = ConnectorVecEnv(n)
    env.set_stage(0)
    obs = env.reset()                                  # (N, obs_dim)
    obs, rew, done, success, depth_mm = env.step(act)  # act (N, 3) in [-1, 1]

Reward (depth/lateral/pitch/contact + seat/latch bonus). "Seated" is defined by
the depth GAP to the fully-seated pose (start-independent, so it fires at every
curriculum stage), not by absolute travel.
"""

from __future__ import annotations

import os
import sys

import newton
import numpy as np
import warp as wp

newton.use_coord_layout_targets = True
from newton.solvers import SolverVBD  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from newton_cabling.connector import cad_rj45_connector, rj45_connector  # noqa: E402
from newton_cabling.sim.safe_vbd import finalize_for_vbd, new_vbd_builder  # noqa: E402
from newton_cabling.sim.scene import add_connector_rig, load_connector_meshes  # noqa: E402

import torch  # noqa: E402

DEV = "cuda:0"

# ── geometry / control (metres; insertion axis is world y, +y = into socket) ──
Z_LIFT = np.array([0.0, 0.0, 0.35])
SPACING = 0.4
SUBSTEPS = 4
MAX_DELTA = 0.002        # m per control step (+-2 mm, mirrors the mbrl action)
BOX = 0.10               # target stays within +-10 cm of seated pose (covers the 85mm travel)
# The plug's authored origin (plug_base + plug_y) is the socket MOUTH (dy=0); the plug
# physically seats ~25mm deeper. Aim the controller +35mm past the mouth (like the
# patch_panel demo's INSERT_CAP) so the plug presses into its ~25mm cap = truly inserted.
SEAT_AIM_DY = 0.035      # m past the mouth that the controller aims for
START_OUT_DY = -0.050    # m: plug starts this far out of the socket (below the mouth)
# Residual RL: total_action = clip(base + RESIDUAL_SCALE * policy_residual). The base
# controller drives the target toward the seated pose (the scripted insert that already
# seats aligned plugs); the policy only learns a bounded correction on top.
RESIDUAL_SCALE = 0.15    # small: the residual makes fine corrections, can't overpower the base
SPRING_KE = 50.0
SPRING_KD = 10.0   # linear damping (sweep showed KD>10 over-damps the insertion -> worse reach).
#                    The wobble at the seat was ANGULAR, fixed by ANGULAR_KD below, not linear.

# ── reverse curriculum on MISALIGNMENT: max lateral start offset (m), per stage ─
# A centered plug slides in with ZERO contact (trivial); the real contact-rich
# task is seating from a lateral offset. Stage 0 is ~aligned so seating bootstraps
# (earns the seat bonus, gives a learning signal); later stages start increasingly
# off-axis, so the plug hits the socket walls and the policy must center under
# contact to seat. Distance is NOT the difficulty axis, so it's held fixed.
# Achievable envelope for the base+residual mechanism: 100% seating up to ~5mm
# lateral offset. 8mm exceeds the socket capture range (plug rams, policy degrades),
# so the curriculum is capped at 5mm. Push RESIDUAL_SCALE / retrain to extend it.
CURRICULUM = [0.0005, 0.0015, 0.003, 0.005, 0.008, 0.011, 0.015]
INSERT_DIST = SEAT_AIM_DY - START_OUT_DY  # start->seated travel along y (0.085 m)

# random_easy_subset: a fixed, reasonable start distribution scaled to THIS small RJ45
# connector (~16mm socket). mbrl's random_easy (±30mm, <=90deg) was for a ~48mm pocket.
# Translation only — rotation is excluded because a d6-angular plug is VBD-unstable.
RE_LAT = 0.008            # m: lateral start offset ~ U[-8, +8] mm (x and z independently)
RE_APPROACH_MIN = 0.020   # m: plug starts 20mm out of the socket mouth ...
RE_APPROACH_MAX = 0.050   # ... up to 50mm out
RE_ROT = np.deg2rad(15.0) # rad: start orientation error, axis ~ S^2, angle ~ U[0, 15deg]
# Curriculum for the subset: scale lateral+rotation from ~aligned up to the full ranges.
# (The full distribution from iter 0 gave PPO no learning signal; ramp it instead.)
RE_CURRICULUM = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
# d6 angular DRIVE (PD toward control.joint_target_q=aligned). A FREE angular axis is
# VBD-unstable (tumbles); a driven one (ke>=0.1) is stable and self-aligns the plug.
ANGULAR_KE = 10.0   # strong+stable (test: 20deg -> 0deg); aligns the plug fast during approach
ANGULAR_KD = 6.0    # raised 2->6: damp the tilt oscillation at the seat (held was tilt-limited)
ROT_CMD_RANGE = np.deg2rad(20.0)  # 6-DOF action: max plug orientation the policy can command
ROT_W = 0.4         # orientation error weight in the reward distance (rad -> m-equivalent)
# hold-to-success: success requires staying seated for HOLD_STEPS consecutive steps, and a
# velocity penalty while seated damps the wobble that drifts the plug back out.
HOLD_STEPS = 20
W_VEL = 50.0        # penalty per (m/s + rad/s) while seated. Raised 5->50: at W_VEL=5 a 0.28
#                    wobble cost only 1.4 vs the +20 seat bonus, so the policy kept commanding
#                    orientation (re-injecting wobble); 50 makes a wobble actually erase the bonus.
OBS_NOISE_POS = 0.0005    # m — Gaussian noise on the observed plug-vs-socket offset
                          # (à la mbrl FoundationPose jitter) so the policy must feel
                          # for the hole via contact, not perfectly center from obs

# ── NON-EXPLOITABLE reward (potential-based shaping, gamma=1) ──────────────────
# The fix for the do-little optimum: the ONLY way to earn reward is to get closer
# to the seated pose. Let d = weighted distance(plug, seated). Per step:
#     r = W_PROG * (d_prev - d)          # progress toward seat; telescopes (non-farmable)
#         + R_SUCCESS    if seated        # the prize
# Doing nothing -> d_prev - d = 0 -> r = 0 (no do-little optimum). Moving away -> r < 0.
# No per-step offset/contact/action penalties (those rewarded staying put). Lateral is
# weighted up in d (LAT_WEIGHT) so centering counts as progress. The argmax of this
# reward IS reaching the seated pose, so PPO maximizing it = solving the task.
W_PROG = 100.0           # * metres of distance reduced toward seat (full insert ~ +2.5)
R_SUCCESS = 20.0         # terminal seat bonus (the prize; dominates over shaping)
LAT_WEIGHT = 2.0         # lateral error weighted up in the distance (centering matters more)

# ── success thresholds (mbrl) ─────────────────────────────────────────────────
# seated = plug within SEAT_DEPTH_TOL of the +35mm aim. The plug caps ~10mm short of
# the aim (at its ~+25mm physical seat), so the tol must exceed that gap.
SEAT_DEPTH_TOL = 0.015   # m
SEAT_OFFSET = 0.003      # m — loosened 2->3mm (physically reasonable for the connector clearance)
SEAT_ANGLE = np.deg2rad(3.0)  # tilt kept tight at 3deg (5deg is too sloppy for a connector)
OOB_LATERAL = 0.06       # m — mbrl MAX_LATERAL

OBS_DIM = 9              # pos_err(3) lin_vel(3) orient_err_rotvec(3)
ACT_DIM = 3              # translation env: 3-DOF position. random_easy: 6 (pos + orientation).

# ── per-asset geometry profiles ───────────────────────────────────────────────
# The RL kernels and control are shared; only these geometry numbers depend on the
# specific connector mesh. "rj45" reproduces the original bundled-asset constants
# EXACTLY so the proven path is unchanged. "cad_rj45" is the real-CAD asset (McMaster
# 9953K216 plug / 1422N17 jack) whose merged USD is built by
# tools/cad_assets/build_cad_rj45_usd.py — plug leading face is authored at the mouth
# (plug_y=0) and the cavity floor is +12 mm past the mouth.
ASSET_PROFILES = {
    "rj45": dict(
        spec=rj45_connector,
        plug_y=(0.0, -0.025, 0.0),
        seat_aim_dy=SEAT_AIM_DY, box=BOX, rigid_gap=0.005,
        re_lat=RE_LAT, re_approach_min=RE_APPROACH_MIN, re_approach_max=RE_APPROACH_MAX,
        re_rot=RE_ROT,
        seat_depth_tol=SEAT_DEPTH_TOL, seat_offset=SEAT_OFFSET, seat_angle=SEAT_ANGLE,
    ),
    "cad_rj45": dict(
        spec=cad_rj45_connector,
        plug_y=(0.0, 0.0, 0.0),
        # sub-mm rigid_gap to match the real (tight) RJ45 clearance — 5mm would hold
        # the plug off the cavity walls and prevent any insertion.
        seat_aim_dy=0.012, box=0.05, rigid_gap=0.00005,
        re_lat=0.004, re_approach_min=0.010, re_approach_max=0.030,
        re_rot=RE_ROT,
        seat_depth_tol=0.005, seat_offset=0.003, seat_angle=SEAT_ANGLE,
    ),
}


@wp.func
def _quat_angle(q: wp.quat) -> float:
    return 2.0 * wp.acos(wp.clamp(wp.abs(q[3]), -1.0, 1.0))


@wp.func
def _quat_rotvec(q: wp.quat) -> wp.vec3:
    # axis*angle of a quaternion (shortest rotation), for orientation error in obs
    w = q[3]
    v = wp.vec3(q[0], q[1], q[2])
    if w < 0.0:
        w = -w
        v = -v
    s = wp.length(v)
    if s < 1.0e-6:
        return v * 2.0
    return v * (2.0 * wp.atan2(s, w) / s)


@wp.kernel
def integrate_target(action: wp.array2d(dtype=float), seated: wp.array(dtype=wp.vec3),
                     max_d: float, box: float, target: wp.array(dtype=wp.vec3)):
    w = wp.tid()
    t = target[w]
    s = seated[w]
    nx = wp.clamp(t[0] + wp.clamp(action[w, 0], -1.0, 1.0) * max_d, s[0] - box, s[0] + box)
    ny = wp.clamp(t[1] + wp.clamp(action[w, 1], -1.0, 1.0) * max_d, s[1] - box, s[1] + box)
    nz = wp.clamp(t[2] + wp.clamp(action[w, 2], -1.0, 1.0) * max_d, s[2] - box, s[2] + box)
    target[w] = wp.vec3(nx, ny, nz)


@wp.kernel
def apply_control(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                  body_f: wp.array(dtype=wp.spatial_vector), body_mass: wp.array(dtype=float),
                  plug_idx: wp.array(dtype=int), latch_idx: wp.array(dtype=int),
                  target: wp.array(dtype=wp.vec3), gravity: wp.vec3, ke: float, kd: float):
    w = wp.tid()
    p = plug_idx[w]
    la = latch_idx[w]
    wp.atomic_add(body_f, p, wp.spatial_vector(-gravity * body_mass[p], wp.vec3(0.0)))
    wp.atomic_add(body_f, la, wp.spatial_vector(-gravity * body_mass[la], wp.vec3(0.0)))
    pos = wp.transform_get_translation(body_q[p])
    vel = wp.spatial_top(body_qd[p])
    f = (10.0 + body_mass[p]) * (ke * (target[w] - pos) - kd * vel)
    wp.atomic_add(body_f, p, wp.spatial_vector(f, wp.vec3(0.0)))


@wp.kernel
def reduce_contact_force(count: wp.array(dtype=int), force: wp.array(dtype=wp.vec3),
                         shape0: wp.array(dtype=int), shape1: wp.array(dtype=int),
                         shape_to_world: wp.array(dtype=int), out: wp.array(dtype=float)):
    i = wp.tid()
    if i >= count[0]:
        return
    w0 = shape_to_world[shape0[i]]
    w1 = shape_to_world[shape1[i]]
    w = wp.max(w0, w1)
    if w >= 0:
        wp.atomic_add(out, w, wp.length(force[i]))


@wp.kernel
def write_obs(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
              plug_idx: wp.array(dtype=int), seated: wp.array(dtype=wp.vec3),
              seated_rot: wp.array(dtype=wp.quat), obs: wp.array2d(dtype=float)):
    w = wp.tid()
    p = plug_idx[w]
    pos = wp.transform_get_translation(body_q[p])
    vel = wp.spatial_top(body_qd[p])
    e = pos - seated[w]
    # orientation error: plug current vs seated (aligned) orientation, as a rotvec
    q_err = wp.mul(wp.quat_inverse(seated_rot[w]), wp.transform_get_rotation(body_q[p]))
    rv = _quat_rotvec(q_err)
    obs[w, 0] = e[0] * 50.0
    obs[w, 1] = e[1] * 50.0
    obs[w, 2] = e[2] * 50.0
    obs[w, 3] = vel[0]
    obs[w, 4] = vel[1]
    obs[w, 5] = vel[2]
    obs[w, 6] = rv[0] * 3.0
    obs[w, 7] = rv[1] * 3.0
    obs[w, 8] = rv[2] * 3.0


@wp.kernel
def write_reward(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                 plug_idx: wp.array(dtype=int),
                 seated: wp.array(dtype=wp.vec3), seated_rot: wp.array(dtype=wp.quat),
                 start_y: wp.array(dtype=float),
                 prev_dist: wp.array(dtype=float), hold_count: wp.array(dtype=int),
                 hold_steps: int, max_steps: int, w_prog: float, r_success: float,
                 lat_weight: float, rot_w: float, w_vel: float, seat_depth_tol: float,
                 seat_offset: float, seat_angle: float, oob_lateral: float,
                 rew: wp.array(dtype=float), done: wp.array(dtype=float),
                 success: wp.array(dtype=float), new_dist: wp.array(dtype=float),
                 depth_mm: wp.array(dtype=float)):
    w = wp.tid()
    p = plug_idx[w]
    pos = wp.transform_get_translation(body_q[p])
    e = pos - seated[w]
    offset = wp.sqrt(e[0] * e[0] + e[2] * e[2])
    ang_err = _quat_angle(wp.mul(wp.quat_inverse(seated_rot[w]),
                                 wp.transform_get_rotation(body_q[p])))
    d = wp.sqrt(lat_weight * lat_weight * (e[0] * e[0] + e[2] * e[2]) + e[1] * e[1]) \
        + rot_w * ang_err
    r = w_prog * (prev_dist[w] - d)

    seat_gap = seated[w][1] - pos[1]  # >0 = still short of full seat
    seated_inst = float(0.0)
    if seat_gap <= seat_depth_tol and offset <= seat_offset and ang_err <= seat_angle:
        seated_inst = 1.0
        # HOLD-TO-SUCCESS: reward being seated AND damp velocity so it settles instead of
        # drifting back out (the seat-then-drift failure mode); ramps with the hold streak.
        speed = wp.length(wp.spatial_top(body_qd[p])) + wp.length(wp.spatial_bottom(body_qd[p]))
        r = r + r_success - w_vel * speed

    # consecutive-seated counter -> success only once held for hold_steps
    hc = int(0)
    if seated_inst > 0.5:
        hc = hold_count[w] + 1
    hold_count[w] = hc
    held = float(0.0)
    if hc >= hold_steps:
        held = 1.0

    # fixed-horizon episodes (reset is per-rollout) -> no mid-rollout termination.
    dd = float(0.0)
    if (pos[0] != pos[0] or pos[1] != pos[1] or pos[2] != pos[2]
            or wp.abs(pos[0]) > 50.0 or wp.abs(pos[1]) > 50.0 or wp.abs(pos[2]) > 50.0):
        r = -1.0
        held = 0.0
        d = 0.0

    rew[w] = r
    done[w] = dd
    success[w] = held  # metric/eval = SUSTAINED seat (held hold_steps), not a transient touch
    new_dist[w] = d
    depth_mm[w] = (pos[1] - start_y[w]) * 1000.0


@wp.kernel
def body_mask_from_done(done: wp.array(dtype=float), plug_idx: wp.array(dtype=int),
                        latch_idx: wp.array(dtype=int), mask: wp.array(dtype=wp.bool)):
    w = wp.tid()
    if done[w] > 0.5:
        mask[plug_idx[w]] = True
        mask[latch_idx[w]] = True


@wp.kernel
def set_angular_target(rot_cmd: wp.array2d(dtype=float), ang_coords: wp.array2d(dtype=int),
                       rng: float, joint_target_q: wp.array(dtype=float)):
    # 6-DOF action: command the plug's orientation via the d6 angular DRIVE targets.
    w = wp.tid()
    for j in range(3):
        joint_target_q[ang_coords[w, j]] = rng * wp.clamp(rot_cmd[w, j], -1.0, 1.0)


@wp.kernel
def place_envs(mask: wp.array(dtype=float), lat_x: wp.array(dtype=float),
               lat_z: wp.array(dtype=float), insert_dist: wp.array(dtype=float),
               rand_rot: wp.array(dtype=wp.quat), lat_weight: float, rot_w: float,
               plug_idx: wp.array(dtype=int), latch_idx: wp.array(dtype=int),
               seated_plug: wp.array(dtype=wp.vec3), seated_latch: wp.array(dtype=wp.vec3),
               plug_rot: wp.array(dtype=wp.quat), latch_rot: wp.array(dtype=wp.quat),
               body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
               target: wp.array(dtype=wp.vec3), start_y: wp.array(dtype=float),
               prev_dist: wp.array(dtype=float), step_count: wp.array(dtype=int)):
    w = wp.tid()
    if mask[w] > 0.5:
        p = plug_idx[w]
        la = latch_idx[w]
        d = insert_dist[w]
        # start: d out along -y, with a lateral (x,z) offset (the misalignment)
        lx = lat_x[w]
        lz = lat_z[w]
        off = wp.vec3(lx, -d, lz)
        pp = seated_plug[w] + off
        # rigid start rotation of the connector about the plug origin (identity if no rot)
        rr = rand_rot[w]
        lp = pp + wp.quat_rotate(rr, seated_latch[w] - seated_plug[w])
        body_q[p] = wp.transform(pp, wp.mul(rr, plug_rot[w]))
        body_q[la] = wp.transform(lp, wp.mul(rr, latch_rot[w]))
        zero = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
        body_qd[p] = zero
        body_qd[la] = zero
        target[w] = pp
        start_y[w] = pp[1]
        # initial weighted distance to seated (matches write_reward's d: pos + rot_w*ang_err)
        prev_dist[w] = wp.sqrt(lat_weight * lat_weight * (lx * lx + lz * lz) + d * d) \
            + rot_w * _quat_angle(rr)
        step_count[w] = 0
    else:
        step_count[w] = step_count[w] + 1


class ConnectorVecEnv:
    def __init__(self, n: int, *, contact_buffer: int = 64, seed: int = 0, random_easy: bool = False,
                 asset: str = "rj45"):
        self.n = n
        self.obs_dim = OBS_DIM
        self.act_dim = 6 if random_easy else ACT_DIM  # 6-DOF (pos+orientation) for the subset
        self.random_easy = random_easy  # set before the rig build (controls 6-DOF rig)
        self.num_stages = len(RE_CURRICULUM if random_easy else CURRICULUM)
        if asset not in ASSET_PROFILES:
            raise ValueError(f"unknown asset {asset!r}; choose from {list(ASSET_PROFILES)}")
        self.asset = asset
        prof = ASSET_PROFILES[asset]
        # geometry constants for THIS asset (rj45 profile == the original constants)
        self.seat_aim_dy = prof["seat_aim_dy"]
        self.box = prof["box"]
        self.seat_depth_tol = prof["seat_depth_tol"]
        self.seat_offset = prof["seat_offset"]
        self.seat_angle = prof["seat_angle"]
        spec = prof["spec"]()
        meshes = load_connector_meshes(spec)
        sb, pb, lb = (meshes.socket.base_position, meshes.plug.base_position,
                      meshes.latch.base_position)
        plug_y = np.array(prof["plug_y"])
        build_start = np.array([0.0, -0.03, 0.0])  # any start; reset() repositions

        builder = new_vbd_builder(gravity=-9.81)
        builder.rigid_gap = prof["rigid_gap"]
        rigs, seat_plug, seat_latch, shape_world = [], [], [], {}
        cols = max(1, int(np.ceil(np.sqrt(n))))
        for i in range(n):
            shift = np.array([(i % cols) * SPACING, 0.0, (i // cols) * SPACING])
            mouth = pb + plug_y + Z_LIFT + shift       # plug origin = socket MOUTH (dy = 0)
            sl_mouth = lb + plug_y + Z_LIFT + shift
            aim = np.array([0.0, self.seat_aim_dy, 0.0])  # the true inserted pose is deeper +y
            # random_easy adds start orientation error, so the plug needs a 6-DOF rig:
            # a DRIVEN d6 angular (auto-aligns toward control target). Else translation-only.
            rig = add_connector_rig(builder, spec, meshes, socket_pos=sb + Z_LIFT + shift,
                                    plug_pos=mouth + build_start, latch_pos=sl_mouth + build_start,
                                    plug_anchor_pos=mouth + build_start,
                                    lock_rotation=not random_easy,
                                    angular_ke=ANGULAR_KE if random_easy else 0.0,
                                    angular_kd=ANGULAR_KD if random_easy else 0.0)
            rigs.append(rig)
            seat_plug.append(mouth + aim)              # seated reference = inserted pose
            seat_latch.append(sl_mouth + aim)
            for s in rig.connector_shapes:
                shape_world[s] = i

        self.model = finalize_for_vbd(builder)
        self.device = self.model.device
        d = self.device
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self._contact_buffer = contact_buffer
        self.solver = self._new_solver()

        self.plug_idx = wp.array([r.plug_body for r in rigs], dtype=int, device=d)
        self.latch_idx = wp.array([r.latch_body for r in rigs], dtype=int, device=d)
        # 6-DOF: the d6 angular drive coords (3 angular coords follow the 3 linear) per env,
        # set via control.joint_target_q to command the plug orientation.
        if random_easy:
            jtype = self.model.joint_type.numpy()
            jchild = self.model.joint_child.numpy()
            jqs = self.model.joint_q_start.numpy()
            d6t = int(newton.JointType.D6)
            ang = []
            for r in rigs:
                d6j = next(j for j in range(self.model.joint_count)
                           if int(jtype[j]) == d6t and int(jchild[j]) == r.plug_body)
                s = int(jqs[d6j])
                ang.append([s + 3, s + 4, s + 5])  # angular coords after the 3 linear coords
            self.ang_coords = wp.array(np.array(ang, dtype=np.int32), dtype=int, device=d)
        self.seated = wp.array([wp.vec3(*s) for s in seat_plug], dtype=wp.vec3, device=d)
        self.seated_latch = wp.array([wp.vec3(*s) for s in seat_latch], dtype=wp.vec3, device=d)
        self.target = wp.zeros(n, dtype=wp.vec3, device=d)
        self.grav = wp.vec3(0.0, 0.0, -9.81)
        self.dt = 1.0 / 60.0 / SUBSTEPS

        s2w = np.full(self.model.shape_count, -1, dtype=np.int32)
        for s, wi in shape_world.items():
            s2w[s] = wi
        self.shape_to_world = wp.array(s2w, dtype=int, device=d)

        q0 = self.state_0.body_q.numpy()
        plug_rot, latch_rot, rel = [], [], []
        for r in rigs:
            pq = q0[r.plug_body][3:7]
            lq = q0[r.latch_body][3:7]
            plug_rot.append(wp.quat(*pq))
            latch_rot.append(wp.quat(*lq))
            rel.append(wp.quat(*_quat_mul(_quat_inv(pq), lq)))
        self.plug_rot = wp.array(plug_rot, dtype=wp.quat, device=d)
        self.latch_rot = wp.array(latch_rot, dtype=wp.quat, device=d)
        self.rel_rest = wp.array(rel, dtype=wp.quat, device=d)

        self.start_y = wp.zeros(n, dtype=float, device=d)
        self.prev_dist = wp.zeros(n, dtype=float, device=d)
        self.step_count = wp.zeros(n, dtype=int, device=d)
        self.obs_wp = wp.zeros((n, OBS_DIM), dtype=float, device=d)
        self.contact_wp = wp.zeros(n, dtype=float, device=d)
        self.rew_wp = wp.zeros(n, dtype=float, device=d)
        self.done_wp = wp.zeros(n, dtype=float, device=d)
        self.succ_wp = wp.zeros(n, dtype=float, device=d)
        self.dist_wp = wp.zeros(n, dtype=float, device=d)
        self.depth_wp = wp.zeros(n, dtype=float, device=d)
        self.hold_count = wp.zeros(n, dtype=int, device=d)  # consecutive-seated steps (hold-to-success)
        self._ones = wp.array(np.ones(n, np.float32), dtype=float, device=d)
        self.ik_mask = wp.zeros(self.model.body_count, dtype=wp.bool, device=d)
        self.max_steps = 200
        self._fixed = None  # set via set_fixed_starts() to replay specific start poses
        self.spring_kd = SPRING_KD  # runtime-tunable (sweep critical damping for the seat hold)

        self.residual_scale = RESIDUAL_SCALE  # set to 0.0 to measure the base controller alone
        # random_easy_subset: fixed reasonable start distribution for THIS small connector,
        # translation-only (rotation is excluded: the d6-angular plug is VBD-unstable).
        self.random_easy = random_easy
        self.re_lat = prof["re_lat"]
        self.re_approach_min = prof["re_approach_min"]
        self.re_approach_max = prof["re_approach_max"]
        self.re_rot = prof["re_rot"]
        self._gen = torch.Generator(device=DEV)
        self._gen.manual_seed(seed)
        self.set_stage(0)

    # ── curriculum (on lateral misalignment magnitude) ──────────────────────────
    def set_stage(self, stage: int):
        self.stage = int(np.clip(stage, 0, self.num_stages - 1))
        if self.random_easy:
            self.re_scale = RE_CURRICULUM[self.stage]  # scales lateral + rotation 0.1 -> 1.0
        else:
            self._mag = CURRICULUM[self.stage]         # max lateral start offset (m)

    def set_fixed_starts(self, latx, latz, insert_dist, rot):
        """Replay exact start poses (each (n,) / rot (n,4)) instead of random sampling."""
        self._fixed = (latx.contiguous(), latz.contiguous(), insert_dist.contiguous(),
                       rot.contiguous())

    def _sample_start(self):
        """Per-env start: lateral (x,z) offset + insertion travel + start rotation quat."""
        g, n = self._gen, self.n
        if self._fixed is not None:                      # replay injected starts
            lx, lz, ins, rot = self._fixed
            self._latx_keep, self._latz_keep, self._ins_keep, self._rot_keep = lx, lz, ins, rot
            return (wp.from_torch(lx, dtype=wp.float32), wp.from_torch(lz, dtype=wp.float32),
                    wp.from_torch(ins, dtype=wp.float32), wp.from_torch(rot, dtype=wp.quat))
        if self.random_easy:                 # subset distribution, scaled by curriculum re_scale
            lat = (torch.rand(n, 2, generator=g, device=DEV) * 2.0 - 1.0) * (self.re_lat * self.re_scale)
            approach = self.re_approach_min + (self.re_approach_max - self.re_approach_min) \
                * torch.rand(n, generator=g, device=DEV)            # approach: full range (not hard)
            axis = torch.randn(n, 3, generator=g, device=DEV)        # rotation axis ~ S^2
            axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-9)
            half = (self.re_rot * self.re_scale * torch.rand(n, generator=g, device=DEV) / 2.0).unsqueeze(1)
            rot = torch.cat([axis * torch.sin(half), torch.cos(half)], dim=1)  # (n,4) xyzw
        else:                                # curriculum: ramp lateral, fixed 50mm approach, no rot
            lat = (torch.rand(n, 2, generator=g, device=DEV) * 2.0 - 1.0) * self._mag
            approach = torch.full((n,), 0.05, device=DEV)
            rot = torch.zeros(n, 4, device=DEV)
            rot[:, 3] = 1.0                                          # identity quat
        insert_dist = self.seat_aim_dy + approach  # start is `approach` below the mouth; seated is deeper
        self._latx_keep = lat[:, 0].contiguous()
        self._latz_keep = lat[:, 1].contiguous()
        self._ins_keep = insert_dist.contiguous()
        self._rot_keep = rot.contiguous()
        return (wp.from_torch(self._latx_keep, dtype=wp.float32),
                wp.from_torch(self._latz_keep, dtype=wp.float32),
                wp.from_torch(self._ins_keep, dtype=wp.float32),
                wp.from_torch(self._rot_keep, dtype=wp.quat))

    def _place(self, mask_wp):
        latx, latz, insd, rot = self._sample_start()
        wp.launch(place_envs, dim=self.n,
                  inputs=(mask_wp, latx, latz, insd, rot, LAT_WEIGHT, ROT_W, self.plug_idx,
                          self.latch_idx, self.seated, self.seated_latch, self.plug_rot,
                          self.latch_rot, self.state_0.body_q, self.state_0.body_qd, self.target,
                          self.start_y, self.prev_dist, self.step_count),
                  device=self.device)
        # CRITICAL: teleporting body_q desyncs the d6/revolute joint coords -> VBD ejects the
        # plug violently. Re-derive joint_q/joint_qd from the new bodies — but ONLY for the
        # reset envs (whole-model eval_ik would overwrite running envs' VBD rest reference).
        self.ik_mask.zero_()
        wp.launch(body_mask_from_done, dim=self.n,
                  inputs=(mask_wp, self.plug_idx, self.latch_idx, self.ik_mask), device=self.device)
        newton.eval_ik(self.model, self.state_0, self.model.joint_q, self.model.joint_qd,
                       mask=self.ik_mask)

    def _add_obs_noise(self, obs):
        # noise on the observed plug-vs-socket offset only (obs[:, :3], scaled by 50)
        obs[:, :3] += torch.randn(self.n, 3, generator=self._gen, device=DEV) * (OBS_NOISE_POS * 50.0)
        return obs

    def _compute_contact(self):
        self.contact_wp.zero_()
        wp.launch(reduce_contact_force, dim=self.contacts.rigid_contact_force.shape[0],
                  inputs=(self.contacts.rigid_contact_count, self.contacts.rigid_contact_force,
                          self.contacts.rigid_contact_shape0, self.contacts.rigid_contact_shape1,
                          self.shape_to_world, self.contact_wp), device=self.device)

    def _write_obs(self):
        wp.launch(write_obs, dim=self.n,
                  inputs=(self.state_0.body_q, self.state_0.body_qd, self.plug_idx,
                          self.seated, self.plug_rot, self.obs_wp), device=self.device)

    def _new_solver(self):
        return SolverVBD(self.model, iterations=8, rigid_contact_hard=False,
                         rigid_body_contact_buffer_size=self._contact_buffer)

    def reset(self):
        self._place(self._ones)
        # SolverVBD caches internal state tied to the config at construction; teleporting the
        # bodies to fresh starts and stepping the OLD solver ejects them. Rebuilding the solver
        # (~5ms) clears that state -> the reset is stable. This is the VBD-safe reset.
        self.solver = self._new_solver()
        self.hold_count.zero_()  # reset the consecutive-seated counter each episode
        self.contact_wp.zero_()
        self._write_obs()
        wp.synchronize()
        obs = torch.nan_to_num(wp.to_torch(self.obs_wp).clone()).clamp_(-50.0, 50.0)
        return self._add_obs_noise(obs)

    def step(self, residual_t):
        residual_t = residual_t.detach()
        # POSITION (3-DOF): base controller (scripted seat) steps the target toward the seated
        # pose; the policy's first 3 outputs are a bounded residual on top.
        tgt = wp.to_torch(self.target)
        seat = wp.to_torch(self.seated)
        base = ((seat - tgt) / MAX_DELTA).clamp(-1.0, 1.0)
        pos_res = residual_t[:, :3] if self.act_dim == 6 else residual_t
        total = (base + self.residual_scale * pos_res.clamp(-1.0, 1.0)).clamp(-1.0, 1.0)
        self._act_keep = total.contiguous()  # keep alive for the from_torch view
        act_wp = wp.from_torch(self._act_keep, dtype=wp.float32)
        wp.launch(integrate_target, dim=self.n,
                  inputs=(act_wp, self.seated, MAX_DELTA, self.box, self.target), device=self.device)
        # ORIENTATION (6-DOF subset only): the policy's last 3 outputs command the plug's
        # orientation via the stable d6 angular DRIVE target (base = aligned).
        if self.act_dim == 6:
            self._rotcmd_keep = residual_t[:, 3:6].contiguous()
            rc_wp = wp.from_torch(self._rotcmd_keep, dtype=wp.float32)
            wp.launch(set_angular_target, dim=self.n,
                      inputs=(rc_wp, self.ang_coords, ROT_CMD_RANGE, self.control.joint_target_q),
                      device=self.device)
        for _ in range(SUBSTEPS):
            self.state_0.clear_forces()
            wp.launch(apply_control, dim=self.n,
                      inputs=(self.state_0.body_q, self.state_0.body_qd, self.state_0.body_f,
                              self.model.body_mass, self.plug_idx, self.latch_idx,
                              self.target, self.grav, SPRING_KE, self.spring_kd), device=self.device)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        wp.launch(write_reward, dim=self.n,
                  inputs=(self.state_0.body_q, self.state_0.body_qd, self.plug_idx,
                          self.seated, self.plug_rot, self.start_y, self.prev_dist,
                          self.hold_count, HOLD_STEPS, self.max_steps,
                          W_PROG, R_SUCCESS, LAT_WEIGHT, ROT_W, W_VEL,
                          self.seat_depth_tol, self.seat_offset, self.seat_angle, OOB_LATERAL,
                          self.rew_wp, self.done_wp, self.succ_wp, self.dist_wp, self.depth_wp),
                  device=self.device)
        wp.copy(self.prev_dist, self.dist_wp)

        rew = wp.to_torch(self.rew_wp).clone()
        done = wp.to_torch(self.done_wp).clone()
        succ = wp.to_torch(self.succ_wp).clone()
        depth_mm = torch.nan_to_num(wp.to_torch(self.depth_wp).clone())

        # NO mid-rollout reset (that teleport is what VBD ejects). Envs run the full
        # rollout; the trainer calls reset() at each rollout boundary (fixed horizon).
        self._write_obs()
        wp.synchronize()
        obs = torch.nan_to_num(wp.to_torch(self.obs_wp).clone(),
                               nan=0.0, posinf=0.0, neginf=0.0).clamp_(-50.0, 50.0)
        return self._add_obs_noise(obs), rew, done, succ, depth_mm

    def contact_count(self) -> tuple[int, int]:
        return (int(self.contacts.rigid_contact_count.numpy()[0]),
                int(self.contacts.rigid_contact_force.shape[0]))


# ── tiny host-side quaternion helpers (xyzw) for rest-pose setup ──────────────
def _quat_inv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])
