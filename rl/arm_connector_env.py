"""Arm-in-the-loop RJ45 insertion env for PPO: the RO1 grips the plug and the policy
predicts GRIPPER (end-effector) motion, not the plug directly.

N (arm + friction-grasped plug + static socket) worlds in ONE batched Newton model. The arm
is position-controlled (kinematic, posed by eval_fk from the commanded arm joint_q); the plug
is a free VBD body held by AG-145 finger friction; the socket is static. Each control step:

    action = [dpos(3), drotvec(3), gripper(1)]     # base-frame EEF delta + gripper (pi05 rj45_sbot)
    -> residual on a scripted base drive toward the seat pose
    -> DLS IK turns the EEF target into arm joint motion
    -> the kinematic arm carries the grasped plug into the socket (VBD contact)
    reward = potential progress toward seated + hold-to-success   (same shape as connector_env)

Reverse curriculum: the EEF START pose is offset from the aligned approach (lateral + angular),
scaled up per stage — the analog of connector_env's lateral-offset misalignment curriculum.

Interface mirrors ConnectorVecEnv:
    env = ArmConnectorVecEnv(n); env.set_stage(0)
    obs = env.reset()
    obs, rew, done, success, depth_mm = env.step(action)   # action (n,7) in [-1,1]

See the sbot-arm-insertion-rl memory. Needs a CUDA GPU + the sim extra + robo_maker sbot USD.
"""
from __future__ import annotations

import math
import os
import sys

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverVBD
from scipy.spatial.transform import Rotation as Rot, Slerp

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from newton_cabling.connector import cad_rj45_connector  # noqa: E402
from newton_cabling.sim.safe_vbd import finalize_for_vbd, new_vbd_builder  # noqa: E402
from newton_cabling.sim.scene import load_connector_meshes, connector_shape_config  # noqa: E402
from newton_cabling.sim.sbot import (  # noqa: E402
    add_sbot, enable_finger_contact, ARM_JOINT_NAMES, WRIST_BODY,
    GRIPPER_COUPLING, GRIPPER_THETA_GRASP, GRASP_FACE_DIST)
from arm_ik import ArmIK  # noqa: E402

import torch  # noqa: E402

DEV = "cuda:0"

# ── control / geometry (metres, radians) ──────────────────────────────────────
ARM_HOME = [math.radians(a) for a in (4.0, -19.5, -113.0, 43.5, -268.9, -178.0)]
SPACING = 2.5                # m between arm bases (RO1 reach ~1.3 m)
SUBSTEPS = 8
MAX_DPOS = 0.002            # m/step EEF translation authority (mirrors the free-plug 2 mm)
MAX_DROT = math.radians(0.75)  # rad/step EEF rotation authority (angular is the sensitive axis)
# Residual gain on the base drive. START SMALL: a full-authority random residual from scratch
# breaks the grasp and ejects the plug (base alone seats ~30-56%; residual=1.0 -> 0%). The
# free-plug study's 1.0 was on an already-stable policy; ramp up after it learns.
RESIDUAL_SCALE = 0.2
EJECT_DIST = 0.15          # m: plug this far from its seat = ejected (grasp broke) -> failed step
APPROACH = 0.030            # m: max start->seat travel (the insertion itself)
APPROACH_MIN = 0.008        # m: stage-0 start distance (near-seated bootstrap; ramps to APPROACH)
# How far the jack sits along the plug's INSERTION AXIS from the nominal grasp (always along the
# tool axis, never world -z, so the approach stays axis-aligned).
#   0.03  = COUPLED: jack just below the fingertips. Unrealistic scene (jack floats at the hand,
#           ~shoulder height) but the PROVEN-STABLE config: base seats ~88%, clean 12mm travel.
#   0.25  = DECOUPLED: static table-height jack (~25cm below the base) the arm reaches down to.
#           Realistic, and validated in isolation (standalone proto seats 0.0mm/0.2mm/0.3deg, 3/3),
#           but IN THE ENV the extended reach-down config is still unstable (base ~12-38% with
#           frequent plug ejections). Root cause not yet isolated -> keep the stable default until
#           it's fixed. See the sbot-arm-insertion-rl memory.
JACK_DROP = 0.03
SEAT_AIM_DY = 0.012         # cad_rj45 canonical seat depth
IK_ITERS = 2                # DLS iters per control step (small deltas -> few iters)

# reverse curriculum: scale on the EEF start misalignment (lateral m + angular rad), per stage
RE_LAT = 0.010             # m max lateral start offset at scale 1.0
RE_ROT = math.radians(12.0)  # rad max angular start offset at scale 1.0
RE_CURRICULUM = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]

# reward (same structure as connector_env) ─────────────────────────────────────
W_PROG = 100.0
R_SUCCESS = 30.0           # seat bonus (raised 20->30 so seating strictly beats hovering)
LAT_WEIGHT = 2.0
ROT_W = 0.4
# Velocity-at-seat penalty. The free-plug env used 50, but a GRASPED plug keeps residual
# jitter (grasp compliance + IK) at the seat, so 50*speed erased the seat bonus and the policy
# learned to AVOID seating (arm_v1 collapse: eval 72%->5%, entropy 2.94->3.49). 2.0 damps
# wobble without ever out-weighing R_SUCCESS. The hold-to-success (20 steps) already gates
# transient touches, so heavy velocity damping isn't needed.
W_VEL = 2.0
HOLD_STEPS = 20
SEAT_DEPTH_TOL = 0.005
SEAT_OFFSET = 0.003
SEAT_ANGLE = math.radians(3.0)

OBS_DIM = 22   # eef_pos(3) eef_rot6d(6) gripper(1) + plug pos_err(3) lin_vel(3) orient_err(3) ang_vel(3)
ACT_DIM = 7    # dpos(3) drotvec(3) gripper(1)


def _all_idx(labels, sfx):
    return [i for i, l in enumerate(labels) if l.endswith(sfx)]


def _compose(A, B):
    pa, qa = A; pb, qb = B; Ra = Rot.from_quat(qa)
    return (pa + Ra.apply(pb), (Ra * Rot.from_quat(qb)).as_quat())


def _inv(A):
    p, q = A; Ri = Rot.from_quat(q).inv(); return (-Ri.apply(p), Ri.as_quat())


class ArmConnectorVecEnv:
    def __init__(self, n: int, *, seed: int = 0, socket_mu: float = 0.5, plug_mu: float = 2.0,
                 residual_scale: float = RESIDUAL_SCALE, contact_buffer_per: int = 64,
                 ik_iters: int = IK_ITERS, jack_drop: float = JACK_DROP, pin_grip: bool = True):
        # pin_grip: after eval_ik, force the gripper joints back to the commanded theta. eval_ik
        # re-derives every joint from body_q; pinning stops the jaws drifting. A/B this — it may
        # snap the fingers against a settled plug and fling it.
        self.pin_grip = pin_grip
        # jack_drop: how far along the insertion axis the jack sits from the nominal grasp.
        #   0.25 (default) = DECOUPLED, table-height jack the arm reaches down to (realistic scene,
        #                    but the extended reach-down config is less stable in VBD).
        #   0.03           = the original COUPLED geometry (jack just below the fingertips) —
        #                    less realistic, but the proven-stable configuration (base seats 31-56%).
        self.jack_drop = jack_drop
        self.n = n
        self.obs_dim = OBS_DIM
        self.act_dim = ACT_DIM
        self.num_stages = len(RE_CURRICULUM)
        self.residual_scale = residual_scale
        self.ik_iters = ik_iters
        self.max_steps = 200
        self._rng = np.random.default_rng(seed)
        self._gen = torch.Generator(device=DEV); self._gen.manual_seed(seed)

        b = new_vbd_builder(gravity=-9.81); b.rigid_gap = 0.005
        cols = max(1, int(math.ceil(math.sqrt(n))))
        for i in range(n):
            base = wp.vec3((i % cols) * SPACING, (i // cols) * SPACING, 0.6)
            add_sbot(b, wp.transform(base, wp.quat_identity()), with_gripper=True)
        JL = list(b.joint_label)
        arm_j = {nm: _all_idx(JL, nm) for nm in ARM_JOINT_NAMES}
        grip_j = {nm: _all_idx(JL, nm) for nm, _ in GRIPPER_COUPLING}
        for i in range(n):
            for nm, q in zip(ARM_JOINT_NAMES, ARM_HOME):
                j = arm_j[nm][i]; b.joint_q[j] = q; b.joint_target_q[j] = q
            for nm, ratio in GRIPPER_COUPLING:
                j = grip_j[nm][i]
                b.joint_q[j] = ratio * GRIPPER_THETA_GRASP; b.joint_target_q[j] = ratio * GRIPPER_THETA_GRASP
        enable_finger_contact(b, mu=plug_mu)
        for bod in range(b.body_count):
            b.body_flags[bod] = int(newton.BodyFlags.KINEMATIC)

        # FK the home pose to build per-arm plug/socket geometry
        fk_m = b.finalize(); fk_s = fk_m.state()
        newton.eval_fk(fk_m, fk_m.joint_q, fk_m.joint_qd, fk_s)
        bq = fk_s.body_q.numpy(); FL = list(fk_m.body_label)
        wristsF = _all_idx(FL, WRIST_BODY)
        tip1 = _all_idx(FL, "gripper_finger1_finger_tip_link")
        tip2 = _all_idx(FL, "gripper_finger2_finger_tip_link")

        spec = cad_rj45_connector(friction=socket_mu)
        meshes = load_connector_meshes(spec)
        sb = np.asarray(meshes.socket.base_position); pb = np.asarray(meshes.plug.base_position)
        off_local = sb - (pb + np.array([0.0, SEAT_AIM_DY, 0.0]))
        plug_cfg = newton.ModelBuilder.ShapeConfig(mu=plug_mu, ke=spec.contact.stiffness, kd=0.0,
                                                   gap=spec.contact.gap_meters, density=1500.0)
        self.seated_pos = np.zeros((n, 3)); self.seated_quat = np.zeros((n, 4))
        self.Pp0 = []  # nominal grasp spawn (pos, quat) per env
        for i in range(n):
            wpos = bq[wristsF[i]][:3]; wq = bq[wristsF[i]][3:7]
            tipm = 0.5 * (bq[tip1[i]][:3] + bq[tip2[i]][:3])
            Rw = Rot.from_quat(wq).as_matrix()
            tool = tipm - wpos; tool /= np.linalg.norm(tool)
            jaw = Rw @ np.array([0., 1., 0.]); jaw -= tool * (jaw @ tool); jaw /= np.linalg.norm(jaw)
            y_p = tool; x_p = jaw - y_p * (jaw @ y_p); x_p /= np.linalg.norm(x_p); z_p = np.cross(x_p, y_p)
            q_plug = Rot.from_matrix(np.column_stack([x_p, y_p, z_p])).as_quat()
            Pp0_pos = wpos + GRASP_FACE_DIST * tool
            Pp0 = (Pp0_pos, q_plug)
            # jack sits jack_drop along the insertion axis -> table height, decoupled from the grasp
            seated = _compose(Pp0, (np.array([0, self.jack_drop, 0.]), np.array([0, 0, 0, 1.])))
            socket_world = _compose(seated, (off_local, np.array([0, 0, 0, 1.])))
            self.seated_pos[i] = seated[0]; self.seated_quat[i] = seated[1]
            self.Pp0.append(Pp0)
            pbody = b.add_body(xform=wp.transform(wp.vec3(*Pp0_pos), wp.quat(*q_plug)), label=f"plug{i}")
            b.add_shape_mesh(pbody, mesh=meshes.plug.mesh, cfg=plug_cfg)
            b.add_joint_free(child=pbody)
            b.add_shape_mesh(-1, mesh=meshes.socket.mesh,
                             xform=wp.transform(wp.vec3(*socket_world[0]), wp.quat(*socket_world[1])),
                             cfg=connector_shape_config(spec), label=f"socket{i}")
            # VISUAL-ONLY table under the jack, top at the jack mesh's measured bottom. The jack is
            # already a static world shape so it needs no support; a COLLIDABLE table would only
            # risk finger/table contact at the seat (the fingertips end ~6mm above it).
            sv_w = Rot.from_quat(socket_world[1]).apply(np.asarray(meshes.socket.mesh.vertices)) \
                + socket_world[0]
            table_top_z = float(sv_w[:, 2].min()) - 0.001
            t_shape = b.add_shape_box(-1, xform=wp.transform(
                wp.vec3(float(seated[0][0]), float(seated[0][1]), table_top_z - 0.15),
                wp.quat_identity()), hx=0.20, hy=0.20, hz=0.15,
                cfg=newton.ModelBuilder.ShapeConfig(mu=0.6, ke=1e5, kd=0.0, gap=0.002, density=1000.0),
                label=f"table{i}")
            b.shape_flags[t_shape] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)

        self.model = finalize_for_vbd(b)
        self.device = self.model.device
        ML = list(self.model.body_label)
        self.plug_body = np.array([_all_idx(ML, f"plug{i}")[0] for i in range(n)])
        self.wrist_body = np.array(_all_idx(ML, WRIST_BODY))
        qstart = self.model.joint_q_start.numpy()
        self.arm_qc = np.array([[qstart[arm_j[nm][i]] for nm in ARM_JOINT_NAMES] for i in range(n)])
        # gripper joint coords + their coupling ratios, so the commanded grip can be RESTORED after
        # any eval_ik (which re-derives every joint from body_q and can drift the jaws open ->
        # the plug slips out). Pinning these keeps the grasp exactly at GRIPPER_THETA_GRASP.
        self.grip_qc = np.array([[qstart[grip_j[nm][i]] for nm, _ in GRIPPER_COUPLING]
                                 for i in range(n)])
        self.grip_ratios = np.array([r for _, r in GRIPPER_COUPLING])
        # plug free-joint q-start (7 coords: pos3 + quat4) for eval_ik-free reset via body_q
        jtype = self.model.joint_type.numpy(); jchild = self.model.joint_child.numpy()
        freej = {int(jchild[j]): j for j in range(self.model.joint_count)
                 if int(jtype[j]) == int(newton.JointType.FREE)}
        self.plug_freejoint = np.array([freej[int(pbi)] for pbi in self.plug_body])

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = SolverVBD(self.model, iterations=16, rigid_contact_hard=False,
                                rigid_body_contact_buffer_size=contact_buffer_per * n)
        self.ik = ArmIK(self.model, self.arm_qc, self.wrist_body)
        self.jq = self.model.joint_q.numpy().copy()
        self.dt = 1.0 / 60.0 / SUBSTEPS

        # nominal seat wrist pose Wseat (per env), + grasp offset G measured after a nominal settle
        self._settle_nominal()
        self.set_stage(0)
        self.eef_pos = self.seated_pos.copy(); self.eef_quat = self.seated_quat.copy()
        self.hold_count = np.zeros(n, dtype=np.int32)
        self.prev_dist = np.zeros(n)
        self.start_pos = np.zeros((n, 3))

    # ── simulation helpers ────────────────────────────────────────────────────
    def _sim(self, nsub=SUBSTEPS):
        jqw = wp.array(self.jq, dtype=float, device=self.device)
        for _ in range(nsub):
            newton.eval_fk(self.model, jqw, self.model.joint_qd, self.state_0,
                           body_flag_filter=int(newton.BodyFlags.KINEMATIC))
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _bodies(self):
        return self.state_0.body_q.numpy(), self.state_0.body_qd.numpy()

    def _set_plug_state(self, poses):
        """Teleport each plug free body to poses[i]=(pos,quat) and zero velocity, then sync joint coords."""
        bqn = self.state_0.body_q.numpy()
        bqd = self.state_0.body_qd.numpy()
        for i in range(self.n):
            p, q = poses[i]
            bqn[self.plug_body[i], :3] = p; bqn[self.plug_body[i], 3:7] = q
            bqd[self.plug_body[i]] = 0.0
        self.state_0.body_q.assign(bqn); self.state_0.body_qd.assign(bqd)
        newton.eval_ik(self.model, self.state_0, self.model.joint_q, self.model.joint_qd)
        # eval_ik also re-derives joint_qd from body velocities, and _sim() feeds joint_qd into
        # eval_fk — which gives the KINEMATIC fingers a velocity. At an extended reach-down config
        # the same joint_qd maps to a much larger fingertip velocity (longer moment arm), so the
        # jaws smack the grasped plug and fling it. The arm is static at reset: zero it.
        self.model.joint_qd.zero_()
        self.jq = self.model.joint_q.numpy().copy()
        # eval_ik re-derived EVERY joint from body_q, including the gripper's — restore the
        # commanded grip so the jaws can't drift open and drop the plug.
        if self.pin_grip:
            self.jq[self.grip_qc] = self.grip_ratios[None, :] * GRIPPER_THETA_GRASP

    def _settle_nominal(self):
        """Set arms to home, place plugs at nominal grasp, settle, measure grasp offset G + Wseat."""
        self.jq = self.model.joint_q.numpy().copy()
        self._set_plug_state([(np.array(self.Pp0[i][0]), np.array(self.Pp0[i][1])) for i in range(self.n)])
        for _ in range(30):
            self._sim()
        bqn, _ = self._bodies()
        self.G = []
        for i in range(self.n):
            W = (bqn[self.wrist_body[i], :3], bqn[self.wrist_body[i], 3:7])
            P = (bqn[self.plug_body[i], :3], bqn[self.plug_body[i], 3:7])
            self.G.append(_compose(_inv(W), P))
        # Wseat = wrist pose that puts a nominally-grasped plug at the seat
        self.Wseat = [_compose((self.seated_pos[i], self.seated_quat[i]), _inv(self.G[i]))
                      for i in range(self.n)]
        self.jq_home = self.model.joint_q.numpy().copy()  # arm-home joint_q reference

    # ── curriculum ────────────────────────────────────────────────────────────
    def set_stage(self, stage: int):
        self.stage = int(np.clip(stage, 0, self.num_stages - 1))
        self.re_scale = RE_CURRICULUM[self.stage]

    # ── reset / step ──────────────────────────────────────────────────────────
    def reset(self):
        n = self.n
        # sample a start EEF pose = Wseat backed off APPROACH along insertion + lateral+angular offset
        lat = (self._rng.random((n, 2)) * 2 - 1) * (RE_LAT * self.re_scale)     # jaw-plane offset
        ax = self._rng.standard_normal((n, 3)); ax /= (np.linalg.norm(ax, axis=1, keepdims=True) + 1e-9)
        ang = (self._rng.random(n)) * (RE_ROT * self.re_scale)
        # reverse curriculum on START DISTANCE too: stage 0 starts near-seated (bootstraps
        # held-success), later stages start further out (up to the built-in APPROACH).
        approach_i = APPROACH_MIN + (APPROACH - APPROACH_MIN) * self.re_scale
        start_poses = []
        Wstart = []
        for i in range(n):
            seatp = np.array(self.seated_pos[i]); seatq = np.array(self.seated_quat[i])
            Rseat = Rot.from_quat(seatq)
            ins = Rseat.apply([0, 1, 0])                       # insertion axis (world)
            xj = Rseat.apply([1, 0, 0]); zj = Rseat.apply([0, 0, 1])
            start_plug_pos = seatp - approach_i * ins + lat[i, 0] * xj + lat[i, 1] * zj
            start_plug_rot = (Rot.from_rotvec(ax[i] * ang[i]) * Rseat).as_quat()
            start_poses.append((start_plug_pos, start_plug_rot))
            Wstart.append(_compose((start_plug_pos, start_plug_rot), _inv(self.G[i])))
        # drive the arm to Wstart via IK, place the grasped plug there, settle briefly
        self.jq = self.jq_home.copy()
        wsp = np.array([w[0] for w in Wstart]); wsq = np.array([w[1] for w in Wstart])
        # NOTE: with the decoupled (table-height) jack this is a big reach-DOWN from ARM_HOME,
        # so it needs enough iterations to converge — too few leaves the arm up top while the plug
        # is placed at the jack, and it just drops (contacts=0, instant ejection).
        # NOTE: tightening this tolerance to 0.05mm was TESTED and does NOT fix the decoupled
        # reset ejection (60% vs 65%) — so IK residual vs the 0.4mm plug clearance is NOT the
        # mechanism. Reverted to keep resets fast. See the memory for what else was ruled out.
        self.jq, ik_ep, ik_er = self.ik.solve(self.jq, wsp, wsq, iters=60)
        if ik_ep.max() > 2e-3:
            print(f"[arm_env] WARNING: reset IK did not converge (max pos err "
                  f"{ik_ep.max()*1000:.1f}mm) — start pose unreachable?", flush=True)
        # CRITICAL: actually pose the arm at the IK'd config in state_0 BEFORE placing the plug.
        # _set_plug_state's (unmasked) eval_ik re-derives joint coords from body_q, so if body_q
        # still holds the OLD arm pose it silently discards this IK solution — the arm snaps back
        # and the plug is placed out of the grip and drops.
        jqw = wp.array(self.jq, dtype=float, device=self.device)
        newton.eval_fk(self.model, jqw, self.model.joint_qd, self.state_0,
                       body_flag_filter=int(newton.BodyFlags.KINEMATIC))
        # Place the plug at the ACTUAL wrist pose ∘ G (not the target start pose): G is the
        # post-settle grip equilibrium, so this puts the plug exactly in the jaws regardless of
        # any IK residual. Placing at the target instead leaves a small mismatch -> weak grip.
        bq_now = self.state_0.body_q.numpy()
        start_poses = [_compose((bq_now[self.wrist_body[i], :3].copy(),
                                 bq_now[self.wrist_body[i], 3:7].copy()), self.G[i])
                       for i in range(n)]
        self._set_plug_state(start_poses)
        self.solver = SolverVBD(self.model, iterations=16, rigid_contact_hard=False,
                                rigid_body_contact_buffer_size=self.solver.rigid_body_contact_buffer_size
                                if hasattr(self.solver, "rigid_body_contact_buffer_size") else 64 * self.n)
        for _ in range(15):          # settle the grip at the start pose (matches the proto)
            self._sim()
        bqn, _ = self._bodies()
        self.eef_pos = bqn[self.wrist_body, :3].copy()
        self.eef_quat = bqn[self.wrist_body, 3:7].copy()
        self.hold_count[:] = 0
        self.start_pos = bqn[self.plug_body, :3].copy()
        self.prev_dist = self._weighted_dist(bqn)
        return self._obs(bqn, self.state_0.body_qd.numpy())

    def _weighted_dist(self, bqn):
        e = bqn[self.plug_body, :3] - self.seated_pos                    # (n,3)
        ins = Rot.from_quat(self.seated_quat).apply(np.tile([0, 1, 0], (self.n, 1)))
        along = np.sum(e * ins, axis=1)
        lateral = np.linalg.norm(e - along[:, None] * ins, axis=1)
        ang = (Rot.from_quat(bqn[self.plug_body, 3:7]) * Rot.from_quat(self.seated_quat).inv()).magnitude()
        return np.sqrt(LAT_WEIGHT**2 * lateral**2 + along**2) + ROT_W * ang

    def _seat_terms(self, bqn):
        e = bqn[self.plug_body, :3] - self.seated_pos
        ins = Rot.from_quat(self.seated_quat).apply(np.tile([0, 1, 0], (self.n, 1)))
        along = np.sum(e * ins, axis=1)                 # >0 past seat, <0 short
        seat_gap = -along                               # >0 short of seat
        lateral = np.linalg.norm(e - along[:, None] * ins, axis=1)
        ang = (Rot.from_quat(bqn[self.plug_body, 3:7]) * Rot.from_quat(self.seated_quat).inv()).magnitude()
        return seat_gap, lateral, ang, along

    def step(self, action):
        a = np.asarray(action.detach().cpu() if hasattr(action, "detach") else action, dtype=np.float64)
        a = np.clip(a, -1.0, 1.0)
        # base drive toward Wseat (pos + rot), policy residual on top
        wseat_p = np.array([w[0] for w in self.Wseat]); wseat_q = np.array([w[1] for w in self.Wseat])
        base_dpos = np.clip((wseat_p - self.eef_pos) / MAX_DPOS, -1, 1)
        rerr = (Rot.from_quat(wseat_q) * Rot.from_quat(self.eef_quat).inv()).as_rotvec()
        base_drot = np.clip(rerr / MAX_DROT, -1, 1)
        tot_dpos = np.clip(base_dpos + self.residual_scale * a[:, 0:3], -1, 1) * MAX_DPOS
        tot_drot = np.clip(base_drot + self.residual_scale * a[:, 3:6], -1, 1) * MAX_DROT
        self.eef_pos = self.eef_pos + tot_dpos
        self.eef_quat = (Rot.from_rotvec(tot_drot) * Rot.from_quat(self.eef_quat)).as_quat()
        # IK the EEF target -> arm joints, then step the sim
        self.jq, _, _ = self.ik.solve(self.jq, self.eef_pos, self.eef_quat, iters=self.ik_iters)
        self._sim()

        bqn, bqd = self._bodies()
        d = self._weighted_dist(bqn)
        seat_gap, lateral, ang, along = self._seat_terms(bqn)
        rew = W_PROG * (self.prev_dist - d)
        seated_inst = (seat_gap <= SEAT_DEPTH_TOL) & (lateral <= SEAT_OFFSET) & (ang <= SEAT_ANGLE)
        speed = (np.linalg.norm(bqd[self.plug_body, 0:3], axis=1)
                 + np.linalg.norm(bqd[self.plug_body, 3:6], axis=1))
        rew = rew + np.where(seated_inst, R_SUCCESS - W_VEL * speed, 0.0)
        self.hold_count = np.where(seated_inst, self.hold_count + 1, 0)
        success = (self.hold_count >= HOLD_STEPS).astype(np.float32)
        # NaN / ejection guard: non-finite OR flung far from the seat (grasp broke) = failed step.
        # Freeze prev_dist for ejected envs so a huge -progress spike can't poison the advantages.
        dist_from_seat = np.linalg.norm(bqn[self.plug_body, :3] - self.seated_pos, axis=1)
        bad = (~np.isfinite(bqn[self.plug_body]).all(axis=1)) | (dist_from_seat > EJECT_DIST)
        rew = np.where(bad, -1.0, rew); success = np.where(bad, 0.0, success)
        d = np.where(bad, self.prev_dist, d)
        self.prev_dist = d
        # honest instantaneous-seated fraction this step (for the trainer's progress/curriculum
        # metric — the held-success rollout mean is horizon-diluted and never triggers advance).
        self.seated_inst_frac = float(np.mean(np.where(bad, False, seated_inst)))
        # travel from start (mm), clamped so an ejected plug can't report metres
        depth_mm = np.clip(np.linalg.norm(bqn[self.plug_body, :3] - self.start_pos, axis=1),
                           0.0, 0.2) * 1000.0
        obs = self._obs(bqn, bqd)
        done = np.zeros(self.n, dtype=np.float32)
        to_t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=DEV)
        return (obs, to_t(np.nan_to_num(rew)), to_t(done),
                to_t(success), to_t(np.nan_to_num(depth_mm)))

    def _obs(self, bqn, bqd):
        # eef state (VLA-aligned): pos(3) rot6d(6) gripper(1)
        eefp = bqn[self.wrist_body, :3]
        R6 = Rot.from_quat(bqn[self.wrist_body, 3:7]).as_matrix()[:, :, :2].reshape(self.n, 6)
        grip = np.full((self.n, 1), 1.0)                         # closed (held); action[6] reserved
        # privileged plug-vs-seat error (like connector_env obs)
        e = (bqn[self.plug_body, :3] - self.seated_pos) * 50.0
        lin = bqd[self.plug_body, 0:3]
        q_err = (Rot.from_quat(self.seated_quat).inv() * Rot.from_quat(bqn[self.plug_body, 3:7])).as_rotvec() * 3.0
        angv = bqd[self.plug_body, 3:6]
        obs = np.concatenate([eefp, R6, grip, e, lin, q_err, angv], axis=1)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clip(-50, 50)
        return torch.as_tensor(obs, dtype=torch.float32, device=DEV)

    def contact_count(self):
        return (int(self.contacts.rigid_contact_count.numpy()[0]),
                int(self.contacts.rigid_contact_force.shape[0]))
