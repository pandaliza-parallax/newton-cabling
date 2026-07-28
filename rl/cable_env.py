"""Batched cable-insertion env: the RO1 grips the FLEXIBLE ethernet cable, the connector
dangles at the end, and the task is a horizontal insertion into a panel jack.

Stage C of the cable track (see the sbot-cable-grasp memory). Everything the single-env
study established is baked in:

  * cable = add_rod capsules; friction grip on a mid segment (kinematic fingers)
  * connector = a mesh SHAPE on the front rod body (a welded rigid body explodes in VBD),
    collision-filtered against the rear rod segments it engulfs
  * BOOT: the connector face sits a stiff strain-relief boot-length ahead of the rod end —
    the real-RJ45 feature that buys finger-to-jack clearance (bare cable can't both clear
    the mouth plane and transmit the push)
  * frames are PLUG-FACE based (the face is the plug mesh origin); the jack is placed from
    the face's seat pose — never from the rod-body origin (the scratch-script bug)
  * base controller = the validated funnel servo: null the connector's lateral error with
    the grip (quasi-static), advance only when inside the gate, and NEVER advance the
    finger front past the jack mouth minus a margin (standoff clamp)
  * finger<->jack contacts are counted per env as VIOLATIONS (both bodies are kinematic, so
    the solver cannot respond — the env must know instead: reward penalty + report)
  * success is angle-gated and hold-based (position-only was gamed by a 59-deg twist)

TILTED-CABLE variant: CableInsertVecEnv(n, cable_tilt_deg=8.0) starts every episode with the
hanging connector drooping that many deg below horizontal (jack stays world-level; same seat
geometry as the level env) — the policy must rotate the wrist to level the cable to seat.

Interface mirrors ConnectorVecEnv / ArmConnectorVecEnv:
    env = CableInsertVecEnv(n); env.set_stage(0)
    obs = env.reset()
    obs, rew, done, success, depth_mm = env.step(action)   # action (n,7) in [-1,1]
Action = [dpos(3), drotvec(3), gripper(1)] EEF residual on the servo base (pi05 rj45_sbot).
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverVBD
from scipy.spatial.transform import Rotation as Rot

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
    GRIPPER_COUPLING, FINGER_LINK_BODIES)
from arm_ik import ArmIK  # noqa: E402

import torch  # noqa: E402

DEV = "cuda:0"

# ── scene constants (validated in the single-env study) ───────────────────────
ARM_HOME = [math.radians(a) for a in (4.0, -19.5, -113.0, 43.5, -268.9, -178.0)]
SPACING = 2.5
SUBSTEPS = 8
THETA_CABLE = -0.008        # jaw angle for the 6.5 mm cable (friction grip)
ARM_PITCH_DEG = 45.0        # ARM approach angle (user's GS-scene reference): the WRIST/fingers
#                             are pitched 45 deg down-forward about the jaw axis, while the cable
#                             + connector stay HORIZONTAL in the jaws and the jack face stays
#                             perpendicular to the ground. The 45 lives in the robot's pose, the
#                             task geometry stays level — no post-hoc cable/wrist compensation.
GRIP_BACK = 0.050           # m of cable between the grip and the rod front end
CABLE_BACK = 0.060          # m of free cable hanging behind the grip
SEG_LEN = 0.010
BOOT = 0.018                # m stiff strain-relief boot: plug FACE this far ahead of the rod end
APPROACH = 0.030            # m the face travels from settle to seat
SEAT_AIM_DY = 0.012         # face ends this far past the jack mouth (cavity depth aim)
N_FILTER_SEGS = 4           # rear rod segments collision-filtered against the connector

# servo base controller
KP = 0.6
KD = 0.3
LAT_CAP = 0.0015            # m max lateral grip correction per control step
ADV = 0.0008                # m grip advance per step when gated open
GATE = 0.0025               # m lateral gate for advancing
MARGIN = 0.003              # m finger-front standoff from the jack mouth plane

# policy residual
MAX_DPOS = 0.002
MAX_DROT = math.radians(0.75)
RESIDUAL_SCALE = 0.2

# curriculum: (approach distance m, jack lateral offset magnitude m) per stage.
# Pure-RL needs the near-seated bootstrap (stage 0 starts close), like the rigid env.
CURRICULUM = [(0.010, 0.0), (0.015, 0.0015), (0.020, 0.003), (0.025, 0.005), (0.030, 0.008)]
ANCHOR_SEGS = (5, 6)        # rod segment indices held in the jaws -> kinematically wrist-anchored
BOOT_JOINTS = 2             # stiffen the LAST k cable joints (the strain-relief boot section)
BOOT_KE_MULT = 20.0         # bend/twist ke multiplier for the boot joints
BOOT_KD_MULT = 4.0

# reward (mirrors the rigid arm env)
W_PROG = 100.0
R_SUCCESS = 30.0
LAT_WEIGHT = 2.0
ROT_W = 0.4
W_VEL = 2.0
W_VIOL = 2.0                # flat penalty per step with any finger<->jack contact
HOLD_STEPS = 20
SEAT_DEPTH_TOL = 0.005
SEAT_OFFSET = 0.003
SEAT_ANGLE = math.radians(8.0)   # face angle vs settled hang (boot end floats a little)
EJECT_DIST = 0.15

OBS_DIM = 22
ACT_DIM = 7


def _all_idx(labels, sfx):
    return [i for i, l in enumerate(labels) if l.endswith(sfx)]


@wp.kernel
def _sync_grip_anchors(body_q: wp.array(dtype=wp.transform),
                       body_qd: wp.array(dtype=wp.spatial_vector),
                       parent_idx: wp.array(dtype=int), anchor_idx: wp.array(dtype=int),
                       off_p: wp.array(dtype=wp.vec3), off_q: wp.array(dtype=wp.quat)):
    """Rigidly carry the GRIPPED cable segments with the wrist (the user's spec: once grasped,
    the cable moves ONLY with the robot's hand — no slip/roll in the jaws; orientation authority
    comes from wrist rotation). Same pattern as Newton's RJ45 example _sync_cable_anchors."""
    tid = wp.tid()
    par = body_q[parent_idx[tid]]
    pp = wp.transform_get_translation(par)
    pr = wp.transform_get_rotation(par)
    aw = pp + wp.quat_rotate(pr, off_p[tid])
    ar = wp.normalize(wp.mul(pr, off_q[tid]))
    body_q[anchor_idx[tid]] = wp.transform(aw, ar)
    body_qd[anchor_idx[tid]] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class CableInsertVecEnv:
    def __init__(self, n: int, *, seed: int = 0, socket_mu: float = 0.5,
                 residual_scale: float = RESIDUAL_SCALE, ik_iters: int = 2,
                 contact_buffer_per: int = 1024, cable_tilt_deg: float = 0.0):
        # NOTE: contact_buffer_per is PER-BODY; deep insertion measured ~300 contacts on the
        # plug body, and an overflowed buffer drops grasp/insertion contacts -> the connector
        # gets flung right at first jack contact. 1024 is the validated value.
        self.n = n
        self.obs_dim = OBS_DIM
        self.act_dim = ACT_DIM
        self.num_stages = len(CURRICULUM)
        self.residual_scale = residual_scale
        self.ik_iters = ik_iters
        self.max_steps = 260
        self._rng = np.random.default_rng(seed)
        # TILTED-CABLE variant: instead of leveling the hang to horizontal, the align loop
        # targets this pitch. Positive = the connector end DROOPS below horizontal. The jack
        # stays world-level regardless (user spec), so the policy must rotate the wrist to
        # level the cable before it can seat. 0.0 = the original level env, bit-identical.
        # A (lo, hi) tuple samples a PER-ENV tilt uniformly — DR across the batch (fixed per
        # env across resets, since the tilt is baked into the settled snapshot).
        if isinstance(cable_tilt_deg, (tuple, list)):
            self.cable_tilt = np.radians(self._rng.uniform(*cable_tilt_deg, n))
        else:
            self.cable_tilt = np.full(n, math.radians(cable_tilt_deg))

        # ── build: N arms, home pose, cable grip ─────────────────────────────
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
                b.joint_q[j] = ratio * THETA_CABLE; b.joint_target_q[j] = ratio * THETA_CABLE
        all_finger_shapes = enable_finger_contact(b, mu=2.0)
        for bod in range(b.body_count):
            b.body_flags[bod] = int(newton.BodyFlags.KINEMATIC)

        # FK home to get per-arm frames
        fk_m = b.finalize(); fk_s = fk_m.state()
        newton.eval_fk(fk_m, fk_m.joint_q, fk_m.joint_qd, fk_s)
        bq = fk_s.body_q.numpy(); FL = list(fk_m.body_label)
        wristsF = _all_idx(FL, WRIST_BODY)
        tip1 = _all_idx(FL, "gripper_finger1_finger_tip_link")
        tip2 = _all_idx(FL, "gripper_finger2_finger_tip_link")

        spec = cad_rj45_connector(friction=socket_mu)
        meshes = load_connector_meshes(spec)
        sbp = np.asarray(meshes.socket.base_position)
        pbp = np.asarray(meshes.plug.base_position)
        self._off_local = sbp - (pbp + np.array([0.0, SEAT_AIM_DY, 0.0]))  # plug-origin -> socket-origin
        plug_cfg = newton.ModelBuilder.ShapeConfig(mu=2.0, ke=spec.contact.stiffness, kd=0.0,
                                                   gap=spec.contact.gap_meters, density=1500.0)
        cable_cfg = dataclasses.replace(b.default_shape_cfg, ke=spec.contact.stiffness,
                                        kd=spec.contact.damping, mu=spec.cable_friction)

        self.rod_front = []          # front rod body per env (carries the connector)
        self.rod_bodies_all = []     # all rod bodies per env (for snapshot bookkeeping)
        self.jack_body = []
        self.plug_local = []         # (pos, quat) of the plug FACE in the front-rod frame
        self.frame = []              # per-env dict: tool/jaw/n/grip_pos
        jack_shape_of_env = {}
        finger_env_of_shape = {}
        for i in range(n):
            wpos = bq[wristsF[i]][:3]; wq = bq[wristsF[i]][3:7]
            tipm = 0.5 * (bq[tip1[i]][:3] + bq[tip2[i]][:3])
            Rw = Rot.from_quat(wq).as_matrix()
            tool = tipm - wpos; tool /= np.linalg.norm(tool)
            jaw = Rw @ np.array([0., 1., 0.]); jaw -= tool * (jaw @ tool); jaw /= np.linalg.norm(jaw)
            nfw = np.cross(tool, jaw); nfw /= np.linalg.norm(nfw)
            # ARM approach angle, ABSOLUTE from vertical: 0 = grab from directly ABOVE (fingers
            # straight down), 45 = the GS-scene diagonal. NOTE the pendant home is itself ~45deg
            # (tool=[0,-0.72,-0.70]) — a relative rotation was the earlier sign fiasco. Rotate the
            # authored frame so tool hits the absolute target; wrist IK'd to match post-finalize.
            fwd_h = tool.copy(); fwd_h[2] = 0.0; fwd_h /= np.linalg.norm(fwd_h)
            sp_ = math.sin(math.radians(ARM_PITCH_DEG)); cpv = math.cos(math.radians(ARM_PITCH_DEG))
            tool_t = np.array([0.0, 0.0, -1.0]) * cpv + fwd_h * sp_
            axis = np.cross(tool, tool_t); s_ = np.linalg.norm(axis)
            Ralign = (Rot.from_rotvec(axis / s_ * math.atan2(s_, float(np.dot(tool, tool_t))))
                      if s_ > 1e-9 else Rot.identity())
            tool = Ralign.apply(tool)
            nfw = Ralign.apply(nfw); nfw[2] = 0.0; nfw /= np.linalg.norm(nfw)
            grip_pos = wpos + 0.192 * tool
            self.frame.append(dict(tool=tool, jaw=jaw, n=nfw, grip=grip_pos))
            pts = []
            d = -CABLE_BACK
            while d <= GRIP_BACK + 1e-6:
                pts.append(grip_pos + d * nfw); d += SEG_LEN
            pts = np.array(pts)
            cp = [wp.vec3(*p) for p in pts]
            cq = newton.utils.create_parallel_transport_cable_quaternions(cp)
            ke_start = len(b.joint_target_ke)
            rb, _ = b.add_rod(positions=cp, quaternions=cq, radius=spec.cable_radius_meters,
                              cfg=cable_cfg, bend_stiffness=1e1, bend_damping=3e-1,
                              label=f"cable{i}")
            # BOOT as a stiff rod section: raise bend/twist (shared slot) on the LAST joints.
            # Cable joints append 2 dofs each (stretch, bend) -> bend slots are odd offsets.
            njoints = (len(b.joint_target_ke) - ke_start) // 2
            for j in range(max(0, njoints - BOOT_JOINTS), njoints):
                b.joint_target_ke[ke_start + 2 * j + 1] *= BOOT_KE_MULT
                b.joint_target_kd[ke_start + 2 * j + 1] *= BOOT_KD_MULT
            # grip anchor: the segments held in the jaws become KINEMATIC, carried by the wrist
            for k in ANCHOR_SEGS:
                b.body_flags[rb[k]] = int(newton.BodyFlags.KINEMATIC)
                for s in b.body_shapes[rb[k]]:
                    b.shape_flags[s] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
            self.rod_bodies_all.append(rb)
            front = rb[-1]; self.rod_front.append(front)
            # connector: face BOOT ahead of the rod front, plug local y (insertion) -> n
            # plug axes: width -> jaw, insertion -> the pitched forward span, height -> completes
            z_p = np.cross(jaw, nfw); z_p /= np.linalg.norm(z_p)
            Rp = np.column_stack([jaw, nfw, z_p]); q_plug = Rot.from_matrix(Rp).as_quat()
            face_w = pts[-1] + BOOT * nfw
            rod_p = np.array(pts[-1])
            rod_q = np.array([cq[-1][0], cq[-1][1], cq[-1][2], cq[-1][3]])
            Rr = Rot.from_quat(rod_q)
            loc_p = Rr.inv().apply(face_w - rod_p)
            loc_q = (Rr.inv() * Rot.from_quat(q_plug)).as_quat()
            self.plug_local.append((loc_p, loc_q))
            plug_shape = b.add_shape_mesh(front, mesh=meshes.plug.mesh,
                                          xform=wp.transform(wp.vec3(*loc_p), wp.quat(*loc_q)),
                                          cfg=plug_cfg)
            for rbdy in rb[max(0, len(rb) - 1 - N_FILTER_SEGS):]:
                for rs in b.body_shapes[rbdy]:
                    if rs != plug_shape:
                        b.add_shape_collision_filter_pair(plug_shape, rs)
            # kinematic jack (parked far below; placed per-episode at reset)
            jb = b.add_body(xform=wp.transform(wp.vec3(0., 0., -5. - 0.5 * i), wp.quat_identity()),
                            label=f"jack{i}")
            js = b.add_shape_mesh(jb, mesh=meshes.socket.mesh, cfg=connector_shape_config(spec))
            b.add_joint_free(child=jb)
            b.body_flags[jb] = int(newton.BodyFlags.KINEMATIC)
            self.jack_body.append(jb)
            jack_shape_of_env[js] = i

        # map finger shapes -> env by body ownership (arm order)
        finger_bodies_per_env = {i: set() for i in range(n)}
        for nm in FINGER_LINK_BODIES:
            for i, bidx in enumerate(_all_idx(list(b.body_label), nm)):
                finger_bodies_per_env[i].add(bidx)
        for s in all_finger_shapes:
            for i in range(n):
                if b.shape_body[s] in finger_bodies_per_env[i]:
                    finger_env_of_shape[s] = i; break

        self.model = finalize_for_vbd(b)
        self.device = self.model.device
        self._contact_buffer = contact_buffer_per
        ML = list(self.model.body_label)
        self.wrist_body = np.array(_all_idx(ML, WRIST_BODY))
        qstart = self.model.joint_q_start.numpy()
        self.arm_qc = np.array([[qstart[arm_j[nm][i]] for nm in ARM_JOINT_NAMES] for i in range(n)])
        jtype = self.model.joint_type.numpy(); jchild = self.model.joint_child.numpy()
        # NOTE: add_body + add_joint_free leaves the jack with TWO free joints; only the FIRST
        # is the effective one in eval_fk. Keep first-match (a last-match dict wrote the shadowed
        # duplicate -> the jack never moved off its parking spot).
        freej = {}
        for j in range(self.model.joint_count):
            c = int(jchild[j])
            if int(jtype[j]) == int(newton.JointType.FREE) and c not in freej:
                freej[c] = j
        self.jack_qs = np.array([int(qstart[freej[jb]]) for jb in self.jack_body])
        # shape -> env lookup arrays for the violation count
        self.fmap = np.full(self.model.shape_count, -1, dtype=np.int64)
        for s, i in finger_env_of_shape.items():
            self.fmap[s] = i
        self.jmap = np.full(self.model.shape_count, -1, dtype=np.int64)
        for s, i in jack_shape_of_env.items():
            self.jmap[s] = i

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = self._new_solver()
        self.ik = ArmIK(self.model, self.arm_qc, self.wrist_body)
        self.jq = self.model.joint_q.numpy().copy()
        self.dt = 1.0 / 60.0 / SUBSTEPS
        self._pll = (np.array([p for p, _ in self.plug_local]),
                     np.array([q for _, q in self.plug_local]))
        self.rod_front = np.array(self.rod_front)

        # grip-anchor offsets: pose the arm at home once (rods keep their build transforms —
        # eval_fk never touches cable-joint bodies), then record each anchored segment's pose
        # in its wrist's frame. The sync kernel replays wrist ∘ offset every substep.
        jqw0 = wp.array(self.jq, dtype=float, device=self.device)
        newton.eval_fk(self.model, jqw0, self.model.joint_qd, self.state_0,
                       body_flag_filter=int(newton.BodyFlags.KINEMATIC))
        bq0 = self.state_0.body_q.numpy()
        # put the wrist at the approach pose the cable was authored in: rotate the HOME wrist so
        # its (home) tool direction lands on the authored frame's tool (absolute-from-vertical).
        if True:
            wq_t = np.zeros((n, 4)); wp_t = bq0[self.wrist_body, :3].copy()
            for i in range(n):
                Rw = Rot.from_quat(bq0[self.wrist_body[i], 3:7])
                # home tool dir from the stored pre-rotation FK is not kept; recover it via the
                # authored tool and the SAME absolute construction: rotation from home-tool to
                # authored tool == frame Ralign; reconstruct from vectors.
                # home tool = wristF frames used at authoring time: derive from tips at home FK.
                pass
            # simpler + exact: rotate so the wrist's CURRENT tool (home) maps to frame tool
            tips1 = _all_idx(ML, "gripper_finger1_finger_tip_link")
            tips2 = _all_idx(ML, "gripper_finger2_finger_tip_link")
            for i in range(n):
                tm = 0.5 * (bq0[tips1[i], :3] + bq0[tips2[i], :3])
                t_home = tm - bq0[self.wrist_body[i], :3]; t_home /= np.linalg.norm(t_home)
                t_want = self.frame[i]["tool"]
                ax = np.cross(t_home, t_want); s_ = np.linalg.norm(ax)
                Rc = (Rot.from_rotvec(ax / s_ * math.atan2(s_, float(np.dot(t_home, t_want))))
                      if s_ > 1e-9 else Rot.identity())
                wq_t[i] = (Rc * Rot.from_quat(bq0[self.wrist_body[i], 3:7])).as_quat()
            self.jq, ik_ep, _ = self.ik.solve(self.jq, wp_t, wq_t, iters=100)
            if ik_ep.max() > 2e-3:
                print(f"[cable_env] WARNING: arm-pitch IK residual {ik_ep.max()*1000:.1f}mm")
            jqw0 = wp.array(self.jq, dtype=float, device=self.device)
            newton.eval_fk(self.model, jqw0, self.model.joint_qd, self.state_0,
                           body_flag_filter=int(newton.BodyFlags.KINEMATIC))
            bq0 = self.state_0.body_q.numpy()
        par, anc, offp, offq = [], [], [], []
        for i in range(n):
            Rw = Rot.from_quat(bq0[self.wrist_body[i], 3:7])
            wpn = bq0[self.wrist_body[i], :3]
            for k in ANCHOR_SEGS:
                ab = self.rod_bodies_all[i][k]
                par.append(int(self.wrist_body[i])); anc.append(int(ab))
                offp.append(Rw.inv().apply(bq0[ab, :3] - wpn))
                offq.append((Rw.inv() * Rot.from_quat(bq0[ab, 3:7])).as_quat())
        self._anc_par = wp.array(np.array(par, dtype=np.int32), dtype=int, device=self.device)
        self._anc_idx = wp.array(np.array(anc, dtype=np.int32), dtype=int, device=self.device)
        self._anc_p = wp.array(np.array(offp), dtype=wp.vec3, device=self.device)
        self._anc_q = wp.array(np.array(offq), dtype=wp.quat, device=self.device)
        self._n_anchors = len(anc)

        # ── align the connector with the WORLD jack axis ──────────────────────
        # The world fixes the jack (horizontal panel insertion); the ROBOT compensates for the
        # in-hand grasp angle: rotate each wrist by -GRASP_PITCH about its jaw axis so the
        # 45-deg-held connector comes out horizontal. The grip anchor carries the cable with it.
        # ── align + settle: rotate each WRIST until the hanging connector is LEVEL ────
        # All orientation authority is the robot's (user spec): the wrist compensates BOTH the
        # 45-deg in-hand grasp AND the free span's gravity sag, so the hanging connector's axis
        # ends up exactly horizontal — matching the world-level jack. Iterate settle -> measure
        # -> rotate wrist by the residual (the grip anchor carries the cable each time).
        rf = np.array(self.rod_front) if not isinstance(self.rod_front, np.ndarray) else self.rod_front
        wq_cur = bq0[self.wrist_body, 3:7].copy()
        wp_cur = bq0[self.wrist_body, :3].copy()
        def _align(target):
            """Settle, then pitch each wrist about its jaw axis until the hang axis's vertical
            angle asin(y_z) hits `target[i]` (damped iterations, jaw-axis only)."""
            errs = [0.0]
            for align_it in range(4):
                for k in range(400):
                    self._sim()
                    if k >= 45 and k % 5 == 0:
                        v = np.linalg.norm(self.state_0.body_qd.numpy()[rf, 0:3], axis=1).max()
                        if v < 5e-4:
                            break
                self._settle_steps = k + 1
                _, fq = self._face_pose(self.state_0.body_q.numpy())
                y_s = Rot.from_quat(fq).apply(np.tile([0, 1, 0.], (n, 1)))
                errs = []
                for i in range(n):
                    # sag = vertical component of the hang axis; correct by DAMPED pitch about
                    # the JAW axis only (full 3D error correction injects roll and diverges —
                    # measured 71 deg residual). Negative-about-jaw raises the connector.
                    sag = math.asin(float(np.clip(y_s[i, 2], -1.0, 1.0)))
                    err = sag - target[i]
                    errs.append(abs(err))
                    if abs(err) > 1e-4:
                        wq_cur[i] = (Rot.from_rotvec(self.frame[i]["jaw"] * (0.7 * err))
                                     * Rot.from_quat(wq_cur[i])).as_quat()
                if max(errs) < math.radians(0.5):
                    break
                self.jq, _, _ = self.ik.solve(self.jq, wp_cur, wq_cur, iters=60)
            return errs, align_it

        errs, align_it = _align(np.zeros(n))
        print(f"[cable_env] hang leveled to {math.degrees(max(errs)):.2f} deg "
              f"({align_it + 1} align iters)", flush=True)
        self.wq_level = wq_cur.copy()   # LEVEL wrist pose: the teacher's rotation target
        # LEVEL reference: the face pose and the level jack frame are recorded at the LEVEL
        # hang. The tilted variant places the seat from THESE, so the task geometry is the
        # level env's exactly — the tilt is purely an initial condition the policy rotates away.
        bqn = self.state_0.body_q.numpy()
        self.face_ref, faceq_ref = self._face_pose(bqn)
        # WORLD-LEVEL jack frame (user spec): bore exactly horizontal, face exactly vertical —
        # never inherit the cable's gravity sag. Project the settled insertion axis onto the
        # horizontal plane and rebuild a level, right-handed plug frame around it.
        y_s = Rot.from_quat(faceq_ref).apply(np.tile([0, 1, 0.], (n, 1)))
        z_s = Rot.from_quat(faceq_ref).apply(np.tile([0, 0, 1.], (n, 1)))
        self.ins = y_s.copy(); self.ins[:, 2] = 0.0
        self.ins /= np.linalg.norm(self.ins, axis=1, keepdims=True)
        self.seat_qw = np.zeros((n, 4))
        for i in range(n):
            z_n = np.array([0., 0., 1.]) * (1.0 if z_s[i, 2] >= 0 else -1.0)
            x_n = np.cross(self.ins[i], z_n); x_n /= np.linalg.norm(x_n)
            self.seat_qw[i] = Rot.from_matrix(
                np.column_stack([x_n, self.ins[i], z_n])).as_quat()
        # finger mesh verts in BODY frame (env 0; identical geometry across arms) — kept so
        # the finger front extent can be RE-measured after every retilt (the wrist rotation
        # swings the fingertips ~20mm along ins, so ffa is snapshot-dependent)
        self._ffa_pts = []
        for bidx in finger_bodies_per_env[0]:
            for sh in b.body_shapes[bidx]:
                src = b.shape_source[sh]
                if src is None or b.shape_type[sh] != int(newton.GeoType.MESH):
                    continue
                v = np.asarray(src.vertices)
                stx = b.shape_transform[sh]
                Rs = Rot.from_quat([stx.q[0], stx.q[1], stx.q[2], stx.q[3]]).as_matrix()
                self._ffa_pts.append(
                    (int(bidx), (Rs @ v.T).T + np.array([stx.p[0], stx.p[1], stx.p[2]])))
        # TILT lives in the CURRICULUM (stage 0 = level near-seated bootstrap, full droop at
        # the top stage): set_stage() re-runs the align to the scaled droop and re-snapshots.
        # cable_v1 lesson: full tilt from iteration zero destroys the bootstrap (tilting
        # swings the grip ~19mm back, so a "10mm near-seated" stage-0 start becomes ~29mm +
        # a leveling rotation — top-stage difficulty) and PPO stalled at 0% held all run.
        self._align_fn = _align
        self._tilt_scale_cur = 0.0
        self._take_snapshot()

        # runtime buffers
        self.servo_rot_gain = 0.4    # teacher's pre-dock orientation gain (0 = old teacher)
        self.seat_pos = np.zeros((n, 3)); self.seat_q = self.faceq0.copy()
        self.wrist_tgt_p = self.W0p.copy(); self.wrist_tgt_q = self.W0q.copy()
        self.prev_lat = np.zeros((n, 3)); self.advanced = np.zeros(n)
        self.hold = np.zeros(n, dtype=np.int64); self.prev_dist = np.zeros(n)
        self.face_start = self.face0.copy()
        self.viol_last = np.zeros(n)
        self.set_stage(0)

    # ── internals ─────────────────────────────────────────────────────────────
    def _new_solver(self):
        return SolverVBD(self.model, iterations=24, rigid_contact_hard=False,
                         rigid_body_contact_buffer_size=self._contact_buffer)

    def _sim(self):
        jqw = wp.array(self.jq, dtype=float, device=self.device)
        for _ in range(SUBSTEPS):
            newton.eval_fk(self.model, jqw, self.model.joint_qd, self.state_0,
                           body_flag_filter=int(newton.BodyFlags.KINEMATIC))
            wp.launch(_sync_grip_anchors, dim=self._n_anchors,
                      inputs=(self.state_0.body_q, self.state_0.body_qd,
                              self._anc_par, self._anc_idx, self._anc_p, self._anc_q),
                      device=self.device)
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _face_pose(self, bqn):
        """World pose of the plug FACE (mesh origin) per env, from the front rod body."""
        rp = bqn[self.rod_front, :3]; rq = bqn[self.rod_front, 3:7]
        Rr = Rot.from_quat(rq)
        pos = rp + Rr.apply(self._pll[0])
        quat = (Rr * Rot.from_quat(self._pll[1])).as_quat()
        return pos, quat

    def _measure_ffa(self, bqn):
        """Finger front extent along ins (env 0), at the CURRENT pose."""
        ffa = -1e9
        for bidx, vb in self._ffa_pts:
            Rb = Rot.from_quat(bqn[bidx][3:7]).as_matrix()
            vw = (Rb @ vb.T).T + bqn[bidx][:3]
            ffa = max(ffa, float((vw @ self.ins[0]).max()))
        return ffa

    def _take_snapshot(self):
        """Record the CURRENT settled scene as the per-episode start state: face pose, wrist
        pose, finger front room (mouth from the LEVEL reference the seat is placed from), and
        the full kinematic snapshot reset() restores."""
        bqn = self.state_0.body_q.numpy()
        self.face0, self.faceq0 = self._face_pose(bqn)
        self.W0p = bqn[self.wrist_body, :3].copy()
        self.W0q = bqn[self.wrist_body, 3:7].copy()
        mouth_along0 = float(self.face_ref[0] @ self.ins[0]) + APPROACH - SEAT_AIM_DY
        self.front_room = mouth_along0 - self._measure_ffa(bqn)
        self._snap_bq = bqn.copy()
        self._snap_bqd = self.state_0.body_qd.numpy().copy()
        self._snap_jq = self.jq.copy()

    def _violations(self):
        nc = int(self.contacts.rigid_contact_count.numpy()[0])
        out = np.zeros(self.n)
        if nc:
            s0 = self.contacts.rigid_contact_shape0.numpy()[:nc]
            s1 = self.contacts.rigid_contact_shape1.numpy()[:nc]
            f0, j1 = self.fmap[s0], self.jmap[s1]
            f1, j0 = self.fmap[s1], self.jmap[s0]
            hit = np.concatenate([f0[(f0 >= 0) & (f0 == j1)], f1[(f1 >= 0) & (f1 == j0)]])
            if len(hit):
                out = np.bincount(hit, minlength=self.n).astype(np.float64)
        return out

    def set_stage(self, stage: int):
        self.stage = int(np.clip(stage, 0, self.num_stages - 1))
        self._approach, self._mag = CURRICULUM[self.stage]
        # tilt curriculum: droop scales with the stage (0 at stage 0 -> full at the top).
        # A stage change re-runs the align to the scaled per-env droop and re-snapshots;
        # callers always reset() after set_stage(), which restores from the new snapshot.
        scale = self.stage / max(1, self.num_stages - 1)
        if np.any(np.abs(self.cable_tilt) > 1e-9) and abs(scale - self._tilt_scale_cur) > 1e-9:
            # re-align from the last snapshot (a mid-rollout state may have the plug in the
            # jack), with all jacks PARKED so the re-drape can't catch on them
            self.state_0.body_q.assign(self._snap_bq)
            self.state_0.body_qd.assign(self._snap_bqd)
            self.state_1.body_q.assign(self._snap_bq)
            self.state_1.body_qd.assign(self._snap_bqd)
            self.jq = self._snap_jq.copy()
            for i in range(self.n):
                qs = self.jack_qs[i]
                self.jq[qs:qs + 3] = [0.0, 0.0, -5.0 - 0.5 * i]
                self.jq[qs + 3:qs + 7] = [0.0, 0.0, 0.0, 1.0]
            errs, _ = self._align_fn(-self.cable_tilt * scale)
            self._tilt_scale_cur = scale
            self._take_snapshot()
            td = np.degrees(self.cable_tilt * scale)
            print(f"[cable_env] stage {self.stage}: droop retilted to "
                  f"[{td.min():.1f}, {td.max():.1f}] deg "
                  f"(residual {math.degrees(max(errs)):.2f} deg)", flush=True)

    # ── reset / step ──────────────────────────────────────────────────────────
    def reset(self):
        n = self.n
        # restore the settled snapshot exactly (bodies + joints); no eval_ik anywhere
        self.state_0.body_q.assign(self._snap_bq)
        self.state_0.body_qd.assign(self._snap_bqd)
        self.state_1.body_q.assign(self._snap_bq)
        self.state_1.body_qd.assign(self._snap_bqd)
        self.jq = self._snap_jq.copy()
        # per-episode jack placement: seat = settled face + APPROACH*ins + lateral offset
        ang = self._rng.uniform(0, 2 * np.pi, n)
        mag = self._rng.uniform(0, self._mag, n)
        # curriculum approach: the jack sits closer at early stages (near-seated bootstrap);
        # the standoff room shrinks by the same amount the mouth moves closer.
        self.front_room_ep = self.front_room - (APPROACH - self._approach)
        for i in range(n):
            off = mag[i] * (math.cos(ang[i]) * self.frame[i]["jaw"]
                            + math.sin(ang[i]) * self.frame[i]["tool"])
            # seat from the LEVEL reference face (== face0 in the level env): the tilted
            # variant keeps the level env's jack geometry, only the start hang differs
            self.seat_pos[i] = self.face_ref[i] + self._approach * self.ins[i] + off
            Rs = Rot.from_quat(self.seat_qw[i])            # LEVEL world frame, not the sagged hang
            jack_p = self.seat_pos[i] + Rs.apply(self._off_local)
            qs = self.jack_qs[i]
            self.jq[qs:qs + 3] = jack_p
            self.jq[qs + 3:qs + 7] = self.seat_qw[i]
        self.seat_q = self.seat_qw.copy()
        # NO solver rebuild here: a fresh SolverVBD initializes cable joints from model.joint_q,
        # which still holds the BUILD-time (straight) rod coords while body_q is draped -> the new
        # solver yanks the rod toward the stale config and scrambles it. eval_fk/eval_ik cannot
        # resync CABLE joints (VBD owns them), so we keep the original solver: the snapshot
        # restore returns to a state it has already integrated through.
        for _ in range(5):
            self._sim()
        bqn = self.state_0.body_q.numpy()
        self.wrist_tgt_p = bqn[self.wrist_body, :3].copy()
        self.wrist_tgt_q = bqn[self.wrist_body, 3:7].copy()
        self.prev_lat[:] = 0; self.advanced[:] = 0; self.hold[:] = 0
        self.docked = np.zeros(n, dtype=bool)
        face, faceq = self._face_pose(bqn)
        self.face_start = face.copy()
        self.prev_dist = self._dist(face, faceq)
        return self._obs(bqn)

    def _terms(self, face, faceq):
        e = face - self.seat_pos
        along = np.sum(e * self.ins, axis=1)
        lat = e - along[:, None] * self.ins
        latn = np.linalg.norm(lat, axis=1)
        ang = (Rot.from_quat(faceq) * Rot.from_quat(self.seat_q).inv()).magnitude()
        return along, lat, latn, ang

    def _dist(self, face, faceq):
        along, _, latn, ang = self._terms(face, faceq)
        return np.sqrt(LAT_WEIGHT ** 2 * latn ** 2 + along ** 2) + ROT_W * ang

    def step(self, action):
        a = np.asarray(action.detach().cpu() if hasattr(action, "detach") else action,
                       dtype=np.float64)
        a = np.clip(a, -1.0, 1.0)
        bqn = self.state_0.body_q.numpy()
        face, faceq = self._face_pose(bqn)
        along, lat, latn, ang = self._terms(face, faceq)
        # PURE RL: the action IS the gripper motion (user spec — no scripted base in the loop).
        # The only env-side modification is the finger<->jack STANDOFF clamp: never advance the
        # finger front past the jack mouth minus MARGIN. That is a physical constraint (a real
        # gripper cannot cross the panel), not a controller; retreating is always free.
        dpos = a[:, 0:3] * MAX_DPOS
        d_along = np.sum(dpos * self.ins, axis=1)
        room = np.maximum(0.0, (self.front_room_ep - MARGIN) - self.advanced)
        excess = np.maximum(0.0, d_along - room)
        dpos = dpos - excess[:, None] * self.ins
        self.advanced += np.sum(dpos * self.ins, axis=1)
        self.wrist_tgt_p = self.wrist_tgt_p + dpos
        drot = a[:, 3:6] * MAX_DROT
        self.wrist_tgt_q = (Rot.from_rotvec(drot) * Rot.from_quat(self.wrist_tgt_q)).as_quat()
        self.jq, _, _ = self.ik.solve(self.jq, self.wrist_tgt_p, self.wrist_tgt_q,
                                      iters=self.ik_iters)
        self._sim()

        bqn = self.state_0.body_q.numpy()
        bqd = self.state_0.body_qd.numpy()
        face, faceq = self._face_pose(bqn)
        along, lat, latn, ang = self._terms(face, faceq)
        d = self._dist(face, faceq)
        rew = W_PROG * (self.prev_dist - d)
        seated = (along >= -SEAT_DEPTH_TOL) & (latn <= SEAT_OFFSET) & (ang <= SEAT_ANGLE)
        speed = (np.linalg.norm(bqd[self.rod_front, 0:3], axis=1)
                 + np.linalg.norm(bqd[self.rod_front, 3:6], axis=1))
        rew = rew + np.where(seated, R_SUCCESS - W_VEL * speed, 0.0)
        viol = self._violations()
        self.viol_last = viol
        rew = rew - W_VIOL * (viol > 0)
        self.hold = np.where(seated, self.hold + 1, 0)
        success = (self.hold >= HOLD_STEPS).astype(np.float32)
        dist_seat = np.linalg.norm(face - self.seat_pos, axis=1)
        bad = (~np.isfinite(face).all(axis=1)) | (dist_seat > EJECT_DIST)
        rew = np.where(bad, -1.0, rew)
        success = np.where(bad, 0.0, success)
        d = np.where(bad, self.prev_dist, d)
        self.prev_dist = d
        self.seated_inst_frac = float(np.mean(np.where(bad, False, seated)))
        depth_mm = np.clip(np.linalg.norm(face - self.face_start, axis=1), 0, 0.2) * 1000.0
        obs = self._obs(bqn)
        z = torch.zeros(self.n, device=DEV)
        to_t = lambda x: torch.as_tensor(np.nan_to_num(x), dtype=torch.float32, device=DEV)
        return obs, to_t(rew), z, to_t(success), to_t(depth_mm)

    def _obs(self, bqn):
        bqd = self.state_0.body_qd.numpy()
        eefp = bqn[self.wrist_body, :3]
        R6 = Rot.from_quat(bqn[self.wrist_body, 3:7]).as_matrix()[:, :, :2].reshape(self.n, 6)
        grip = np.full((self.n, 1), 1.0)
        face, faceq = self._face_pose(bqn)
        e = (face - self.seat_pos) * 50.0
        lin = bqd[self.rod_front, 0:3]
        q_err = (Rot.from_quat(self.seat_q).inv() * Rot.from_quat(faceq)).as_rotvec() * 3.0
        angv = bqd[self.rod_front, 3:6]
        obs = np.concatenate([eefp, R6, grip, e, lin, q_err, angv], axis=1)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clip(-50, 50)
        return torch.as_tensor(obs, dtype=torch.float32, device=DEV)

    def contact_count(self):
        return (int(self.contacts.rigid_contact_count.numpy()[0]),
                int(self.contacts.rigid_contact_force.shape[0]))

    def servo_action(self):
        """The scripted funnel-servo TEACHER, expressed as a pure action in [-1,1].

        The env itself is pure-RL (step() executes the action verbatim); this exists only for
        demos, eval baselines, and BC warm-start data. Lateral PD toward the seat line, advance
        gated on lateral+angle, hold-regulate ~2mm compression once docked.
        """
        bqn = self.state_0.body_q.numpy()
        face, faceq = self._face_pose(bqn)
        along, lat, latn, ang = self._terms(face, faceq)
        # orientation: close the rotation loop on the WRIST, never on the hanging face — a
        # face-angle servo chases the free span's pendulum with lag and PUMPS it (measured
        # divergence 7 -> 36 deg at stage 0 of the tilted variant). The wrist target is the
        # LEVEL-align pose captured at build; the grip anchor carries the cable with it, the
        # free span follows quasi-statically. In the level env rv ~ 0 and this is a no-op.
        rv = (Rot.from_quat(self.wq_level) * Rot.from_quat(self.wrist_tgt_q).inv()).as_rotvec()
        leveling = np.linalg.norm(rv, axis=1) > math.radians(1.0)
        corr = -KP * lat - KD * (lat - self.prev_lat)
        cn = np.linalg.norm(corr, axis=1, keepdims=True)
        corr = np.where(cn > LAT_CAP, corr * (LAT_CAP / np.maximum(cn, 1e-12)), corr)
        # damp the lateral pull while still leveling: a full-cap vertical yank on the grip
        # swings the dangling connector like a pendulum right when precision is coming up
        corr = np.where(leveling[:, None], 0.4 * corr, corr)
        self.prev_lat = lat.copy()
        ang_ok = (ang < math.radians(6.0)) & ~leveling
        adv = np.where((latn < GATE) & ang_ok & (along < -0.001), ADV, 0.0)
        docked = along >= -(SEAT_DEPTH_TOL - 0.0005)
        # hold-regulate only when ALIGNED: pushing a crooked plug at the mouth buckles the
        # cable (measured runaway 6 -> 21 deg). Crooked at the mouth -> back off gently.
        adv = np.where(docked & ang_ok, np.clip(0.3 * (-0.002 - along), 0.0, 0.5 * ADV), adv)
        adv = np.where(docked & ~ang_ok, -0.5 * ADV, adv)
        dpos = adv[:, None] * self.ins + np.where(docked[:, None], 0.5, 1.0) * corr
        act = np.zeros((self.n, ACT_DIM))
        act[:, 0:3] = np.clip(dpos / MAX_DPOS, -1.0, 1.0)
        act[:, 3:6] = np.clip(rv * (self.servo_rot_gain / MAX_DROT), -1.0, 1.0)
        return torch.as_tensor(act, dtype=torch.float32, device=DEV)
