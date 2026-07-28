"""Batched cable-insertion env, RIGID-cable variant: the loose cable is a SOLID rod.

Differences vs cable_env.py (user request 2026-07-21, "make it non deformable; just a solid
cable" + "the plug's head should remain as is"):
  * the add_rod capsule chain + grip-anchor + boot-joint machinery is replaced by ONE
    DYNAMIC rigid body per env: a 6.5mm capsule (the cable) + the UNCHANGED cad_rj45 plug
    mesh at its tip (same contact config, same face-origin frames)
  * the body is held by the PHYSICAL AG-145 friction grasp (fingers squeeze the capsule) —
    chosen over welding so plug<->jack KEEPS full contact physics (chamfer guidance, seating
    resistance); the grasp can slip under enough torque, which is real
  * tilt variant = the same align machinery, now leveling/pitching a rigid in-hand grasp
    angle instead of gravity droop (no sag, no pendulum, no buckling)
Everything else (frames, curriculum incl. tilt scaling, reward, standoff clamp, violation
counting, servo teacher, snapshot resets, obs/action) is identical to cable_env.py.

Interface (identical to CableInsertVecEnv):
    env = RigidCableVecEnv(n, cable_tilt_deg=8.0); env.set_stage(0)
    obs = env.reset()
    obs, rew, done, success, depth_mm = env.step(action)   # action (n,7) in [-1,1]
Action = [dpos(3), drotvec(3), gripper(1)] pure gripper motion (pi05 rj45_sbot).
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
THETA_CABLE = -0.0152       # jaw angle for the 6.5 mm cable, PHYSICAL friction grasp.
#                             MEASURED pad gap (FK, 2026-07-21): -0.008 -> 4.46mm (the
#                             deformable env could use that because its jaw segments were
#                             non-collidable anchors; a COLLIDABLE rigid capsule explodes at
#                             2mm crush), -0.016 -> 6.55mm; linear ~0.26mm/0.001.
#                             -0.0152 -> ~6.35mm = 0.15mm squeeze on the 6.5mm capsule.
ARM_PITCH_DEG = 45.0        # ARM approach angle (user's GS-scene reference): the WRIST/fingers
#                             are pitched 45 deg down-forward about the jaw axis, while the cable
#                             + connector stay HORIZONTAL in the jaws and the jack face stays
#                             perpendicular to the ground. The 45 lives in the robot's pose, the
#                             task geometry stays level — no post-hoc cable/wrist compensation.
GRIP_BACK = 0.050           # m of cable between the grip and the capsule front end
CABLE_BACK = 0.060          # m of cable sticking out behind the grip
GRIP_AT = 0.1975            # m wrist->grip along tool: the AG-145 pads' converging FLAT is
#                             at d=0.195-0.200 (measured gap profile; 6.34mm there at theta
#                             -0.0152 = 0.16mm squeeze). The deformable env's 0.192 sits in
#                             an 8-10mm-wide region — a rigid capsule there is NOT gripped.
BOOT = 0.018                # m boot gap: plug FACE this far ahead of the capsule front
PAD_INSET = 0.0007          # m each pad box is authored INSIDE the measured pad face: the
#                             0.15mm face-gap squeeze leaves only ~0.075mm/side penetration,
#                             and AVBD's ramped penalty at that depth gives ~0.01N of friction
#                             vs the 0.06N cable weight -> the cable slides out (smoke 13:
#                             611m drift, 1.3 deg rot slip = clean slip-and-fall). Insetting
#                             the boxes 0.7mm/side gives ~1.5mm total squeeze, the arm env's
#                             validated grasp depth, without moving the (visual) finger pose.
APPROACH = 0.030            # m the face travels from settle to seat
SEAT_AIM_DY = 0.012         # face ends this far past the jack mouth (cavity depth aim)

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
def _sync_pad_anchors(body_q: wp.array(dtype=wp.transform),
                      body_qd: wp.array(dtype=wp.spatial_vector),
                      parent_idx: wp.array(dtype=int), anchor_idx: wp.array(dtype=int),
                      off_p: wp.array(dtype=wp.vec3), off_q: wp.array(dtype=wp.quat)):
    """Carry the kinematic PAD PROXY bodies with the wrist (wrist ∘ fixed offset each
    substep) — the same pattern as the deformable env's grip anchors and the jack."""
    tid = wp.tid()
    par = body_q[parent_idx[tid]]
    pp = wp.transform_get_translation(par)
    pr = wp.transform_get_rotation(par)
    aw = pp + wp.quat_rotate(pr, off_p[tid])
    ar = wp.normalize(wp.mul(pr, off_q[tid]))
    body_q[anchor_idx[tid]] = wp.transform(aw, ar)
    body_qd[anchor_idx[tid]] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class RigidCableVecEnv:
    def __init__(self, n: int, *, seed: int = 0, socket_mu: float = 0.5,
                 residual_scale: float = RESIDUAL_SCALE, ik_iters: int = 2,
                 contact_buffer_per: int = 1024, cable_tilt_deg: float = 0.0,
                 cable_mesh: bool = False, _dbg_shift: float = 0.0,
                 _dbg_no_pad_collide: bool = False, _dbg_skip_align: bool = False):
        # cable_mesh: use a trimesh cylinder (SDF mesh-mesh contact, like the validated plug
        # grasp) instead of a capsule primitive. _dbg_shift: debug-only — author the cable
        # this far (+x) away from the jaws and skip the teleport (isolation experiments).
        # _dbg_no_pad_collide: debug-only — pad boxes don't collide (cable falls free).
        self._cable_mesh = cable_mesh
        self._dbg_shift = _dbg_shift
        self._dbg_no_pad_collide = _dbg_no_pad_collide
        self._dbg_skip_align = _dbg_skip_align
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
        # rigid_gap 0.005 (deformable env value) wraps every rigid shape in a 5mm contact
        # shell — INSIDE the hand that generates shell contacts at +4..11mm "separation" on
        # curved tip/knuckle surfaces with wild normals, torquing the light cable body.
        # 1mm still covers the ~0.25mm/substep max relative motion (no tunneling).
        b = new_vbd_builder(gravity=-9.81); b.rigid_gap = 0.001
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
        # TWO-PASS AUTHORING: solve the ARM-PITCH pose on this throwaway model, then author
        # the cable at the jaws' ACTUAL post-IK pose. (The first build authored an analytic
        # frame and TELEPORTED the body onto the real jaws after the solver existed — VBD
        # reads that rewrite as a ~30deg-in-one-substep motion and injects ~370 rad/s spin.
        # Authoring at the true pose means NO teleport, no injection.)
        fk_qs = fk_m.joint_q_start.numpy()
        fk_arm_qc = np.array([[fk_qs[arm_j[nm][i]] for nm in ARM_JOINT_NAMES]
                              for i in range(n)])
        fk_ik = ArmIK(fk_m, fk_arm_qc, np.array(wristsF))
        wq_t = np.zeros((n, 4)); wp_t = bq[wristsF, :3].copy()
        for i in range(n):
            # ARM approach angle, ABSOLUTE from vertical (0 = fingers straight down, 45 =
            # the GS-scene diagonal); rotate the HOME wrist so its tool hits the target.
            wpos = bq[wristsF[i]][:3]
            tipm = 0.5 * (bq[tip1[i]][:3] + bq[tip2[i]][:3])
            tool = tipm - wpos; tool /= np.linalg.norm(tool)
            fwd_h = tool.copy(); fwd_h[2] = 0.0; fwd_h /= np.linalg.norm(fwd_h)
            sp_ = math.sin(math.radians(ARM_PITCH_DEG))
            cpv = math.cos(math.radians(ARM_PITCH_DEG))
            tool_t = np.array([0.0, 0.0, -1.0]) * cpv + fwd_h * sp_
            ax = np.cross(tool, tool_t); s_ = np.linalg.norm(ax)
            Rc = (Rot.from_rotvec(ax / s_ * math.atan2(s_, float(np.dot(tool, tool_t))))
                  if s_ > 1e-9 else Rot.identity())
            wq_t[i] = (Rc * Rot.from_quat(bq[wristsF[i], 3:7])).as_quat()
        fk_jq = fk_m.joint_q.numpy().copy()
        fk_jq, ik_ep, _ = fk_ik.solve(fk_jq, wp_t, wq_t, iters=100)
        if ik_ep.max() > 2e-3:
            print(f"[rigid_cable_env] WARNING: arm-pitch IK residual {ik_ep.max()*1000:.1f}mm")
        self._arm_pitch_q = np.array([[fk_jq[fk_qs[arm_j[nm][i]]] for nm in ARM_JOINT_NAMES]
                                      for i in range(n)])
        jqw_fk = wp.array(fk_jq, dtype=float, device=fk_m.device)
        newton.eval_fk(fk_m, jqw_fk, fk_m.joint_qd, fk_s)
        bq = fk_s.body_q.numpy()   # ACTUAL approach-pose frames; author from these below

        spec = cad_rj45_connector(friction=socket_mu)
        meshes = load_connector_meshes(spec)
        sbp = np.asarray(meshes.socket.base_position)
        pbp = np.asarray(meshes.plug.base_position)
        self._off_local = sbp - (pbp + np.array([0.0, SEAT_AIM_DY, 0.0]))  # plug-origin -> socket-origin
        plug_cfg = newton.ModelBuilder.ShapeConfig(mu=2.0, ke=spec.contact.stiffness, kd=0.0,
                                                   gap=spec.contact.gap_meters, density=1500.0)
        # kd MUST be 0 like plug_cfg: under Newton 1.4 damping semantics, a nonzero kd acts
        # through SDF contacts that are generated at +4..11mm SEPARATION (measured) — random-
        # normal impulses on the light body -> the deterministic ~400 rad/s spin injection
        cable_cfg = dataclasses.replace(b.default_shape_cfg, ke=spec.contact.stiffness,
                                        kd=0.0, mu=spec.cable_friction)

        self.rod_front = []          # cable body per env (carries the connector)
        self.rod_bodies_all = []     # kept as 1-lists for interface parity with the rod env
        self._pad_bodies = []        # (env, body, authored pos, authored quat) per pad proxy
        self.jack_body = []
        self.plug_local = []         # (pos, quat) of the plug FACE in the cable-body frame
        self.frame = []              # per-env dict: tool/jaw/n/grip_pos
        jack_shape_of_env = {}
        finger_env_of_shape = {}
        # finger bodies per env, needed inside the loop for the capsule<->knuckle filters
        finger_bodies_per_env = {i: set() for i in range(n)}
        for nm in FINGER_LINK_BODIES:
            for i, bidx in enumerate(_all_idx(list(b.body_label), nm)):
                finger_bodies_per_env[i].add(bidx)
        def _tip_pts(bidx, bq_arr):
            pts = []
            for sh in b.body_shapes[bidx]:
                src = b.shape_source[sh]
                if src is None or b.shape_type[sh] != int(newton.GeoType.MESH):
                    continue
                v = np.asarray(src.vertices)
                stx = b.shape_transform[sh]
                Rs = Rot.from_quat([stx.q[0], stx.q[1], stx.q[2], stx.q[3]]).as_matrix()
                vb = (Rs @ v.T).T + np.array([stx.p[0], stx.p[1], stx.p[2]])
                Rb = Rot.from_quat(bq_arr[bidx][3:7]).as_matrix()
                pts.append((Rb @ vb.T).T + bq_arr[bidx][:3])
            return np.vstack(pts)

        for i in range(n):
            # frames measured at the SOLVED approach pose (no analytic re-derivation)
            wpos = bq[wristsF[i]][:3]; wq = bq[wristsF[i]][3:7]
            tipm = 0.5 * (bq[tip1[i]][:3] + bq[tip2[i]][:3])
            Rw = Rot.from_quat(wq).as_matrix()
            tool = tipm - wpos; tool /= np.linalg.norm(tool)
            jaw = Rw @ np.array([0., 1., 0.]); jaw -= tool * (jaw @ tool); jaw /= np.linalg.norm(jaw)
            nfw = np.cross(tool, jaw); nfw /= np.linalg.norm(nfw)
            # grip point = MEASURED midpoint between the two pads' inner faces (fixed-tool-
            # line placement left the capsule 1.6-3.2mm off the pads — not gripped at all).
            # Pad region: verts beyond GRIP_AT-0.02 along tool; inner face: the 1mm of verts
            # facing the other finger. The cable keeps the TRUE jaw-frame nfw (no z-flatten:
            # flattening drifts the capsule off the pad flats); the align loop levels it.
            v1 = _tip_pts(tip1[i], bq); v2 = _tip_pts(tip2[i], bq)
            j1 = (v1 - wpos) @ jaw; j2 = (v2 - wpos) @ jaw
            t1 = (v1 - wpos) @ tool; t2 = (v2 - wpos) @ tool
            m1 = t1 > GRIP_AT - 0.02; m2 = t2 > GRIP_AT - 0.02
            if np.median(j1[m1]) > np.median(j2[m2]):
                f1 = v1[m1][j1[m1] < j1[m1].min() + 0.001]
                f2 = v2[m2][j2[m2] > j2[m2].max() - 0.001]
            else:
                f1 = v1[m1][j1[m1] > j1[m1].max() - 0.001]
                f2 = v2[m2][j2[m2] < j2[m2].min() + 0.001]
            pad_gap = abs(float((f1.mean(axis=0) - f2.mean(axis=0)) @ jaw))
            if i == 0:
                print(f"[rigid_cable_env] measured pad gap {pad_gap*1000:.2f}mm "
                      f"(cable {2*spec.cable_radius_meters*1000:.1f}mm)", flush=True)
            mid = 0.5 * (f1.mean(axis=0) + f2.mean(axis=0))
            grip_pos = mid + np.array([self._dbg_shift, 0.0, 0.0])
            self.frame.append(dict(tool=tool, jaw=jaw, n=nfw, grip=grip_pos, tipm=tipm))
            # PAD PROXIES: standalone kinematic free bodies (jack-like) with primitive box
            # shapes at the measured pad faces, carried by the wrist via _sync_pad_anchors.
            # This is the ONLY grasp configuration that is stable: every variant with the
            # cable contacting ARTICULATION-member bodies exploded (capsule/mesh vs finger
            # SDF meshes AND box pads mounted on the fingertip links), while the identical
            # box-vs-capsule squeeze against standalone kinematic bodies is rock stable —
            # same pattern as the jack, which has been stable for weeks.
            # kd MUST be 0 here too (builder default kd=100!): the nonzero-kd separation-
            # contact damping is the proven ~400 rad/s injector (see cable_cfg note above),
            # and it acts on whichever SIDE of the pair carries it — pads included.
            pad_cfg = dataclasses.replace(b.default_shape_cfg, mu=2.0,
                                          ke=spec.contact.stiffness, kd=0.0)
            for k_, fc in ((0, f1.mean(axis=0)), (1, f2.mean(axis=0))):
                outj = jaw * np.sign(float((fc - mid) @ jaw))
                c_box = fc + (0.005 - PAD_INSET) * outj   # inner surface INSET past the face
                qb_ = Rot.from_matrix(np.column_stack([jaw, nfw, tool])).as_quat()
                pb = b.add_body(xform=wp.transform(wp.vec3(*c_box), wp.quat(*qb_)),
                                label=f"pad{i}_{k_}")
                sbox = b.add_shape_box(pb, hx=0.005, hy=0.008, hz=0.010, cfg=pad_cfg)
                if self._dbg_no_pad_collide:
                    b.shape_flags[sbox] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
                b.add_joint_free(child=pb)
                b.body_flags[pb] = int(newton.BodyFlags.KINEMATIC)
                finger_env_of_shape[sbox] = i
                self._pad_bodies.append((i, pb, c_box.copy(), qb_.copy()))
            # ONE dynamic rigid body: cable capsule + the UNCHANGED plug mesh. Body frame =
            # plug frame (x = jaw/width, y = insertion/forward, z = height), origin at the
            # grip point. Added AFTER the kinematic-flag loop -> stays DYNAMIC: it is held by
            # the physical finger friction grasp, and plug<->jack keeps full contact physics.
            z_p = np.cross(jaw, nfw); z_p /= np.linalg.norm(z_p)
            # ROLL-FREE body frame: the jaw axis is ~33 deg off horizontal at this IK pose,
            # so [jaw, nfw, jaw x nfw] hangs the plug ROLLED 33 deg about the cable axis —
            # the world-level seat is unreachable (the teacher can only move the wrist, and
            # with the grip box locking roll the error is permanent; measured ang 32.6 deg
            # at t=0). Author the plug UP = world up (projected +nfw-orthogonal); the grip
            # box below gets the opposite local roll so its flats still meet the pad faces.
            up_ = np.array([0.0, 0.0, 1.0])
            z_v = up_ - float(up_ @ nfw) * nfw; z_v /= np.linalg.norm(z_v)
            jaw_v = np.cross(nfw, z_v)
            Rp = np.column_stack([jaw_v, nfw, z_v]); q_plug = Rot.from_matrix(Rp).as_quat()
            R_grip = np.column_stack([jaw, nfw, z_p])
            q_gb = Rot.from_matrix(Rp.T @ R_grip).as_quat()   # grip-box local: pad-aligned
            body = b.add_body(xform=wp.transform(wp.vec3(*grip_pos), wp.quat(*q_plug)),
                              label=f"cable{i}")
            # a FREE joint is REQUIRED for a dynamic body in this stack (the arm env's plug
            # has one too): without it the body is not integrated properly and explodes even
            # contact-free. eval_fk skips it via the KINEMATIC filter; VBD owns its motion.
            b.add_joint_free(child=body)
            # capsule extends along local Z -> rotate onto the body's +y (forward) axis;
            # front end stops half a boot short of the plug so it can't touch the jack mouth
            cap_lo, cap_hi = -CABLE_BACK, GRIP_BACK + 0.5 * BOOT
            if self._cable_mesh:
                import trimesh
                cyl = trimesh.creation.cylinder(radius=spec.cable_radius_meters,
                                                height=cap_hi - cap_lo, sections=24)
                cyl.apply_transform(
                    trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
                cyl.apply_translation([0.0, 0.5 * (cap_lo + cap_hi), 0.0])
                cap_shape = b.add_shape_mesh(
                    body, mesh=newton.Mesh(np.asarray(cyl.vertices),
                                           np.asarray(cyl.faces).flatten()),
                    cfg=cable_cfg)
            else:
                cap_shape = b.add_shape_capsule(
                    body,
                    xform=wp.transform(
                        wp.vec3(0.0, 0.5 * (cap_lo + cap_hi), 0.0),
                        wp.quat_from_axis_angle(wp.vec3(1., 0., 0.), math.pi / 2)),
                    radius=spec.cable_radius_meters,
                    half_height=0.5 * (cap_hi - cap_lo) - spec.cable_radius_meters,
                    cfg=cable_cfg)
            # GRIP BOX: a smooth capsule between two flat pads has NO roll lock (line
            # contact, ~zero torque arm) — the dangling plug slowly rolls/yaws away
            # (measured: 6.6 deg/4s slip, ~70 deg by the end of the align). The real
            # boot/cable flattens under squeeze and locks roll; proxy it with a small box
            # at the grip, slightly PROUD of the capsule radius so the pads land on its
            # flats first. Same contact cfg as the cable.
            gb = b.add_shape_box(body, hx=0.0033, hy=0.008, hz=0.0033,
                                 xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(*q_gb)),
                                 cfg=cable_cfg)
            if self._dbg_no_pad_collide:
                b.shape_flags[gb] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
            # ALL finger MESH shapes leave collision (SDF mesh contact vs the cable body is
            # the proven injector; the box pads above are the physical grip + jack sensor)
            for s in all_finger_shapes:
                if b.shape_body[s] in finger_bodies_per_env[i]:
                    b.shape_flags[s] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
            loc_p = np.array([0.0, GRIP_BACK + BOOT, 0.0])   # plug FACE in the body frame
            loc_q = np.array([0.0, 0.0, 0.0, 1.0])
            self.plug_local.append((loc_p, loc_q))
            ps_ = b.add_shape_mesh(body, mesh=meshes.plug.mesh,
                                   xform=wp.transform(wp.vec3(*loc_p), wp.quat(*loc_q)),
                                   cfg=plug_cfg)
            if self._dbg_no_pad_collide:   # truly contact-free body: cable + plug off too
                b.shape_flags[cap_shape] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
                b.shape_flags[ps_] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
            self.rod_bodies_all.append([body])
            self.rod_front.append(body)
            # kinematic jack (parked far below; placed per-episode at reset)
            jb = b.add_body(xform=wp.transform(wp.vec3(0., 0., -5. - 0.5 * i), wp.quat_identity()),
                            label=f"jack{i}")
            js = b.add_shape_mesh(jb, mesh=meshes.socket.mesh, cfg=connector_shape_config(spec))
            b.add_joint_free(child=jb)
            b.body_flags[jb] = int(newton.BodyFlags.KINEMATIC)
            self.jack_body.append(jb)
            jack_shape_of_env[js] = i

        # map finger shapes -> env by body ownership (arm order)
        for s in all_finger_shapes:
            for i in range(n):
                if b.shape_body[s] in finger_bodies_per_env[i]:
                    finger_env_of_shape[s] = i; break

        self.model = finalize_for_vbd(b)
        self.device = self.model.device
        _bm = self.model.body_mass.numpy(); _bi = self.model.body_inertia.numpy()
        _cb = self.rod_front[0]
        print(f"[rigid_cable_env] cable body mass {_bm[_cb]*1000:.1f}g | inertia diag "
              f"{np.diag(_bi[_cb])} kg m^2", flush=True)
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
        # NOTE: the solver is constructed BELOW, after the cable teleport + free-joint sync
        self.ik = ArmIK(self.model, self.arm_qc, self.wrist_body)
        self.jq = self.model.joint_q.numpy().copy()
        # ALL free joints of each cable body (add_body leaves an implicit one + our explicit
        # add_joint_free = TWO, same duplicate quirk as the jack) — both must be synced
        self._cable_fj_qs = {int(bd): [] for bd in self.rod_front}
        for j in range(self.model.joint_count):
            c = int(jchild[j])
            if int(jtype[j]) == int(newton.JointType.FREE) and c in self._cable_fj_qs:
                self._cable_fj_qs[c].append(int(qstart[j]))
        self.dt = 1.0 / 60.0 / SUBSTEPS
        self._pll = (np.array([p for p, _ in self.plug_local]),
                     np.array([q for _, q in self.plug_local]))
        self.rod_front = np.array(self.rod_front)

        # pose the arm at home once (the free cable body keeps its build transform — eval_fk
        # only touches KINEMATIC bodies), then IK the wrist to the authored approach pose so
        # the jaws arrive exactly where the cable body was authored.
        jqw0 = wp.array(self.jq, dtype=float, device=self.device)
        newton.eval_fk(self.model, jqw0, self.model.joint_qd, self.state_0,
                       body_flag_filter=int(newton.BodyFlags.KINEMATIC))
        bq0 = self.state_0.body_q.numpy()
        # apply the SAME arm-pitch joint solution to the real model (identical arm-joint
        # layout as the throwaway model; the cable was AUTHORED at the resulting jaw pose,
        # so there is NO teleport — the solver never sees a body_q rewrite)
        for i in range(n):
            self.jq[self.arm_qc[i]] = self._arm_pitch_q[i]
        jqw0 = wp.array(self.jq, dtype=float, device=self.device)
        newton.eval_fk(self.model, jqw0, self.model.joint_qd, self.state_0,
                       body_flag_filter=int(newton.BodyFlags.KINEMATIC))
        bq0 = self.state_0.body_q.numpy()
        tips1 = _all_idx(ML, "gripper_finger1_finger_tip_link")
        tips2 = _all_idx(ML, "gripper_finger2_finger_tip_link")
        gp_err = max(
            np.linalg.norm(0.5 * (bq0[tips1[i], :3] + bq0[tips2[i], :3])
                           - self.frame[i]["tipm"]) for i in range(n))
        if gp_err > 1e-3 and not self._dbg_shift:
            print(f"[rigid_cable_env] WARNING: real-vs-throwaway jaw mismatch "
                  f"{gp_err*1000:.2f}mm")
        # sync BOTH free joints of each cable body (add_body implicit + explicit) to the
        # authored pose, THEN construct the solver: VBD captures joint/body state at
        # construction, and stale coords there were an earlier ~400 rad/s injector.
        for i in range(n):
            bd = int(self.rod_front[i])
            for qs in self._cable_fj_qs[bd]:
                self.jq[qs:qs + 3] = bq0[bd, :3]
                self.jq[qs + 3:qs + 7] = bq0[bd, 3:7]
        # pad proxies: sync their free joints + state rows to the AUTHORED poses (they are
        # eval_fk-posed from joint coords like the jack), and record wrist-frame offsets for
        # the per-substep carry kernel
        pad_par, pad_anc, pad_op, pad_oq = [], [], [], []
        for i, pb, pp_, pq_ in self._pad_bodies:
            for j in range(self.model.joint_count):
                if (int(jtype[j]) == int(newton.JointType.FREE) and int(jchild[j]) == pb):
                    qs = int(qstart[j])
                    self.jq[qs:qs + 3] = pp_
                    self.jq[qs + 3:qs + 7] = pq_
            bq0[pb, :3] = pp_; bq0[pb, 3:7] = pq_
            Rw = Rot.from_quat(bq0[self.wrist_body[i], 3:7])
            wpn = bq0[self.wrist_body[i], :3]
            pad_par.append(int(self.wrist_body[i])); pad_anc.append(int(pb))
            pad_op.append(Rw.inv().apply(pp_ - wpn))
            pad_oq.append((Rw.inv() * Rot.from_quat(pq_)).as_quat())
        self._anc_par = wp.array(np.array(pad_par, dtype=np.int32), dtype=int, device=self.device)
        self._anc_idx = wp.array(np.array(pad_anc, dtype=np.int32), dtype=int, device=self.device)
        self._anc_p = wp.array(np.array(pad_op), dtype=wp.vec3, device=self.device)
        self._anc_q = wp.array(np.array(pad_oq), dtype=wp.quat, device=self.device)
        self._n_anchors = len(pad_anc)
        self.state_0.body_q.assign(bq0)
        self.state_1.body_q.assign(bq0)
        self.model.joint_q.assign(self.jq)
        # model.body_q must ALSO match: the solver clones body_q_prev from model.body_q at
        # construction and uses model.body_q/joint_q as the joint rest reference. Leaving the
        # arm at HOME rest there while the states hold the PITCHED pose gives every body a
        # phantom (pitch-delta/dt) first-step velocity in the BDF1 update.
        self.model.body_q.assign(bq0)
        self.solver = self._new_solver()
        # ── align + settle: rotate each WRIST until the grasped connector is LEVEL ────
        # The world fixes the jack (horizontal panel insertion); the ROBOT compensates any
        # in-hand grasp pitch AND any settle-slip of the friction grasp: iterate settle ->
        # measure the face pitch -> rotate the wrist by the residual (friction carries the
        # rigid body with the jaws). Same loop as the deformable env, converges faster here.
        rf = np.array(self.rod_front) if not isinstance(self.rod_front, np.ndarray) else self.rod_front
        wq_cur = bq0[self.wrist_body, 3:7].copy()
        wp_cur = bq0[self.wrist_body, :3].copy()
        def _align(target):
            """Settle, then pitch each wrist about its jaw axis until the hang axis's vertical
            angle asin(y_z) hits `target[i]` (damped iterations, jaw-axis only)."""
            errs = [0.0]
            for align_it in range(6):
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
                # RAMP the wrist to the IK solution instead of teleporting it: a ~19 deg
                # instantaneous wrist rotation moves the pads ~60mm in one substep (0.2m
                # lever) and the 0.8mm-deep friction grasp cannot follow. ~0.3 deg/frame,
                # per-substep interpolated, lets friction carry the cable.
                jq_t, _, _ = self.ik.solve(self.jq, wp_cur, wq_cur, iters=60)
                jq_from = self.jq.copy()
                RAMP = 60
                for k in range(RAMP):
                    prev = self.jq
                    self.jq = jq_from + (k + 1) / RAMP * (jq_t - jq_from)
                    self._sim(prev)
            return errs, align_it

        if self._dbg_skip_align:
            errs, align_it = [0.0], 0   # debug: snapshot the AUTHORED state, no settling
        else:
            errs, align_it = _align(np.zeros(n))
        print(f"[rigid_cable_env] hang leveled to {math.degrees(max(errs)):.2f} deg "
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

    def _sim(self, jq_prev=None):
        """One control frame (SUBSTEPS substeps). jq_prev: when given, the ARM joints are
        linearly interpolated jq_prev -> self.jq across the substeps instead of jumping in
        substep 1. The friction grasp is only ~0.8mm deep — a whole-frame wrist jump moves
        the (kinematic) pads several squeeze-depths in one substep and the cable is simply
        left behind (measured: t=0 hold is rock solid, the align teleport dropped it)."""
        if jq_prev is None:
            jqw = wp.array(self.jq, dtype=float, device=self.device)
        for s in range(SUBSTEPS):
            if jq_prev is not None:
                a = (s + 1) / SUBSTEPS
                jqw = wp.array(jq_prev + a * (self.jq - jq_prev), dtype=float,
                               device=self.device)
            newton.eval_fk(self.model, jqw, self.model.joint_qd, self.state_0,
                           body_flag_filter=int(newton.BodyFlags.KINEMATIC))
            wp.launch(_sync_pad_anchors, dim=self._n_anchors,
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

    def _sync_teleport(self):
        """MUST follow every state_0.body_q rewrite (snapshot restore): VBD's BDF1 velocity
        update reads (accepted - body_q_prev)/dt, so a dynamic-body teleport that skips
        body_q_prev is read as one-substep motion -> a (restore_delta/dt) velocity injection
        on the cable at the first step of EVERY episode. Kinematic bodies don't need it."""
        self.solver.body_q_prev.assign(self.state_0.body_q)

    def _realize_jq_teleport(self):
        """Realize a KINEMATIC teleport written into self.jq (jack placement/parking) NOW,
        then re-sync body_q_prev. Without this, the first _sim step computes the jack's
        kinematic velocity as (new_pose - old_pose)/dt: at stage 0 the jack lands within
        SDF contact range of the plug (~10mm approach vs +4..11mm band) with a 5m/substep
        'velocity' -> a huge friction kick on the plug at every reset (smoke 16: stage-0
        start-ang 147 deg while stage 4 with its farther spawn was perfect)."""
        jqw = wp.array(self.jq, dtype=float, device=self.device)
        newton.eval_fk(self.model, jqw, self.model.joint_qd, self.state_0,
                       body_flag_filter=int(newton.BodyFlags.KINEMATIC))
        wp.launch(_sync_pad_anchors, dim=self._n_anchors,
                  inputs=(self.state_0.body_q, self.state_0.body_qd,
                          self._anc_par, self._anc_idx, self._anc_p, self._anc_q),
                  device=self.device)
        self.state_1.body_q.assign(self.state_0.body_q)
        self._sync_teleport()

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
            self._sync_teleport()
            self.jq = self._snap_jq.copy()
            for i in range(self.n):
                qs = self.jack_qs[i]
                self.jq[qs:qs + 3] = [0.0, 0.0, -5.0 - 0.5 * i]
                self.jq[qs + 3:qs + 7] = [0.0, 0.0, 0.0, 1.0]
            self._realize_jq_teleport()
            errs, _ = self._align_fn(-self.cable_tilt * scale)
            self._tilt_scale_cur = scale
            self._take_snapshot()
            td = np.degrees(self.cable_tilt * scale)
            print(f"[rigid_cable_env] stage {self.stage}: droop retilted to "
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
        self._sync_teleport()
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
        self._realize_jq_teleport()   # land the jacks NOW, before any solver step sees them
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
        jq_prev = self.jq.copy()
        self.jq, _, _ = self.ik.solve(self.jq, self.wrist_tgt_p, self.wrist_tgt_q,
                                      iters=self.ik_iters)
        self._sim(jq_prev)   # per-substep interp: don't jump the pads a whole frame at once

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
