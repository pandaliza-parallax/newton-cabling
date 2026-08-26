"""Behaviour tests for the align-then-insert scripted controller.

The centrepiece is :class:`RigidGraspPlant` — an idealised stand-in for the
simulation in which the plug is welded to the wrist. It exercises the real code
path end to end (measure -> phase machine -> face command -> wrist action) with
no GPU, so a broken state machine or a sign error in the wrist conversion is
caught here rather than 30 s into a Newton build.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque

import numpy as np
import pytest

from newton_cabling.scripted_controller import (
    AlignInsertConfig,
    AlignInsertController,
    InsertionObs,
    InsertPhase,
)
from newton_cabling.scripted_controller.quaternion import (
    quat_conjugate,
    quat_multiply,
    quat_rotate,
    quat_to_rotvec,
    rotate_by_rotvec,
    rotvec_to_quat,
)

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


# ── quaternion helpers ────────────────────────────────────────────────────────
def test_rotvec_quaternion_round_trip() -> None:
    rng = np.random.default_rng(0)
    axis = rng.normal(size=(32, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = rng.uniform(-math.pi * 0.99, math.pi * 0.99, (32, 1))
    rotvec = axis * angle
    assert np.allclose(quat_to_rotvec(rotvec_to_quat(rotvec)), rotvec, atol=1e-9)


def test_rotvec_round_trip_survives_the_zero_rotation() -> None:
    zero = np.zeros((4, 3))
    assert np.allclose(rotvec_to_quat(zero), IDENTITY)
    assert np.allclose(quat_to_rotvec(np.tile(IDENTITY, (4, 1))), 0.0)


def test_quat_to_rotvec_takes_the_short_way_round() -> None:
    """A negated quaternion is the same rotation and must give the same log map."""
    q = rotvec_to_quat(np.array([[0.0, 0.0, math.radians(1.0)]]))
    assert np.allclose(quat_to_rotvec(-q), quat_to_rotvec(q), atol=1e-12)


def test_quat_multiply_composes_right_to_left() -> None:
    a = rotvec_to_quat(np.array([[0.0, 0.0, math.pi / 2]]))
    b = rotvec_to_quat(np.array([[math.pi / 2, 0.0, 0.0]]))
    x = np.array([[0.0, 0.0, 1.0]])
    assert np.allclose(quat_rotate(quat_multiply(a, b), x), quat_rotate(a, quat_rotate(b, x)))


def test_rotate_by_rotvec_matches_quaternion_rotation() -> None:
    rng = np.random.default_rng(1)
    rotvec = rng.normal(scale=0.4, size=(16, 3))
    x = rng.normal(size=(16, 3))
    assert np.allclose(rotate_by_rotvec(rotvec, x), quat_rotate(rotvec_to_quat(rotvec), x))


# ── config ────────────────────────────────────────────────────────────────────
def test_pre_dock_pose_sits_outside_the_jack_mouth() -> None:
    c = AlignInsertConfig(mouth_depth_m=0.012, align_standoff_m=0.005)
    assert c.mouth_along_m == pytest.approx(-0.012)
    assert c.align_along_m == pytest.approx(-0.017)
    # the whole point of the standoff: align strictly further out than the mouth
    assert c.align_along_m < c.mouth_along_m


def test_config_rejects_aligning_inside_the_jack() -> None:
    with pytest.raises(ValueError, match="align_standoff_m"):
        AlignInsertConfig(align_standoff_m=0.0)


def test_config_rejects_an_out_of_range_push_correction() -> None:
    with pytest.raises(ValueError, match="push_correction"):
        AlignInsertConfig(push_correction=1.5)


def test_config_rejects_an_unknown_servo_source() -> None:
    with pytest.raises(ValueError, match="servo_source"):
        AlignInsertConfig(servo_source="predicted")


# ── an idealised plant: the plug is rigidly held by the wrist ─────────────────
class RigidGraspPlant:
    """Wrist that moves exactly as commanded, carrying the plug on a fixed offset.

    Deliberately frictionless and lag-free. The simulation's grasp compliance and
    IK lag make the real plant harder, never differently shaped, so anything that
    fails here is a controller bug rather than a physics interaction.
    """

    def __init__(self, n: int, cfg: AlignInsertConfig, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.n = n
        # jack: level frame, insertion axis +y, seat at the origin of each cell
        self.seat_pos = np.zeros((n, 3))
        self.seat_quat = np.tile(IDENTITY, (n, 1))
        self.ins = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
        # grasp offset: the wrist sits 0.21 m back from the face, the lever arm that
        # makes rotation dominate translation in the real environment
        self.grasp_offset = np.tile(np.array([0.0, -0.21, 0.0]), (n, 1))
        # start: well short of the mouth, laterally offset and tilted
        start_along = -0.043
        self.face_pos = self.seat_pos + start_along * self.ins
        self.face_pos[:, 0] += rng.uniform(-0.008, 0.008, n)
        self.face_pos[:, 2] += rng.uniform(-0.008, 0.008, n)
        tilt = rng.uniform(-math.radians(8.0), math.radians(8.0), (n, 1))
        self.face_quat = rotvec_to_quat(np.array([[1.0, 0.0, 0.0]]) * tilt)

    def observe(self) -> InsertionObs:
        return InsertionObs(
            face_pos=self.face_pos.copy(),
            face_quat=self.face_quat.copy(),
            wrist_pos=self.face_pos + quat_rotate(self.face_quat, self.grasp_offset),
            seat_pos=self.seat_pos.copy(),
            seat_quat=self.seat_quat.copy(),
            ins=self.ins.copy(),
        )

    def apply(self, action: np.ndarray) -> None:
        """Move the wrist by the action, then carry the plug rigidly with it."""
        c = self.cfg
        obs = self.observe()
        drot = action[:, 3:6] * c.max_drot_rad
        dpos = action[:, 0:3] * c.max_dpos_m
        wrist_pos = obs.wrist_pos + dpos
        self.face_quat = quat_multiply(rotvec_to_quat(drot), self.face_quat)
        # the face hangs off the moved, rotated wrist by the same body-frame offset
        self.face_pos = wrist_pos - quat_rotate(self.face_quat, self.grasp_offset)

    def along(self) -> np.ndarray:
        return np.sum((self.face_pos - self.seat_pos) * self.ins, axis=1)


def _run(plant: RigidGraspPlant, ctrl: AlignInsertController, steps: int) -> None:
    for _ in range(steps):
        plant.apply(ctrl.act(plant.observe()))


# ── the machine ───────────────────────────────────────────────────────────────
def test_controller_aligns_then_inserts_from_a_tilted_offset_start() -> None:
    cfg = AlignInsertConfig()
    plant = RigidGraspPlant(8, cfg, seed=0)
    ctrl = AlignInsertController(8, cfg)
    _run(plant, ctrl, 260)

    obs = plant.observe()
    along, lat, latn, _, ang = ctrl._errors(obs)
    assert np.all(ctrl.phase == int(InsertPhase.HOLD)), ctrl.phase_counts()
    # seated: past the environment's -5 mm depth gate, within 3 mm laterally and 8 deg
    assert np.all(along >= -0.005)
    assert np.all(latn <= 0.003)
    assert np.all(ang <= math.radians(8.0))
    assert np.all(ctrl.attempts == 0)  # a clean run needs no jam recovery
    assert lat.shape == (8, 3)


def test_it_parks_outside_the_mouth_before_ever_pushing() -> None:
    """The defining property: no advance past the mouth until alignment has converged."""
    cfg = AlignInsertConfig()
    plant = RigidGraspPlant(4, cfg, seed=1)
    ctrl = AlignInsertController(4, cfg)
    entered_push = np.zeros(4, dtype=bool)
    for _ in range(260):
        action = ctrl.act(plant.observe())
        just_now = (ctrl.phase == int(InsertPhase.PUSH)) & ~entered_push
        if np.any(just_now):
            # the measurements carried on the controller are the ones the transition
            # was taken on: still outside the mouth, already inside the pre-dock band
            assert np.all(ctrl.along[just_now] < cfg.mouth_along_m)
            assert np.all(ctrl.latn[just_now] <= cfg.lat_tol_m)
            assert np.all(ctrl.ang[just_now] <= cfg.ang_tol_rad)
            entered_push |= just_now
        plant.apply(action)
    assert np.all(entered_push)


def test_actions_never_exceed_the_environments_per_step_caps() -> None:
    cfg = AlignInsertConfig()
    plant = RigidGraspPlant(8, cfg, seed=2)
    ctrl = AlignInsertController(8, cfg)
    for _ in range(260):
        a = ctrl.act(plant.observe())
        assert np.all(np.abs(a) <= 1.0 + 1e-12)
        # capped by NORM, not per component: the commanded direction is preserved
        assert np.all(np.linalg.norm(a[:, 0:3], axis=1) <= 1.0 + 1e-9)
        assert np.all(np.linalg.norm(a[:, 3:6], axis=1) <= 1.0 + 1e-9)
        plant.apply(a)


def test_rotation_is_rate_limited_below_the_action_cap() -> None:
    """The grasp only follows ~0.3 deg/step; a full-scale 0.75 deg command would slip."""
    cfg = AlignInsertConfig()
    plant = RigidGraspPlant(4, cfg, seed=3)
    ctrl = AlignInsertController(4, cfg)
    worst = 0.0
    for _ in range(120):
        a = ctrl.act(plant.observe())
        worst = max(worst, float(np.max(np.linalg.norm(a[:, 3:6] * cfg.max_drot_rad, axis=1))))
        plant.apply(a)
    assert worst <= cfg.align_rot_rate_rad + 1e-9


def test_settle_phase_commands_nothing() -> None:
    cfg = AlignInsertConfig(settle_steps=3)
    plant = RigidGraspPlant(4, cfg, seed=4)
    ctrl = AlignInsertController(4, cfg)
    for _ in range(3):
        assert np.allclose(ctrl.act(plant.observe()), 0.0)
    assert np.all(ctrl.phase == int(InsertPhase.ALIGN))


def test_an_open_loop_push_issues_no_rotation() -> None:
    cfg = AlignInsertConfig(push_correction=0.0)
    plant = RigidGraspPlant(4, cfg, seed=5)
    ctrl = AlignInsertController(4, cfg)
    saw_push = False
    for _ in range(260):
        # the phase BEFORE the call is the one the returned action was computed under;
        # act() leaves ctrl.phase holding the phase for the NEXT step
        pushing = ctrl.phase == int(InsertPhase.PUSH)
        a = ctrl.act(plant.observe())
        if np.any(pushing):
            saw_push = True
            assert np.allclose(a[pushing, 3:6], 0.0)
        plant.apply(a)
    assert saw_push


def test_a_jammed_push_retreats_and_realigns() -> None:
    """Freeze the plug mid-push: the controller must back out to the pre-dock pose."""
    cfg = AlignInsertConfig()
    plant = RigidGraspPlant(2, cfg, seed=6)
    ctrl = AlignInsertController(2, cfg)
    jammed_at = None
    ever_seized = False
    for _ in range(400):
        a = ctrl.act(plant.observe())
        if np.all(ctrl.phase == int(InsertPhase.PUSH)):
            if jammed_at is None:  # seize the plug the moment the push starts
                jammed_at = (plant.face_pos.copy(), plant.face_quat.copy())
                ever_seized = True
            plant.face_pos = jammed_at[0].copy()
            plant.face_quat = jammed_at[1].copy()
        else:
            jammed_at = None
            plant.apply(a)
    assert ever_seized
    assert np.all(ctrl.attempts >= 1), ctrl.attempts


def test_retreat_backs_straight_out_without_twisting() -> None:
    """Twisting a plug jammed inside the mouth levers it against the cavity walls."""
    cfg = AlignInsertConfig()
    ctrl = AlignInsertController(2, cfg)
    ctrl.phase[:] = int(InsertPhase.RETREAT)
    # sitting inside the cavity, badly misaligned so any correction term would show
    obs = InsertionObs(
        face_pos=np.array([[0.004, -0.004, 0.003], [0.004, -0.004, 0.003]]),
        face_quat=rotvec_to_quat(np.tile([0.0, 0.0, math.radians(9.0)], (2, 1))),
        wrist_pos=np.array([[0.0, -0.214, 0.0], [0.0, -0.214, 0.0]]),
        seat_pos=np.zeros((2, 3)),
        seat_quat=np.tile(IDENTITY, (2, 1)),
        ins=np.tile([0.0, 1.0, 0.0], (2, 1)),
    )
    a = ctrl.act(obs)
    assert np.allclose(a[:, 3:6], 0.0)
    # motion is purely backwards along the insertion axis
    dpos = a[:, 0:3] * cfg.max_dpos_m
    assert np.all(dpos[:, 1] < 0.0)
    assert np.allclose(dpos[:, [0, 2]], 0.0, atol=1e-12)


def test_reset_restores_a_fresh_machine() -> None:
    cfg = AlignInsertConfig()
    plant = RigidGraspPlant(4, cfg, seed=7)
    ctrl = AlignInsertController(4, cfg)
    _run(plant, ctrl, 200)
    assert np.any(ctrl.phase != int(InsertPhase.SETTLE))
    ctrl.reset()
    assert np.all(ctrl.phase == int(InsertPhase.SETTLE))
    assert np.all(ctrl.attempts == 0)
    assert ctrl.phase_counts()["SETTLE"] == 4


def test_the_wrist_command_produces_the_intended_face_motion() -> None:
    """A pure face rotation must come out as the wrist arc that swings the face in place."""
    cfg = AlignInsertConfig()
    ctrl = AlignInsertController(1, cfg)
    face_pos = np.zeros((1, 3))
    wrist_pos = np.array([[0.0, -0.21, 0.0]])
    obs = InsertionObs(
        face_pos=face_pos,
        face_quat=IDENTITY[None],
        wrist_pos=wrist_pos,
        seat_pos=face_pos,
        seat_quat=IDENTITY[None],
        ins=np.array([[0.0, 1.0, 0.0]]),
    )
    dr = np.array([[math.radians(0.2), 0.0, 0.0]])
    action = ctrl._to_action(np.zeros((1, 3)), dr, obs)
    # replay it through the rigid relation and confirm the face held still
    moved_wrist = obs.wrist_pos + action[:, 0:3] * cfg.max_dpos_m
    new_face_quat = quat_multiply(rotvec_to_quat(action[:, 3:6] * cfg.max_drot_rad), obs.face_quat)
    offset = quat_rotate(quat_conjugate(obs.face_quat), obs.wrist_pos - obs.face_pos)
    new_face = moved_wrist - quat_rotate(new_face_quat, offset)
    assert np.allclose(new_face, face_pos, atol=1e-12)


# ── R11: servo on the commanded pose, gate on the measured one ────────────────
class LaggedGraspPlant(RigidGraspPlant):
    """RigidGraspPlant behind a cheap model of the calibrated servo arm.

    An internal COMMANDED pose integrates the wrist action exactly — this is the
    env's open command integrator (``wrist_tgt_p/q``), the state the ``cmd_*``
    observation fields report. The PHYSICAL wrist/face (what ``face_pos`` and the
    plain observation report) tracks that command first-order (``alpha`` per step)
    through a ``delay``-step transport deque, the shape of the lag measured in
    tests/gates/REPORT.md. ``observe`` therefore returns exactly the split the
    rigid-cable adapter provides on a real env: measured truth in the ordinary
    fields, the exactly-known command in ``cmd_*``.
    """

    def __init__(
        self, n: int, cfg: AlignInsertConfig, *, seed: int = 0, alpha: float = 0.6, delay: int = 2
    ) -> None:
        super().__init__(n, cfg, seed=seed)
        self.alpha = float(alpha)
        self.cmd_face_pos = self.face_pos.copy()
        self.cmd_face_quat = self.face_quat.copy()
        # poses commanded `delay` steps ago, primed with the rest pose
        self._pipe = deque((self.face_pos.copy(), self.face_quat.copy()) for _ in range(delay))

    def cmd_wrist_pos(self) -> np.ndarray:
        return self.cmd_face_pos + quat_rotate(self.cmd_face_quat, self.grasp_offset)

    def cmd_along(self) -> np.ndarray:
        return np.sum((self.cmd_face_pos - self.seat_pos) * self.ins, axis=1)

    def observe(self) -> InsertionObs:
        return dataclasses.replace(
            super().observe(),
            cmd_wrist_pos=self.cmd_wrist_pos(),
            cmd_face_pos=self.cmd_face_pos.copy(),
            cmd_face_quat=self.cmd_face_quat.copy(),
        )

    def apply(self, action: np.ndarray) -> None:
        """Integrate the command exactly; let the physical pose trail it."""
        c = self.cfg
        dpos = action[:, 0:3] * c.max_dpos_m
        drot = action[:, 3:6] * c.max_drot_rad
        # the command integrator moves exactly as commanded, like RigidGraspPlant.apply
        cmd_wrist = self.cmd_wrist_pos() + dpos
        self.cmd_face_quat = quat_multiply(rotvec_to_quat(drot), self.cmd_face_quat)
        self.cmd_face_pos = cmd_wrist - quat_rotate(self.cmd_face_quat, self.grasp_offset)
        # the physical plant: first-order tracking of the delay line's output
        self._pipe.append((self.cmd_face_pos.copy(), self.cmd_face_quat.copy()))
        tgt_pos, tgt_quat = self._pipe.popleft()
        self.face_pos = self.face_pos + self.alpha * (tgt_pos - self.face_pos)
        err = quat_to_rotvec(quat_multiply(tgt_quat, quat_conjugate(self.face_quat)))
        self.face_quat = quat_multiply(rotvec_to_quat(self.alpha * err), self.face_quat)


def _mirror_cmd(obs: InsertionObs) -> InsertionObs:
    """Commanded fields byte-equal to the measured ones — a lag-free plant."""
    return dataclasses.replace(
        obs, cmd_wrist_pos=obs.wrist_pos, cmd_face_pos=obs.face_pos, cmd_face_quat=obs.face_quat
    )


def test_commanded_mode_demands_the_cmd_observation_fields() -> None:
    """Silently servoing on nothing would be the worst outcome; it must raise instead."""
    cfg = AlignInsertConfig(servo_source="commanded")
    plant = RigidGraspPlant(2, cfg, seed=8)
    ctrl = AlignInsertController(2, cfg)
    with pytest.raises(ValueError, match="cmd_wrist_pos"):
        ctrl.act(plant.observe())


def test_commanded_mode_collapses_to_measured_on_a_lag_free_plant() -> None:
    """Degeneracy proof: with cmd == measured the two modes are bit-identical.

    Guarantees `servo_source="commanded"` is the same controller looking at a
    different estimate of the same face — not a second controller — so everything
    the 18 measured-mode tests establish carries over.
    """
    plant_m = RigidGraspPlant(8, AlignInsertConfig(), seed=0)
    plant_c = RigidGraspPlant(8, AlignInsertConfig(servo_source="commanded"), seed=0)
    ctrl_m = AlignInsertController(8, plant_m.cfg)
    ctrl_c = AlignInsertController(8, plant_c.cfg)
    for _ in range(260):
        a_m = ctrl_m.act(plant_m.observe())
        a_c = ctrl_c.act(_mirror_cmd(plant_c.observe()))
        assert np.array_equal(a_m, a_c)
        plant_m.apply(a_m)
        plant_c.apply(a_c)
    assert np.array_equal(ctrl_m.phase, ctrl_c.phase)


def test_measured_mode_under_lag_drives_the_command_past_the_seat() -> None:
    """The baseline that motivates servo_source="commanded", documented as measured.

    This mild first-order lag still converges — the 77% seat-rate gap of
    tests/gates/REPORT.md needs the real jerk-limited plant — but the pathology is
    already unmistakable: the depth loop keeps commanding advance until the STALE
    measurement reads seated, by which time the command integrator has run
    ~lag x push_rate past the seat and dragged the physical plug > 1 mm beyond it.
    Against a real cavity bottom that is ramming through the friction grasp.
    """
    cfg = AlignInsertConfig()  # servo_source="measured"
    plant = LaggedGraspPlant(8, cfg, seed=0)
    ctrl = AlignInsertController(8, cfg)
    max_cmd_along = np.full(8, -np.inf)
    max_meas_along = np.full(8, -np.inf)
    for _ in range(400):
        plant.apply(ctrl.act(plant.observe()))
        max_cmd_along = np.maximum(max_cmd_along, plant.cmd_along())
        max_meas_along = np.maximum(max_meas_along, plant.along())
    assert np.all(ctrl.phase == int(InsertPhase.HOLD)), ctrl.phase_counts()
    # the overshoot: the command runs > 1 mm deep, the physical face follows it in
    assert np.all(max_cmd_along >= 0.001), max_cmd_along
    assert np.all(max_meas_along >= 0.0005), max_meas_along


def test_commanded_mode_under_the_same_lag_seats_without_overshoot() -> None:
    """Servoing on the command closes the loop around a state that cannot lag.

    Same plant, same lag, same stock rates as the measured-mode baseline above:
    the machine reaches HOLD 8/8 with no retreats on a kinematic-like schedule,
    and the command integrator never crosses the seat target at all — the PUSH
    advance is clipped against the commanded `along`, which is exact.
    """
    cfg = AlignInsertConfig(servo_source="commanded")
    plant = LaggedGraspPlant(8, cfg, seed=0)
    ctrl = AlignInsertController(8, cfg)
    max_cmd_along = np.full(8, -np.inf)
    max_meas_along = np.full(8, -np.inf)
    for _ in range(400):
        plant.apply(ctrl.act(plant.observe()))
        max_cmd_along = np.maximum(max_cmd_along, plant.cmd_along())
        max_meas_along = np.maximum(max_meas_along, plant.along())
    obs = plant.observe()
    along, _, latn, _, ang = ctrl._errors(obs)
    assert np.all(ctrl.phase == int(InsertPhase.HOLD)), ctrl.phase_counts()
    assert np.all(ctrl.attempts == 0)
    # seated to the same standard as the lag-free machine test
    assert np.all(along >= -0.005)
    assert np.all(latn <= 0.003)
    assert np.all(ang <= math.radians(8.0))
    # and with no ram: neither the command nor the plug ever passes the seat
    assert np.all(max_cmd_along <= 1e-9), max_cmd_along
    assert np.all(max_meas_along <= 1e-9), max_meas_along


def test_commanded_mode_still_parks_outside_the_mouth_before_pushing() -> None:
    """The defining safety property survives the mode switch, gated on MEASURED truth.

    The servo drives the command, but the ALIGN->PUSH commitment must wait for the
    PHYSICAL plug — settling behind the parked command — to dock. At every PUSH
    entry the controller's carried measurements are the plant's physical state,
    outside the mouth and inside the pre-dock band.
    """
    cfg = AlignInsertConfig(servo_source="commanded")
    plant = LaggedGraspPlant(4, cfg, seed=1)
    ctrl = AlignInsertController(4, cfg)
    entered_push = np.zeros(4, dtype=bool)
    for _ in range(400):
        action = ctrl.act(plant.observe())
        just_now = (ctrl.phase == int(InsertPhase.PUSH)) & ~entered_push
        if np.any(just_now):
            # ctrl.along/latn/ang are the MEASURED errors the transition fired on
            assert np.all(ctrl.along[just_now] < cfg.mouth_along_m)
            assert np.all(ctrl.latn[just_now] <= cfg.lat_tol_m)
            assert np.all(ctrl.ang[just_now] <= cfg.ang_tol_rad)
            # and they are the plant's physical truth, not the commanded pose
            assert np.allclose(plant.along()[just_now], ctrl.along[just_now], atol=1e-12)
            assert np.all(plant.along()[just_now] < cfg.mouth_along_m)
            entered_push |= just_now
        plant.apply(action)
    assert np.all(entered_push)


def test_a_held_plug_that_twists_out_of_tolerance_retreats() -> None:
    """HOLD must abort on attitude, not regulate depth forever.

    Discovered on the r180 (camera-side) grasp under the calibrated arm: contact +
    gravity ratchet the friction grasp ~0.2 deg/frame while the plug sits at seat
    depth, so the mid-push `lost` abort never fires and HOLD ground on to a ~50 deg
    wedge. A held plug past the abort thresholds must RETREAT (straight back out —
    any further correction levers it on the cavity walls).
    """
    cfg = AlignInsertConfig()
    ctrl = AlignInsertController(2, cfg)
    ctrl.phase[:] = int(InsertPhase.HOLD)
    # env 0: seated but twisted 12 deg (> abort_ang 10) ; env 1: seated and healthy
    obs = InsertionObs(
        face_pos=np.array([[0.0, 0.001, 0.0], [0.0, 0.001, 0.0]]),
        face_quat=np.concatenate(
            [rotvec_to_quat(np.array([[0.0, 0.0, math.radians(12.0)]])),
             rotvec_to_quat(np.array([[0.0, 0.0, math.radians(2.0)]]))]),
        wrist_pos=np.array([[0.0, -0.209, 0.0], [0.0, -0.209, 0.0]]),
        seat_pos=np.zeros((2, 3)),
        seat_quat=np.tile(IDENTITY, (2, 1)),
        ins=np.tile([0.0, 1.0, 0.0], (2, 1)),
    )
    ctrl.act(obs)
    assert ctrl.phase[0] == int(InsertPhase.RETREAT), ctrl.phase_counts()
    assert ctrl.phase[1] == int(InsertPhase.HOLD), ctrl.phase_counts()
