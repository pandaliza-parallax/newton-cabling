"""Behaviour tests for the declarative cabling timeline.

These assert observable scheduling behaviour (offsets, weights, validation),
not the internal interpolation helpers.
"""

from __future__ import annotations

import math

import pytest

from newton_cabling.timeline import (
    CableTimeline,
    GraspState,
    LatchState,
    Phase,
    proven_cycle_timeline,
)


def _grasped(name: str, duration: float, offset: float) -> Phase:
    return Phase(name, duration, offset, GraspState.GRASPED, LatchState.NEUTRAL)


def test_plug_offset_ramps_between_phase_targets() -> None:
    timeline = CableTimeline.build([_grasped("hold", 2.0, 0.0), _grasped("advance", 2.0, 1.0)])

    assert timeline.sample(2.0).plug_offset_meters == pytest.approx(0.0)
    assert timeline.sample(4.0).plug_offset_meters == pytest.approx(1.0)
    midpoint = timeline.sample(3.0).plug_offset_meters
    assert 0.0 < midpoint < 1.0


def test_total_seconds_is_sum_of_durations() -> None:
    timeline = CableTimeline.build([_grasped("a", 1.5, 0.0), _grasped("b", 2.5, 0.0)])
    assert timeline.total_seconds == pytest.approx(4.0)


def test_sampling_clamps_beyond_the_run() -> None:
    timeline = CableTimeline.build([_grasped("only", 2.0, 0.3)])
    assert timeline.sample(99.0).plug_offset_meters == pytest.approx(0.3)
    assert timeline.sample(-5.0).plug_offset_meters == pytest.approx(0.3)


def test_grasp_weight_is_one_while_grasped_and_zero_while_released() -> None:
    timeline = CableTimeline.build(
        [
            Phase("hold", 2.0, 0.0, GraspState.GRASPED, LatchState.NEUTRAL),
            Phase("drop", 2.0, 0.0, GraspState.RELEASED, LatchState.NEUTRAL),
        ],
        grasp_blend_seconds=0.4,
    )

    assert timeline.sample(1.0).grasp_weight == pytest.approx(1.0)
    # Fully past the 0.4s release blend that starts at t=2.0.
    assert timeline.sample(3.5).grasp_weight == pytest.approx(0.0)


def test_grasp_weight_blends_across_a_state_change() -> None:
    timeline = CableTimeline.build(
        [
            Phase("hold", 2.0, 0.0, GraspState.GRASPED, LatchState.NEUTRAL),
            Phase("drop", 2.0, 0.0, GraspState.RELEASED, LatchState.NEUTRAL),
        ],
        grasp_blend_seconds=0.4,
    )

    mid_blend = timeline.sample(2.2).grasp_weight
    assert 0.0 < mid_blend < 1.0


def test_latch_weight_presses_only_in_pressed_phases() -> None:
    timeline = CableTimeline.build(
        [
            Phase("neutral", 2.0, 0.0, GraspState.GRASPED, LatchState.NEUTRAL),
            Phase("press", 2.0, 0.0, GraspState.GRASPED, LatchState.PRESSED),
        ],
        latch_blend_seconds=1.0,
    )

    assert timeline.sample(1.0).latch_weight == pytest.approx(0.0)
    assert timeline.sample(4.0).latch_weight == pytest.approx(1.0)


def test_empty_timeline_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one phase"):
        CableTimeline.build([])


def test_non_positive_duration_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-positive duration"):
        CableTimeline.build([_grasped("bad", 0.0, 0.0)])


def test_loop_requires_matching_start_and_end_offset() -> None:
    with pytest.raises(ValueError, match="same plug offset"):
        CableTimeline.build(
            [_grasped("start", 1.0, 0.0), _grasped("end", 1.0, 0.5)],
            loop=True,
        )


def test_loop_requires_matching_start_and_end_state() -> None:
    with pytest.raises(ValueError, match="same grasp/latch state"):
        CableTimeline.build(
            [
                Phase("start", 1.0, 0.0, GraspState.GRASPED, LatchState.NEUTRAL),
                Phase("end", 1.0, 0.0, GraspState.RELEASED, LatchState.NEUTRAL),
            ],
            loop=True,
        )


def test_proven_cycle_is_a_valid_nineteen_second_loop() -> None:
    timeline = proven_cycle_timeline()
    assert timeline.total_seconds == pytest.approx(19.0)

    start = timeline.sample(0.0)
    end = timeline.sample(timeline.total_seconds)
    assert start.plug_offset_meters == pytest.approx(end.plug_offset_meters)
    assert start.grasp_weight == pytest.approx(end.grasp_weight)
    assert start.latch_weight == pytest.approx(end.latch_weight)


def test_proven_cycle_offsets_stay_within_commanded_bounds() -> None:
    timeline = proven_cycle_timeline(start_offset_meters=-0.05, insert_cap_meters=0.035)
    samples = [timeline.sample(0.05 * step) for step in range(int(19.0 / 0.05) + 1)]
    offsets = [sample.plug_offset_meters for sample in samples]
    assert min(offsets) >= -0.05 - 1e-9
    assert max(offsets) <= 0.035 + 1e-9
    assert all(math.isfinite(offset) for offset in offsets)
