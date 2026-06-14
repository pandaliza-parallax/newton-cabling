"""Behaviour tests for cycle evaluation and outcome serialisation."""

from __future__ import annotations

import pytest

from newton_cabling.report import (
    CycleMeasurements,
    CycleSucceeded,
    CycleThresholds,
    ExtractionIncomplete,
    InsertionFailed,
    LatchSlipped,
    evaluate_cycle,
    outcome_from_dict,
    outcome_to_dict,
)


def _measurements(
    *,
    start: float = -0.0287,
    seated: float = -0.0037,
    after_release: float = -0.0040,
    end: float = -0.0785,
) -> CycleMeasurements:
    return CycleMeasurements(
        plug_y_start_meters=start,
        plug_y_seated_meters=seated,
        plug_y_after_release_meters=after_release,
        plug_y_end_meters=end,
        max_tracking_error_meters=0.001,
        max_plug_pitch_degrees=0.0,
    )


def test_clean_cycle_succeeds() -> None:
    outcome = evaluate_cycle(_measurements())
    assert isinstance(outcome, CycleSucceeded)
    assert outcome.metrics.insertion_travel_meters == pytest.approx(0.025, abs=1e-3)


def test_short_insertion_is_reported_as_insertion_failure() -> None:
    outcome = evaluate_cycle(_measurements(seated=-0.027))
    assert isinstance(outcome, InsertionFailed)
    assert outcome.insertion_travel_meters < CycleThresholds().min_insertion_travel_meters


def test_plug_falling_out_when_released_is_a_latch_slip() -> None:
    outcome = evaluate_cycle(_measurements(after_release=-0.20))
    assert isinstance(outcome, LatchSlipped)
    assert outcome.slip_meters > CycleThresholds().max_latch_slip_meters


def test_plug_left_seated_after_pull_is_incomplete_extraction() -> None:
    # End barely out of the socket relative to the start -> not extracted.
    outcome = evaluate_cycle(_measurements(end=-0.0287 + 0.01))
    assert isinstance(outcome, ExtractionIncomplete)
    assert outcome.residual_meters > CycleThresholds().max_extract_residual_meters


def test_checks_run_in_cycle_order_insertion_takes_precedence() -> None:
    # Both insertion and latch would fail; the earliest stage names the outcome.
    outcome = evaluate_cycle(_measurements(seated=-0.027, after_release=-0.30))
    assert isinstance(outcome, InsertionFailed)


def test_outcome_round_trips_through_dict() -> None:
    original = evaluate_cycle(_measurements())
    restored = outcome_from_dict(outcome_to_dict(original))
    assert restored == original


def test_failure_outcome_round_trips_with_its_extra_field() -> None:
    original = evaluate_cycle(_measurements(after_release=-0.20))
    restored = outcome_from_dict(outcome_to_dict(original))
    assert restored == original


def test_unknown_outcome_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown cycle outcome kind"):
        outcome_from_dict({"kind": "Exploded", "metrics": {}})
