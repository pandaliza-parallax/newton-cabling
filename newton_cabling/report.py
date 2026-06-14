"""Structured outcome of one cabling cycle.

The original script printed a wall of per-frame telemetry and three ad-hoc
booleans (``inserted=...``, ``latch_held=...``, ``extracted=...``) computed
inline at the end. That makes a tuning loop impossible to automate: you have to
eyeball a log to know whether a parameter set worked.

This module makes the pass/fail logic a single source of truth
(:func:`evaluate_cycle`) and the verdict an explicit set of outcome variants the
caller pattern-matches on. The parameter-sweep harness consumes the dict form.

Plug positions follow the simulation convention used throughout the project:
the insertion axis is world ``y`` and a *more negative* plug ``y`` is further out
of the socket (toward the operator), so insertion travel is ``seated - start``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CycleThresholds:
    """Tolerances that turn raw measurements into a pass/fail verdict.

    Defaults are the values proven on ``panda_cycle.rrd``.
    """

    min_insertion_travel_meters: float = 0.020
    max_latch_slip_meters: float = 0.005
    max_extract_residual_meters: float = 0.002


@dataclass(frozen=True)
class CycleMeasurements:
    """Raw measurements taken at the four key instants of a cycle.

    Parsed into a :data:`CycleOutcome` at the boundary by :func:`evaluate_cycle`;
    downstream code branches on the outcome, never on these raw numbers.
    """

    plug_y_start_meters: float
    plug_y_seated_meters: float
    plug_y_after_release_meters: float
    plug_y_end_meters: float
    max_tracking_error_meters: float
    max_plug_pitch_degrees: float


@dataclass(frozen=True)
class CycleMetrics:
    """Quantitative quality metrics carried by every outcome variant."""

    insertion_travel_meters: float
    seated_depth_meters: float
    max_tracking_error_meters: float
    max_plug_pitch_degrees: float


@dataclass(frozen=True)
class CycleSucceeded:
    metrics: CycleMetrics


@dataclass(frozen=True)
class InsertionFailed:
    metrics: CycleMetrics
    insertion_travel_meters: float


@dataclass(frozen=True)
class LatchSlipped:
    metrics: CycleMetrics
    slip_meters: float


@dataclass(frozen=True)
class ExtractionIncomplete:
    metrics: CycleMetrics
    residual_meters: float


# Recoverable outcomes live on the success side as variants the tuning harness
# branches on; genuine errors (a crashed sim, a missing recording) propagate as
# ordinary exceptions and never reach here.
CycleOutcome = CycleSucceeded | InsertionFailed | LatchSlipped | ExtractionIncomplete


_DEFAULT_THRESHOLDS = CycleThresholds()


def evaluate_cycle(
    measurements: CycleMeasurements,
    thresholds: CycleThresholds = _DEFAULT_THRESHOLDS,
) -> CycleOutcome:
    """Derive the single pass/fail verdict for a cycle.

    Checks run in cycle order, so the first stage that fails names the outcome:
    insertion, then latch hold while released, then extraction.
    """
    insertion_travel = measurements.plug_y_seated_meters - measurements.plug_y_start_meters
    metrics = CycleMetrics(
        insertion_travel_meters=insertion_travel,
        seated_depth_meters=measurements.plug_y_seated_meters,
        max_tracking_error_meters=measurements.max_tracking_error_meters,
        max_plug_pitch_degrees=measurements.max_plug_pitch_degrees,
    )

    if insertion_travel < thresholds.min_insertion_travel_meters:
        return InsertionFailed(metrics, insertion_travel)

    slip = abs(measurements.plug_y_after_release_meters - measurements.plug_y_seated_meters)
    if slip > thresholds.max_latch_slip_meters:
        return LatchSlipped(metrics, slip)

    extract_residual = measurements.plug_y_end_meters - measurements.plug_y_start_meters
    if extract_residual > thresholds.max_extract_residual_meters:
        return ExtractionIncomplete(metrics, extract_residual)

    return CycleSucceeded(metrics)


def outcome_to_dict(outcome: CycleOutcome) -> dict[str, object]:
    """Serialise an outcome (for the parameter-sweep harness) with a kind tag."""
    payload = asdict(outcome)
    payload["kind"] = type(outcome).__name__
    return payload


_KNOWN_KINDS = frozenset(
    {"CycleSucceeded", "InsertionFailed", "LatchSlipped", "ExtractionIncomplete"}
)


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"expected a number, got {value!r}")


def _parse_metrics(payload: object) -> CycleMetrics:
    if not isinstance(payload, dict):
        raise ValueError(f"expected a metrics object, got {payload!r}")
    return CycleMetrics(
        insertion_travel_meters=_as_float(payload.get("insertion_travel_meters")),
        seated_depth_meters=_as_float(payload.get("seated_depth_meters")),
        max_tracking_error_meters=_as_float(payload.get("max_tracking_error_meters")),
        max_plug_pitch_degrees=_as_float(payload.get("max_plug_pitch_degrees")),
    )


def outcome_from_dict(payload: dict[str, object]) -> CycleOutcome:
    """Parse a serialised outcome, rejecting unknown kinds explicitly.

    The kind is validated before the metrics are parsed, and each variant is
    reconstructed by name so every field is type-checked at the boundary rather
    than splatting an untyped dict into a constructor.
    """
    kind = payload.get("kind")
    if kind not in _KNOWN_KINDS:
        raise ValueError(f"unknown cycle outcome kind: {kind!r}")
    metrics = _parse_metrics(payload.get("metrics"))
    match kind:
        case "InsertionFailed":
            return InsertionFailed(metrics, _as_float(payload.get("insertion_travel_meters")))
        case "LatchSlipped":
            return LatchSlipped(metrics, _as_float(payload.get("slip_meters")))
        case "ExtractionIncomplete":
            return ExtractionIncomplete(metrics, _as_float(payload.get("residual_meters")))
        case _:
            return CycleSucceeded(metrics)
