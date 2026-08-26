"""Reusable building blocks for cabling simulation on the Newton physics engine.

The pure-logic surface (timeline, report, connector specs) imports no GPU code
and is safe to use anywhere. The Newton-backed scene builder, grasp spring, and
recording helpers live behind the ``sim`` optional dependency and are imported
lazily by the runner scripts.
"""

from newton_cabling.cable import route_cable_from_boot
from newton_cabling.connector import (
    ConnectorSpec,
    ContactSpec,
    LatchSpec,
    cad_rj45_connector,
    rj45_connector,
)
from newton_cabling.report import (
    CycleMeasurements,
    CycleMetrics,
    CycleOutcome,
    CycleSucceeded,
    CycleThresholds,
    ExtractionIncomplete,
    InsertionFailed,
    LatchSlipped,
    evaluate_cycle,
    outcome_from_dict,
    outcome_to_dict,
)
from newton_cabling.scripted_controller import (
    AlignInsertConfig,
    AlignInsertController,
    InsertionObs,
    InsertPhase,
    config_for_rigid_cable_env,
    observe_rigid_cable_env,
)
from newton_cabling.timeline import (
    CableTimeline,
    GraspState,
    LatchState,
    Phase,
    TimelineSample,
    proven_cycle_timeline,
)

__all__ = [
    "AlignInsertConfig",
    "AlignInsertController",
    "CableTimeline",
    "ConnectorSpec",
    "ContactSpec",
    "CycleMeasurements",
    "CycleMetrics",
    "CycleOutcome",
    "CycleSucceeded",
    "CycleThresholds",
    "ExtractionIncomplete",
    "GraspState",
    "InsertPhase",
    "InsertionFailed",
    "InsertionObs",
    "LatchSlipped",
    "LatchSpec",
    "LatchState",
    "Phase",
    "TimelineSample",
    "cad_rj45_connector",
    "config_for_rigid_cable_env",
    "evaluate_cycle",
    "observe_rigid_cable_env",
    "outcome_from_dict",
    "outcome_to_dict",
    "proven_cycle_timeline",
    "rj45_connector",
    "route_cable_from_boot",
]
