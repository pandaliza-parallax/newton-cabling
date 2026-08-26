"""Scripted (non-learned) controllers for connector insertion.

:class:`~.align_insert.AlignInsertController` implements the align-then-insert
strategy: park the plug face a standoff clear of the jack mouth, converge on the
seat pose there in free space, then push straight in — retreating and re-aligning
if the push jams. Pure NumPy, so it lints, type-checks and unit-tests with no GPU
and no Newton install; bind it to a simulation with an adapter.
"""

from newton_cabling.scripted_controller.align_insert import (
    AlignInsertConfig,
    AlignInsertController,
    InsertionObs,
    InsertPhase,
)
from newton_cabling.scripted_controller.rigid_cable_adapter import (
    config_for_rigid_cable_env,
    observe_rigid_cable_env,
)

__all__ = [
    "AlignInsertConfig",
    "AlignInsertController",
    "InsertPhase",
    "InsertionObs",
    "config_for_rigid_cable_env",
    "observe_rigid_cable_env",
]
