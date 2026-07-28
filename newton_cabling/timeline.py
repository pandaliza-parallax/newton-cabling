"""Declarative phase timeline for scripted cabling skills.

The arm-driven cabling cycle is a sequence of timed phases (approach, insert,
hold, release, retreat, return, re-grasp, press, extract). Encoding the schedule
as one ordered list of phases keeps the plug trajectory, the grasp state, and the
latch state in a single source of truth.

This replaces the original script's three hand-synchronised functions
(``hand_dy``, ``grasp_weight``, ``latch_target``) whose hard-coded time
boundaries drifted apart every time the schedule changed: making the recording
loop seamlessly previously meant editing six matching numbers across three
functions, and a single missed edit produced a silent jump on playback.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto


class GraspState(Enum):
    """Whether the gripper is holding the plug during a phase.

    A two-state variant rather than a bool so the schedule reads in domain terms
    and so a future third state (e.g. ``REGRASPING``) cannot collide with an
    unrelated boolean flag.
    """

    GRASPED = auto()
    RELEASED = auto()


class LatchState(Enum):
    """Whether the connector latch tab is held depressed during a phase."""

    NEUTRAL = auto()
    PRESSED = auto()


def _smoothstep(start: float, end: float, progress: float) -> float:
    """Hermite smoothstep from ``start`` to ``end`` as ``progress`` goes 0 -> 1.

    Owned locally rather than imported: it is three lines of well-understood math,
    so a dependency would cost more than the code (see the dependency guidance in
    the style guide). ``progress`` is clamped, so callers may pass any real value.
    """
    clamped = min(1.0, max(0.0, progress))
    weight = clamped * clamped * (3.0 - 2.0 * clamped)
    return start + (end - start) * weight


@dataclass(frozen=True)
class Phase:
    """One leg of the cabling schedule.

    ``plug_offset_meters`` is the commanded plug position along the insertion
    axis at the *end* of this phase; the timeline ramps to it with a smoothstep
    over the phase duration. ``grasp`` and ``latch`` are the states this phase
    holds; transitions between adjacent phases are blended over the timeline's
    short blend windows.
    """

    name: str
    duration_seconds: float
    plug_offset_meters: float
    grasp: GraspState
    latch: LatchState


@dataclass(frozen=True)
class TimelineSample:
    """The commanded state at a single instant, derived from the timeline.

    Weights are in ``[0.0, 1.0]`` by construction (smoothstep clamps), so
    consumers can lerp with them directly:

    * ``grasp_weight``: 1.0 fully grasped (fingers closed, arm follows the plug),
      0.0 released (fingers open, arm follows its own trajectory).
    * ``latch_weight``: 1.0 latch tab fully pressed, 0.0 neutral.
    """

    phase_name: str
    plug_offset_meters: float
    grasp_weight: float
    latch_weight: float


def _grasp_target(phase: Phase) -> float:
    return 1.0 if phase.grasp is GraspState.GRASPED else 0.0


def _latch_target(phase: Phase) -> float:
    return 1.0 if phase.latch is LatchState.PRESSED else 0.0


class CableTimeline:
    """An ordered, validated cabling schedule that can be sampled at any time.

    Construct through :meth:`build` so invalid schedules cannot be represented:
    empty phase lists, non-positive durations, and (for loops) start/end states
    that would jump on playback are all rejected at construction.
    """

    def __init__(
        self,
        phases: tuple[Phase, ...],
        grasp_blend_seconds: float,
        latch_blend_seconds: float,
    ) -> None:
        # Trusts already-validated inputs; external callers use build().
        self._phases = phases
        self._grasp_blend_seconds = grasp_blend_seconds
        self._latch_blend_seconds = latch_blend_seconds
        starts: list[float] = []
        cumulative = 0.0
        for phase in phases:
            starts.append(cumulative)
            cumulative += phase.duration_seconds
        self._phase_start_seconds = tuple(starts)
        self._total_seconds = cumulative

    @classmethod
    def build(
        cls,
        phases: Sequence[Phase],
        *,
        grasp_blend_seconds: float = 0.4,
        latch_blend_seconds: float = 1.0,
        loop: bool = False,
    ) -> CableTimeline:
        """Validate a schedule and return a sampleable timeline.

        When ``loop`` is set, the first and last phases must share plug offset,
        grasp state, and latch state, so the generated recording wraps without a
        visible jump. Raises ``ValueError`` on any invalid schedule.
        """
        phase_tuple = tuple(phases)
        if not phase_tuple:
            raise ValueError("a timeline needs at least one phase")
        for phase in phase_tuple:
            if phase.duration_seconds <= 0.0:
                raise ValueError(
                    f"phase {phase.name!r} has non-positive duration {phase.duration_seconds}"
                )
        if grasp_blend_seconds < 0.0 or latch_blend_seconds < 0.0:
            raise ValueError("blend durations must be non-negative")
        if loop:
            first, last = phase_tuple[0], phase_tuple[-1]
            if abs(first.plug_offset_meters - last.plug_offset_meters) > 1e-9:
                raise ValueError(
                    "loop timeline must start and end at the same plug offset "
                    f"({first.plug_offset_meters} != {last.plug_offset_meters})"
                )
            if first.grasp is not last.grasp or first.latch is not last.latch:
                raise ValueError("loop timeline must start and end in the same grasp/latch state")
        return cls(phase_tuple, grasp_blend_seconds, latch_blend_seconds)

    @property
    def total_seconds(self) -> float:
        return self._total_seconds

    @property
    def phases(self) -> tuple[Phase, ...]:
        return self._phases

    def _phase_index_at(self, time_seconds: float) -> int:
        index = 0
        for candidate, start in enumerate(self._phase_start_seconds):
            if time_seconds >= start:
                index = candidate
            else:
                break
        return index

    def _blended_state_weight(
        self,
        index: int,
        time_seconds: float,
        target_of: Callable[[Phase], float],
        blend_seconds: float,
    ) -> float:
        current = target_of(self._phases[index])
        if index == 0 or blend_seconds <= 0.0:
            return current
        previous = target_of(self._phases[index - 1])
        if previous == current:
            return current
        elapsed_in_phase = time_seconds - self._phase_start_seconds[index]
        return _smoothstep(previous, current, elapsed_in_phase / blend_seconds)

    def sample(self, time_seconds: float) -> TimelineSample:
        """Return the commanded state at ``time_seconds`` (clamped to the run)."""
        clamped_time = min(self._total_seconds, max(0.0, time_seconds))
        index = self._phase_index_at(clamped_time)
        phase = self._phases[index]

        phase_start = self._phase_start_seconds[index]
        progress = (clamped_time - phase_start) / phase.duration_seconds
        previous_offset = (
            self._phases[index - 1].plug_offset_meters if index > 0 else phase.plug_offset_meters
        )
        plug_offset = _smoothstep(previous_offset, phase.plug_offset_meters, progress)

        grasp_weight = self._blended_state_weight(
            index, clamped_time, _grasp_target, self._grasp_blend_seconds
        )
        latch_weight = self._blended_state_weight(
            index, clamped_time, _latch_target, self._latch_blend_seconds
        )
        return TimelineSample(
            phase_name=phase.name,
            plug_offset_meters=plug_offset,
            grasp_weight=grasp_weight,
            latch_weight=latch_weight,
        )


def proven_cycle_timeline(
    *,
    start_offset_meters: float = -0.050,
    insert_cap_meters: float = 0.035,
) -> CableTimeline:
    """The full insert/extract cycle schedule verified in ``panda_cycle.rrd``.

    ``start_offset_meters`` is the retracted pose the loop starts and ends at;
    ``insert_cap_meters`` is the commanded forward stroke (the runtime clamps the
    actual advance to the measured seat depth via closed-loop detection, so this
    is an upper bound, not the final depth).

    Note: the durations and states mirror the verified 19s run, but re-wiring
    ``examples/record_panda_cycle.py`` to consume this timeline and re-confirming the
    dynamics on a GPU box is still pending. The schedule's *structure* (valid
    seamless loop, continuous trajectory) is covered by the unit tests.
    """
    g, r = GraspState.GRASPED, GraspState.RELEASED
    neutral, pressed = LatchState.NEUTRAL, LatchState.PRESSED
    phases = [
        Phase("settle", 1.0, start_offset_meters, g, neutral),
        Phase("approach", 3.0, 0.0, g, neutral),
        Phase("insert", 3.5, insert_cap_meters, g, neutral),
        Phase("hold", 1.0, insert_cap_meters, g, neutral),
        Phase("retreat", 2.5, start_offset_meters, r, neutral),
        Phase("return", 2.5, insert_cap_meters, r, neutral),
        Phase("regrasp_press", 1.5, insert_cap_meters, g, pressed),
        Phase("extract", 3.0, start_offset_meters, g, pressed),
        Phase("settle_end", 1.0, start_offset_meters, g, neutral),
    ]
    return CableTimeline.build(phases, loop=True)
