"""Step scheduling primitives."""

from __future__ import annotations

from enum import Enum
from warnings import warn


class ProfilerAction(str, Enum):
    """Action selected for one profiler step."""

    NONE = "none"
    WARMUP = "warmup"
    RECORD = "record"
    RECORD_AND_SAVE = "record_and_save"
    DEVICE_STOPPED = "device_stopped"


def schedule(
    *,
    wait: int,
    warmup: int,
    active: int,
    repeat: int = 0,
    skip_first: int = 0,
    skip_first_wait: int = 0,
):
    """Build a callable that maps step numbers to profiler actions."""
    if wait < 0 or warmup < 0 or active <= 0:
        raise AssertionError("Invalid schedule: wait and warmup must be non-negative, active must be positive")
    if repeat < 0 or skip_first < 0:
        raise AssertionError("Invalid schedule: repeat and skip_first must be non-negative")
    if skip_first_wait not in (0, 1, False, True):
        raise AssertionError("skip_first_wait must be a boolean")
    if warmup == 0:
        warn(
            "Profiler schedule has no warmup steps; measured results may be noisy",
            stacklevel=2,
        )

    cycle = wait + warmup + active

    def schedule_fn(step: int):
        if step < 0:
            raise AssertionError(f"Step must be non-negative. Got {step}.")
        if step < skip_first:
            return ProfilerAction.NONE
        step -= skip_first
        if skip_first_wait:
            step += wait
        if repeat > 0 and step >= repeat * cycle:
            return ProfilerAction.NONE
        position = step % cycle
        if position < wait:
            return ProfilerAction.NONE
        if position < wait + warmup:
            return ProfilerAction.WARMUP
        return ProfilerAction.RECORD

    return schedule_fn


__all__ = ["ProfilerAction", "schedule"]
