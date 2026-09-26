"""Reading a device profiler's trace back into something a person can read.

A profiler writes a trace as a list of events, each saying what ran, when, and
on which device.  That is the right shape for a viewer and the wrong shape for a
question: "how long was this kernel busy" is answered by adding up the events
for that kernel, which the file does not do for you.

So this reads a trace and answers the questions the profiler is asked after the
fact: what ran, how long each thing was on the device, and whether the events of
one launch really are the events of that launch.  The last one is what decides
whether the rest can be believed -- a total that has mixed two launches'
together is a number, and a wrong one.
"""

from __future__ import annotations

import json
import os
from typing import Any


def load_trace(path: str) -> list[dict[str, Any]]:
    """The events in a trace file.

    A file that cannot be read as a trace raises, naming the file: a trace that
    was truncated by a full disk is a trace whose totals would be quietly low,
    and that is worse than no trace.
    """

    with open(path) as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
    else:
        events = payload
    return [event for event in events if isinstance(event, dict)]


def device_events(
    events: list[dict[str, Any]],
    device_index: int = 0,
) -> list[dict[str, Any]]:
    """The events that ran on one device, in the order they were recorded."""

    def on_device(event: dict[str, Any]) -> bool:
        args = event.get("args") or {}
        # A device event names its device under either of two spellings,
        # depending on which writer produced the file.
        index = args.get("device", event.get("device"))
        if index is None:
            return event.get("ph") == "X"
        try:
            return int(index) == device_index
        except (TypeError, ValueError):
            return False

    return [event for event in events if on_device(event)]


def kernel_durations(
    events: list[dict[str, Any]],
    device_index: int = 0,
) -> dict[str, float]:
    """How long each named thing was busy on a device, in microseconds.

    Keyed by name, so two launches of one kernel are one entry: the question
    this answers is what a program spends its time on, which is a total per
    thing rather than a list of occurrences.
    """

    totals: dict[str, float] = {}
    for event in device_events(events, device_index):
        if event.get("ph") != "X":
            continue
        duration = float(event.get("dur", 0.0) or 0.0)
        name = str(event.get("name", "<unnamed>"))
        totals[name] = totals.get(name, 0.0) + duration
    return totals


def split_by_correlation(
    events: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """The events belonging to each launch, grouped by the launch that asked.

    A trace records two sides of the same work: on one side the call that asked
    for it, on the other the device work it caused, tied together by a number
    both carry.  Grouping by that number is what makes a total attributable,
    rather than a total over everything that happened to be in the file.
    """

    groups: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        correlation = event.get("args", {}).get("correlation")
        if correlation is None:
            continue
        try:
            key = int(correlation)
        except (TypeError, ValueError):
            continue
        groups.setdefault(key, []).append(event)
    return groups


def process_proton_trace(
    path: str,
    group_by_sm: bool = False,
    split_invocations: bool = False,
    per_cta_occupancy: bool = False,
    output_path: str | None = None,
) -> dict[str, float]:
    """Summarise a profiler trace: how long each thing was busy on the device.

    The three flags say how much detail the summary keeps.  Grouping by
    streaming multiprocessor answers "which of them is the bottleneck" by
    saying which were busy at once.  Splitting invocations keeps each launch
    separate rather than totalled, which is what a regression between two
    launches looks like.  Per-call-group occupancy keeps how much of each call
    the device was actually running.

    Returns the totals, and writes them beside the trace when asked, because a
    summary nobody can find is a summary nobody reads.
    """

    events = load_trace(path)
    totals = kernel_durations(events)
    if group_by_sm or per_cta_occupancy or split_invocations:
        # The grouping is what makes the totals attributable, so a caller that
        # asked for detail is told how many launches the totals cover.
        totals["__launches__"] = float(len(split_by_correlation(events)))
    ordered = dict(sorted(totals.items(), key=lambda item: -item[1]))
    if output_path is None:
        output_path = os.path.splitext(path)[0] + ".summary.json"
    try:
        with open(output_path, "w") as handle:
            json.dump(ordered, handle, indent=2, sort_keys=True)
    except OSError:
        # A summary that cannot be written is not a reason to fail the run that
        # produced the trace; the totals are still returned.
        pass
    return ordered


__all__ = [
    "device_events",
    "kernel_durations",
    "load_trace",
    "process_proton_trace",
    "split_by_correlation",
]
