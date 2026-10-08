"""Shared helpers for the profiler package."""

from __future__ import annotations

import collections
import os


def rank_world():
    """Return the optional process rank and world size."""
    rank = os.environ.get("RANK", os.environ.get("TP_RANK"))
    world = os.environ.get("WORLD_SIZE", os.environ.get("TP_WORLD_SIZE"))
    try:
        return (
            int(rank) if rank is not None else None,
            int(world) if world is not None else None,
        )
    except (TypeError, ValueError):
        return None, None


#: What a device record carries in place of an operation when no
#: operation was being dispatched, which is every record the collector
#: gathered outside a dispatch.
NO_EXTERNAL_ID = 0xFFFFFFFFFFFFFFFF


def event_gpu_us(event):
    """Return one event's device duration in microseconds."""
    if len(event) <= 8 or event[8] is None or event[8] < 0:
        return 0.0
    return float(event[8]) * 1000.0


def self_times(events):
    """Compute each span's duration excluding same-thread child spans."""
    by_tid = collections.defaultdict(list)
    for index, event in enumerate(events):
        if len(event) < 5:
            continue
        start_ns, end_ns, tid = event[2], event[3], event[4]
        by_tid[tid].append((start_ns, end_ns, index))

    self_ns = [0] * len(events)
    for spans in by_tid.values():
        spans.sort(key=lambda item: (item[0], -item[1], item[2]))
        stack = []
        for start_ns, end_ns, index in spans:
            while stack and stack[-1][1] <= start_ns:
                stack.pop()
            duration = max(end_ns - start_ns, 0)
            self_ns[index] += duration
            if stack and end_ns <= stack[-1][1]:
                self_ns[stack[-1][2]] -= duration
            stack.append((start_ns, end_ns, index))
    return self_ns


def self_cuda_us(events):
    """Compute each span's device duration excluding nested child spans."""
    by_tid = collections.defaultdict(list)
    for index, event in enumerate(events):
        if len(event) < 5:
            continue
        by_tid[event[4]].append((event[2], event[3], index, event_gpu_us(event)))

    self_us = [0.0] * len(events)
    for spans in by_tid.values():
        spans.sort(key=lambda item: (item[0], -item[1], item[2]))
        stack = []
        for start_ns, end_ns, index, device_us in spans:
            while stack and stack[-1][1] <= start_ns:
                stack.pop()
            self_us[index] += device_us
            if stack and end_ns <= stack[-1][1] and device_us:
                self_us[stack[-1][2]] -= device_us
            stack.append((start_ns, end_ns, index))
    return [max(value, 0.0) for value in self_us]


def dtype_name(value):
    """Convert a native dtype value to its short Python spelling."""
    try:
        import tensorplay as tp

        return str(tp.DType(value)).rsplit(".", 1)[-1]
    except Exception:
        return None


def nested_sums(children, own):
    """For each span, ``own`` plus the totals of every span nested in it.

    A parent's total is only complete once every span inside it is complete, so
    the walk repeats until a pass finds nothing left to add.  Doing it by rounds
    rather than by walking children first is what keeps a parent that appears
    before its own children -- which containment does not prevent, since both
    can open at the same instant -- from being settled with the partial totals
    of a child that has not been reached yet.
    """

    totals = list(own)
    for _ in range(len(children)):
        changed = False
        for index, kids in enumerate(children):
            nested = own[index] + sum(totals[child] for child in kids)
            if nested != totals[index]:
                totals[index] = nested
                changed = True
        if not changed:
            break
    return totals


def event_out_bytes(event):
    """Return one span's own output allocation volume in bytes.

    Zero when the session recorded none: the volume is stamped on a span when
    shape capture is on, so zero means either that the operation allocated
    nothing or that nothing was asked of it.
    """

    if len(event) <= 9 or event[9] is None:
        return 0
    return max(int(event[9]), 0)


def nested_output_bytes(events, children=None):
    """For each span, the output allocation volume of itself and its children.

    A span that is asked about is often a region rather than an operation, so
    the volume that belongs to it is the whole of what was allocated under it;
    the volume of one operation is its own.
    """

    own = [event_out_bytes(event) for event in events]
    if children is None:
        children = nested_spans(events)
    return nested_sums(children, own)


__all__ = [
    "dtype_name",
    "event_gpu_us",
    "event_out_bytes",
    "nested_output_bytes",
    "nested_spans",
    "nested_sums",
    "rank_world",
    "self_cuda_us",
    "self_times",
]


def nested_spans(events):
    """For each span, the spans dispatched inside it, in dispatch order.

    A span on one thread contains the spans that started after it started and
    finished before it finished, which is what nesting means for spans that do
    not wait for each other.  Two spans on one thread that overlap are not nested
    -- a span that waits for the device overlaps whatever else is in flight
    without containing it -- so an overlapping span is not a child, and the
    parent of a span is the innermost span that contains it rather than merely
    the earliest one.

    Returned as a list of child-index lists, parallel to ``events``, so that a
    caller walking the tree does not have to search for a parent twice.
    """

    children: list[list[int]] = [[] for _ in events]
    by_tid = collections.defaultdict(list)
    for index, event in enumerate(events):
        if len(event) < 5:
            continue
        by_tid[event[4]].append((event[2], event[3], index))

    for spans in by_tid.values():
        # Outermost first among spans that start together, so that a span opened
        # inside another is placed after it rather than before.
        spans.sort(key=lambda item: (item[0], -item[1], item[2]))
        stack: list[int] = []
        for start_ns, end_ns, index in spans:
            while stack and events[stack[-1]][3] <= start_ns:
                stack.pop()
            if stack:
                children[stack[-1]].append(index)
            stack.append(index)
    return children


def device_us_by_op(gpu_activities):
    """For each operation slot, the device time of the work it launched.

    A kernel record does not name the operation that launched it; it shares a
    correlation id with the launch record that did, and that record names the
    operation.  So the join is made through the correlation id, which is why it
    is exact where a span's timestamps would not be: a kernel can run long after
    the operation that launched it returned, and long after the span that
    surrounded the launch closed.
    """

    launch_to_slot: dict[int, int] = {}
    for activity in gpu_activities:
        if len(activity) > 7 and activity[7] != NO_EXTERNAL_ID:
            launch_to_slot.setdefault(activity[6], activity[7])

    totals: dict[int, float] = collections.defaultdict(float)
    for activity in gpu_activities:
        if activity[1] == "r" or activity[1] == "d":
            continue
        slot = launch_to_slot.get(activity[6])
        if slot is not None:
            totals[slot] += (activity[3] - activity[2]) / 1000.0
    return totals


def nested_cuda_us(events, gpu_activities=()):
    """For each span, the device time of the work dispatched inside it.

    A span that dispatches nothing of its own has no device time of its own and
    yet is exactly the span whose device time gets asked for: an annotated region
    wraps work rather than being work.  So a span's device time is the device
    time of the operation it is, plus that of every operation under it.

    The per-operation device time is the one the collector already joined to the
    operation by correlation id, so what is added up here is exact; what this
    adds is only which operations belong to which span.
    """

    own = device_us_by_op(gpu_activities) if gpu_activities else {}
    children = nested_spans(events)

    totals = [
        (own.get(event[13], 0.0) if len(event) > 13 else 0.0)
        for event in events
    ]
    return nested_sums(children, totals)


