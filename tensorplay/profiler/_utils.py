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


__all__ = [
    "dtype_name",
    "event_gpu_us",
    "rank_world",
    "self_cuda_us",
    "self_times",
]
