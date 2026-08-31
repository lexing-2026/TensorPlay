"""Memory accounting and lightweight allocation summaries."""

from __future__ import annotations

import collections


FACTORY_PREFIXES = (
    "empty",
    "zeros",
    "ones",
    "full",
    "rand",
    "randn",
    "randint",
    "randperm",
    "arange",
    "linspace",
    "logspace",
    "eye",
    "tensor",
    "scalar_tensor",
)


def export_memory_timeline(mem_events, path):
    """Write a CSV timeline and return its rows."""
    live = collections.defaultdict(int)
    rows = []
    for ts, _ptr, nbytes, allocated, is_cuda, device, _stream, _tid in sorted(mem_events, key=lambda event: event[0]):
        live[(is_cuda, device)] = max(0, live[(is_cuda, device)] + (nbytes if allocated else -nbytes))
        device_name = f"cuda:{device}" if is_cuda else "cpu"
        rows.append((ts, device_name, sum(live.values())))

    with open(path, "w") as file:
        file.write("timestamp_ns,device,allocated_bytes\n")
        for ts, device_name, total in rows:
            file.write(f"{ts},{device_name},{total}\n")
    return rows


def allocator_memory_summary(mem_events):
    """Return total allocated bytes, peak live bytes and a compact timeline."""
    timeline = []
    total = 0
    peak = 0
    live = 0
    for ts, _ptr, nbytes, allocated, *_rest in sorted(mem_events, key=lambda event: event[0]):
        delta = nbytes if allocated else -nbytes
        if allocated:
            total += nbytes
        live = max(live + delta, 0)
        peak = max(peak, live)
        timeline.append((ts, nbytes, "alloc" if allocated else "free"))
    return total, peak, timeline


def factory_memory_summary(events, factory_prefixes=FACTORY_PREFIXES):
    """Estimate allocation volume from output sizes stored on factory events."""
    timeline = []
    for event in events:
        if len(event) <= 9:
            continue
        name, start_ns, nbytes = event[0], event[2], event[9]
        base_name = name.split(".", 1)[0]
        if base_name not in factory_prefixes and not base_name.endswith("_like"):
            continue
        if nbytes is None or nbytes <= 0:
            continue
        timeline.append((start_ns, nbytes, name))

    timeline.sort()
    total = sum(nbytes for _ts, nbytes, _name in timeline)
    peak = 0
    live = 0
    for _ts, nbytes, _name in timeline:
        live += nbytes
        peak = max(peak, live)
    return total, peak, timeline


def memory_summary(mem_events, events):
    """Select exact allocator accounting or the shape-based fallback."""
    if mem_events:
        return allocator_memory_summary(mem_events)
    return factory_memory_summary(events)


__all__ = [
    "FACTORY_PREFIXES",
    "allocator_memory_summary",
    "export_memory_timeline",
    "factory_memory_summary",
    "memory_summary",
]
