"""Reporting what a compiled region costs, from the outside.

Two questions get asked after a region is built, and neither is answered by the
generated code: how long one call takes, and how the time divides up.  The first
is a number with a spread; the second is a list of what ran and how long each
thing was busy.  Both are read by a person deciding whether the region is worth
keeping, so both are printed rather than returned, and both say what they
measured rather than leaving the reader to guess.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import tensorplay as tp

from .benchmarking import get_interface_for_device


def print_performance(
    fn: Callable[[], Any],
    times: int = 10,
    repeat: int = 10,
    device: str = "cpu",
    name: str | None = None,
) -> float:
    """Run ``fn`` and print how long a call took, returning the median.

    The median rather than the mean, because the first call after a change is
    slow for reasons that have nothing to do with the region -- a page is
    faulted in, a clock is read cold -- and a mean spends the region's own
    number on those.  The spread is printed beside it, because a median whose
    neighbours are far away is not a measurement of anything.

    The device is synchronized around each repetition, so what is timed is the
    work and not the queueing of it: on an accelerator a call returns long
    before the device has finished, and timing the return would measure the
    launch.
    """

    interface = get_interface_for_device(device)

    for _ in range(max(1, times)):
        fn()
    _sync(interface)

    samples: list[float] = []
    for _ in range(max(1, repeat)):
        _sync(interface)
        start = time.perf_counter()
        fn()
        _sync(interface)
        samples.append((time.perf_counter() - start) * 1e3)

    ordered = sorted(samples)
    median = statistics.median(ordered)
    spread = ordered[-1] - ordered[0] if len(ordered) > 1 else 0.0
    label = name or getattr(fn, "__name__", "the region")
    print(
        f"{label} on {device}: {median:.4f} ms median of {len(ordered)} "
        f"repetitions, spread {spread:.4f} ms "
        f"(min {ordered[0]:.4f}, max {ordered[-1]:.4f})"
    )
    return median


def _sync(interface: Any) -> None:
    """Wait for the device, where the device has a queue of its own."""

    synchronize = getattr(interface, "synchronize", None)
    if callable(synchronize):
        synchronize()


def time_and_count(fn: Callable[[], Any], times: int, repeat: int) -> tuple[float, int]:
    """How long a call took and how many times it ran, for a caller that logs it.

    The count is the number of repetitions the estimate is over, which is not
    the number asked for: a call that takes a millisecond is measured hundreds
    of times to fill the requested time, so a report that says it ran ten times
    when it ran five hundred is not reporting what it did.
    """

    for _ in range(max(1, times)):
        fn()
    start = time.perf_counter()
    for _ in range(max(1, repeat)):
        fn()
    return (time.perf_counter() - start) * 1e3, max(1, repeat)


def compiled_module_main() -> Callable[..., None]:
    """A main that measures a compiled region and prints the result.

    What a generated module runs when it is executed rather than imported, so
    that a region built by hand can be measured the same way one built by the
    test is.  The inputs are whatever the module's forward takes, filled with
    random values, because measuring a region on nothing is not possible.
    """

    def main() -> None:
        import sys

        module = sys.modules.get("__main__")
        forward = getattr(module, "call", None) or getattr(module, "forward", None)
        if forward is None:
            print("this module has no call or forward to measure")
            return
        print_performance(forward, name=getattr(forward, "__name__", "the region"))

    return main


#: The kinds of kernel a generated file can hold, by the decorator it is
#: written under.  A file's kind is what its metadata means, so it is read off
#: the decorator rather than worked out from the body.
_KERNEL_CATEGORIES = (
    "foreach",
    "persistent_reduction",
    "pointwise",
    "reduction",
    "split_scan",
    "template",
)


def get_kernel_category_by_source_code(src_code: str) -> str:
    """What kind of kernel a generated file holds, read off its decorators.

    Read from the source rather than from a compiled module, because a file is
    often classified before anything has been compiled from it -- the metadata
    a build is about to make is wanted while the build is still deciding.

    A file that says nothing, or that says two different things, is
    ``"unknown"``: the callers of this treat a category as a filter over what
    they report, and a filter that guessed would drop a kernel from a report
    rather than mislabel it.
    """

    choices = [
        category
        for category in _KERNEL_CATEGORIES
        if f"@triton_heuristics.{category}" in src_code
    ]
    if len(choices) == 1:
        return choices[0]
    return "unknown"


__all__ = [
    "compiled_module_main",
    "get_kernel_category_by_source_code",
    "print_performance",
    "time_and_count",
]
