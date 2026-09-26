"""What is known about a kernel after it has been written, kept by name.

A compiled region is not always called under the name it was generated with.  A
call written for one device is handed to another, a template's chosen candidate
is invoked through the template's own name, and a graph that is partitioned
calls a piece under the name the partition gave it.  Each of those is a second
name for the same generated code, and a report about that code -- where it came
from, which piece of the graph produced it, what it was measured at -- is filed
under the name it was generated with.

So the second name is recorded alongside the first, and a lookup by either finds
the same record.  Nothing is copied: the alias points at the same list, so a
record filled in after the alias was made is visible through both.
"""

from __future__ import annotations

from typing import Any

#: Where each generated kernel was written from, by the name it was generated
#: under.  A stack of frames, innermost last, so the entry answers "what was
#: being lowered when this kernel was written".
_kernel_stack_trace: dict[str, list] = {}

#: Which post-grad node each generated kernel came from, by generated name.
#: A template's candidates and a partition's pieces all reach this under more
#: than one name, which is what the aliases below are for.
_kernel_to_post_grad_node_info: dict[str, list] = {}

#: What is known about each generated kernel, by generated name: its shape of
#: choices, the measured cost of each, and the configuration that was chosen.
_kernel_information_jsons: dict[str, dict[str, Any]] = {}


def alias_kernel_provenance(original_kernel_name: str, alias_kernel_name: str) -> None:
    """Expose one generated kernel's record under a second name.

    Only the names that already have a record gain an alias.  Aliasing a name
    that was never recorded would create an entry that looks complete and says
    nothing, which is worse than a lookup that misses.
    """

    if original_kernel_name in _kernel_stack_trace:
        _kernel_stack_trace[alias_kernel_name] = _kernel_stack_trace[
            original_kernel_name
        ]
    if original_kernel_name in _kernel_to_post_grad_node_info:
        _kernel_to_post_grad_node_info[alias_kernel_name] = (
            _kernel_to_post_grad_node_info[original_kernel_name]
        )


def record_kernel_stack_trace(kernel_name: str, frames: list) -> None:
    """File where a generated kernel was written from, under its own name."""

    _kernel_stack_trace[kernel_name] = frames


def record_kernel_post_grad_node_info(kernel_name: str, nodes: list) -> None:
    """File which post-grad node a generated kernel came from."""

    _kernel_to_post_grad_node_info[kernel_name] = nodes


def record_kernel_information(kernel_name: str, information: dict[str, Any]) -> None:
    """File what is known about one generated kernel's choices."""

    _kernel_information_jsons[kernel_name] = information


def kernel_stack_trace(kernel_name: str) -> list:
    """Where a generated kernel was written from, by either of its names."""

    return _kernel_stack_trace.get(kernel_name, [])


def kernel_post_grad_node_info(kernel_name: str) -> list:
    """Which post-grad node a generated kernel came from, by either name."""

    return _kernel_to_post_grad_node_info.get(kernel_name, [])


def kernel_information_jsons() -> dict[str, dict[str, Any]]:
    """What is known about every generated kernel, by generated name."""

    return _kernel_information_jsons


def reset_provenance() -> None:
    """Forget every record.

    Called between regions: the records name the region that produced them, so
    a record left over from the previous one would be attributed to whatever
    came next.
    """

    _kernel_stack_trace.clear()
    _kernel_to_post_grad_node_info.clear()
    _kernel_information_jsons.clear()


__all__ = [
    "alias_kernel_provenance",
    "kernel_information_jsons",
    "kernel_post_grad_node_info",
    "kernel_stack_trace",
    "record_kernel_information",
    "record_kernel_post_grad_node_info",
    "record_kernel_stack_trace",
    "reset_provenance",
]
