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

import logging
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


#: A counter that makes every kernel's record name unique.  Two kernels
#: generated from the same body would otherwise file under one name, and the
#: second would overwrite the first's record.
_provenance_handle = 0

#: Whether the record is being kept at all.  Keeping it costs a stack walk per
#: kernel, which is why it is off unless something is watching.
_provenance_enabled = False

#: The two views of a region, one taken before its pieces are fused and one
#: after, so that a fusion decision can be read back against what it was
#: deciding between.
ir_pre_fusion_log = logging.getLogger("tensorplay.stax.ir_pre_fusion")
ir_post_fusion_log = logging.getLogger("tensorplay.stax.ir_post_fusion")


def _write_ir(nodes) -> str:
    """The region's pieces as text, one per line, each naming its own result."""

    lines = []
    for node in nodes:
        name = getattr(node, "get_name", lambda: "<unnamed>")()
        lines.append(f"  {name}")
    return "\n".join(lines)


def set_provenance_tracing(enabled: bool) -> None:
    """Keep a record of what each kernel was made from, or stop keeping one.

    Off by default: the record costs a stack walk per kernel, and a run that
    nobody is going to read the record of should not pay for it.
    """

    global _provenance_enabled
    _provenance_enabled = bool(enabled)


def provenance_tracing_enabled() -> bool:
    """Whether a record is being kept."""

    return _provenance_enabled


def set_kernel_post_grad_provenance_tracing(
    node_schedule,
    kernel_name: str,
    is_extern: bool = False,
) -> int | None:
    """Record which pieces of the region a generated kernel came from.

    Returns a number that is unique to this kernel, or nothing when no record is
    being kept.  The number goes into the generated code as a comment, so that
    source read later can be tied back to the record without having to match
    the two by name.
    """

    global _provenance_handle
    if not _provenance_enabled:
        return None

    _provenance_handle += 1
    # The name carries the handle, so two kernels made from the same body are
    # still told apart.
    recorded_name = f"{kernel_name}:{_provenance_handle}"
    names = [
        getattr(node, "get_name", lambda: "<unnamed>")() for node in node_schedule
    ]
    if is_extern:
        _kernel_to_post_grad_node_info[recorded_name] = names
    else:
        _kernel_to_post_grad_node_info[recorded_name] = names
    # The stack is what says which call site produced the piece, which the list
    # of names alone does not.
    record_kernel_stack_trace(recorded_name, _stack())
    return _provenance_handle


def _stack() -> list:
    """The calls that led here, innermost last, without this module's frames."""

    import traceback

    frames = []
    # walk_stack yields (frame, lineno) pairs, outermost first.
    for frame, lineno in traceback.walk_stack(None):
        name = frame.f_globals.get("__name__", "")
        if name == __name__:
            continue
        frames.append(f"{name}:{lineno}")
    return frames


def log_ir_pre_fusion(nodes) -> None:
    """The region's pieces as they were before they were fused.

    Written to its own log rather than the ordinary one, because what is wanted
    is the state on one side of a decision, and the ordinary log is read for
    what went wrong rather than for what was considered.
    """

    if ir_pre_fusion_log.isEnabledFor(logging.INFO):
        ir_pre_fusion_log.info("BEFORE FUSION\n%s", _write_ir(nodes))


def log_ir_post_fusion(nodes) -> None:
    """The region's pieces as they were after they were fused."""

    if ir_post_fusion_log.isEnabledFor(logging.INFO):
        ir_post_fusion_log.info("AFTER FUSION\n%s", _write_ir(nodes))


__all__ = [
    "alias_kernel_provenance",
    "ir_post_fusion_log",
    "ir_pre_fusion_log",
    "log_ir_post_fusion",
    "log_ir_pre_fusion",
    "provenance_tracing_enabled",
    "set_kernel_post_grad_provenance_tracing",
    "set_provenance_tracing",
    "kernel_information_jsons",
    "kernel_post_grad_node_info",
    "kernel_stack_trace",
    "record_kernel_information",
    "record_kernel_post_grad_node_info",
    "record_kernel_stack_trace",
    "reset_provenance",
]
