"""Apply tensor-parallel styles to modules and submodules."""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

from .._api import _current_mesh
from .style import ParallelStyle

__all__ = ["parallelize_module"]


def parallelize_module(
    module: Any,
    device_mesh: Any = None,
    parallelize_plan: ParallelStyle | dict[str, ParallelStyle] | None = None,
    *,
    src_data_rank: int | None = 0,
) -> Any:
    mesh = device_mesh or _current_mesh()
    if parallelize_plan is None:
        return module
    if isinstance(parallelize_plan, ParallelStyle):
        parallelize_plan.src_data_rank = src_data_rank
        return parallelize_plan._apply(module, mesh)
    if not isinstance(parallelize_plan, dict):
        raise TypeError("parallelize_plan must be a ParallelStyle or a mapping")
    for path, style in parallelize_plan.items():
        if not isinstance(style, ParallelStyle):
            raise TypeError(f"parallelize plan entry {path!r} is not a ParallelStyle")
        if path == "":
            parallelize_module(module, mesh, style, src_data_rank=src_data_rank)
            continue
        tokens = path.split(".")
        matched = [(name, child) for name, child in module.named_children() if fnmatch(name, tokens[0])]
        if not matched:
            raise ValueError(f"parallelize plan path {path!r} does not match a child module")
        tail = ".".join(tokens[1:])
        for _, child in matched:
            parallelize_module(child, mesh, {tail: style} if tail else style, src_data_rank=src_data_rank)
    return module
