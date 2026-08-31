"""Reusable error types and settings for graph minimization tools."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "GraphNetMinimizerBadModuleError",
    "GraphNetMinimizerResultMismatchError",
    "GraphNetMinimizerRunFuncError",
]


class GraphNetMinimizerBadModuleError(Exception):
    pass


class GraphNetMinimizerRunFuncError(Exception):
    pass


class GraphNetMinimizerResultMismatchError(Exception):
    pass


@dataclass
class _MinimizerSettingBase:
    accumulate_error: bool = False
    traverse_method: str = "sequential"
    find_all: bool = False
    return_intermediate: bool = False
    all_outputs: bool = False
