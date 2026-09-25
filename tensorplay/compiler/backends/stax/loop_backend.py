"""Whole-graph loop compilation, exposed as a compiler backend.

The captured training region is wrapped ahead of time: the joint forward and
backward graph is split by the min-cut partitioner, and each half is lowered
into loop IR, scheduled into kernels and compiled.  A region that needs no
gradient compiles the forward half alone.

This is the same shape as the flat-program backend, but the unit of work is
a loop nest rather than an operator chain, so a normalization, its moments,
the activation after it and the casts around them can end up in one kernel.
"""

from __future__ import annotations

from typing import Any, Sequence

from .loop_compile import compile_graph


def _compile_half(graph_module, example_inputs: Sequence[Any]):
    """Compile one partitioned graph half into a callable."""

    return compile_graph(graph_module, list(example_inputs))


def compile_module(module, example_inputs: Sequence[Any], **options: Any):
    """Compile ``module`` with loop-IR kernels; returns a callable."""

    from ..._core.aot_autograd import (
        aot_module_simplified,
        min_cut_rematerialization_partition,
    )

    return aot_module_simplified(
        module,
        example_inputs,
        fw_compiler=_compile_half,
        bw_compiler=_compile_half,
        partition_fn=min_cut_rematerialization_partition,
        **options,
    )


__all__ = ["compile_module"]
