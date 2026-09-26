"""Compile a captured graph into scheduled Triton kernels.

The path is: lower the graph into loop IR, group the nests into kernels, emit
and compile each kernel, then hand the ordered steps to the runtime.  A group
that cannot be expressed as one kernel falls back to compiling its nodes as
separate kernels rather than failing the whole program.
"""

from __future__ import annotations

import typing
from typing import Any, Callable

from .codegen.loop_triton import (
    LaunchConfig,
    PlanError,
    compile_group,
    config_candidates,
)
from .graph_lowering import GraphLowering
from .kernel_scheduler import ExternNode, FusedGroup, KernelScheduler
from .loop_runtime import ExternStep, FusedStep, LoopProgram


def stored_names(scheduler: KernelScheduler, group: FusedGroup) -> set:
    """Buffers of ``group`` that something outside it still needs.

    Everything else stays in registers: a value nobody else reads is never
    written to memory.
    """

    inside = {id(node) for node in group.nodes}
    out = set()
    for name in group.names:
        if name in scheduler.output_names:
            out.add(name)
        readers = scheduler.users.get(name)
        if not readers or not readers <= inside:
            out.add(name)
    return out


def _steps_for(graph, groups) -> list:
    scheduler: KernelScheduler = graph.scheduler
    buffers = graph.buffers
    steps = []
    for group in groups:
        if isinstance(group, ExternNode):
            steps.append(ExternStep(group.kernel))
            continue
        stored = stored_names(scheduler, group)
        candidates = config_candidates(group, buffers)
        launcher = None
        ptr_names: list = []
        for config in candidates:
            try:
                launcher, ptr_names = compile_group(group, buffers, stored, config)
                break
            except PlanError:
                continue
        if launcher is None:
            raise PlanError(
                f"no launchable kernel for a group of {len(group.nodes)} nests"
            )
        steps.append(FusedStep(launcher, ptr_names, stored, buffers))
    return steps


def compile_graph(graph_module, example_inputs, *, scheduler: KernelScheduler | None = None) -> LoopProgram:
    """Lower, schedule and compile ``graph_module``; returns a callable."""

    graph = GraphLowering(graph_module, list(example_inputs)).run()
    plan = scheduler if scheduler is not None else KernelScheduler(graph)
    graph.scheduler = plan
    groups = plan.fuse()
    steps = _steps_for(graph, groups)
    return LoopProgram(graph, steps)


__all__ = ["compile_graph", "compile_region", "stored_names"]


def compile_region(module, example_inputs, **options):
    """Compile one captured region with loop-IR kernels; returns a callable.

    The training region is wrapped ahead of time: the joint forward and
    backward graph is split by the min-cut partitioner and each half is
    lowered into loop IR, scheduled into kernels and compiled.  A region that
    needs no gradient compiles the forward half alone.
    """

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


def _compile_half(graph_module, example_inputs):
    return compile_graph(graph_module, list(example_inputs))
