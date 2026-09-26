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

#: What "this route cannot express this region" looks like from outside.
NotLowerable = PlanError
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
    program = LoopProgram(graph, steps)
    # This artifact runs generated kernels, and it carries no reverse pass.
    program._tensorplay_codegen = "triton"  # type: ignore[attr-defined]
    program._tensorplay_backward_codegen = None  # type: ignore[attr-defined]
    return program


__all__ = ["compile_graph", "compile_half", "stored_names"]



class NotLowerable(Exception):
    """This region is not one the loop IR can express.

    The caller keeps its other routes.  Anything else that goes wrong is a
    bug and is left to surface, rather than being read as "unsupported".
    """


def compile_half(graph_module, example_inputs, **options):
    """Compile one graph half into a callable.

    A half is one piece of a region: the forward, the backward, or a region
    that does not differentiate at all.  Compiling it means lowering it into
    loop IR, grouping the nests into kernels, emitting and compiling each
    kernel, and handing the ordered steps to the runtime.  A group that
    cannot be expressed as one kernel falls back to compiling its nodes as
    separate kernels rather than failing the whole half.

    The returned callable takes exactly the inputs it was handed here.  Which
    values those are for a given half -- a forward's placeholders, a
    backward's saved values and incoming gradients -- is the caller's
    business: this function compiles a half and nothing else.
    """

    del options
    return compile_graph(graph_module, list(example_inputs))


def _traced_value_of(joint, name: str):
    """The value a saved node held when the joint graph was traced."""

    for node in joint.graph.nodes:
        if node.name == name or node.target == name:
            value = node.meta.get("val")
            if value is not None:
                return value
    raise KeyError(f"no traced value for saved node {name!r}")
