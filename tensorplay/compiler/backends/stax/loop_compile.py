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
#:
#: One name for the whole route rather than one per stage: a plan this route
#: declines is a plan it declines, whichever stage declined it, so a caller
#: that keeps its other routes catches this and nothing has to know which
#: stage was reached.  Anything else that goes wrong is a bug and is left to
#: surface, rather than being read as "unsupported".
NotLowerable = PlanError
from .graph_lowering import GraphLowering
from .loop_fusion import ExternNode, FusedGroup, KernelScheduler
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


_tp_debug_plans = __import__('os').environ.get('TP_DEBUG_PLANS') == '1'


def _steps_for(graph, groups) -> list:
    scheduler: KernelScheduler = graph.scheduler
    buffers = graph.name_to_buffer
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
                launcher, ptr_names = compile_group(
                    group, buffers, stored, config, graph.sizevars
                )
                break
            except PlanError as exc:
                if __debug__ and _tp_debug_plans:
                    print(f"[stax] candidate declined: {exc}")
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


def compile_half_host(graph_module, example_inputs, **options):
    """Compile one graph half for the host, from the same IR.

    The host prints a scheduled group with its own emitter and runs the result
    through the same runtime the accelerator's kernels run through, so what a
    region does is decided in one place and only the printing differs.
    """

    del options
    from .codegen import loop_cpp
    from .codegen.cpp import build_cpu_native_kernel
    from .loop_runtime import ExternStep, HostStep, LoopProgram

    graph = GraphLowering(graph_module, list(example_inputs)).run()
    plan = KernelScheduler(graph)
    graph.scheduler = plan
    steps = []
    for group in plan.fuse():
        if isinstance(group, ExternNode):
            steps.append(ExternStep(group.kernel))
            continue
        stored = stored_names(plan, group)
        try:
            program = loop_cpp.flatten(group, graph.name_to_buffer, stored, graph)
        except loop_cpp.HostPlanError as exc:
            raise PlanError(str(exc)) from exc
        shapes, strides = [], []
        for name in program["inputs"]:
            buffer = graph.name_to_buffer[name]
            shapes.append(tuple(int(s) for s in buffer.get_size()))
            strides.append(tuple(int(s) for s in buffer.layout.stride))
        # The layouts pin the specialization, so a call whose operands do not
        # match them is not this kernel.
        launch = build_cpu_native_kernel(
            program["instructions"], program["constants"],
            program["input_count"], program["output_ref"],
            input_shapes=tuple(shapes), input_strides=tuple(strides),
        )
        if launch is None:
            raise PlanError("the host emitter declined a group")
        steps.append(HostStep(launch, program["inputs"], _stored_name(stored)))
    program = LoopProgram(graph, steps)
    program._tensorplay_codegen = "cpp"  # type: ignore[attr-defined]
    program._tensorplay_backward_codegen = "cpp"  # type: ignore[attr-defined]
    return program


def _stored_name(stored) -> str:
    names = sorted(stored)
    if len(names) != 1:
        raise PlanError("only a single-result group is printed on the host")
    return names[0]


__all__ = ["compile_graph", "compile_half", "compile_half_host", "stored_names"]


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
