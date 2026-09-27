"""Compile a captured graph into scheduled Triton kernels.

The path is: lower the graph into loop IR, group the nests into kernels, emit
and compile each kernel, then hand the ordered steps to the runtime.  A group
that cannot be expressed as one kernel falls back to compiling its nodes as
separate kernels rather than failing the whole program.
"""

from __future__ import annotations

from tensorplay._higher_order_ops._hop_base import FakeTensorMode

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

    from .loops import set_fake_mode
    from .virtualized import V as shared_V

    with set_fake_mode(FakeTensorMode(allow_non_fake_inputs=True)), \
            shared_V.set_fake_mode(FakeTensorMode(allow_non_fake_inputs=True)):
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


def host_launches(graph, plan=None, groups=None):
    """One entry per fused group of a lowered region, in run order.

    An entry is the group's built launcher, or nothing where the group is a call
    that is run rather than printed.  The printing is shared by every way of
    asking for a built form of a region, so that what a caller gets back does
    not depend on which of them it asked: a group is grouped the same way, laid
    out the same way, and printed the same way whichever way it was asked for.
    """

    from .codegen import loop_cpp
    from .codegen.cpp import build_cpu_native_kernel
    if plan is None:
        plan = KernelScheduler(graph)
        graph.scheduler = plan
        groups = plan.fuse()
    elif groups is None:
        graph.scheduler = plan
        groups = plan.fuse()
    launches = []
    for group in groups:
        if isinstance(group, ExternNode):
            # A call that is run rather than printed has nothing built to launch,
            # but it still takes its place in the run order, so the position is
            # kept rather than dropped: a caller that walks the groups alongside
            # this list is then walking them in the order they run in.
            launches.append(None)
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
        result = graph.name_to_buffer[_stored_name(stored)]
        launch = build_cpu_native_kernel(
            program["instructions"], program["constants"],
            program["input_count"], program["output_ref"],
            input_shapes=tuple(shapes), input_strides=tuple(strides),
            shape=tuple(int(s) for s in result.get_size()),
        )
        if launch is None:
            raise PlanError("the host emitter declined a group")
        launches.append((launch, program["inputs"], _stored_name(stored)))
    return launches


def compile_half_host(graph_module, example_inputs, **options):
    """Compile one graph half for the host, from the same IR.

    The host prints a scheduled group with its own emitter and runs the result
    through the same runtime the accelerator's kernels run through, so what a
    region does is decided in one place and only the printing differs.
    """

    del options
    from .loop_runtime import ExternStep, HostStep, LoopProgram
    from .loops import set_fake_mode
    from .virtualized import V as shared_V

    with set_fake_mode(FakeTensorMode(allow_non_fake_inputs=True)), \
            shared_V.set_fake_mode(FakeTensorMode(allow_non_fake_inputs=True)):
        graph = GraphLowering(graph_module, list(example_inputs)).run()
    plan = KernelScheduler(graph)
    graph.scheduler = plan
    # Fusing is asked once and the groups are kept: fusing walks the region and
    # settles it, so asking twice would be asking about a region that has already
    # been through it.
    groups = plan.fuse()
    # One step per group, taken in the order the groups run in, so that what
    # produces a value is never behind what reads it.
    steps = []
    for group, launch in zip(groups, host_launches(graph, plan, groups)):
        if launch is None:
            steps.append(ExternStep(group.kernel))
        else:
            built, inputs, output = launch
            steps.append(HostStep(built, inputs, output))
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
    try:
        return compile_graph(graph_module, list(example_inputs))
    except NotLowerable:
        raise
    except NotImplementedError:
        raise
    except (AttributeError, KeyError, TypeError, AssertionError, IndexError) as exc:
        # A form this route's own machinery does not cover yet.  The route says
        # a group it cannot express is compiled as separate kernels rather than
        # failing the program, and a region it cannot lower at all is the same
        # answer one level up: the caller keeps its other route, and saying
        # which form arrived is what lets that coverage be measured.
        raise NotLowerable(
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _traced_value_of(joint, name: str):
    """The value a saved node held when the joint graph was traced."""

    for node in joint.graph.nodes:
        if node.name == name or node.target == name:
            value = node.meta.get("val")
            if value is not None:
                return value
    raise KeyError(f"no traced value for saved node {name!r}")


def record_original_output_strides(gm) -> None:
    """Note the strides each of a graph's outputs had when the graph was traced.

    Recorded once and then left alone, because a pass over the graph may pad a
    result to make it easier to compute, and a later reader asking what the graph
    produces is asking what the program asked for rather than what the pass made
    of it.  Overwriting on a second call would replace the answer to that question
    with the intermediate one.
    """

    import tensorplay as tp

    output_node = gm.graph.find_nodes(op="output")[0]
    if "original_output_strides" in output_node.meta:
        return

    # The output node wraps what it yields, so a graph yielding one value still
    # holds it in a one-element sequence, and a graph yielding several holds them
    # bare.
    outputs = output_node.args[0]
    if not _is_fx_node(outputs):
        outputs = (outputs,)

    strides = []
    for output in outputs:
        val = output.meta.get("val") if _is_fx_node(output) else None
        strides.append(
            tuple(int(x) for x in val.stride())
            if val is not None and isinstance(val, tp.Tensor)
            else None
        )
    output_node.meta["original_output_strides"] = strides


def _is_fx_node(value) -> bool:
    """Whether a value is a graph node rather than what a node stands for."""

    return hasattr(value, "op") and hasattr(value, "users")


