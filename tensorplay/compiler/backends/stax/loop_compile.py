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



def _compile_half(graph_module, example_inputs):
    """Compile one partitioned graph half into a callable."""

    return compile_graph(graph_module, list(example_inputs))


def _joint_example(bw_module, input_kinds, input_keys, saved, tangents, primals, primal_names):
    from ..._core.aot_autograd import _bind_backward_inputs

    named = list(zip([name for name, _ in tangents], [value for _, value in tangents]))
    return _bind_backward_inputs(
        bw_module, input_kinds, input_keys, saved, named, primals, primal_names
    )


def compile_region(module, example_inputs, **options):
    """Compile one captured region with loop-IR kernels.

    The training region is wrapped ahead of time: the joint forward and
    backward graph is split by the min-cut partitioner, and each half is
    lowered into loop IR, scheduled into kernels and compiled.

    Both halves are compiled *here*, before the region is handed back.  A
    backward half is otherwise only built on the first backward call, which
    would turn "this route cannot express that half" into a failure at
    training time instead of a fallback at compile time.

    Returns ``None`` when the region is not one this route can express, so
    the caller keeps its other routes.
    """

    import tensorplay
    from tensorplay.autograd import Function
    from tensorplay.graph._pytree import tree_unflatten

    from ..._core.aot import partition_min_cut
    from ..._core.aot_autograd import (
        _call,
        _functionalize,
        _state_access,
        _trace_inputs,
        _trace_joint,
    )

    names, read_state, substitute = _state_access(module)
    count = len(names)

    def flat_fn(*flat):
        if count == 0:
            return module(*flat)
        with substitute(dict(zip(names, flat[:count]))):
            return module(*flat[count:])

    primals = read_state() + list(example_inputs)
    joint, out_spec, flat_out, num_fwd, trace_primals, tangents = _trace_joint(
        flat_fn, primals, options.get("decompositions")
    )
    joint = _functionalize(
        joint, _trace_inputs(trace_primals) + [t.clone() for t in tangents]
    )
    fw_module, bw_module, input_kinds, input_keys, saved_names = partition_min_cut(
        joint, trace_primals + tangents, num_fwd_outputs=num_fwd
    )
    primal_names = [
        node.name
        for node in joint.graph.placeholders
        if not node.meta.get("is_backward")
    ]
    tangent_names = [
        node.name for node in joint.graph.placeholders if node.meta.get("is_backward")
    ]
    fw_order = [primal_names.index(node.name) for node in fw_module.graph.placeholders]
    fw_inputs = [trace_primals[i] for i in fw_order]
    compiled_fw = _compile_half(fw_module, fw_inputs)

    # A backward that reads a value the forward did not save would fail only
    # when it runs, so the traced values stand in for the saved ones here.
    saved_pairs = [(name, _traced_value_of(joint, name)) for name in saved_names]
    grad_pairs = [
        (tangent_names[i], flat_out[i].detach())
        for i in range(num_fwd)
        if flat_out[i].requires_grad
    ]
    bw_inputs = _joint_example(
        bw_module, input_kinds, input_keys, saved_pairs, grad_pairs, trace_primals, primal_names
    )
    compiled_bw = _compile_half(bw_module, bw_inputs)

    diff_out = [tensorplay.is_tensor(o) and o.requires_grad for o in flat_out]
    grad_mask = [tensorplay.is_tensor(p) and p.requires_grad for p in primals]

    class CompiledRegion(Function):
        @staticmethod
        def forward(ctx, *run_primals):
            outputs = _call(compiled_fw, [run_primals[i] for i in fw_order])
            user = outputs[:num_fwd]
            ctx.saved_pairs = list(zip(saved_names, outputs[num_fwd:]))
            ctx.run_primals = run_primals
            ctx.run_outputs = user
            plain = [
                out
                for out, diff in zip(user, diff_out)
                if tensorplay.is_tensor(out) and not diff
            ]
            if plain:
                ctx.mark_non_differentiable(*plain)
            return tuple(user)

        @staticmethod
        def backward(ctx, *grad_outputs):
            grads = [
                g if g is not None else tensorplay.zeros_like(out)
                for g, out, diff in zip(grad_outputs, ctx.run_outputs, diff_out)
                if diff
            ]
            inputs = _joint_example(
                bw_module,
                input_kinds,
                input_keys,
                ctx.saved_pairs,
                list(zip(tangent_names, grads)),
                ctx.run_primals,
                primal_names,
            )
            produced = iter(_call(compiled_bw, inputs))
            return tuple(
                next(produced) if need else None for need in grad_mask
            )

    state = {"apply": CompiledRegion.apply}

    def call(*args):
        if not tensorplay.is_grad_enabled():
            outputs = _call(compiled_fw, [args[i] for i in fw_order])
            return tree_unflatten(list(outputs[:num_fwd]), out_spec)
        user = state["apply"](*args)
        if not isinstance(user, tuple):
            user = (user,)
        return tree_unflatten(list(user), out_spec)

    return call


def _traced_value_of(joint, name: str):
    """The value a saved node held when the joint graph was traced."""

    for node in joint.graph.nodes:
        if node.name == name or node.target == name:
            value = node.meta.get("val")
            if value is not None:
                return value
    raise KeyError(f"no traced value for saved node {name!r}")
