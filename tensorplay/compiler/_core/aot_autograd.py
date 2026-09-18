"""Ahead-of-time autograd over dispatcher-level graphs.

``aot_module_simplified(module, example_inputs, fw_compiler=..., ...)``
turns a module (typically a captured :class:`GraphModule`) into a compiled
callable:

* Parameters and buffers are lifted into explicit inputs, so the traced
  graphs are pure functions of ``(parameters, buffers, user inputs)`` and see
  the module's current state on every call.
* Inference (no input requires grad, or grad mode is off): the forward is
  traced at dispatcher level and handed to ``inference_compiler``.
* Training: the forward and its backward are traced together -- the
  backward half is what the autograd engine dispatches inside
  ``tensorplay.autograd.grad`` -- into one joint graph whose backward nodes
  and tangent inputs are tagged ``is_backward``.  ``partition_fn`` splits it
  into a forward graph (user outputs plus saved values) and a backward graph
  (saved values plus tangents to input gradients); ``fw_compiler`` and
  ``bw_compiler`` compile them, and a TensorPlay ``autograd.Function`` runs
  the pair.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import tensorplay
from tensorplay.graph import GraphModule
from tensorplay.graph.experimental._dispatch_trace import (
    DispatchTracer,
    ProxyTensorDispatchMode,
)
from tensorplay.graph._pytree import tree_flatten, tree_unflatten
import tensorplay.random

from .aot import partition_default, partition_min_cut

__all__ = [
    "aot_function",
    "aot_module_simplified",
    "default_partition",
    "min_cut_rematerialization_partition",
]

def _is_tensor(value: Any) -> bool:
    return isinstance(value, tensorplay.Tensor)


def default_partition(joint_module: GraphModule, _joint_inputs: Sequence[Any], *, num_fwd_outputs: int):
    """Save every forward value the backward reads."""

    return partition_default(joint_module, num_fwd_outputs=num_fwd_outputs)


def min_cut_rematerialization_partition(
    joint_module: GraphModule,
    _joint_inputs: Sequence[Any],
    *,
    num_fwd_outputs: int,
    memory_budget: int | None = None,
):
    """Save the cheapest cut of forward values; recompute the rest in backward."""

    return partition_min_cut(
        joint_module, num_fwd_outputs=num_fwd_outputs, memory_budget=memory_budget
    )


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


def _placeholders(tracer: DispatchTracer, values: Sequence[Any], prefix: str, *, backward: bool = False):
    nodes = []
    for index, value in enumerate(values):
        node = tracer.graph.placeholder(f"{prefix}_{index + 1}")
        if backward:
            node.meta["is_backward"] = True
        if _is_tensor(value):
            tracer.track(value, node)
        else:
            node.meta["val"] = value
        nodes.append(node)
    return nodes


class _TaggingTracer(DispatchTracer):
    """Tags every node recorded while ``backward`` is set."""

    backward = False

    def record(self, func, args, kwargs, out):
        before = len(list(self.graph.nodes))
        node = super().record(func, args, kwargs, out)
        if self.backward:
            for recorded in list(self.graph.nodes)[before:]:
                recorded.meta["is_backward"] = True
        return node


def _trace_devices(values: Sequence[Any]) -> list[int]:
    devices = []
    for value in values:
        if _is_tensor(value) and value.device.type == "cuda":
            index = value.device.index or 0
            if index not in devices:
                devices.append(index)
    return devices


def _trace_inputs(primals: Sequence[Any]) -> list[Any]:
    """Private copies of the primals for a compile-time run.

    Tracing executes the program; running it on copies keeps in-place
    updates (running statistics, caches) away from the caller's tensors.
    """

    return [
        p.detach().clone().requires_grad_(p.requires_grad) if _is_tensor(p) else p
        for p in primals
    ]


def _trace_forward(fn: Callable[..., Any], primals: Sequence[Any], decompositions):
    tracer = DispatchTracer()
    trace_primals = _trace_inputs(primals)
    _placeholders(tracer, trace_primals, "primals")
    with tensorplay.random.fork_rng(devices=_trace_devices(primals)):
        with ProxyTensorDispatchMode(tracer, decompositions):
            out = fn(*trace_primals)
    flat_out, out_spec = tree_flatten(out)
    tracer.graph.output(tuple(tracer.map_value(v) for v in flat_out))
    return GraphModule(tracer.root, tracer.graph), out_spec, flat_out


def _trace_joint(fn: Callable[..., Any], primals: Sequence[Any], decompositions):
    """Trace forward + backward into one tagged joint graph.

    Returns ``(joint, out_spec, flat_out, num_fwd_outputs, tangent_examples)``.
    Tangent placeholders are created once the forward outputs are known and
    placed ahead of every operator node.
    """

    from tensorplay.utils._dispatch import _disable_current_modes

    tracer = _TaggingTracer()
    trace_primals = _trace_inputs(primals)
    primal_nodes = _placeholders(tracer, trace_primals, "primals")
    diff_inputs = [p for p in trace_primals if _is_tensor(p) and p.requires_grad]
    with tensorplay.random.fork_rng(devices=_trace_devices(primals)):
        with ProxyTensorDispatchMode(tracer, decompositions):
            with tensorplay.enable_grad():
                out = fn(*trace_primals)
            flat_out, out_spec = tree_flatten(out)
            diff_outputs = [o for o in flat_out if _is_tensor(o) and o.requires_grad]
            with _disable_current_modes():
                tangents = [tensorplay.ones_like(o.detach()) for o in diff_outputs]
            anchor = next(
                (n for n in tracer.graph.nodes if n.op != "placeholder"), None
            )
            with tracer.graph.inserting_before(anchor):
                _placeholders(tracer, tangents, "tangents", backward=True)
            tracer.backward = True
            grads = (
                tensorplay.autograd.grad(
                    diff_outputs, diff_inputs, tangents, allow_unused=True
                )
                if diff_outputs and diff_inputs
                else [None] * len(diff_inputs)
            )
            tracer.backward = False
    del primal_nodes
    fwd_values = [tracer.map_value(v) for v in flat_out]
    bwd_values = [None if g is None else tracer.map_value(g) for g in grads]
    tracer.graph.output(tuple(fwd_values + bwd_values))
    joint = GraphModule(tracer.root, tracer.graph)
    return joint, out_spec, flat_out, len(fwd_values), trace_primals, tangents


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def _call(compiled: Callable[..., Any], args: Sequence[Any]) -> tuple[Any, ...]:
    if getattr(compiled, "_boxed_call", False):
        result = compiled(list(args))
    else:
        result = compiled(*args)
    if isinstance(result, (tuple, list)):
        return tuple(result)
    return (result,)


def _bind_backward_inputs(bw_module, input_kinds, input_keys, saved, grad_outputs, primals, primal_names):
    saved_by_name = dict(saved)
    primal_by_name = dict(zip(primal_names, primals))
    tangent_by_name = dict(grad_outputs)
    values = []
    for placeholder, kind, key in zip(bw_module.graph.placeholders, input_kinds, input_keys):
        if kind == "tangent":
            values.append(tangent_by_name[placeholder.name])
        elif kind == "saved":
            values.append(saved_by_name[key])
        else:
            values.append(primal_by_name[key])
    return values


def aot_function(
    fn: Callable[..., Any],
    example_primals: Sequence[Any],
    *,
    fw_compiler: Callable[..., Any],
    bw_compiler: Callable[..., Any] | None = None,
    inference_compiler: Callable[..., Any] | None = None,
    partition_fn: Callable[..., Any] | None = None,
    decompositions: Mapping[Any, Callable[..., Any]] | None = None,
    keep_inference_input_mutations: bool = False,
) -> Callable[..., Any]:
    """Compile ``fn(*primals)`` (flat primals) with an AOT forward/backward."""

    del keep_inference_input_mutations  # mutations stay in the traced graph order
    bw_compiler = bw_compiler or fw_compiler
    inference_compiler = inference_compiler or fw_compiler
    partition_fn = partition_fn or default_partition
    primals = list(example_primals)
    needs_grad = tensorplay.is_grad_enabled() and any(
        _is_tensor(p) and p.requires_grad for p in primals
    )

    if not needs_grad:
        with tensorplay.no_grad():
            fw_module, out_spec, _ = _trace_forward(fn, primals, decompositions)
        compiled_fw = inference_compiler(fw_module, primals)

        def run_inference(*args: Any) -> Any:
            with tensorplay.no_grad():
                return tree_unflatten(list(_call(compiled_fw, args)), out_spec)

        run_inference._tensorplay_aot_graphs = (fw_module,)  # type: ignore[attr-defined]
        return run_inference

    joint, out_spec, flat_out, num_fwd, trace_primals, tangents = _trace_joint(
        fn, primals, decompositions
    )
    diff_out_mask = [_is_tensor(o) and o.requires_grad for o in flat_out]
    grad_mask = [_is_tensor(p) and p.requires_grad for p in primals]

    fw_module, bw_module, input_kinds, input_keys, saved_names = partition_fn(
        joint, trace_primals + tangents, num_fwd_outputs=num_fwd
    )
    primal_names = [p.name for p in joint.graph.placeholders if not p.meta.get("is_backward")]
    tangent_names = [p.name for p in joint.graph.placeholders if p.meta.get("is_backward")]

    fw_inputs = [
        trace_primals[primal_names.index(p.name)] for p in fw_module.graph.placeholders
    ]
    compiled_fw = fw_compiler(fw_module, fw_inputs)
    compiled_bw_box: list[Any] = []

    fw_input_order = [primal_names.index(p.name) for p in fw_module.graph.placeholders]

    def bw_example_inputs(saved, grad_outputs, run_primals):
        return _bind_backward_inputs(
            bw_module, input_kinds, input_keys, saved, grad_outputs, run_primals, primal_names
        )

    from tensorplay.autograd import Function

    class CompiledFunction(Function):
        @staticmethod
        def forward(ctx, *run_primals):
            outputs = _call(compiled_fw, [run_primals[i] for i in fw_input_order])
            user = outputs[:num_fwd]
            saved = list(zip(saved_names, outputs[num_fwd:]))
            ctx.saved_pairs = saved
            ctx.run_primals = run_primals
            ctx.run_outputs = user
            non_diff = [
                out for out, diff in zip(user, diff_out_mask) if _is_tensor(out) and not diff
            ]
            if non_diff:
                ctx.mark_non_differentiable(*non_diff)
            return tuple(user)

        @staticmethod
        def backward(ctx, *grad_outputs):
            diff_grads = [
                g if g is not None else tensorplay.zeros_like(out)
                for g, out, diff in zip(grad_outputs, ctx.run_outputs, diff_out_mask)
                if diff
            ]
            named = list(zip(tangent_names, diff_grads))
            inputs = bw_example_inputs(ctx.saved_pairs, named, ctx.run_primals)
            if not compiled_bw_box:
                compiled_bw_box.append(bw_compiler(bw_module, inputs))
            grads = iter(_call(compiled_bw_box[0], inputs))
            return tuple(next(grads) if needed else None for needed in grad_mask)

    def run_training(*args: Any) -> Any:
        if not tensorplay.is_grad_enabled():
            # Called under no_grad after a training compile: the forward graph
            # alone produces the outputs.
            outputs = _call(compiled_fw, [args[i] for i in fw_input_order])
            return tree_unflatten(list(outputs[:num_fwd]), out_spec)
        user = CompiledFunction.apply(*args)
        if not isinstance(user, tuple):
            user = (user,)
        return tree_unflatten(list(user), out_spec)

    run_training._tensorplay_aot_graphs = (joint, fw_module, bw_module)  # type: ignore[attr-defined]
    return run_training


def aot_module_simplified(
    module: Any,
    example_inputs: Sequence[Any],
    fw_compiler: Callable[..., Any],
    bw_compiler: Callable[..., Any] | None = None,
    inference_compiler: Callable[..., Any] | None = None,
    partition_fn: Callable[..., Any] | None = None,
    decompositions: Mapping[Any, Callable[..., Any]] | None = None,
    keep_inference_input_mutations: bool = False,
    **_: Any,
) -> Callable[..., Any]:
    """Compile a module whose parameters and buffers become graph inputs."""

    from tensorplay.func import functional_call

    named_state = list(_named_state(module))
    names = [name for name, _ in named_state]
    count = len(names)

    def flat_fn(*flat: Any) -> Any:
        state = dict(zip(names, flat[:count]))
        return functional_call(module, state, tuple(flat[count:]))

    example = [value for _, value in named_state] + list(example_inputs)
    compiled = aot_function(
        flat_fn,
        example,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
        partition_fn=partition_fn,
        decompositions=decompositions,
        keep_inference_input_mutations=keep_inference_input_mutations,
    )

    def forward(*args: Any) -> Any:
        state = [value for _, value in _named_state(module)]
        return compiled(*state, *args)

    forward._tensorplay_aot_graphs = getattr(compiled, "_tensorplay_aot_graphs", ())  # type: ignore[attr-defined]
    return forward


def _named_state(module: Any):
    named_parameters = getattr(module, "named_parameters", None)
    named_buffers = getattr(module, "named_buffers", None)
    if named_parameters is None:
        return []
    seen: set[int] = set()
    state = []
    for name, value in itertools.chain(named_parameters(), named_buffers() if named_buffers else ()):
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        state.append((name, value))
    return state
