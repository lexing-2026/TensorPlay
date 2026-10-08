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

import contextlib
import functools

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

def _mark_user_outputs(module: Any, count: int) -> None:
    """Record which of a region's results the program's caller reads.

    A backend that chooses how values are laid out leaves the results a
    caller reads as the program produced them and is free with the rest.  The
    first ``count`` results are the caller's; the strides every result was
    traced with are recorded beside them.
    """

    graph = getattr(module, "graph", None)
    find_nodes = getattr(graph, "find_nodes", None)
    if find_nodes is None:
        return
    output_nodes = find_nodes(op="output")
    if not output_nodes:
        return
    output_node = output_nodes[0]
    args = output_node.args
    if args and isinstance(args[0], (tuple, list)):
        args = args[0]
    strides = []
    for arg in args:
        val = getattr(arg, "meta", {}).get("val") if hasattr(arg, "meta") else None
        try:
            strides.append(tuple(int(s) for s in val.stride()) if _is_tensor(val) else None)
        except (TypeError, ValueError):
            strides.append(None)
    output_node.meta["user_visible_output_idxs"] = list(range(min(count, len(args))))
    output_node.meta["original_output_strides"] = strides


def _holds_tensor(value: Any) -> bool:
    """Whether a list, tuple or dict argument has a tensor somewhere inside."""
    if isinstance(value, (list, tuple)):
        return any(_is_tensor(v) or _holds_tensor(v) for v in value)
    if isinstance(value, dict):
        return any(_is_tensor(v) or _holds_tensor(v) for v in value.values())
    return False


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
    tracer.graph.eliminate_dead_code()
    tracer._tracked.clear()
    return GraphModule(tracer.root, tracer.graph), out_spec, flat_out


def _trace_joint(fn: Callable[..., Any], primals: Sequence[Any], decompositions):
    """Trace forward + backward into one tagged joint graph.

    Returns ``(joint, out_spec, flat_out, num_fwd_outputs, trace_primals,
    tangent_examples, tangent_names, diff_outputs)``.  Tangent placeholders
    are created once the forward outputs are known and placed ahead of every
    operator node.  Their names and the outputs they stand for travel back
    with the graph: a caller that re-derived the set from the returned
    outputs would be reading ``requires_grad`` off proxies whose graph has
    since been rebuilt, and would come back with a different answer than the
    one the placeholders were created from.
    """

    from tensorplay.primitives.rng_prims import PhiloxStateTracker
    from tensorplay.primitives.common import CUDARngStateHelper
    from tensorplay.utils._dispatch import _disable_current_modes

    tracer = _TaggingTracer()
    trace_primals = _trace_inputs(primals)
    primal_nodes = _placeholders(tracer, trace_primals, "primals")
    diff_inputs = [p for p in trace_primals if _is_tensor(p) and p.requires_grad]
    # A traced program that reads a random value reads it at a position rather
    # than from a generator, so the position each pass starts from is what the
    # reads are written against: the forward and the backward each get their
    # own, taken from where the generators stand now.  Only device streams
    # with a Philox/counter-based generator expose a readable position; a
    # CPU-only trace keeps its native generator and leaves the state unset.
    if tensorplay.cuda.is_available():
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple()
        bwd_seed, bwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple()
        as_state = lambda pair: (
            tensorplay.tensor(pair[0], dtype=tensorplay.int64),
            tensorplay.tensor(pair[1], dtype=tensorplay.int64),
        )
        fwd_seed, fwd_base_offset = as_state((fwd_seed, fwd_base_offset))
        bwd_seed, bwd_base_offset = as_state((bwd_seed, bwd_base_offset))
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, "forward")
        PhiloxStateTracker.record_state(bwd_seed, bwd_base_offset, "backward")
    with tensorplay.random.fork_rng(devices=_trace_devices(primals)):
        with ProxyTensorDispatchMode(tracer, decompositions):
            with tensorplay.enable_grad():
                out = fn(*trace_primals)
            flat_out, out_spec = tree_flatten(out)
            # Settled before the backward runs: an operation that hands back
            # the very value it was given (a cast to the type it already has)
            # makes the backward's node the one that value stands for, and a
            # forward result read after that would name a backward node.
            fwd_values = [tracer.map_value(v) for v in flat_out]
            diff_outputs = [o for o in flat_out if _is_tensor(o) and o.requires_grad]
            with _disable_current_modes():
                tangents = [tensorplay.ones_like(o.detach()) for o in diff_outputs]
            anchor = next(
                (n for n in tracer.graph.nodes if n.op != "placeholder"), None
            )
            with tracer.graph.inserting_before(anchor):
                tangent_nodes = _placeholders(
                    tracer, tangents, "tangents", backward=True
                )
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
    bwd_values =[None if g is None else tracer.map_value(g) for g in grads]
    tracer.graph.output(tuple(fwd_values + bwd_values))
    tracer.graph.eliminate_dead_code()
    tracer._tracked.clear()
    joint = GraphModule(tracer.root, tracer.graph)
    return (
        joint,
        out_spec,
        flat_out,
        len(fwd_values),
        trace_primals,
        tangents,
        [node.name for node in tangent_nodes],
        diff_outputs,
    )


def _functionalize(gm: GraphModule, example_inputs: Sequence[Any]) -> GraphModule:
    """Mutation-free equivalent of ``gm`` with fresh value metadata.

    The metadata pass runs the graph, so it gets private copies of the
    inputs: the trailing input updates must not reach the caller's tensors.
    """

    from tensorplay.graph.passes.functionalize import functionalize
    from tensorplay.graph.passes.shape_prop import ShapeProp
    from .api import _release_recorded_values

    functional = functionalize(gm)
    _release_recorded_values(gm)
    with tensorplay.no_grad():
        ShapeProp(list(example_inputs))(functional)
    return functional


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


def _restore_saved(ctx, names) -> list:
    """The saved pairs, in the order the forward listed them."""

    held = iter(ctx.saved_tensors)
    plain = iter(ctx.saved_plain)
    return [
        (name, next(held) if is_tensor else next(plain))
        for name, is_tensor in zip(names, ctx.saved_is_tensor)
    ]


def _bind_backward_inputs(bw_module, input_kinds, input_keys, saved, grad_outputs, primals, primal_names):
    saved_by_name = dict(saved)
    primal_by_name = dict(zip(primal_names, primals))
    tangent_values = [value for _, value in grad_outputs]
    values = []
    tangent_position = 0
    for placeholder, kind, key in zip(bw_module.graph.placeholders, input_kinds, input_keys):
        if kind == "tangent":
            # Paired by position.  A half that was partitioned out builds its
            # own placeholders and names them again, so the name a tangent
            # carried when it was created is not a key anything holds; the
            # order the tangents were created in is the order the half reads
            # them in.
            values.append(tangent_values[tangent_position])
            tangent_position += 1
        elif kind == "saved":
            values.append(saved_by_name[key])
        elif key in primal_by_name:
            values.append(primal_by_name[key])
        else:
            # A graph constant (get_attr) read by the backward.
            values.append(bw_module._get_attr(key))
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

    primals = list(example_primals)
    if any(not _is_tensor(p) and _holds_tensor(p) for p in primals):
        # A tensor inside a list, tuple or dict argument is an input like any
        # other: it is given a placeholder of its own, so whether it requires
        # grad is seen and its gradient is returned to it.  Left inside its
        # container it would be one opaque input, and a region whose only
        # differentiable inputs sit in containers would be compiled as if it
        # had none.
        leaves, spec = tree_flatten(tuple(primals))

        def flat_fn(*flat: Any) -> Any:
            return fn(*tree_unflatten(list(flat), spec))

        inner = aot_function(
            flat_fn,
            leaves,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            inference_compiler=inference_compiler,
            partition_fn=partition_fn,
            decompositions=decompositions,
            keep_inference_input_mutations=keep_inference_input_mutations,
        )

        def run_flattened(*args: Any) -> Any:
            run_leaves, run_spec = tree_flatten(tuple(args))
            if run_spec != spec:
                raise TypeError(
                    "compiled region called with arguments nested differently "
                    "from the ones it was compiled for"
                )
            return inner(*run_leaves)

        for name in (
            "_tensorplay_aot_graphs",
            "_tensorplay_codegen",
            "_tensorplay_backward_codegen",
        ):
            if hasattr(inner, name):
                setattr(run_flattened, name, getattr(inner, name))
        run_flattened._tensorplay_flattened = inner  # type: ignore[attr-defined]
        return run_flattened

    del keep_inference_input_mutations  # mutations stay in the traced graph order
    bw_compiler = bw_compiler or fw_compiler
    inference_compiler = inference_compiler or fw_compiler
    partition_fn = partition_fn or default_partition
    needs_grad = tensorplay.is_grad_enabled() and any(
        _is_tensor(p) and p.requires_grad for p in primals
    )

    if not needs_grad:
        with tensorplay.no_grad():
            fw_module, out_spec, _ = _trace_forward(fn, primals, decompositions)
            fw_module = _functionalize(fw_module, _trace_inputs(primals))
        compiled_fw = inference_compiler(fw_module, primals)
        from .api import _release_recorded_values

        _release_recorded_values(fw_module)

        def run_inference(*args: Any) -> Any:
            with tensorplay.no_grad():
                return tree_unflatten(list(_call(compiled_fw, args)), out_spec)

        run_inference._tensorplay_aot_graphs = (fw_module,)  # type: ignore[attr-defined]
        run_inference._tensorplay_codegen = getattr(
            compiled_fw, "_tensorplay_codegen", None
        )  # type: ignore[attr-defined]
        run_inference._tensorplay_backward_codegen = None  # type: ignore[attr-defined]
        return run_inference

    (
        joint,
        out_spec,
        flat_out,
        num_fwd,
        trace_primals,
        tangents,
        _tangent_names,
        _diff_outputs,
    ) = _trace_joint(
        fn, primals, decompositions
    )
    joint = _functionalize(
        joint, _trace_inputs(trace_primals) + [t.clone() for t in tangents]
    )
    diff_out_mask = [_is_tensor(o) and o.requires_grad for o in flat_out]
    grad_mask = [_is_tensor(p) and p.requires_grad for p in primals]
    # Which differentiable inputs the program reached.  One it never read has
    # no gradient, and the backward graph produces nothing for it: its results
    # are the gradients of the reached inputs alone, in input order.
    joint_out = list(joint.graph.output_node.args)
    if joint_out and isinstance(joint_out[0], (tuple, list)):
        joint_out = list(joint_out[0])
    grad_reached = [arg is not None for arg in joint_out[num_fwd:]]

    fw_module, bw_module, input_kinds, input_keys, saved_names = partition_fn(
        joint, trace_primals + tangents, num_fwd_outputs=num_fwd
    )
    primal_names = [p.name for p in joint.graph.placeholders if not p.meta.get("is_backward")]
    tangent_names = [p.name for p in joint.graph.placeholders if p.meta.get("is_backward")]

    fw_inputs = [
        trace_primals[primal_names.index(p.name)] for p in fw_module.graph.placeholders
    ]
    # The forward returns the program's results followed by what it keeps for
    # the backward; only the first are read by the caller.  The backward's
    # results go to the engine, which lays each gradient out for its leaf.
    _mark_user_outputs(fw_module, num_fwd)
    _mark_user_outputs(bw_module, 0)
    compiled_fw = fw_compiler(fw_module, fw_inputs)
    from .api import _release_recorded_values

    _release_recorded_values(joint)
    _release_recorded_values(fw_module)
    _release_recorded_values(bw_module)
    compiled_bw_box: list[Any] = []
    fw_codegen = getattr(compiled_fw, "_tensorplay_codegen", None)

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
            # Saved through the context, so the engine releases them when it
            # releases the graph.  A list hung on the context is invisible to
            # it: the values outlive every backward pass, and a step's saved
            # activations are most of a region's memory.
            held = [value for _, value in saved if _is_tensor(value)]
            ctx.save_for_backward(*held)
            ctx.saved_is_tensor = tuple(_is_tensor(value) for _, value in saved)
            ctx.saved_plain = tuple(
                value for _, value in saved if not _is_tensor(value)
            )
            ctx.run_primals = run_primals
            # Record only the shapes and dtypes of the differentiable outputs.
            # Holding the output tensors on the context would close a
            # reference cycle (output -> grad_fn -> ctx -> output) that the
            # Python collector cannot see through the C++ node, so a compiled
            # forward called without a following backward would retain every
            # saved activation forever.  The engine zero-fills a missing
            # gradient from this metadata instead.
            ctx.diff_output_metas = [
                None if not (diff and _is_tensor(out)) else (
                    out.shape, out.dtype, out.device
                )
                for out, diff in zip(user, diff_out_mask)
            ]
            non_diff = [
                out for out, diff in zip(user, diff_out_mask) if _is_tensor(out) and not diff
            ]
            if non_diff:
                ctx.mark_non_differentiable(*non_diff)
            return tuple(user)

        @staticmethod
        def backward(ctx, *grad_outputs):
            diff_grads = [
                tensorplay.zeros(meta[0], dtype=meta[1], device=meta[2]) if g is None else g
                for g, diff, meta in zip(
                    grad_outputs, diff_out_mask, ctx.diff_output_metas
                )
                if diff
            ]
            named = list(zip(tangent_names, diff_grads))
            inputs = bw_example_inputs(
                _restore_saved(ctx, saved_names), named, ctx.run_primals
            )
            if not compiled_bw_box:
                from tensorplay.graph.passes.shape_prop import ShapeProp

                with tensorplay.no_grad():
                    ShapeProp(inputs)(bw_module)
                compiled_bw_box.append(bw_compiler(bw_module, inputs))
                _release_recorded_values(bw_module)
                run_training._tensorplay_backward_codegen = getattr(
                    compiled_bw_box[0], "_tensorplay_codegen", None
                )
            grads = iter(_call(compiled_bw_box[0], inputs))
            reached = iter(grad_reached)
            out = []
            for needed in grad_mask:
                if needed and next(reached):
                    out.append(next(grads))
                else:
                    out.append(None)
            out = tuple(out)
            # What this pass kept for itself is its own bookkeeping, and the
            # forward's outputs are its largest entry: they are read only to
            # decide which incoming gradients matter, which is now decided.
            # The values the context holds for its backward are released by
            # the engine, which knows whether the graph is kept for another
            # pass.
            ctx.saved_plain = ()
            ctx.diff_output_metas = None
            ctx.run_primals = None
            return out

    def run_training(*args: Any) -> Any:
        if not tensorplay.is_grad_enabled():
            # Called under no_grad after a training compile: the forward graph
            # alone produces the outputs.
            outputs = _call(compiled_fw, [args[i] for i in fw_input_order])
            if len(outputs) < num_fwd:
                raise AssertionError(
                    "compiled forward returned fewer outputs than the trace "
                    f"promised: expected at least {num_fwd}, got {len(outputs)}"
                )
            return tree_unflatten(list(outputs[:num_fwd]), out_spec)
        user = CompiledFunction.apply(*args)
        if not isinstance(user, tuple):
            user = (user,)
        return tree_unflatten(list(user), out_spec)

    run_training._tensorplay_aot_graphs = (joint, fw_module, bw_module)  # type: ignore[attr-defined]
    run_training._tensorplay_codegen = fw_codegen  # type: ignore[attr-defined]
    run_training._tensorplay_backward_codegen = None  # type: ignore[attr-defined]
    return run_training


def _bind_graph_inputs(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    """Call arguments as the module's graph inputs, in placeholder order.

    A captured graph module takes its program's parameters: keywords and
    defaults are resolved against its signature.
    """

    signature = getattr(module, "signature", None)
    placeholders = getattr(getattr(module, "graph", None), "placeholders", None)
    if signature is None or placeholders is None:
        if kwargs:
            raise TypeError("keyword arguments need a module with a signature")
        return list(args)
    if not kwargs and len(args) == len(placeholders):
        return list(args)
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    values = []
    for node in placeholders:
        key = node.target if isinstance(node.target, str) else node.name
        values.append(bound.arguments[key] if key in bound.arguments else bound.arguments[node.name])
    return values


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
    """Compile a module whose parameters and buffers become graph inputs.

    A module's state is what ``named_parameters``/``named_buffers`` report.
    A captured :class:`GraphModule` reads its state through ``get_attr``
    nodes resolved against its root, so its state is the tensors those nodes
    name; they are substituted on the root for each call.
    """

    names, read_state, substitute = _state_access(module)
    from .api import _release_recorded_values

    _release_recorded_values(module)
    count = len(names)

    def flat_fn(*flat: Any) -> Any:
        if count == 0:
            return module(*flat)
        with substitute(dict(zip(names, flat[:count]))):
            return module(*flat[count:])

    example = read_state() + list(example_inputs)
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

    class _AotForward:
        """Callable wrapper whose codegen reports mirror the compiled artifact.

        The compiled forward and backward artifacts report which route
        produced them; the wrapper a caller holds is not one of those, so a
        report read through the wrapper reaches the artifact's live answer,
        including the backward's route once a backward has run.
        """

        def __init__(
            self,
            compiled: Callable[..., Any],
            module: Any,
            read_state: Callable[[], list[Any]],
            bind_graph_inputs: Callable[..., list[Any]],
        ) -> None:
            self._compiled = compiled
            self._module = module
            self._read_state = read_state
            self._bind_graph_inputs = bind_graph_inputs
            self._tensorplay_aot_graphs = getattr(
                compiled, "_tensorplay_aot_graphs", ()
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self._compiled(
                *self._read_state(),
                *self._bind_graph_inputs(self._module, args, kwargs),
            )

        def __getattr__(self, name: str) -> Any:
            return getattr(self._compiled, name)

    return _AotForward(compiled, module, read_state, _bind_graph_inputs)


def _lift_literal_tensors(module: GraphModule) -> None:
    """Turn tensors embedded in node arguments into ``get_attr`` nodes.

    Capture records tensors a program closes over (parameters of a module it
    calls, constants) as literal arguments.  As graph attributes they become
    inputs of the compiled function, so gradients reach the original tensors.
    """

    by_id: dict[int, Any] = {}
    changed = False

    def lift(value: Any, before: Any) -> Any:
        nonlocal changed
        if isinstance(value, tensorplay.Tensor):
            node = by_id.get(id(value))
            if node is None:
                name = f"_lifted_tensor_{len(by_id)}"
                # Instance attribute: generated code and _get_attr both
                # resolve it without going through the root.
                object.__setattr__(module, name, value)
                with module.graph.inserting_before(before):
                    node = module.graph.get_attr(name)
                node.meta["val"] = value
                by_id[id(value)] = node
            changed = True
            return node
        if isinstance(value, tuple):
            return tuple(lift(v, before) for v in value)
        if isinstance(value, list):
            return [lift(v, before) for v in value]
        if isinstance(value, dict):
            return {k: lift(v, before) for k, v in value.items()}
        return value

    for node in list(module.graph.nodes):
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        node.args = lift(node.args, node)
        node.kwargs = lift(dict(node.kwargs), node)
    if changed:
        module.recompile()


def _swap_state_tables(module: Any, state: Mapping[str, Any]) -> list[tuple[Any, Any] | None]:
    """Write tensors into the parameter/buffer tables generated code reads.

    The generated forward resolves ``get_attr`` through the module's own
    ``_parameters``/``_buffers`` (and nested ``_modules``) before the root,
    so state written only to the root never reaches those reads.  Returns
    ``(container, key, old_value)`` triples (``None`` for names that resolve
    through the root) for restoration.
    """

    saved: list[tuple[Any, Any] | None] = []

    def swap_holder(holder: Any, key: str, value: Any) -> None:
        for table_name in ("_parameters", "_buffers"):
            table = holder.__dict__.get(table_name)
            if table is not None and key in table:
                saved.append((table, key, table[key]))
                table[key] = value
                return
        saved.append(None)

    for name, value in state.items():
        parts = name.split(".")
        holder = module
        for part in parts[:-1]:
            modules = holder.__dict__.get("_modules", {})
            if part in modules:
                holder = modules[part]
            else:
                holder = None
                break
        if holder is None:
            saved.append(None)
            continue
        swap_holder(holder, parts[-1], value)
    return saved


_NO_ENTRY = object()


def _state_reader(module: Any, target: str) -> Callable[[], Any]:
    """A reader of one tensor a graph module reads, fetched on every call.

    The full lookup walks a dotted path through attribute access, which for
    modules runs Python attribute hooks at every level -- a hundred-odd
    parameters cost more than the region's own launch.  A path that resolves
    through modules' child and state tables is walked through those tables
    directly, which is where the module API keeps what it registers, so a
    parameter or submodule assigned after compiling is still the one read.
    The graph's own attribute table is consulted first on every call, as the
    full lookup does (substituted state lives there), and any path the tables
    do not answer falls back to the full lookup.
    """

    full = functools.partial(module._get_attr, target)
    parts = target.split(".")
    path, leaf = parts[:-1], parts[-1]
    table = module.__dict__

    def walk(base: Any) -> Any:
        current = base
        for name in path:
            current = current._modules[name]
        state = current._parameters
        if leaf in state:
            return state[leaf]
        return current._buffers[leaf]

    try:
        expected = full()
    except (AttributeError, KeyError, IndexError, TypeError):
        return full
    base = None
    for candidate in (module, table.get("_root")):
        if not isinstance(candidate, tensorplay.nn.Module):
            continue
        try:
            if walk(candidate) is expected:
                base = candidate
                break
        except (AttributeError, KeyError, TypeError):
            continue
    if base is None:
        return full

    def read() -> Any:
        attrs = table.get("_graph_attrs")
        if attrs:
            hit = attrs.get(target, _NO_ENTRY)
            if hit is not _NO_ENTRY:
                return hit
        try:
            return walk(base)
        except (AttributeError, KeyError, TypeError):
            return full()

    return read


def _state_access(module: Any):
    """``(names, read_state, substitute)`` for the tensors a module reads."""

    from tensorplay.nn.utils.stateless import _reparametrize_module

    if isinstance(module, GraphModule):
        _lift_literal_tensors(module)
        targets: list[str] = []
        seen: set[int] = set()
        for node in module.graph.nodes:
            if node.op != "get_attr" or node.target in targets:
                continue
            value = module._get_attr(node.target)
            if isinstance(value, tensorplay.Tensor) and id(value) not in seen:
                seen.add(id(value))
                targets.append(node.target)

        readers = [_state_reader(module, target) for target in targets]

        def read_graph_state() -> list[Any]:
            return [read() for read in readers]

        root = module.root
        owned = module.__dict__

        @contextlib.contextmanager
        def substitute(state: dict[str, Any]):
            # Lifted literals live on the graph module itself; module state
            # is read through the graph attribute table and through direct
            # parameter/buffer lookup in the generated forward, so the state
            # rides in every table those reads can hit.
            local = {k: v for k, v in state.items() if k in owned}
            rooted = {k: v for k, v in state.items() if k not in owned}
            saved = {k: owned[k] for k in local}
            owned.update(local)
            try:
                if rooted:
                    attrs = module._graph_attrs
                    saved_attrs = {k: attrs[k] for k in rooted if k in attrs}
                    attrs.update(rooted)
                    table_saved = _swap_state_tables(module, rooted)
                    try:
                        if root is not None and isinstance(root, tensorplay.nn.Module):
                            # Keep the root's own bindings consistent for
                            # any read that falls through the module's
                            # attribute tables.
                            with _reparametrize_module(root, rooted):
                                yield
                            return
                        yield
                    finally:
                        for k in saved_attrs:
                            attrs[k] = saved_attrs[k]
                        for k in rooted:
                            if k not in saved_attrs:
                                attrs.pop(k, None)
                        for entry in table_saved:
                            if entry is None:
                                continue
                            table, key, old = entry
                            table[key] = old
                else:
                    yield
            finally:
                owned.update(saved)

        return targets, read_graph_state, substitute

    named = list(_named_state(module))
    names = [name for name, _ in named]

    def read_module_state() -> list[Any]:
        return [value for _, value in _named_state(module)]

    return names, read_module_state, lambda state: _reparametrize_module(module, state)


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
