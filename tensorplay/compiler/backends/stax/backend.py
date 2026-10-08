"""Compiler entry for the Stax backend.

The layers below hold the work: :mod:`loops` for the IR every device is
lowered to, :mod:`graph_lowering` for the captured-graph walk, and
and re-exports their names, so ``backend`` stays the single import
surface for the code generators and the tests.
"""
from __future__ import annotations

import inspect
import os
import sys

from typing import Any

from ....graph import GraphModule
# Option patch each compile ``mode`` selects, keyed by the backend option
# namespace.  ``default`` leaves every knob at its built-in value;
# ``reduce-overhead`` replays the artifact through CUDA graphs;
# ``max-autotune(-no-cudagraphs)`` additionally selects the exhaustive
# search tier: the widened pointwise candidate table plus coordinate-descent
# refinement of every benchmark winner.
_MODE_OPTIONS: dict[str, dict[str, bool]] = {
    "default": {},
    "reduce-overhead": {
        "stax.cudagraphs": True,
    },
    "max-autotune-no-cudagraphs": {
        "stax.max_autotune": True,
        "stax.coordinate_descent_tuning": True,
    },
    "max-autotune": {
        "stax.max_autotune": True,
        "stax.cudagraphs": True,
        "stax.coordinate_descent_tuning": True,
    },
}

# Every backend option key accepted by ``stax`` (and therefore by explicit
# ``options`` dicts and mode patches alike).
_STAX_OPTIONS = (
    "stax.fusion",
    "stax.cuda_codegen",
    "stax.triton",
    "stax.cudagraphs",
    "stax.max_autotune",
    "stax.coordinate_descent_tuning",
)

def _is_scalar(value: Any) -> bool:
    """Whether a captured graph argument is a plain Python number.

    A graph walk has to tell a constant apart from a produced value before it
    can emit one: a constant becomes a literal inside the kernel body, while a
    produced value becomes a load.  Booleans count as numbers here because a
    captured ``True``/``False`` is written into the body as 1.0/0.0, and
    complex is excluded because a kernel body has no complex literal form.
    """

    return isinstance(value, (bool, int, float))


def list_mode_options(mode: str | None = None) -> dict[str, Any]:
    """Return the optimization options each compile ``mode`` selects.

    With ``mode`` set, returns that mode's option patch; with ``mode``
    unset, returns the full mode-to-options mapping.  Unknown modes raise.
    The patch feeds the same validation path as explicit backend options,
    so options supplied by a caller override the mode patch per key.
    """

    try:
        return dict(_MODE_OPTIONS[mode]) if mode else dict(_MODE_OPTIONS)
    except KeyError as exc:
        raise RuntimeError(
            f"Unrecognized mode={mode}, should be one of: "
            f"{', '.join(_MODE_OPTIONS)}"
        ) from exc

def stax(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    mode: str | None = None,
    options: dict[str, Any] | None = None,
    name: str | None = None,
    dynamic: bool | None = None,
    strict_native: bool = False,
    **kwargs: Any,
):
    """Compile one canonical graph and return an executable callable.

    ``example_inputs`` and backend options are part of the same contract as
    metadata in the frontend and uses the native graph when its lowering
    contract is satisfied.  ``strict_native`` makes a failed lowering a hard
    compiler error, so a benchmark can never report the Python GraphModule
    executor as compiled performance.
    """
    del name, kwargs
    if mode not in (None, *_MODE_OPTIONS):
        raise RuntimeError(f"unknown Stax optimization mode: {mode!r}")
    # A mode's option patch is applied first; explicit options overlay it
    # per key (a mode selects defaults, an explicit option wins).
    resolved = dict(_MODE_OPTIONS[mode or "default"])
    if options is not None:
        if not isinstance(options, dict):
            raise TypeError(f"options must be a dict, got {type(options)!r}")
        unknown = set(options).difference(_STAX_OPTIONS)
        if unknown:
            raise RuntimeError(
                f"Unexpected Stax optimization option(s): {sorted(unknown)!r}"
            )
        if any(not isinstance(value, bool) for value in options.values()):
            raise RuntimeError("Stax optimization options must be bool values")
        resolved.update(options)
    use_fusion = resolved.get("stax.fusion", True)
    use_cuda_codegen = resolved.get("stax.cuda_codegen", False)
    use_triton = resolved.get("stax.triton", True)
    max_autotune = resolved.get("stax.max_autotune", False)
    coordinate_descent_tuning = resolved.get(
        "stax.coordinate_descent_tuning", False
    )
    cudagraphs_requested = resolved.get("stax.cudagraphs", False)
    compiled = _lower_stax_region(
        graph_module,
        example_inputs,
        use_fusion=use_fusion,
        use_cuda_codegen=use_cuda_codegen,
        use_triton=use_triton,
        max_autotune=max_autotune,
        coordinate_descent_tuning=coordinate_descent_tuning,
        dynamic=dynamic,
        strict_native=strict_native,
    )
    if not cudagraphs_requested or isinstance(compiled, GraphModule):
        return compiled
    from ..cudagraphs import cudagraph_wrap

    wrapped, reason = cudagraph_wrap(
        compiled, graph_module, example_inputs, dynamic=dynamic
    )
    if reason is not None:
        graph_module._stax_cudagraph_skip_reason = reason
        return compiled
    return wrapped


# This backend lowers operator overloads, never the Python functions a capture
# happened to call.  A region with nothing to differentiate is therefore traced
# down to the operators it runs before it is handed over, exactly as a training
# region is: a function's name does not say which overload it reached, nor what
# a composite is made of.
stax.lowers_operator_graphs = True


def _trace_decompositions():
    from .op_lowerings import trace_decompositions

    return trace_decompositions()


# An operation with no lowering, template or library call of its own is traced
# as the parts it decomposes into, which are lowered, instead of reaching the
# region as a call the framework makes.
stax.trace_decompositions = _trace_decompositions


def _publish_codegen(compiled, tag: str, *, backward: bool) -> None:
    """Report on the artifact itself which route produced it.

    One place, so every route's answer is visible the same way, and a caller
    can tell a compiled region from an interpreted one without knowing which
    route claimed it.
    """

    try:
        compiled._tensorplay_codegen = tag  # type: ignore[attr-defined]
        compiled._tensorplay_backward_codegen = tag if backward else None  # type: ignore[attr-defined]
    except AttributeError:  # noqa: BLE001 - an artifact that cannot carry it
        pass


def _lower_stax_region(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    use_fusion: bool,
    use_cuda_codegen: bool,
    use_triton: bool,
    max_autotune: bool,
    coordinate_descent_tuning: bool,
    dynamic: bool | None,
    strict_native: bool,
):
    """Lower one canonical graph and return an executable callable.

    ``example_inputs`` and backend options are part of the same contract as
    metadata in the frontend and uses the native graph when its lowering
    contract is satisfied.  ``strict_native`` makes a failed lowering a hard
    compiler error, so a benchmark can never report the Python GraphModule
    executor as compiled performance.
    """
    strict = bool(strict_native)
    dynamic_shapes = bool(dynamic is True)
    training = getattr(graph_module.root, "training", False) or any(
        getattr(value, "requires_grad", False) for value in example_inputs
    )
    try:
        on_cuda = example_inputs[0].device.is_cuda()
    except (AttributeError, IndexError):
        on_cuda = False

    if strict and training and any(
        getattr(graph_module._get_attr(node.target), "requires_grad", False)
        for node in graph_module.graph.nodes
        if node.op == "get_attr"
    ):
        raise RuntimeError("AOT backward graph for the captured training region")
    # The region is lowered, scheduled and printed, and what that produces is
    # what the caller calls.  A region that cannot be lowered is a region the
    # framework runs itself, which is what falling back is for.
    from .graph_lowering import GraphLowering

    if any(
        getattr(node.target, "__name__", None) == "auto_functionalized"
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    ):
        # A mutating operator with no functional form reaches this backend as
        # its functional stand-in; it is lowered as the copies and the
        # operator writing into them.
        from .graph_passes.post_grad import decompose_auto_functionalized

        decompose_auto_functionalized(graph_module.graph)
        graph_module.recompile()

    graph = GraphLowering(
        graph_module,
        example_inputs,
        shape_env=None,
        cpp_wrapper=use_cuda_codegen,
        aot_mode=training,
        extern_node_serializer=None,
        is_inference=not training,
        is_backward=False,
    )
    # Everything from here on is being done about this region, and anything
    # asked along the way -- what a kernel reads, how big an extent is, what a
    # buffer is written into -- is answered by the region it belongs to.  So
    # the region is the one that is current for all of it.
    from .loops import set_graph
    from .loops import set_fake_mode
    from tensorplay._higher_order_ops._hop_base import FakeTensorMode

    from .loops import Debug
    from .virtualized import V as VirtualMachine
    from tensorplay._C import (
        _enter_lowered_graph_scope as _enter_lowered_graph_scope,
        _exit_lowered_graph_scope as _exit_lowered_graph_scope,
    )

    # Layouts and timings asked of operators while planning are asked under
    # the conditions the written-out code runs them in.
    _enter_lowered_graph_scope()
    try:
        # A recorder of what was emitted is current for the whole of the
        # compilation, because both the schedule and the region's own picture
        # are asked of it by name while they are being decided, and neither can
        # say where to write unless something has already said that it wants to
        # be written at all.  The fake mode is entered as well as installed:
        # fallback operators dispatch through the Python layer only while a
        # mode sits on the dispatch stack, and a stack with no mode makes the
        # generated Python-dispatch kernel pop nothing on entry.
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        with (
            set_fake_mode(fake_mode),
            VirtualMachine.set_fake_mode(fake_mode),
            FakeTensorMode(allow_non_fake_inputs=True),
            set_graph(graph),
            VirtualMachine.set_debug_handler(Debug()),
        ):
            graph.run(*example_inputs)
            compiled_module = graph.compile_to_module()
    except NotImplementedError as exc:
        # A region whose printed form this compiler does not cover has no built
        # artifact, and an artifact that cannot be entered is worse than none:
        # the failure would arrive at the first call rather than here.  So a
        # caller that did not ask for a built form gets the region run as it was
        # written, and a caller that did asks for the failure instead.
        if os.environ.get("TP_STAX_LOWER_DEBUG"):
            print(
                f"[stax-lower] uncovered: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        if strict:
            raise RuntimeError(
                "strict_native Stax lowering failed: captured graph has no built form"
            ) from exc
        return graph_module.recompile()
    except Exception as exc:
        if os.environ.get("TP_STAX_LOWER_DEBUG"):
            import traceback as _tb

            print(
                f"[stax-lower] unbuilt: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            _tb.print_exc(limit=30, file=sys.stderr)
        if strict:
            raise RuntimeError(
                "strict_native Stax lowering failed: captured graph has no built form"
            ) from exc
        return graph_module.recompile()
    finally:
        _exit_lowered_graph_scope()
    # The generated entry point takes the region's arguments as one sequence,
    # because that is how the written-out code receives them, and it hands back
    # a sequence of results for the same reason.  Everything that calls a
    # compiled region passes ordinary arguments and expects an ordinary result,
    # so the two conventions are met here rather than making every caller know
    # which one this one happens to use -- and a region that produced one result
    # produces that result, rather than a sequence holding it.
    module_call = compiled_module.call
    single_output = bool(getattr(graph, "single_output", False))
    # The generated entry point takes the region's arguments as one sequence,
    # in the order the region's placeholders stand.  The artifact is reached
    # through the same signature the region was captured under, so keyword
    # and default arguments are bound back to their parameter names and
    # re-ordered into that sequence, mirroring the capture-time binding.
    signature = graph_module.signature

    # A call that hands every captured argument by position, in the order
    # the placeholders stand, needs no binding: that is how the autograd
    # wrapper and every compiled caller call a region.
    placeholder_names = [
        node.target if isinstance(node.target, str) else node.name
        for node in graph_module.graph.placeholders
    ]
    positional = signature is not None and [
        name for name, parameter in signature.parameters.items()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
    ] == placeholder_names and len(signature.parameters) == len(placeholder_names)
    num_placeholders = len(placeholder_names)

    def compiled(*args, **kwargs):
        if signature is None or (
            positional and not kwargs and len(args) == num_placeholders
        ):
            ordered = list(args)
        else:
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            ordered = []
            for node in graph_module.graph.placeholders:
                parameter_name = (
                    node.target if isinstance(node.target, str) else node.name
                )
                if parameter_name in bound.arguments:
                    ordered.append(bound.arguments[parameter_name])
                elif node.name in bound.arguments:
                    ordered.append(bound.arguments[node.name])
                else:
                    raise TypeError(
                        f"compiled region is missing a value for "
                        f"captured argument {parameter_name!r}"
                    )
        # A list rather than a tuple, because the written-out code empties what
        # it is given once it has taken it -- which is how a caller that holds
        # the same values does not keep them alive for the call.
        #
        # The call runs inside a lowered-graph scope: the written-out code
        # reads every result with the strides recorded when it was planned,
        # so the operators it hands work to take their operands as they lie
        # instead of repacking them into an order of their own choosing.
        _enter_lowered_graph_scope()
        try:
            result = module_call(ordered)
        finally:
            _exit_lowered_graph_scope()
        if single_output and isinstance(result, tuple) and len(result) == 1:
            return result[0]
        return result

    def boxed_call(args):
        _enter_lowered_graph_scope()
        try:
            result = module_call(args)
        finally:
            _exit_lowered_graph_scope()
        if single_output and isinstance(result, tuple) and len(result) == 1:
            return result[0]
        return result

    compiled._tensorplay_boxed_call = boxed_call  # type: ignore[attr-defined]
    compiled._tensorplay_module_call = module_call  # type: ignore[attr-defined]
    # The report is on what the caller actually receives, which is this and not
    # the module behind it -- a caller asking which route produced its callable
    # would otherwise be told nothing, since the module is not what it holds.
    _publish_codegen(compiled, "triton" if on_cuda else "stax-cpu", backward=training)
    return compiled


#: Everything the layers own, re-exported under one name so the
#: code generators, the lazy ``_stax`` proxy, and the tests keep
#: importing from ``backend``.
__all__ = [
    "list_mode_options",
    "stax",
]
