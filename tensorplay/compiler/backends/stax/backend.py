"""Compiler entry for the Stax backend.

The layers below hold the work: :mod:`ir` for values and node
attributes, :mod:`pointwise` for the planners and fusers,
:mod:`lowering` for the captured-graph walk, and
:mod:`aot_autograd` for the reverse pass.  This module drives them
and re-exports their names, so ``backend`` stays the single import
surface for the code generators and the tests.
"""
from __future__ import annotations

from typing import Any

from ....graph import GraphModule
from .aot_autograd import *  # noqa: F401,F403 - reverse pass
from .aot_autograd import _lower_aot_native
from . import ir as _ir
from . import pointwise as _pointwise
from . import lowering as _lowering
from . import aot_autograd as _aot
from .ir import *  # noqa: F401,F403 - shared surface
from .lowering import *  # noqa: F401,F403 - captured-graph walk
from .lowering import _lower_native
from .pointwise import *  # noqa: F401,F403 - planner surface
from .pointwise import (
    _lower_cpu_fused_pointwise,
    _lower_cpu_fused_reduction,
    _lower_cpu_row_fusion,
    _lower_cpu_segmented,
    _lower_cuda_fused_pointwise,
    _lower_cuda_row_fusion,
)


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
    "stax.native",
    "stax.fusion",
    "stax.cuda_codegen",
    "stax.triton",
    "stax.cudagraphs",
    "stax.max_autotune",
    "stax.coordinate_descent_tuning",
)

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
    use_native = resolved.get("stax.native", True)
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
        use_native=use_native,
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

def _lower_stax_region(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    use_native: bool,
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
    if use_native and use_fusion:
        fused_cpu_graph = _lower_cpu_fused_pointwise(
            graph_module,
            example_inputs,
            strict_native=strict_native,
            dynamic=bool(dynamic is True),
        )
        if fused_cpu_graph is not None:
            graph_module._stax_native_graph = fused_cpu_graph.graph
            return fused_cpu_graph
        # A region whose tail is a reduction folds the whole expression into
        # the reduction loop: one pass over the input, no intermediate.
        fused_cpu_reduction = _lower_cpu_fused_reduction(
            graph_module,
            example_inputs,
            strict_native=strict_native,
            dynamic=bool(dynamic is True),
        )
        if fused_cpu_reduction is not None:
            graph_module._stax_codegen = "stax-fused-cpu-reduce"
            return fused_cpu_reduction
        # A region whose reductions sit in the middle stages per row: each
        # reduction folds the row to one value that the following work reads
        # as a broadcast, so the region still reads its inputs once.
        row_fused_cpu = _lower_cpu_row_fusion(
            graph_module,
            example_inputs,
            strict_native=strict_native,
            dynamic=bool(dynamic is True),
        )
        if row_fused_cpu is not None:
            graph_module._stax_codegen = "stax-fused-cpu-rowfuse"
            return row_fused_cpu
    if use_native and use_fusion and use_cuda_codegen:
        fused_cuda_graph = _lower_cuda_fused_pointwise(
            graph_module,
            example_inputs,
            strict_native=strict_native,
            dynamic=bool(dynamic is True),
        )
        if fused_cuda_graph is not None:
            graph_module._stax_codegen = "stax-cuda"
            return fused_cuda_graph
    if use_native and use_triton:
        # Keep Triton optional and lazy.  Importing tensorplay on a CPU-only
        # machine must not import Triton or its compiler toolchain.
        try:
            first = example_inputs[0]
            is_cuda = first.device.is_cuda()
        except (AttributeError, IndexError):
            is_cuda = False
        if is_cuda:
            from .codegen.triton import (
                compile_graph_module as compile_triton_graph,
            )

            # The per-segment emitter is the source of fusion truth for the
            # shapes it accepts: one trailing reduction per segment plus a
            # store-time epilogue.  The row-staged kernel answers only for
            # what that form cannot express -- a reduction whose result
            # feeds elementwise work feeding another reduction (softmax,
            # normalization) -- so it claims the region last.
            try:
                triton_graph = compile_triton_graph(
                    graph_module,
                    example_inputs,
                    max_autotune=max_autotune,
                    coordinate_descent_tuning=coordinate_descent_tuning,
                    strict_native=strict_native,
                )
            except Exception:
                # Unsupported constants or shape forms belong on the native
                # graph path and must not abort compilation; generated-kernel
                # build failures (toolchain, unsupported op) degrade the same
                # way.
                triton_graph = None
            if triton_graph is not None:
                graph_module._stax_codegen = "triton"
                return triton_graph
            if use_fusion:
                row_fused_cuda = _lower_cuda_row_fusion(
                    graph_module,
                    example_inputs,
                    strict_native=strict_native,
                    dynamic=bool(dynamic is True),
                )
                if row_fused_cuda is not None:
                    graph_module._stax_codegen = "stax-fused-cuda-rowfuse"
                    return row_fused_cuda
    if use_native and use_fusion and not use_cuda_codegen:
        fused_cuda_graph = _lower_cuda_fused_pointwise(
            graph_module,
            example_inputs,
            strict_native=strict_native,
            dynamic=bool(dynamic is True),
        )
        if fused_cuda_graph is not None:
            graph_module._stax_codegen = "stax-cuda"
            return fused_cuda_graph
    # The AOT boundary is a property of the graph's gradient surface, not of
    # the callable's shape: bare functions carry no training flag, so a
    # grad-carrying input list must select the split forward/backward route
    # exactly as a training module does.  The builder re-checks grad mode and
    # returns None for inference calls, leaving the routes below untouched.
    if use_native and (
        getattr(graph_module.root, "training", False)
        or any(
            getattr(value, "requires_grad", False) for value in example_inputs
        )
    ):
        aot_graph = _lower_aot_native(
            graph_module, example_inputs, use_fusion=use_fusion
        )
        if aot_graph is not None:
            graph_module._stax_native_graph = aot_graph.forward_graph
            return aot_graph
        if strict_native and any(
            getattr(graph_module._get_attr(node.target), "requires_grad", False)
            for node in graph_module.graph.nodes
            if node.op == "get_attr"
        ):
            raise RuntimeError(
                "AOT backward graph for the captured training region"
            )
    if use_native and use_fusion:
        # Nothing claimed the region whole.  Its fusible runs are still worth
        # compiling: each becomes one kernel, and the operators between them
        # run as captured instead of the region losing every compiled route.
        segmented = _lower_cpu_segmented(
            graph_module,
            example_inputs,
            strict_native=strict_native,
            dynamic=bool(dynamic is True),
        )
        if segmented is not None:
            graph_module._stax_codegen = "stax-fused-cpu-segments"
            return segmented
    # Fused native pointwise nodes are forward execution primitives.  A
    # training graph that reaches this fallback did not obtain an AOT reverse
    # graph, so keep ordinary tensor operators here to preserve autograd
    # recording for every parameter.
    native_fusion = use_fusion and not getattr(graph_module.root, "training", False)
    native_graph = (
        _lower_native(graph_module, example_inputs, use_fusion=native_fusion)
        if use_native
        else None
    )
    if native_graph is not None:
        graph_module._stax_native_graph = native_graph.graph
        return native_graph
    if strict_native:
        raise RuntimeError(
            "strict_native Stax lowering failed: captured graph has no native executable"
        )
    # No native executable exists for this graph (scalar placeholders,
    # factory-only regions, unsupported surface).  Fall back to the
    # generated Python executor so the region still runs with captured
    # semantics instead of failing to compile.
    return graph_module.recompile()


#: Everything the layers own, re-exported under one name so the
#: code generators, the lazy ``_stax`` proxy, and the tests keep
#: importing from ``backend``.
__all__ = [
    *_ir.__all__,
    *_pointwise.__all__,
    *_lowering.__all__,
    *_aot.__all__,
    "list_mode_options",
    "stax",
]
