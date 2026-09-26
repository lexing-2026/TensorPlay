"""Compiler entry for the Stax backend.

The layers below hold the work: :mod:`ir` for values and node
attributes, :mod:`pointwise` for the planners and fusers,
:mod:`lowering` for the captured-graph walk, and
:mod:`aot_autograd` for the reverse pass.  This module drives them
and re-exports their names, so ``backend`` stays the single import
surface for the code generators and the tests.
"""
from __future__ import annotations

import os
import sys

from typing import Any

from ....graph import GraphModule
from .aot_autograd import *  # noqa: F401,F403 - reverse pass
from . import ir as _ir
from . import pointwise as _pointwise
from . import lowering as _lowering
from . import aot_autograd as _aot
from .ir import *  # noqa: F401,F403 - shared surface
from .lowering import *  # noqa: F401,F403 - captured-graph walk
from .pointwise import *  # noqa: F401,F403 - planner surface
from .pointwise import (
    _lower_cpu_fused_pointwise,
    _lower_cpu_fused_reduction,
    _lower_cpu_row_fusion,
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
    strict = bool(strict_native)
    dynamic_shapes = bool(dynamic is True)
    training = getattr(graph_module.root, "training", False) or any(
        getattr(value, "requires_grad", False) for value in example_inputs
    )
    try:
        on_cuda = example_inputs[0].device.is_cuda()
    except (AttributeError, IndexError):
        on_cuda = False

    routes = []

    def route(tag, enabled, build, codegen=None):
        """Register one lowering: it answers whether it claims the region.

        ``tag`` names the route, which is an internal identity.  ``codegen``
        names what the region ends up running -- generated kernels, a native
        graph, the interpreted executor -- and is what the artifact reports,
        because that is the part a caller can act on.
        """

        if enabled:
            routes.append((tag, build, codegen or tag))

    if use_native:
        # Keep Triton optional and lazy.  Importing tensorplay on a CPU-only
        # machine must not import Triton or its compiler toolchain.
        def build_loop_region():
            from ..._core.aot_autograd import aot_module_simplified
            from .loop_compile import (
                NotLowerable,
                compile_half,
                compile_half_host,
            )

            def compile_half_or_none(half, half_inputs, **_ignored):
                # The half arrives with the values it reads; what they are is
                # the boundary's decision, and this only compiles the half.
                # Which printer runs is the device's: the same IR, printed as
                # Triton for an accelerator and as a host kernel otherwise.
                if on_cuda:
                    return compile_half(
                        half, half_inputs,
                        max_autotune=max_autotune,
                        coordinate_descent_tuning=coordinate_descent_tuning,
                    )
                return compile_half_host(half, half_inputs)

            try:
                # One boundary owns the region: it traces the joint graph,
                # splits it, hands each half to the compiler here, and keeps
                # the saved values and the gradients.  A compiler that cannot
                # express a half says so from inside that walk.
                return aot_module_simplified(
                    graph_module,
                    list(example_inputs),
                    fw_compiler=compile_half_or_none,
                    bw_compiler=compile_half_or_none,
                )
            except (NotLowerable, NotImplementedError) as exc:
                # An operator or a form this lowering does not cover: the next
                # route is the answer, and only for that reason.  Saying which
                # form is what lets the coverage of this route be measured
                # instead of guessed.
                if os.environ.get("TP_LOOP_ROUTE_DEBUG"):
                    print(
                        f"[stax-loops] declined: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                return None

        # One lowering per device.  On an accelerator the loop IR is it: the
        # boundary traces the region, splits it and hands each half to the
        # compiler above.  The routes below are the host backend, chosen by
        # the device the tensors live on rather than by which one answers
        # first, so a form this device's lowering does not cover is a form the
        # framework runs itself -- not a hop to a second emitter for the same
        # device.
        route("stax-loops", True, build_loop_region, codegen="triton")

    # One lowering per device, and it is the one above: a region it cannot
    # print is a region the framework runs itself, which is what a fall back to
    # the framework is for.  It is not answered by a second program
    # representation for the same device -- that is the branch this replaced.

    for tag, build, codegen in routes:
        compiled = build()
        if compiled is None:
            continue
        graph_module._stax_codegen = tag
        _publish_codegen(compiled, codegen, backward=training)
        return compiled
    if strict and training and any(
        getattr(graph_module._get_attr(node.target), "requires_grad", False)
        for node in graph_module.graph.nodes
        if node.op == "get_attr"
    ):
        raise RuntimeError("AOT backward graph for the captured training region")
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
