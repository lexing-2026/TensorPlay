"""Public compiler orchestration for TensorPlay.

Capture, backend selection, and execution concerns are separate. A backend is
never asked to discover Python control flow; it only receives a captured
``GraphModule`` and example inputs.
"""

from __future__ import annotations

import contextlib

import functools
import inspect
import threading
from typing import Any, Callable
from weakref import WeakSet

from tensorplay.graph import (
    GraphCaptureError,
    GraphModule,
    Tracer,
    compiler_context,
)
from tensorplay.graph._utils import _capture_disabled
from .guards import GuardChain, build_guard_chain, format_recompile_reasons
from tensorplay.graph.passes import (
    ConstFold,
    DeadCodeElimination,
    DecomposePass,
    NormalizeOperators,
    PassManager,
    PointwiseFusionHint,
    ShapeProp,
)
from tensorplay.graph.passes.dialect.common import CSEPass, get_CSE_banned_ops
from tensorplay.graph.passes.regional_compile_invoke_subgraph import (
    regional_compile_invoke_subgraph,
)
from .registry import CompilerFn, get_default_backend, lookup_backend
from .region_cache import (
    load_region,
    region_key as make_region_key,
    store_region,
)


_compiled_wrappers: WeakSet[Any] = WeakSet()
_DEFAULT_RECOMPILE_LIMIT = 8


def _compiler_context() -> Any:
    """Capture context owned by the graph namespace."""

    return compiler_context(require_native=True)


def _tensor_signature(value: Any, *, dynamic: bool) -> tuple[Any, ...] | None:
    module_name = type(value).__module__
    if not module_name.startswith("tensorplay"):
        return None
    shape = getattr(value, "shape", None)
    if callable(shape):
        shape = shape()
    try:
        shape = tuple(int(item) for item in shape)
        # Dynamic mode keeps rank specialization but removes concrete sizes.
        # shape policy; operations still receive the real runtime tensors.
        shape_key = ("dynamic", len(shape)) if dynamic else shape
    except (TypeError, ValueError):
        shape_key = repr(shape)
    dtype = getattr(value, "dtype", None)
    if callable(dtype):
        dtype = dtype()
    device = getattr(value, "device", None)
    if callable(device):
        device = device()
    requires_grad = getattr(value, "requires_grad", None)
    if callable(requires_grad):
        requires_grad = requires_grad()
    return (
        "tensor",
        type(value),
        shape_key,
        repr(dtype),
        repr(device),
        bool(requires_grad),
    )


def _value_signature(value: Any, *, dynamic: bool) -> Any:
    tensor_key = _tensor_signature(value, dynamic=dynamic)
    if tensor_key is not None:
        return tensor_key
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return (type(value), value)
    if isinstance(value, tuple):
        return (tuple, tuple(_value_signature(item, dynamic=dynamic) for item in value))
    if isinstance(value, list):
        return (list, tuple(_value_signature(item, dynamic=dynamic) for item in value))
    if isinstance(value, dict):
        items = sorted(
            (
                (
                    _value_signature(key, dynamic=dynamic),
                    _value_signature(item, dynamic=dynamic),
                )
                for key, item in value.items()
            ),
            key=repr,
        )
        return (dict, tuple(items))
    return (type(value), id(value))


def _input_signature(
    args: tuple[Any, ...], kwargs: dict[str, Any], *, dynamic: bool
) -> Any:
    return (
        tuple(_value_signature(item, dynamic=dynamic) for item in args),
        tuple(
            sorted(
                (key, _value_signature(value, dynamic=dynamic))
                for key, value in kwargs.items()
            )
        ),
    )


_grad_enabled: Any = None


def _grad_state_component() -> tuple[Any, ...]:
    """The ambient autograd state at call time.

    A region captured with autograd enabled wires its outputs for backward;
    replaying it with the state flipped (or the reverse) would hand the
    caller an output under the wrong autograd contract, so the state is
    part of every specialization key and is read on every call.
    """
    global _grad_enabled
    if _grad_enabled is None:
        import tensorplay

        _grad_enabled = tensorplay.is_grad_enabled
    return (_grad_enabled(),)


def _quick_value_signature(value: Any, *, dynamic: bool) -> Any:
    """Build the hot-path guard key without repr-heavy metadata formatting."""

    if type(value).__module__.startswith("tensorplay"):
        shape = getattr(value, "shape", None)
        try:
            shape_key = ("dynamic", len(shape)) if dynamic else tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            shape_key = repr(shape)
        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)
        device_type = getattr(device, "type", None)
        if device_type is None:
            device_type = repr(device)
        device_key = (
            device_type,
            getattr(device, "index", None),
        )
        requires_grad = getattr(value, "requires_grad", False)
        return (type(value), shape_key, dtype, device_key, bool(requires_grad))
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return (type(value), value)
    if isinstance(value, tuple):
        return (tuple, tuple(_quick_value_signature(item, dynamic=dynamic) for item in value))
    if isinstance(value, list):
        return (list, tuple(_quick_value_signature(item, dynamic=dynamic) for item in value))
    if isinstance(value, dict):
        return (
            dict,
            tuple(
                sorted(
                    (
                        key,
                        _quick_value_signature(item, dynamic=dynamic),
                    )
                    for key, item in value.items()
                )
            ),
        )
    return (type(value), id(value))


def _quick_input_signature(
    args: tuple[Any, ...], kwargs: dict[str, Any], *, dynamic: bool
) -> Any:
    return (
        tuple(_quick_value_signature(item, dynamic=dynamic) for item in args),
        tuple(
            sorted(
                (key, _quick_value_signature(value, dynamic=dynamic))
                for key, value in kwargs.items()
            )
        ),
    )


def _arg_fingerprint(value: Any) -> Any:
    """Cheap per-call identity probe for the hot-path key memo.

    ``(id, version)`` for tensors: in-place mutation bumps ``_version`` so a
    cached key component is never reused across mutated inputs; fresh tensors
    have fresh ids.  Inference tensors carry no version counter and are
    immutable, so their identity alone keys the entry.  Other inputs without
    a version counter use tensor metadata.  Scalars compare by value.  This
    replaces per-call shape/dtype/device reads and tuple rebuilding, which
    profiling showed at ~40% of steady-state compiled-call time.
    """

    module = type(value).__module__
    if module.startswith("tensorplay"):
        try:
            version = value._version
        except RuntimeError:
            if getattr(value, "is_inference", lambda: False)():
                return ("t", id(value), None)
            version = ("metadata", _quick_value_signature(value, dynamic=False))
        return (
            "t",
            id(value),
            version,
            bool(getattr(value, "requires_grad", False)),
        )
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return ("v", type(value).__name__, value)
    return ("o", id(value))


def _call_fingerprint(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple:
    items = [_arg_fingerprint(item) for item in args]
    if kwargs:
        items.extend(
            (key, _arg_fingerprint(kwargs[key])) for key in sorted(kwargs)
        )
    return tuple(items)


def _backend_kwargs(
    *,
    mode: str | None,
    options: dict[str, Any] | None,
    name: str | None,
    dynamic: bool | None,
    strict_native: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if mode is not None and mode != "default":
        kwargs["mode"] = mode
    if options:
        kwargs["options"] = dict(options)
    if name is not None:
        kwargs["name"] = name
    if dynamic is not None:
        kwargs["dynamic"] = dynamic
    if strict_native:
        # Capture may run in Python, but the returned callable must be a
        # native backend executable when this contract is requested.
        kwargs["strict_native"] = True
    return kwargs


def _log_recompiles(config_verbose: bool) -> bool:
    import os

    return config_verbose or os.environ.get("TP_LOG_RECOMPILES", "") not in ("", "0")


def compile(
    model: Callable[..., Any] | None = None,
    *,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    backend: str | CompilerFn | None = None,
    mode: str | None = None,
    options: dict[str, Any] | None = None,
    name: str | None = None,
    disable: bool = False,
    recompile_limit: int | None = None,
    isolate_recompiles: bool = False,
    strict_native: bool = False,
    dynamic_shapes: Any = None,
) -> Callable[..., Any]:
    """Compile a callable through the TensorPlay compiler frontend.

    ``backend`` may be a registered name or a callable with the contract
    ``backend(graph_module, example_inputs, **kwargs) -> callable``.  The
    frontend caches specializations by input metadata; capture and backend
    failures are surfaced as compiler errors.
    """

    normalized_dynamic_shapes = dynamic_shapes
    if dynamic_shapes is not None:
        if dynamic is not None:
            raise RuntimeError("dynamic and dynamic_shapes cannot both be specified")
        if not isinstance(dynamic_shapes, bool):
            raise TypeError(
                "TensorPlay dynamic_shapes currently accepts only a bool; "
                "use a bool dynamic policy for this frontend"
            )
        dynamic = dynamic_shapes

    if mode is not None and options is not None:
        raise RuntimeError("Either mode or options can be specified, but not both")
    if mode is None and options is None:
        mode = "default"
    if options is not None and not isinstance(options, dict):
        raise TypeError(f"options must be a dict, got {type(options)!r}")

    from tensorplay.compiler import config as compiler_config

    configured_dynamic = compiler_config.dynamic_shapes
    if dynamic is None and normalized_dynamic_shapes is None:
        if configured_dynamic is not None:
            dynamic = configured_dynamic
        elif not compiler_config.assume_static_by_default:
            dynamic = True
    if not isinstance(compiler_config.assume_static_by_default, bool):
        raise TypeError("config.assume_static_by_default must be a bool")
    if not isinstance(compiler_config.verbose, bool):
        raise TypeError("config.verbose must be a bool")
    if not isinstance(compiler_config.fail_on_recompile_limit_hit, bool):
        raise TypeError("config.fail_on_recompile_limit_hit must be a bool")
    if not isinstance(compiler_config.force_disable_caches, bool):
        raise TypeError("config.force_disable_caches must be a bool")
    accumulated_limit = compiler_config.accumulated_recompile_limit
    if (
        not isinstance(accumulated_limit, int)
        or isinstance(accumulated_limit, bool)
        or accumulated_limit < 1
    ):
        raise ValueError("config.accumulated_recompile_limit must be a positive integer")

    if model is None:
        return lambda actual_model: compile(
            actual_model,
            fullgraph=fullgraph,
            backend=backend,
            dynamic=dynamic,
            mode=mode,
            options=options,
            name=name,
            disable=disable,
            recompile_limit=recompile_limit,
            isolate_recompiles=isolate_recompiles,
            strict_native=strict_native,
            # ``dynamic_shapes`` has already been normalized into ``dynamic``
            # above.  Passing it again would look like the user supplied both
            # mutually exclusive knobs on the recursive decorator call.
            dynamic_shapes=None,
        )
    if not callable(model):
        raise TypeError(f"compile() expected a callable, got {type(model)!r}")
    if disable:
        return model
    if recompile_limit is not None and recompile_limit < 1:
        raise ValueError("recompile_limit must be positive")

    backend_spec = get_default_backend() if backend is None else backend
    compiler_fn = lookup_backend(backend_spec)
    backend_kwargs = _backend_kwargs(
        mode=mode,
        options=options,
        name=name,
        dynamic=dynamic,
        strict_native=strict_native,
    )
    specialization_dynamic = dynamic is True
    specialization_limit = (
        _DEFAULT_RECOMPILE_LIMIT if recompile_limit is None else recompile_limit
    )
    if recompile_limit is None:
        specialization_limit = compiler_config.recompile_limit
    if (
        not isinstance(specialization_limit, int)
        or isinstance(specialization_limit, bool)
        or specialization_limit < 1
    ):
        raise ValueError("recompile_limit must be a positive integer")
    cache_enabled = not compiler_config.force_disable_caches
    compile_attempts = 0
    cache: dict[Any, Callable[..., Any]] = {}
    guard_chains: dict[Any, GuardChain] = {}
    lock = threading.RLock()
    last_quick_key: Any = object()
    last_compiled_fn: Callable[..., Any] | None = None
    guard_param_names: tuple[str, ...] = ()
    gate_evaluator: Callable[..., tuple] | None = None
    target_cache = model.forward if _is_module_like(model) else model
    try:
        target_signature: Any = inspect.signature(target_cache)
    except (TypeError, ValueError):
        target_signature = None
    last_call_fp: Any = None
    last_quick_parts: tuple[Any, ...] | None = None

    def _guard_component(
        args_: tuple[Any, ...],
        kwargs_: dict[str, Any],
        builder: Callable[..., Any],
    ) -> tuple[Any, ...]:
        if not guard_param_names:
            return ()
        if target_signature is None:
            return ("shape-guards", "unbound")
        try:
            bound = target_signature.bind_partial(*args_, **kwargs_)
            bound.apply_defaults()
        except (TypeError, ValueError):
            return ("shape-guards", "unbound")
        return (
            "shape-guards",
            tuple(
                builder(bound.arguments.get(name), dynamic=False)
                for name in guard_param_names
            ),
        )

    def _bind_dispatcher_fast(fast: Any, nargs: int) -> None:
        """Hand a resolved steady-state entry to the outer C dispatcher.

        The dispatcher re-enters it directly once its argument memo passes;
        any slow call routes through here again, so the binding stays
        current across recompiles and specialization switches.

        The argument memo cannot see the ambient autograd state, so the
        entry carries the state it was resolved under and falls back to the
        trampoline when a call arrives under a flipped state; the trampoline
        then resolves (and binds) the specialization for that state.
        """

        bound_grad_state = _grad_state_component()

        def _fast(*args: Any) -> Any:
            if _grad_state_component() == bound_grad_state:
                return fast(*args)
            return optimized(*args)

        dispatcher.tpx_set_fast(_fast, nargs)

    @functools.wraps(model)
    def optimized(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_quick_key, last_compiled_fn
        nonlocal guard_param_names, gate_evaluator
        nonlocal last_call_fp, last_quick_parts
        nonlocal compile_attempts
        if _capture_disabled.get():
            return model(*args, **kwargs)
        # Steady-state memo: identical objects with unchanged versions (or
        # unchanged scalars) cannot produce different signatures or gate
        # outcomes -- skip metadata reads and evaluator replay entirely.
        # Tensor-positional calls ask the C dispatcher for a read-only
        # certificate against the trampoline fingerprint (one probe loop in
        # C replaces the Python (id, version) walk); every other call shape
        # falls back to the Python fingerprint below.
        if dispatcher is not None:
            matched = not kwargs and dispatcher.tpx_fingerprint_matches(*args)
            if not matched:
                # Refresh the certificate for this call up front so it stays
                # paired with the quick memo built below even when resolution
                # raises.  Calls the fingerprint cannot describe (scalars,
                # kwargs) clear it instead of leaving a stale certificate.
                if not kwargs:
                    dispatcher.tpx_fingerprint_store(*args)
                else:
                    dispatcher.tpx_fingerprint_clear()
        else:
            matched = False
        if last_quick_parts is not None and matched:
            input_signature, shape_component, data_component = last_quick_parts
        else:
            call_fp = _call_fingerprint(args, kwargs)
            if last_quick_parts is not None and call_fp == last_call_fp:
                input_signature, shape_component, data_component = last_quick_parts
            else:
                input_signature = _quick_input_signature(
                    args, kwargs, dynamic=specialization_dynamic
                )
                shape_component = _guard_component(
                    args, kwargs, _quick_value_signature
                )
                data_component = gate_evaluator(args, kwargs) if gate_evaluator else ()
            last_quick_parts = (
                input_signature,
                shape_component,
                data_component,
            )
            last_call_fp = call_fp
        grad_state = _grad_state_component()
        quick_key = (input_signature, shape_component, data_component, grad_state)
        with lock:
            if cache_enabled and cache and last_compiled_fn is not None and quick_key == last_quick_key:
                compiled_fn = last_compiled_fn
            else:
                key = (
                    _input_signature(args, kwargs, dynamic=specialization_dynamic),
                    _guard_component(args, kwargs, _value_signature),
                    data_component,
                    grad_state,
                )
                compiled_fn = cache.get(key) if cache_enabled else None
            store_compiled = cache_enabled
            if compiled_fn is None:
                if len(cache) >= specialization_limit:
                    if fullgraph or compiler_config.fail_on_recompile_limit_hit:
                        raise RuntimeError(
                            "TensorPlay compile specialization limit reached"
                        )
                    store_compiled = False
                # Explain the miss against every stored specialization before
                # recompiling, retaining each guard mismatch.
                reasons: list[Any] = []
                for chain in guard_chains.values():
                    reasons.extend(chain.explain(args, kwargs))
                if reasons:
                    optimized._tensorplay_last_recompile_reasons = tuple(reasons)
                    if _log_recompiles(compiler_config.verbose):
                        import warnings

                        warnings.warn(
                            "recompiling "
                            f"{getattr(model, '__name__', model)!r}: "
                            + format_recompile_reasons(reasons),
                            stacklevel=2,
                        )
                if compile_attempts >= accumulated_limit:
                    raise RuntimeError(
                        "TensorPlay accumulated recompilation limit reached"
                    )
                compile_attempts += 1
                compiled_fn, captured_gm = _compile_region(
                    model,
                    compiler_fn,
                    args,
                    kwargs,
                    fullgraph=fullgraph,
                    backend_kwargs=backend_kwargs,
                    # An autograd artifact reads recorded values off the nodes
                    # while it runs, so a training region keeps them; an
                    # inference artifact consumes none and drops them here.
                    preserve_recorded_values=bool(grad_state[0]),
                    region_key=make_region_key(
                        target_cache,
                        model if _is_module_like(model) else None,
                        key,
                    ),
                )
                # Keys gain a guard component once capture reveals metadata
                # reads or control-flow gates; invalidate so entries are stored
                # uniformly.
                promoted = _extract_shape_guard_params(captured_gm)
                replay = captured_gm.meta.get("guard_replay")
                if promoted - set(guard_param_names) or (
                    replay is not None and gate_evaluator is None
                ):
                    guard_param_names = tuple(sorted({*guard_param_names, *promoted}))
                    if replay is not None:
                        gate_target = model.forward if _is_module_like(model) else model
                        gate_evaluator = _make_gate_evaluator(replay, gate_target)
                    if cache_enabled:
                        cache.clear()
                        guard_chains.clear()
                    last_compiled_fn = None
                    last_call_fp = None
                    last_quick_parts = None
                    if gate_evaluator is not None:
                        data_component = gate_evaluator(args, kwargs)
                key = (
                    _input_signature(args, kwargs, dynamic=specialization_dynamic),
                    _guard_component(args, kwargs, _value_signature),
                    data_component,
                    grad_state,
                )
                if store_compiled:
                    cache[key] = compiled_fn
                    guard_chains[key] = build_guard_chain(
                        key,
                        args=args,
                        kwargs=kwargs,
                        dynamic=specialization_dynamic,
                        target=model.forward if _is_module_like(model) else model,
                        gate_evaluator=gate_evaluator,
                    )
            last_quick_key = quick_key
            last_compiled_fn = compiled_fn
        if not kwargs:
            fast = getattr(compiled_fn, "_fast_call", None)
            if fast is not None:
                _bind_dispatcher_fast(fast, len(args))
                return compiled_fn(*args)
        return compiled_fn(*args, **kwargs)

    from tensorplay._C import _stax as _stax_native

    dispatcher: Any = _stax_native.make_call_dispatcher(optimized)

    optimized._tensorplay_backend = backend_spec  # type: ignore[attr-defined]
    optimized._tensorplay_cache = cache  # type: ignore[attr-defined]
    optimized._tensorplay_guard_chains = guard_chains  # type: ignore[attr-defined]
    optimized._tensorplay_last_recompile_reasons = ()  # type: ignore[attr-defined]
    optimized._tensorplay_original = model  # type: ignore[attr-defined]
    optimized._tensorplay_dynamic = dynamic  # type: ignore[attr-defined]
    optimized._tensorplay_dynamic_shapes = normalized_dynamic_shapes  # type: ignore[attr-defined]
    optimized._tensorplay_isolate_recompiles = isolate_recompiles  # type: ignore[attr-defined]
    optimized._tensorplay_recompile_limit = specialization_limit  # type: ignore[attr-defined]
    _compiled_wrappers.add(optimized)
    if dispatcher is not None:
        dispatcher._tensorplay_dispatcher = optimized  # type: ignore[attr-defined]
        return dispatcher
    return optimized


def _bind_sample_arguments(
    model: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Bind example call arguments to parameter names for the tracer.

    Metadata reads on placeholders (``x.shape[0] > 2``, ``range(x.ndim)``)
    then specialize statically during capture; the compile signature already
    keys on these fields, so no additional recompile conditions appear.
    """

    target = model.forward if _is_module_like(model) else model
    try:
        signature = inspect.signature(target)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError):
        return None
    return dict(bound.arguments)


def _is_module_like(value: Any) -> bool:
    return hasattr(value, "named_modules") and callable(
        getattr(value, "forward", None)
    )


def _region_is_training(
    example_inputs: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    graph_module: GraphModule | None = None,
) -> bool:
    """Whether any example input carries an autograd requirement.

    A region needs the autograd pipeline when any example input requires
    grad, when the captured module root is in training mode, or when any
    captured attribute (module parameter or buffer) requires grad.
    """

    import tensorplay

    def touches_autograd(value: Any) -> bool:
        if isinstance(value, tensorplay.Tensor):
            return bool(value.requires_grad)
        if isinstance(value, (tuple, list)):
            return any(touches_autograd(item) for item in value)
        if isinstance(value, dict):
            return any(touches_autograd(item) for item in value.values())
        return False

    if any(touches_autograd(item) for item in example_inputs) or any(
        touches_autograd(item) for item in example_kwargs.values()
    ):
        return True
    if graph_module is None:
        return False
    # A module left in training mode is a training region even when a call's
    # inputs carry no gradient requirement: its parameters still do, and the
    # backward must be compiled alongside the forward.
    root = getattr(graph_module, "root", None)
    if isinstance(root, tensorplay.nn.Module) and root.training:
        return True
    for node in graph_module.graph.nodes:
        if node.op != "get_attr":
            continue
        try:
            value = graph_module._get_attr(node.target)
        except (AttributeError, KeyError, IndexError, TypeError):
            continue
        if touches_autograd(value):
            return True
    return False


def _adapt_backend_to_region(
    compiler_fn: CompilerFn,
    example_inputs: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    graph_module: GraphModule | None = None,
    backend_kwargs: dict[str, Any] | None = None,
) -> CompilerFn:
    """Match a backend to the region being compiled.

    Training regions are routed through ahead-of-time autograd so parameters
    are lifted into explicit inputs before the backend sees them (the same
    pipeline the reference ``compile_fx`` runs); inference regions go to the
    backend directly.
    """

    if not _region_is_training(example_inputs, example_kwargs, graph_module):
        return compiler_fn
    from .aot_autograd import default_partition
    from .common import aot_autograd

    forward_compiler = compiler_fn
    if backend_kwargs:
        kwargs = dict(backend_kwargs)

        def forward_compiler(graph, inputs):
            return compiler_fn(graph, inputs, **kwargs)

        forward_compiler.__name__ = getattr(compiler_fn, "__name__", "compiler_fn")

    from tensorplay._decomp import get_decompositions
    import tensorplay

    _GN_DECOMPOSITION_OPS = (
        tensorplay.ops.tp.native_group_norm,
        tensorplay.ops.tp.native_group_norm_backward,
    )

    return aot_autograd(
        fw_compiler=forward_compiler,
        bw_compiler=forward_compiler,
        partition_fn=default_partition,
        decompositions=get_decompositions(_GN_DECOMPOSITION_OPS),
    )


@contextlib.contextmanager
def _preserve_module_state(program: Any) -> Any:
    """Undo the state updates an executing capture makes.

    Capture runs the program to record it; in-place updates it performs on
    the module's parameters and buffers (running statistics, counters) must
    not count as a call, so their values are restored afterwards.
    """

    import tensorplay

    module = program if isinstance(program, tensorplay.nn.Module) else getattr(program, "__self__", None)
    if not isinstance(module, tensorplay.nn.Module):
        yield
        return
    with tensorplay.no_grad():
        saved = [(t, t.clone()) for t in list(module.parameters()) + list(module.buffers())]
    try:
        yield
    finally:
        with tensorplay.no_grad():
            for tensor, value in saved:
                if tuple(tensor.shape) == tuple(value.shape):
                    tensor.copy_(value)


@contextlib.contextmanager
def _preserve_input_values(*values: Any) -> Any:
    """Undo the input mutations an executing capture makes.

    Capture, propagation and lowering all execute the captured program to
    record or measure it; in-place updates it performs on its input tensors
    must not count as a call, so the input values are restored afterwards.
    The same values stand in for every compile-time run, which is what lets a
    program that mutates one of its arguments behave identically at compile
    time and at runtime without rewriting the caller's tensor.
    """

    import tensorplay

    with tensorplay.no_grad():
        saved: list[tuple[Any, Any]] = []

        def _save(value: Any) -> None:
            if isinstance(value, tensorplay.Tensor):
                saved.append((value, value.clone()))
            elif isinstance(value, (tuple, list)):
                for item in value:
                    _save(item)
            elif isinstance(value, dict):
                for item in value.values():
                    _save(item)

        _save(values)
    try:
        yield
    finally:
        with tensorplay.no_grad():
            for tensor, value in saved:
                if tuple(tensor.shape) == tuple(value.shape):
                    tensor.copy_(value)


def _compile_region(
    model: Callable[..., Any],
    compiler_fn: CompilerFn,
    example_inputs: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    *,
    fullgraph: bool,
    backend_kwargs: dict[str, Any],
    region_key: str | None = None,
    preserve_recorded_values: bool = False,
) -> tuple[Callable[..., Any], GraphModule]:
    stored = load_region(region_key, model)
    if stored is not None:
        graph_module = stored
    else:
        with _preserve_input_values(example_inputs, example_kwargs):
            try:
                with _compiler_context(), _preserve_module_state(model):
                    tracer = Tracer(execute=True)
                    # An operator that stands for a region of the program -- rather
                    # than for one operation -- has to be told that a capture is
                    # running, or it runs itself and the region it stands for is
                    # inlined into the graph as the operations that implement it.
                    # That inlined form is correct, and it is exactly what a backend
                    # holding a schedule for that region can never be handed.  The
                    # state is entered around the trace alone: the passes that
                    # follow rewrite the graph that came out, and one of them
                    # evaluates values, which must not be recorded as part of the
                    # region.
                    with tracer.proxy_mode:
                        graph_module = tracer.trace(
                            model,
                            sample_inputs=_bind_sample_arguments(
                                model, example_inputs, example_kwargs
                            ),
                        )
                    # Default capture pipeline: canonicalize operators, then constant
                    # folding, then decomposition, common-subexpression elimination
                    # and dead code elimination; fusion hints are stamped last so
                    # they see the final graph.  CSE runs after decomposition so
                    # shared sub-chains across rewritten composites collapse too
                    # (two gelu sites share one erf chain).  Backends always receive
                    # a folded, linted, hint-annotated graph; ShapeProp below
                    # additionally annotates tensor shapes.
                    pass_result = PassManager(
                        [
                            NormalizeOperators(),
                            ConstFold(),
                            DecomposePass(),
                            CSEPass(get_CSE_banned_ops()),
                            DeadCodeElimination(),
                            PointwiseFusionHint(),
                        ]
                    )(graph_module)
                    graph_module = pass_result.graph_module
            except GraphCaptureError as exc:
                raise GraphCaptureError(
                    "TensorPlay could not capture the requested compiler region"
                ) from exc
            store_region(region_key, graph_module)

    # Capture, propagation and lowering all execute the program to record or
    # measure it.  The generator they advance belongs to the caller, so the
    # whole region is compiled inside a forked RNG: the draws the graph reads
    # at runtime start from where the caller left the generator, not from
    # where the compiler's own sample pass stopped.
    import tensorplay

    devices = []
    for value in example_inputs:
        if isinstance(value, tensorplay.Tensor) and value.device.type == "cuda":
            index = value.device.index or 0
            if index not in devices:
                devices.append(index)
    with _preserve_input_values(example_inputs, example_kwargs):
        with tensorplay.random.fork_rng(devices=devices):
            # Backend failures are compiler failures, not graph breaks.  In
            # particular, a Stax lowering error must not silently turn a requested
            # compiled region into an uncompiled call.
            # Registered backends receive graph inputs in placeholder order, including
            # values supplied through keywords and defaults.  Passing only
            # positional arguments makes a keyword-only/scalar placeholder appear
            # to be missing and is especially harmful for native Stax lowering.
            bound = graph_module.signature.bind_partial(*example_inputs, **example_kwargs)
            bound.apply_defaults()
            # Numeric-gate placeholders ride the contract as synthetic inputs; their
            # trace-time values stand in at lowering so kernels see real 0-d tensors.
            backend_inputs = []
            for node in graph_module.graph.placeholders:
                parameter_name = node.target if isinstance(node.target, str) else node.name
                try:
                    backend_inputs.append(bound.arguments[parameter_name])
                except KeyError:
                    if node.name not in bound.arguments:
                        raise GraphCaptureError(
                            f"missing sample value for graph placeholder {node.name!r}"
                        ) from None
                    backend_inputs.append(bound.arguments[node.name])

            # Advisory shape/value metadata for backends and visualization; never a
            # reason to reject an otherwise compilable region.
            # Propagation executes the graph: it must not count as a call either.
            try:
                with _preserve_module_state(model):
                    ShapeProp(backend_inputs)(graph_module)
            except (GraphCaptureError, RuntimeError):
                pass

            compiler_fn = _adapt_backend_to_region(
                compiler_fn,
                example_inputs,
                example_kwargs,
                graph_module,
                backend_kwargs,
            )

            regional_compile_invoke_subgraph(
                graph_module,
                compiler=compiler_fn,
                compiler_kwargs=backend_kwargs,
            )

            with _compiler_context():
                compiled = compiler_fn(graph_module, backend_inputs, **backend_kwargs)

    if not callable(compiled):
        raise TypeError(
            f"compiler backend returned {type(compiled)!r}; expected a callable"
        )
    if not preserve_recorded_values:
        _release_recorded_values(graph_module)
    return compiled, graph_module


def _release_recorded_values(graph_module: Any) -> None:
    """Drop execution artifacts that capture-time propagation parked on nodes.

    Shape propagation executes the region on real tensors and leaves every
    intermediate on its node for the lowering pipeline to inspect.  Once the
    backend has produced its artifact those tensors are dead weight: one
    region can hold the entire forward's activation set, and every cached
    specialization would keep its own copy alive.  The cheap metadata each
    node carries (shape, dtype, device) stays behind, and propagation on a
    later compile of the same region regenerates whatever values the
    pipeline consumes.
    """
    from tensorplay import Tensor

    try:
        nodes = graph_module.graph.nodes
    except AttributeError:
        return

    def strip(value: Any) -> Any:
        return None if isinstance(value, Tensor) else value

    for node in nodes:
        meta = getattr(node, "meta", None)
        if not meta:
            continue
        value = meta.get("val")
        if isinstance(value, Tensor):
            meta["val"] = None
        elif isinstance(value, tuple):
            meta["val"] = tuple(strip(item) for item in value)


_SHAPE_GUARD_ATTRS = frozenset({"shape", "len", "ndim"})


def _make_gate_evaluator(replay: dict[str, Any], target: Any) -> Callable[..., tuple]:
    """Build the per-call gate re-evaluator for one specialization (L1-D1).

    Replays the extracted condition subgraph on live inputs and returns the
    branch-deciding outcomes (``bool``/``iter`` gates) as the cache-key tail.
    Numeric-gate values never fragment the cache: they stay live inside the
    captured graph itself (GateValue proxies keep the condition subgraph
    reachable from the output), so the artifact recomputes them per call.
    """

    from tensorplay.graph import GraphModule, gate_outcome

    mini = GraphModule(
        target,
        replay["graph"],
        inspect.Signature(
            [
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in replay["placeholders"]
            ]
        ),
    )
    gates = replay["gates"]
    target_signature = None
    try:
        target_signature = inspect.signature(target)
    except (TypeError, ValueError):
        pass

    def evaluate(args_: tuple[Any, ...], kwargs_: dict[str, Any]) -> tuple:
        if target_signature is not None:
            try:
                bound = target_signature.bind_partial(*args_, **kwargs_)
                bound.apply_defaults()
            except (TypeError, ValueError):
                return ("gates", "unbound")
            feeds_src = {
                name: bound.arguments.get(name) for name in replay["placeholders"]
            }
        else:
            feeds_src = dict(zip(replay["placeholders"], args_))
        values = mini._interpret(**feeds_src)
        outputs = values if isinstance(values, tuple) else (values,)
        # graph.gate() nodes stay symbolic inside the captured graph, so
        # their concrete values never fragment reuse; plain int()/float()
        # consumption bakes constants into the artifact and MUST key.
        symbolic = set(replay.get("symbolic") or ())
        key_tail: list[Any] = ["gates"]
        for output, (node_name, kind) in zip(outputs, gates):
            if kind in ("int", "float", "index") and node_name in symbolic:
                continue
            key_tail.append(gate_outcome(kind, output))
        return tuple(key_tail)

    return evaluate


def _extract_shape_guard_params(graph_module: GraphModule) -> frozenset[str]:
    """Parameters whose captured metadata reads require exact-shape guards.

    Branching on ``x.shape[0]`` bakes one side of the branch into the graph,
    so a dynamic-mode cache entry may only be reused while that placeholder's
    shape stays identical.  dtype/device/reads need no extra guards: they are
    already part of every specialization signature.
    """

    touches = getattr(graph_module, "meta", {}).get("metadata_touches") or ()
    names = {name for name, attr in touches if attr in _SHAPE_GUARD_ATTRS}
    if not names or graph_module.signature is None:
        return frozenset()
    return frozenset(
        name for name in graph_module.signature.parameters if name in names
    )


def reset() -> None:
    """Clear all per-wrapper compiler specializations and backend state."""

    from .registry import reset_backends

    reset_backends()
    for wrapper in list(_compiled_wrappers):
        cache = getattr(wrapper, "_tensorplay_cache", None)
        if cache is not None:
            cache.clear()
        chains = getattr(wrapper, "_tensorplay_guard_chains", None)
        if chains is not None:
            chains.clear()
        try:
            wrapper._tensorplay_last_recompile_reasons = ()
        except Exception:
            pass
