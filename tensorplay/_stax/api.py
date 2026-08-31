"""Public compiler orchestration for TensorPlay.

Capture, backend selection, and execution concerns are separate. A backend is
never asked to discover Python control flow; it only receives a captured
``GraphModule`` and example inputs.
"""

from __future__ import annotations

import functools
import inspect
import threading
import weakref
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
from tensorplay.graph.passes.regional_inductor_invoke_subgraph import (
    regional_inductor_invoke_subgraph,
)
from .registry import CompilerFn, get_default_backend, lookup_backend


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
    have fresh ids.  Scalars compare by value.  This replaces per-call
    shape/dtype/device reads and tuple rebuilding, which profiling showed at
    ~40% of steady-state compiled-call time.
    """

    module = type(value).__module__
    if module.startswith("tensorplay") and hasattr(value, "_version"):
        return (
            "t",
            id(value),
            getattr(value, "_version", None),
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
    last_arg_refs: tuple[weakref.ReferenceType[Any], ...] | None = None
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

    @functools.wraps(model)
    def optimized(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_quick_key, last_compiled_fn, last_arg_refs
        nonlocal guard_param_names, gate_evaluator
        nonlocal last_call_fp, last_quick_parts
        nonlocal compile_attempts
        if _capture_disabled.get():
            return model(*args, **kwargs)
        # Identity reuse is unsound once control-flow gates exist: in-place
        # mutation preserves weakref identity but can flip the traced branch,
        # so gate outcomes are always re-evaluated.
        same_last_args = (
            gate_evaluator is None
            and not kwargs
            and last_arg_refs is not None
            and len(last_arg_refs) == len(args)
            and all(reference() is value for reference, value in zip(last_arg_refs, args))
        )
        # Steady-state memo: identical objects with unchanged versions (or
        # unchanged scalars) cannot produce different signatures or gate
        # outcomes -- skip metadata reads and evaluator replay entirely.
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
        quick_key = (input_signature, shape_component, data_component)
        with lock:
            if cache_enabled and cache and last_compiled_fn is not None and quick_key == last_quick_key:
                compiled_fn = last_compiled_fn
            else:
                key = (
                    _input_signature(args, kwargs, dynamic=specialization_dynamic),
                    _guard_component(args, kwargs, _value_signature),
                    data_component,
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
                try:
                    last_arg_refs = tuple(weakref.ref(value) for value in args)
                except TypeError:
                    last_arg_refs = None
            else:
                last_arg_refs = None
        if not kwargs:
            fast = getattr(compiled_fn, "_fast_call", None)
            if fast is not None:
                return fast(*args)
        return compiled_fn(*args, **kwargs)

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


def _compile_region(
    model: Callable[..., Any],
    compiler_fn: CompilerFn,
    example_inputs: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    *,
    fullgraph: bool,
    backend_kwargs: dict[str, Any],
) -> tuple[Callable[..., Any], GraphModule]:
    try:
        with _compiler_context():
            graph_module = Tracer(execute=True).trace(
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

    # Backend failures are compiler failures, not graph breaks.  In
    # particular, a Stax lowering error must not silently turn a requested
    # compiled region into an uncompiled call.
    # Registered backends receive graph inputs in placeholder order, including
    # values supplied through keywords and defaults.  Passing only positional
    # arguments makes a keyword-only/scalar placeholder appear to be missing
    # and is especially harmful for native Stax lowering.
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
    try:
        ShapeProp(backend_inputs)(graph_module)
    except (GraphCaptureError, RuntimeError):
        pass

    regional_inductor_invoke_subgraph(
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
    return compiled, graph_module


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
    """Clear all per-wrapper compiler specializations."""

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
