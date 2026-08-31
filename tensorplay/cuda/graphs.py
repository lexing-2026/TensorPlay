# mypy: allow-untyped-defs
r"""CUDA graph capture and replay helpers.

Backed by the native :class:`tensorplay._C.CUDAGraph` class (capture on a
dedicated side stream or a user-supplied one, graph-private allocator pools
shareable across graphs, eager instantiation at ``capture_end``, replay of the
cached executable, graph-safe RNG refresh per replay).


    g = tensorplay.cuda.CUDAGraph()
    static_input = x.clone()
    # warm up on the capture stream so lazy state lands outside the capture
    with tensorplay.cuda.graph(g):
        static_output = model(static_input)
    # ...later, refresh inputs and re-run without host-side launch overhead:
    static_input.copy_(new_batch)
    g.replay()

For bulk replays where every input is a fresh CUDA tensor,
:meth:`CUDAGraph.stage_and_launch` stages all inputs with raw async copies
(dispatcher bypass) and launches in one call - measurably less host overhead
than per-tensor ``copy_`` plus ``replay``.

Static input/output tensors must stay alive for as long as the graph; their
addresses are baked into the executable.  Random ops captured inside the graph
read their (seed, offset) from a graph-owned device buffer that every
:meth:`CUDAGraph.replay` refreshes from the generator, so each replay draws a
"""

import contextlib

__all__ = [
    "CUDAGraph",
    "graph",
    "graph_pool_handle",
    "is_current_stream_capturing",
    "make_graphed_callables",
    "make_graphed_autograd_function",
    "export_dot",
]


def _native():
    try:
        from .. import _C
    except Exception as exc:  # pragma: no cover - import failure diagnostics
        raise RuntimeError(
            "CUDA graphs require tensorplay._C; import failed: "
            f"{exc!r}. Was TensorPlay built with CUDA support?"
        ) from exc
    if not hasattr(_C, "CUDAGraph"):
        raise RuntimeError(
            "CUDA graphs are not supported by this TensorPlay build "
            "(tensorplay._C exposes no CUDAGraph class)"
        )
    return _C


def graph_pool_handle():
    r"""Return an opaque token representing the id of a graph memory pool.

    Pass it as the ``pool=`` argument of :func:`graph` so several graphs
    capture into (and reuse memory from) one shared private pool.
    """

    return _native().graph_pool_handle()


def _resolve_pool(pool):
    if pool is None:
        return 0
    if isinstance(pool, CUDAGraph):
        return pool.pool_id
    if isinstance(pool, int):
        return pool
    raise TypeError(
        "pool must be None, an id from graph_pool_handle(), or another "
        f"CUDAGraph; got {type(pool).__name__}"
    )


class CUDAGraph:

    def __init__(self):
        self._c = _native().CUDAGraph()

    def capture_begin(self, pool=None, capture_error_mode="global", stream=None):
        """Begin capture.

        Args:
            pool: ``None`` captures into a fresh private pool; an id from
                :func:`graph_pool_handle`, another graph's ``pool_id``, or
                another :class:`CUDAGraph` shares that pool instead.
            capture_error_mode (str): ``"global"`` fails the capture if any
                unsafe CUDA call happens anywhere in the process;
                ``"thread_local"`` only watches this thread; ``"relaxed"``
                does not guard against unsafe calls.
            stream: custom capture stream (a ``tensorplay.cuda.Stream``).
                Defaults to the runtime's dedicated per-device side stream;
                the legacy default stream cannot participate in capture.
        """

        if stream is not None:
            from .streams import Stream

            if not isinstance(stream, Stream):
                raise TypeError(
                    f"stream must be a tensorplay.cuda.Stream; got "
                    f"{type(stream).__name__}"
                )
            native_stream = stream._stream
        else:
            native_stream = None
        self._c.capture_begin(
            _resolve_pool(pool), capture_error_mode, native_stream
        )

    def capture_end(self):
        """End capture and compile the executable (paid here, not on first
        replay)."""

        self._c.capture_end()

    def instantiate(self):
        """No-op once instantiated; kept for late callers."""

        self._c.instantiate()

    def replay(self, stream=None):
        """Run the graph: launch the cached executable on the current stream.

        Args:
            stream (Stream, optional): launch on this explicit stream instead
                of querying the current one - shaves a TLS lookup off hot
                loops pinned to a single stream.
        """

        if stream is not None:
            from .streams import Stream

            if not isinstance(stream, Stream):
                raise TypeError(
                    f"stream must be a tensorplay.cuda.Stream; got "
                    f"{type(stream).__name__}"
                )
            self._c.replay(stream._stream)
        else:
            self._c.replay()

    def stage_and_launch(self, static_inputs, inputs):
        r"""Stage every input onto its static buffer and replay in one call.

        Args:
            static_inputs: buffers captured by the graph (kept alive by the
                caller).
            inputs: fresh tensors whose contents overwrite the matching
                static buffer this iteration.  Contiguous same-dtype/
                same-device pairs take a raw async device-to-device copy;
                anything else falls back to full copy semantics.

        This is the low-overhead bulk entry used by
        :mod:`tensorplay._stax.cudagraphs`: one Python-to-native crossing
        for the whole replay instead of one dispatcher round trip per input.
        """

        self._c.stage_and_launch(list(static_inputs), list(inputs))

    def reset(self):
        """Destroy the executable and release the pool reference.

        All tensors allocated during the capture must be released first.
        """

        self._c.reset()

    @property
    def pool_id(self):
        """Allocator pool id this graph captured against."""

        return self._c.pool_id()

    def enable_debug_mode(self):
        self._c.enable_debug_mode()

    def debug_dump(self, path):
        r"""Write a DOT rendering of the captured graph to ``path``.

        Call :meth:`enable_debug_mode` before capturing for a dump that
        includes full node attributes.
        """

        self._c.debug_dump(str(path))

    # --- conditional nodes (CUDA >= 12.4) -----------------------------------

    def _require_conditional_support(self):
        from .. import _C

        probe = getattr(_C, "conditional_nodes_supported", None)
        if probe is not None and not probe():
            raise RuntimeError(
                "CUDA graphs conditional nodes require CUDA >= 12.4"
            )

    def begin_capture_to_if_node(self, scalar_pred):
        r"""Inside an open capture, gate the following work on an ``if`` node.

        ``scalar_pred`` must be a single-element CUDA Bool tensor; at replay
        time the driver samples it and runs the body captured between this
        call and :meth:`end_capture_to_conditional_node` only when true.
        """

        self._require_conditional_support()
        self._c.begin_capture_to_if_node(scalar_pred)

    def begin_capture_to_while_node(self, scalar_pred):
        r"""Like :meth:`begin_capture_to_if_node`, but the body loops while
        the predicate stays true (driver-level while node)."""

        self._require_conditional_support()
        self._c.begin_capture_to_while_node(scalar_pred)

    def set_conditional_handle_for_current_node(self, scalar_pred):
        r"""Refresh the predicate consumed by the innermost open conditional
        node (used for nested conditionals)."""

        self._c.set_conditional_handle_for_current_node(scalar_pred)

    def end_capture_to_conditional_node(self):
        r"""Close the open conditional body; subsequent capture returns to
        the parent stream."""

        self._c.end_capture_to_conditional_node()

    def __del__(self):
        try:
            self.reset()
        except Exception:
            pass


# The most recently completed capture, for module-level DOT export.
_last_captured_graph = None


@contextlib.contextmanager
def graph(cuda_graph, pool=None, stream=None, capture_error_mode="global"):
    r"""Context-manager that captures CUDA work into a ``tensorplay.cuda.CUDAGraph``.

    Args:
        cuda_graph (CUDAGraph): the graph object to capture into.
        pool: ``None`` gives the graph its own private memory pool; an id
            from :func:`graph_pool_handle` (or another graph / its
            ``pool_id`` property) shares that pool so allocations from both
            captures can recycle each other's space.
        stream (Stream, optional): custom capture stream; defaults to the
            runtime's dedicated per-device side stream (the legacy default
            stream cannot capture).
        capture_error_mode (str, optional): see
            :meth:`CUDAGraph.capture_begin`.
    """

    global _last_captured_graph
    if not isinstance(cuda_graph, CUDAGraph):
        raise TypeError(
            "cuda_graph must be a tensorplay.cuda.CUDAGraph; got "
            f"{type(cuda_graph).__name__}"
        )
    cuda_graph.capture_begin(
        pool=pool, capture_error_mode=capture_error_mode, stream=stream
    )
    entered = True
    try:
        yield cuda_graph
    finally:
        # Only close the window when capture_begin succeeded, so a failed
        # begin propagates its own error instead of a confusing secondary
        # "no live capture" from capture_end.
        if entered:
            cuda_graph.capture_end()
            _last_captured_graph = cuda_graph


def is_current_stream_capturing():
    r"""Return True if CUDA graph capture is underway on the current thread."""

    from . import is_initialized

    if not is_initialized():
        return False
    try:
        from .. import _C
    except ImportError:
        return False
    probe = getattr(_C, "cuda_stream_is_capturing", None)
    if probe is not None:
        return bool(probe())
    process_wide = getattr(_C, "cuda_is_capturing", None)
    return bool(process_wide()) if process_wide is not None else False


def export_dot(file_path: str) -> str:
    r"""Export the most recently completed capture to a DOT file.

    Returns the path written.  For richer node attributes call
    :meth:`CUDAGraph.enable_debug_mode` before that capture.
    """

    if _last_captured_graph is None:
        raise RuntimeError("no CUDA graph has been captured yet")
    file_path = str(file_path)
    _last_captured_graph.debug_dump(file_path)
    return file_path


# --- minimal pytree (tree_flatten/tree_unflatten compatibility) -------------
#
# small structural flatten covers the tensor/tuple/list/dict shapes that
# make_graphed_callables deals with.

_Spec = tuple  # ("seq", type, (specs...)) / ("dict", keys, (specs...)) / None=leaf


def _flatten(obj) -> tuple:
    if isinstance(obj, (tuple, list)):
        leaves, specs = [], []
        for item in obj:
            sub_leaves, sub_spec = _flatten(item)
            leaves.extend(sub_leaves)
            specs.append(sub_spec)
        return leaves, ("seq", type(obj), tuple(specs))
    if isinstance(obj, dict):
        leaves, specs, keys = [], [], []
        for key, item in obj.items():
            sub_leaves, sub_spec = _flatten(item)
            leaves.extend(sub_leaves)
            specs.append(sub_spec)
            keys.append(key)
        return leaves, ("dict", tuple(keys), tuple(specs))
    return [obj], None


def _unflatten(leaves, index, spec):
    if spec is None:
        return leaves[index[0]], index[0] + 1
    kind = spec[0]
    if kind == "dict":
        out = {}
        for key, sub_spec in zip(spec[1], spec[2]):
            out[key], index[0] = _unflatten(leaves, index, sub_spec)
        return out, index[0]
    items = []
    for sub_spec in spec[2]:
        item, index[0] = _unflatten(leaves, index, sub_spec)
        items.append(item)
    return (spec[1](items)), index[0]


def make_graphed_callables(
    callables,
    sample_args,
    num_warmup_iters=3,
    allow_unused_input=False,
    pool=None,
    capture_error_mode="global",
):
    r"""Callables that run per-iteration with CUDA graph capture.

    forward (and backward, via :func:`tensorplay.autograd.grad`) into CUDA
    graphs sharing one private memory pool, then wraps them in autograd
    Functions whose forward/backward are graph replays.  Per-iteration host
    overhead drops to two graph launches.

    carried over verbatim: ``sample_args`` must contain only Tensors whose
    ``requires_grad`` matches the live workload; modules may not carry hooks
    or trainable buffers; arguments must keep their order and shapes.

    Args:
        callables: function or ``tensorplay.nn.Module``, or a tuple of them
            in live-workload order.
        sample_args: matching tuple of argument-tuples of CUDA Tensors.
        num_warmup_iters: warmup iterations run on the capture stream before
            capturing (flushes lazy cuDNN/cuBLAS state).
        allow_unused_input: passed through to :func:`tensorplay.autograd.grad`.
        pool: share an existing graph pool instead of allocating one.
    """

    import tensorplay as tp
    from ..autograd import grad as _grad

    if tp.is_autocast_enabled() and tp.is_autocast_cache_enabled():
        raise RuntimeError(
            "make_graphed_callables does not support the autocast caching. "
            "Please set cache_enabled=False."
        )

    just_one_callable = False
    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        _sample_args = (sample_args,)
    else:
        _sample_args = tuple(sample_args)

    flatten_sample_args = []
    for c, args in zip(callables, _sample_args):
        if hasattr(c, "_backward_hooks") or isinstance(c, tp.nn.Module):
            hooks = (
                len(getattr(c, "_backward_hooks", ()) or ())
                + len(getattr(c, "_forward_hooks", ()) or ())
                + len(getattr(c, "_forward_pre_hooks", ()) or ())
            )
            if hooks:
                raise AssertionError(
                    "Modules must not have hooks registered at the time they "
                    "are passed to make_graphed_callables."
                )
            if any(b.requires_grad for b in c.buffers()):
                raise AssertionError(
                    "In any nn.Module passed to make_graphed_callables, only "
                    "parameters may be trainable; all buffers must have "
                    "requires_grad=False."
                )
        flat, _ = _flatten(args)
        flatten_sample_args.append(tuple(flat))
        for arg in flat:
            if not isinstance(arg, tp.Tensor):
                raise AssertionError(
                    "In this API, sample_args for each callable must contain "
                    "only Tensors; got " + type(arg).__name__
                )

    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    per_callable_module_params = [
        tuple(c.parameters()) if isinstance(c, tp.nn.Module) else ()
        for c in callables
    ]
    per_callable_static_input_surfaces = [
        flatten_sample_args[i] + per_callable_module_params[i]
        for i in range(len(callables))
    ]

    fwd_graphs = [CUDAGraph() for _ in callables]
    bwd_graphs = [CUDAGraph() for _ in callables]
    mempool = graph_pool_handle() if pool is None else _resolve_pool(pool)

    from .streams import Stream
    from . import stream as _stream_ctx

    tp.cuda.synchronize()
    stream = Stream()
    with _stream_ctx(stream):
        # Warmup: flushes lazy-init work so it cannot land mid-capture.
        for func, args, surface in zip(
            callables, _sample_args, per_callable_static_input_surfaces
        ):
            grad_inputs = None
            for _ in range(num_warmup_iters):
                leaves, _spec = _flatten(func(*args))
                outputs_grad = tuple(o for o in leaves if o.requires_grad)
                if len(outputs_grad) > 0:
                    grad_inputs = _grad(
                        outputs=outputs_grad,
                        inputs=tuple(i for i in surface if i.requires_grad),
                        grad_outputs=tuple(
                            tp.empty_like(o) for o in leaves if o.requires_grad
                        ),
                        allow_unused=allow_unused_input,
                    )
            del grad_inputs

    tp.cuda.synchronize()

    # Captures share one mempool, so capture in live order:

    per_callable_static_outputs = []
    per_callable_output_unflatten_spec = []
    for func, args, fwd_graph in zip(callables, _sample_args, fwd_graphs):
        with graph(
            fwd_graph,
            pool=mempool,
            stream=stream,
            capture_error_mode=capture_error_mode,
        ):
            func_outputs = func(*args)
        leaves, spec = _flatten(func_outputs)
        per_callable_static_outputs.append(tuple(leaves))
        per_callable_output_unflatten_spec.append(spec)

    per_callable_static_grad_outputs = []
    per_callable_static_grad_inputs = []
    for surface, static_outputs, bwd_graph in zip(
        reversed(per_callable_static_input_surfaces),
        reversed(per_callable_static_outputs),
        reversed(bwd_graphs),
    ):
        static_grad_outputs = tuple(
            tp.empty_like(o) if o.requires_grad else None for o in static_outputs
        )
        outputs_grad = tuple(o for o in static_outputs if o.requires_grad)
        grad_inputs = None
        if len(outputs_grad) > 0:
            with graph(
                bwd_graph,
                pool=mempool,
                stream=stream,
                capture_error_mode=capture_error_mode,
            ):
                grad_inputs = _grad(
                    outputs=outputs_grad,
                    inputs=tuple(i for i in surface if i.requires_grad),
                    grad_outputs=tuple(
                        o for o in static_grad_outputs if o is not None
                    ),
                    allow_unused=allow_unused_input,
                )

        static_grad_inputs = []
        grad_idx = 0
        for arg in surface:
            if arg.requires_grad and grad_inputs is not None:
                static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)
        per_callable_static_grad_outputs.append(tuple(static_grad_outputs))
        per_callable_static_grad_inputs.append(tuple(static_grad_inputs))

    per_callable_static_grad_outputs.reverse()
    per_callable_static_grad_inputs.reverse()

    ret = []
    for i, func in enumerate(callables):
        graphed = make_graphed_autograd_function(
            fwd_graphs[i],
            bwd_graphs[i],
            per_callable_module_params[i],
            per_callable_len_user_args[i],
            per_callable_output_unflatten_spec[i],
            per_callable_static_input_surfaces[i],
            per_callable_static_outputs[i],
            per_callable_static_grad_outputs[i],
            per_callable_static_grad_inputs[i],
        )
        if isinstance(func, tp.nn.Module):
            original_forward = func.forward
            graph_training_state = func.training

            def new_fwd(*user_args, _func=func, _graphed=graphed,
                        _training=graph_training_state, _orig=original_forward):
                if _func.training == _training:
                    return _graphed(*user_args)
                return _orig(*user_args)

            func.forward = new_fwd
            ret.append(func)
        else:
            ret.append(graphed)

    return ret[0] if just_one_callable else tuple(ret)


def make_graphed_autograd_function(
    fwd_graph,
    bwd_graph,
    module_params,
    len_user_args,
    output_unflatten_spec,
    static_input_surface,
    static_outputs,
    static_grad_outputs,
    static_grad_inputs,
):
    r"""Wrap captured forward/backward graphs in one autograd Function.

    whose forward stages fresh user args onto the captured static inputs and
    replays the forward graph; its backward (once-differentiable) stages
    incoming grads and replays the backward graph.

    The callable's full input surface is ``user_args + module_params``;
    parameters are assumed unchanged since capture.
    """

    from ..autograd.function import Function, once_differentiable

    class Graphed(Function):
        @staticmethod
        def forward(ctx, *inputs):
            for i in range(len_user_args):
                if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                    static_input_surface[i].copy_(inputs[i])
            fwd_graph.replay()
            return tuple(o.detach() for o in static_outputs)

        @staticmethod
        @once_differentiable
        def backward(ctx, *grads):
            # The engine may append trailing None placeholders beyond the
            # per-output gradients; only the leading per-output entries are
            # meaningful here.
            if len(grads) < len(static_grad_outputs):
                raise AssertionError(
                    f"len(grads)={len(grads)} < "
                    f"{len(static_grad_outputs)} static_grad_outputs"
                )
            grads = grads[: len(static_grad_outputs)]
            for g, incoming in zip(static_grad_outputs, grads):
                if g is not None:
                    if g.data_ptr() != incoming.data_ptr():
                        g.copy_(incoming)
            bwd_graph.replay()
            return tuple(b.detach() if b is not None else b
                         for b in static_grad_inputs)

    def functionalized(*user_args):
        flat_user_args, _ = _flatten(user_args)
        leaves_out = Graphed.apply(*(tuple(flat_user_args) + tuple(module_params)))
        index = [0]
        out, _ = _unflatten(leaves_out, index, output_unflatten_spec)
        return out

    return functionalized
