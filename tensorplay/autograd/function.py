import functools
import warnings
import weakref

import tensorplay
import tensorplay._C._autograd as _autograd


_BACKWARD_TWICE_MESSAGE = (
    "Trying to backward through the graph a second time (or directly access "
    "saved tensors after they have already been freed). Saved intermediate "
    "values of the graph are freed when you call .backward() or "
    "autograd.grad(). Specify retain_graph=True if you need to backward "
    "through the graph a second time or if you need to access saved tensors "
    "after calling backward."
)


def _current_saved_hooks_pair():
    """Active (pack, unpack) pair from an enclosing saved_tensors_hooks
    context, or None.  Late import avoids a graph<->function cycle."""
    from .graph import _hook_stack

    return _hook_stack[-1] if _hook_stack else None


def _native_saved_hooks_active() -> bool:
    return bool(getattr(
        _autograd, "_saved_variable_hooks_active", lambda: False)())


def _native_pack_saved_tensor(tensor):
    return _autograd._pack_saved_tensor(tensor)


def _native_unpack_saved_tensor(token):
    return _autograd._unpack_saved_tensor(token)


# build without them is loaded, the generic Python fallbacks run instead.
_FAST_GRAPH = hasattr(_autograd, "setup_custom_function_graph")
_FAST_ATTACH = hasattr(_autograd, "PyNode") and hasattr(
    getattr(_autograd, "PyNode", None), "attach_outputs"
)
_PyNode = _autograd.PyNode
_setup_graph = getattr(_autograd, "setup_custom_function_graph", None)
_APPLY_ALL = getattr(_autograd, "custom_function_apply", None)
_RUN_FWD = getattr(_autograd, "run_custom_function_forward", None)


def _fast_capable():
    return _FAST_GRAPH and _RUN_FWD is not None and _FAST_ATTACH


def _collect_edges(t):
    return _autograd.collect_next_edges(t)


def _materialize(ctx, grads):
    """Zero-fill missing output gradients using lazily captured outputs."""
    outputs = getattr(ctx, "_outputs", None)
    out = []
    metas = ctx._output_grad_metas
    for i, g in enumerate(grads):
        if g is not None:
            out.append(g)
            continue
        if i < len(metas):
            shape, dtype, device = metas[i]
            out.append(tensorplay.zeros(shape, dtype=dtype, device=device))
        elif outputs is not None and i < len(outputs) and outputs[i] is not None:
            o = outputs[i]
            meta = (tuple(o.shape), o.dtype, o.device)
            while len(metas) <= i:
                metas.append(None)
            metas[i] = meta
            out.append(tensorplay.zeros(shape=meta[0], dtype=meta[1],
                                        device=meta[2]))
        else:
            out.append(None)
    return out


def _make_backward(ctx, cls):
    """Resolve the generic backward entry for ``ctx``.

    All per-apply state (hooks, materialization flags, gradient arity)
    lives on the context, so the entry reads it at backward time instead
    of the forward pass building a closure per call.  Returns a bound
    method: the engine invokes ``ctx.backward(*grads)``.
    """
    return ctx._backward_entry


def _make_direct_backward(ctx, cls):
    """Bind a fixed custom backward without retaining its context cycle."""

    context_ref = weakref.ref(ctx)

    def backward(*grads):
        context = context_ref()
        if context is None:
            raise RuntimeError("autograd context was released before backward")
        return cls.backward(context, *grads)

    return backward


def _collect_needs(data, out: list) -> None:
    """Append ``requires_grad`` per input position, recursing into
    nested structures (single-pass variant of the old flat walk)."""
    if isinstance(data, tensorplay.Tensor):
        out.append(bool(data.requires_grad))
    elif isinstance(data, dict):
        for value in data.values():
            _collect_needs(value, out)
    elif isinstance(data, (list, tuple)):
        for item in data:
            _collect_needs(item, out)
    else:
        out.append(False)


class _Context:
    """
    Records information needed for computing gradients.
    """

    # Immutable per-apply defaults live on the class so the constructor
    # only allocates the mutable holders (hooks, gradient metas, sets);
    # instance assignment shadows these on first write.
    materialize_grads = True
    backward_fn = None
    _metadata = None
    requires_grad = False
    next_functions: tuple = ()
    _saved_tensors: tuple = ()
    _saved_anchors: tuple = ()
    _saved_released: bool = False
    _to_save_for_backward: tuple = ()
    _outputs: tuple = ()

    def __init__(self):
        self.dirty_tensors = set()
        self._non_differentiable = set()
        # Outputs captured lazily for gradient materialization; metas are
        # only computed if a None grad actually arrives in backward.
        self._output_grad_metas: list = []
        # Kept as real lists: the C++ PyNode register_hook bindings append
        # into them directly.
        self._hooks: list = []
        self._prehooks: list = []

    def __del__(self):
        # Adoption, not teardown: the saved tensors stay untouched because a
        # backward pass that has not run yet still reads them.  While saved
        # tensors exist they carry this context back (``_tp_saved_ctx``), so
        # reaching the finalizer means either the node is already gone or
        # the node is owned from elsewhere without holding any of this
        # context's tensors -- in both cases adopting (taking the node's
        # keep-alive) is safe and cannot close a cycle the collector cannot
        # see.
        node_id = getattr(self, "_node_id", None)
        if node_id is None:
            return
        try:
            _autograd._adopt_node_if_needed(node_id, self)
        except Exception:
            pass

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = {}
        return self._metadata

    @property
    def non_differentiable(self):
        return self._non_differentiable

    def release_saved(self) -> None:
        """Let go of the tensors saved for backward.

        A saved tensor is held for the backward pass that reads it, and that
        pass is its last reader: once it has run, the value is the graph's
        memory for nothing.  The engine knows when a graph is being kept for
        another pass, so this is one call at the point where the engine stops
        needing them, and a function whose backward may run again does not
        make it.  The back-references each saved tensor carries to this
        context (see :meth:`save_for_backward`) go with them.
        """
        anchors = getattr(self, "_saved_anchors", None)
        if anchors:
            for t in anchors:
                if getattr(t, "_tp_saved_ctx", None) is self:
                    try:
                        del t._tp_saved_ctx
                    except AttributeError:
                        pass
        self._saved_anchors = ()
        self._saved_tensors = ()
        self._saved_released = True
        for name in ("_saved_versions", "_saved_native_tokens", "_saved_pack"):
            if isinstance(getattr(self, name, None), tuple):
                setattr(self, name, ())

    def maybe_clear_saved_tensors(self) -> None:
        """Release saved values when this backward does not retain its graph."""
        if not _autograd._get_current_graph_task_keep_graph():
            self.release_saved()

    @property
    def to_save(self):
        return self._saved_tensors

    @to_save.setter
    def to_save(self, tensors):
        if not isinstance(tensors, (tuple, list)):
            raise TypeError(
                "to_save attribute is expected to be a tuple but is "
                f"{type(tensors)}")
        self.save_for_backward(*tensors)

    def register_hook(self, hook):
        """
        ``(grad_inputs, grad_outputs)`` after :meth:`Function.backward`;
        may return a replacement for ``grad_inputs``."""
        self._hooks.append(hook)

    def register_prehook(self, hook):
        """
        ``(grad_outputs,)`` before :meth:`Function.backward` runs; may
        return replacement for ``grad_outputs``."""
        self._prehooks.append(hook)

    def _backward_entry(self, *grads):
        """Engine-invoked backward: complete/mask the gradient tuple, run
        prehooks, materialize missing grads, call the user formula, then
        posthooks.  Every piece of state is read off this context so the
        forward pass binds this method instead of building a closure."""
        # Complete missing trailing slots with None before anything else.
        # Output count is provided by the C++ fast path (which no longer
        # stores the output tensors on ctx); fall back to the stored tuple
        # for the slow path / materialize_grads=False case.
        n_out = getattr(self, "_n_outputs", None)
        if n_out is None:
            n_out = len(self._outputs)
        if n_out == 0:
            n_out = 1
        if len(grads) > n_out:
            grads = grads[:n_out]
        if len(grads) < n_out:
            grads = grads + (None,) * (n_out - len(grads))
        for ph in self._prehooks:
            replaced = ph((grads,))
            if replaced is not None:
                grads = tuple(replaced[0])
        engine_materializes = getattr(self, "_engine_materializes", False)
        if self.materialize_grads and not engine_materializes \
                and any(g is None for g in grads):
            grads = tuple(_materialize(self, grads))
        results = self.backward_fn(self, *grads)
        if not isinstance(results, tuple):
            results = (results,)
        n = len(results)
        n_in = len(self.needs_input_grad)
        if n != n_in and not (n > n_in and all(r is None for r in results[n_in:])):
            raise RuntimeError(
                f"function {getattr(self, '_node_name', 'Function')} returned "
                f"an incorrect number of gradients (expected {n_in}, got {n})")
        if n > n_in:
            results = results[:n_in]
        for hk in self._hooks:
            replaced = hk(results, grads)
            if replaced is not None:
                results = tuple(replaced)
        return results

    def save_for_backward(self, *tensors):
        r"""Saves given tensors to be accessed via ``ctx.saved_tensors`` in backward.

        When a ``saved_tensors_hooks`` context is active, each tensor is
        passed through the pack hook at save time (and through the unpack
        """
        for t in tensors:
            if t is not None and not isinstance(t, tensorplay.Tensor):
                raise TypeError(
                    "save_for_backward only accepts Tensors or None")
        # Each saved tensor carries this context back: the backward pass may
        # still have to run after the caller drops the context, and a tensor
        # that outlives it (a kept output, a live parameter) must keep the
        # context -- with the backward entry and the saved state -- reachable.
        # The reference is a plain instance attribute, so a context whose
        # saved tensors all became unreachable collects together with them
        # instead of pinning a dead graph.
        anchors = tuple(t for t in tensors if t is not None)
        for t in anchors:
            t._tp_saved_ctx = self
        self._saved_anchors = anchors
        if _native_saved_hooks_active():
            self._saved_native_tokens = tuple(
                None if t is None else _native_pack_saved_tensor(t)
                for t in tensors
            )
            self._saved_pack = None
            self._saved_unpack = None
            self._saved_tensors = tuple(None for _ in tensors)
        else:
            self._saved_native_tokens = None
            pair = _current_saved_hooks_pair()
            if pair is not None:
                pack_fn, unpack_fn = pair
                self._saved_pack = tuple(
                    None if t is None else pack_fn(t) for t in tensors)
                self._saved_unpack = unpack_fn
            else:
                self._saved_pack = None
                self._saved_unpack = None
            self._saved_tensors = tensors
        if not _native_saved_hooks_active():
            self._saved_versions = tuple(
                None if t is None else t._version for t in tensors)
        else:
            self._saved_versions = tuple(None for _ in tensors)

    @property
    def saved_tensors(self):
        r"""Returns saved tensors.

        Raises if any saved tensor was modified in-place since saving,
        """
        if self._saved_released:
            raise RuntimeError(_BACKWARD_TWICE_MESSAGE)
        native_tokens = getattr(self, "_saved_native_tokens", None)
        if native_tokens is not None:
            return tuple(
                None if token is None else _native_unpack_saved_tensor(token)
                for token in native_tokens
            )
        tensors = self._saved_tensors
        versions = getattr(self, "_saved_versions", ())
        for t, v in zip(tensors, versions):
            if t is None or v is None:
                continue
            if t._version != v:
                raise RuntimeError(
                    "one of the variables needed for gradient computation has "
                    "been modified by an inplace operation: "
                    f"[Tensor (version {t._version})] is at version "
                    f"{t._version}; expected version {v} instead."
                )
        unpack_fn = getattr(self, "_saved_unpack", None)
        packed = getattr(self, "_saved_pack", None)
        if unpack_fn is not None and packed is not None:
            return tuple(None if p is None else unpack_fn(p) for p in packed)
        return tuple(tensors)

    def save_for_forward(self, *tensors):
        r"""Saves given tensors for use in the ``vjp`` computation."""
        self._to_save_for_forward = tensors

    @property
    def saved_for_forward(self):
        r"""Returns tensors saved via :meth:`save_for_forward`."""
        return tuple(self._to_save_for_forward)

    def set_materialize_grads(self, value: bool):
        r"""Sets whether None output gradients are materialized into zero tensors."""
        self.materialize_grads = value

    def mark_dirty(self, *args):
        r"""Marks given tensors as modified in an in-place operation.

        immediately (``_mark_dirty`` in python_function.cpp), so later
        ``saved_tensors`` access and double-backward detect the mutation.
        """
        for arg in args:
            if not isinstance(arg, tensorplay.Tensor):
                raise RuntimeError("mark_dirty only accepts Tensor arguments")
            arg._bump_version()
            self.dirty_tensors.add(id(arg))

    def mark_non_differentiable(self, *args):
        r"""Marks outputs as non-differentiable."""
        for arg in args:
            if not isinstance(arg, tensorplay.Tensor):
                raise RuntimeError("mark_non_differentiable only accepts Tensors")
            self._non_differentiable.add(id(arg))

def once_differentiable(fn):
    r"""Decorator to make a custom autograd Function's backward run once,
    with gradients detached and grad-mode disabled inside."""

    @functools.wraps(fn)
    def wrapper(ctx, *grad_inputs):
        prev = _autograd.is_grad_enabled()
        _autograd.set_grad_enabled(False)
        try:
            detached = tuple(
                g.detach() if isinstance(g, tensorplay.Tensor) else g
                for g in grad_inputs
            )
            return fn(ctx, *detached)
        finally:
            _autograd.set_grad_enabled(prev)

    return wrapper


class FunctionMeta(type):
    """
    the ``name`` classproperty (``"<Cls>Backward"``, used for node naming)
    and a friendlier repr for subclasses."""

    def __new__(mcls, name, bases, namespace, **kwds):
        cls = super().__new__(mcls, name, bases, namespace, **kwds)
        cls._node_name = f"{name}Backward"
        return cls

    @property
    def name(cls):
        return cls._node_name


def _maybe_process_forward_ad(cls, ctx, args, output):
    """Forward-mode AD for custom Functions.

    When any input carries a tangent at the active forward level, the user's
    ``jvp`` computes the output tangents, which are attached to the outputs
    one-for-one.  Tangent reads stay disabled while ``jvp`` runs so the
    tangent arithmetic itself never re-enters forward propagation.
    """
    if cls.jvp is _BASE_JVP:
        return output

    grad_inputs = []
    any_fw = False

    def collect(a):
        nonlocal any_fw
        if isinstance(a, tensorplay.Tensor):
            g = tensorplay._C._fw_grad(a, 0)
            if g.defined():
                any_fw = True
                grad_inputs.append(g)
            else:
                grad_inputs.append(None)
        elif isinstance(a, (list, tuple)):
            for item in a:
                collect(item)
        elif isinstance(a, dict):
            for v in a.values():
                collect(v)
        else:
            grad_inputs.append(None)

    for a in args:
        collect(a)
    if not any_fw:
        return output

    from .forward_ad import _set_fwd_grad_enabled

    outs = output if isinstance(output, (tuple, list)) else (output,)
    with _set_fwd_grad_enabled(False):
        tangent_outs = cls.jvp(ctx, *grad_inputs)
    if not isinstance(tangent_outs, (tuple, list)):
        tangent_outs = (tangent_outs,)
    if len(tangent_outs) != len(outs):
        raise RuntimeError(
            f"jvp for {cls.__name__} returned {len(tangent_outs)} tangents "
            f"but the forward returned {len(outs)} outputs")
    for out, t in zip(outs, tangent_outs):
        if (isinstance(out, tensorplay.Tensor) and t is not None
                and out.defined() and t.defined()):
            tensorplay._C._set_fw_grad(out, t, 0, False)
    return output


class Function(metaclass=FunctionMeta):
    r"""Records operation history and defines formulas for differentiating ops.


    1. Legacy style: ``forward(ctx, ...)`` / ``backward(ctx, ...)``
       (forward receives a context object).
    2. Combined-forward style: define ``forward(*args, **kwargs)``,
       ``setup_context(ctx, inputs, output)`` and use
       ``save_for_backward``/``save_for_forward`` inside ``setup_context``
       instead of receiving a ``ctx`` argument in ``forward``.
    """

    generate_vmap_rule = False

    auto_setup_ctx = False

    @staticmethod
    def forward(ctx, *args, **kwargs):
        r"""Performs the operation.

        This function is to be overridden by all subclasses. There are two ways
        to define forward:

        Usage 1 (Combined forward and ctx)::

            @staticmethod
            def forward(ctx, input1, input2):
                ...
                return output

        Usage 2 (Separated forward and ctx)::

            @staticmethod
            def forward(input1, input2):
                ...
                return output

            @staticmethod
            def setup_context(ctx, inputs, output):
                ...
        """
        raise NotImplementedError(
            "You must implement the forward function for your custom autograd Function."
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        r"""Sets up the context object (Usage 2 above).

        Arguments:
            ctx (_Context): context object to modify in-place
            inputs (tuple): inputs to :meth:`forward`
            output (Any): output of :meth:`forward`
        """
        raise NotImplementedError(
            "You must implement the setup_context function for your custom "
            "autograd Function if you define forward without a ctx argument."
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        r"""Defines a formula for differentiating the operation."""
        raise NotImplementedError(
            "You must implement either the backward or vjp method "
            "for your custom autograd Function to use it with autograd."
        )

    @staticmethod
    def jvp(ctx, *grad_inputs):
        r"""Defines a formula for computing the jacobian-vector product.

        Called by forward-mode AD when at least one input of this Function
        carries a tangent: ``grad_inputs`` holds the tangent of each input
        (``None`` for inputs without one), and the returned tangents are
        attached to the outputs one-for-one.
        """
        raise NotImplementedError(
            "You must implement the jvp method for your custom autograd "
            "Function to use it with forward-mode AD."
        )

    @staticmethod
    def vmap(info, in_dims, *args):
        r"""Defines a formula for vectorizing the operation.

        Not yet supported by this engine; provided for API compatibility.
        """
        raise RuntimeError(
            "You tried to vmap over a custom Function that does not have "
            "vmap support. Please override and implement the vmap "
            "staticmethod or set generate_vmap_rule=True."
        )

    @classmethod
    def apply(cls, *args, **kwargs):
        r"""Runs the operation and attaches gradient bookkeeping to outputs.

        flat arguments computes ``needs_input_grad`` and wires next-edges
        BEFORE forward; outputs are marked and attached AFTER
        ``setup_context``.  When the fused C++ apply is present the hot
        path makes a single pybind crossing that also wires the backward
        entry; otherwise a generic Python fallback runs.
        """
        uses_setup_context = cls.setup_context is not _BASE_SETUP_CONTEXT
        grad_enabled = _autograd.is_grad_enabled()

        # Containers need the flattening fallback below; the fused helper
        # records one needs-bit per top-level slot only.
        flat = True
        for a in args:
            if isinstance(a, (list, tuple, dict)):
                flat = False
                break

        # ---- C++ boundary: ONE crossing ----
        if _APPLY_ALL is not None and flat and not kwargs and grad_enabled:
            output, ctx, needs, executable, fn = _APPLY_ALL(
                _Context,
                cls.forward,
                cls.setup_context if uses_setup_context else None,
                args,
                cls.backward,
                cls._node_name)
            if executable and getattr(
                    cls, "_tensorplay_direct_backward", False):
                ctx.backward = _make_direct_backward(ctx, cls)
            if cls.jvp is _BASE_JVP:
                return output
            return _maybe_process_forward_ad(cls, ctx, args, output)

        fast = (
            _FAST_GRAPH and _RUN_FWD is not None and _FAST_ATTACH
            and flat and not kwargs and grad_enabled
        )

        ctx = _Context()

        # ---- unpack_input path: needs bits + next_edges pre-forward ----
        if fast:
            fn = _PyNode(ctx)
            needs, any_rg = _setup_graph(fn, args)
            needs = tuple(needs)
        else:
            needs_l: list[bool] = []
            _collect_needs(args, needs_l)
            needs = tuple(needs_l)
            any_rg = any(needs)
            fn = _PyNode(ctx) if any_rg else None
        ctx.needs_input_grad = needs

        executable = grad_enabled and any_rg
        if not grad_enabled and any_rg:
            warnings.warn(
                "An output of the user-provided Function seems to not "
                "require grad while at least one input requires grad. "
                "The autograd engine will not track this op.",
                stacklevel=2,
            )

        if executable:
            ctx.requires_grad = True
            ctx.backward_fn = cls.backward
            ctx._node_name = f"{cls.__name__}Backward"

        # Run forward with grad disabled (engine semantics).  The fused
        # Crossing the C++ autograd boundary disables gradient recording for
        # the forward call, then restores the previous state.
        if fast:
            output = _RUN_FWD(
                ctx, cls.forward,
                cls.setup_context if uses_setup_context else None,
                args,
            )
        else:
            if grad_enabled:
                _autograd.set_grad_enabled(False)
            try:
                if uses_setup_context:
                    output = cls.forward(*args, **kwargs)
                else:
                    output = cls.forward(ctx, *args, **kwargs)
            finally:
                if grad_enabled:
                    _autograd.set_grad_enabled(True)

            if uses_setup_context:
                cls.setup_context(ctx, args, output)

        if not executable:
            return _maybe_process_forward_ad(cls, ctx, args, output)

        # choice (possibly set inside setup_context) to the ENGINE, so
        # zero-filling of missing gradient slots happens in C++.
        if fn is not None and hasattr(fn, "set_materialize_grads"):
            fn.set_materialize_grads(bool(ctx.materialize_grads))
            ctx._engine_materializes = bool(ctx.materialize_grads)

        # ---- _wrap_outputs path: mark + attach in one pass ----
        if isinstance(output, tuple):
            n_out = len(output)
        elif isinstance(output, list):
            n_out = len(output)
        else:
            n_out = 1
        ctx._n_outputs = n_out
        # The fused attach assumes edges were already wired by the fused
        # setup above; never mix fast-attach with slow wiring (or vice
        # versa) or the node reaches the engine with a wrong input arity.
        if fast:
            fn.attach_outputs(output)
        else:
            next_fns: list = []

            def connect(arg):
                if isinstance(arg, tensorplay.Tensor):
                    if arg.requires_grad:
                        edges = _collect_edges(arg)
                        if edges:
                            for e in edges:
                                fn.add_next_edge(e[0], e[1])
                                next_fns.append(e)
                        else:
                            fn.add_next_edge(None)
                            next_fns.append(None)
                    else:
                        fn.add_next_edge(None)
                        next_fns.append(None)
                elif isinstance(arg, dict):
                    for v in arg.values():
                        connect(v)
                elif isinstance(arg, (list, tuple)):
                    for item in arg:
                        connect(item)
                else:
                    fn.add_next_edge(None)
                    next_fns.append(None)

            for arg in args:
                connect(arg)
            ctx.next_functions = tuple(next_fns)

            idx = 0

            def attach_all(obj):
                nonlocal idx
                if isinstance(obj, tensorplay.Tensor):
                    if id(obj) not in ctx._non_differentiable:
                        obj.requires_grad = True
                        obj._set_grad_fn(fn, idx)
                    idx += 1
                elif isinstance(obj, (list, tuple)):
                    for o in obj:
                        attach_all(o)

            attach_all(output)

        ctx.backward = (
            _make_direct_backward(ctx, cls)
            if getattr(cls, "_tensorplay_direct_backward", False)
            else _make_backward(ctx, cls)
        )
        return _maybe_process_forward_ad(cls, ctx, args, output)


# Base-class markers for the per-call style checks in apply; module-level
# so the hot path compares against a plain global instead of walking the
# base class each time.
_BASE_SETUP_CONTEXT = Function.setup_context
_BASE_JVP = Function.jvp


class InplaceFunction(Function):
    """

    In-place operations must call ``ctx.mark_dirty`` on the mutated inputs
    inside ``forward``; this subclass exists only so historical code that
    subclasses it keeps working.
    """


class NestedIOFunction(Function):
    """

    Kept only for import compatibility; the modern contract is to define
    ``forward`` + ``backward`` on :class:`Function` directly.
    """

    def _nested_io(self, *inputs):
        raise RuntimeError("NestedIOFunction is legacy and unsupported")

    forward = _nested_io
    backward = _nested_io
