import functools
import warnings

import tensorplay
import tensorplay._C._autograd as _autograd


def _current_saved_hooks_pair():
    """Active (pack, unpack) pair from an enclosing saved_tensors_hooks
    context, or None.  Late import avoids a graph<->function cycle."""
    from .graph import _hook_stack

    return _hook_stack[-1] if _hook_stack else None


# build without them is loaded, the generic Python fallbacks run instead.
_FAST_GRAPH = hasattr(_autograd, "setup_custom_function_graph")
_FAST_ATTACH = hasattr(_autograd, "PyNode") and hasattr(
    getattr(_autograd, "PyNode", None), "attach_outputs"
)
_PyNode = _autograd.PyNode
_setup_graph = getattr(_autograd, "setup_custom_function_graph", None)
_APPLY_ALL = getattr(_autograd, "custom_function_apply", None)
_RUN_FWD = getattr(_autograd, "run_custom_function_forward", None)
_NODE_FACTORY = (lambda c: _PyNode(c)) if _APPLY_ALL is not None else None


def _fast_capable():
    return _FAST_GRAPH and _RUN_FWD is not None and _FAST_ATTACH


def _collect_edges(t):
    return _autograd.collect_next_edges(t)


def _materialize(ctx, grads):
    """Zero-fill missing output gradients using lazily captured outputs."""
    outputs = ctx._outputs
    out = []
    metas = ctx._output_grad_metas
    for i, g in enumerate(grads):
        if g is not None:
            out.append(g)
            continue
        if i < len(metas):
            shape, dtype, device = metas[i]
            out.append(tensorplay.zeros(shape, dtype=dtype, device=device))
        elif i < len(outputs) and outputs[i] is not None:
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
    """
    prehook(grads_tuple) -> replacement; hook(grad_inputs, grad_outputs)
    -> replacement grad_inputs.

    When the engine materializes missing gradients itself (Node::
    zero-fill branch is compiled out entirely.
    """
    hooks = ctx._hooks
    prehooks = ctx._prehooks
    engine_materializes = getattr(ctx, "_engine_materializes", False)
    materialize_default = getattr(ctx, "materialize_grads", True) and not engine_materializes
    backward_fn = ctx.backward_fn
    n_in = len(ctx.needs_input_grad)
    # CustomFunctionNode semantics); unused outputs arrive absent/None.
    n_out = max(len(getattr(ctx, "_outputs", ())), n_in)

    def backward(*grads):
        # Complete missing trailing slots with None before anything else.
        if len(grads) < n_out:
            grads = grads + (None,) * (n_out - len(grads))
        for ph in prehooks:
            replaced = ph((grads,))
            if replaced is not None:
                grads = tuple(replaced[0])
        if materialize_default and any(g is None for g in grads):
            grads = tuple(_materialize(ctx, grads))
        results = backward_fn(ctx, *grads)
        if not isinstance(results, tuple):
            results = (results,)
        n = len(results)
        if n != n_in and not (n > n_in and all(r is None for r in results[n_in:])):
            raise RuntimeError(
                f"function {cls.name} returned an incorrect number of "
                f"gradients (expected {n_in}, got {n})")
        if n > n_in:
            results = results[:n_in]
        for hk in hooks:
            replaced = hk(results, grads)
            if replaced is not None:
                results = tuple(replaced)
        return results

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

    def __init__(self):
        self._saved_tensors = ()
        self._to_save_for_forward = ()
        self.materialize_grads = True
        self.dirty_tensors = set()
        self._non_differentiable = set()
        # Outputs captured lazily for gradient materialization; metas are
        # only computed if a None grad actually arrives in backward.
        self._outputs: tuple = ()
        self._output_grad_metas: list = []
        self.backward_fn = None
        self._metadata = None
        self.requires_grad = False
        self.next_functions: tuple = ()
        # Kept as real lists: the C++ PyNode register_hook bindings append
        # into them directly.
        self._hooks: list = []
        self._prehooks: list = []

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = {}
        return self._metadata

    @property
    def non_differentiable(self):
        return self._non_differentiable

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
        return replacement ``grad_outputs``."""
        self._prehooks.append(hook)

    def save_for_backward(self, *tensors):
        r"""Saves given tensors to be accessed via ``ctx.saved_tensors`` in backward.

        When a ``saved_tensors_hooks`` context is active, each tensor is
        passed through the pack hook at save time (and through the unpack
        """
        for t in tensors:
            if t is not None and not isinstance(t, tensorplay.Tensor):
                raise TypeError(
                    "save_for_backward only accepts Tensors or None")
        pair = _current_saved_hooks_pair()
        if pair is not None:
            pack_fn, unpack_fn = pair
            self._saved_pack = tuple(
                None if t is None else pack_fn(t) for t in tensors)
            self._saved_unpack = unpack_fn
            self._saved_versions = tuple(
                None if t is None else t._version for t in tensors)
        else:
            self._saved_pack = None
            self._saved_unpack = None
            self._saved_versions = tuple(
                None if t is None else t._version for t in tensors)
        self._saved_tensors = tensors

    @property
    def saved_tensors(self):
        r"""Returns saved tensors.

        Raises if any saved tensor was modified in-place since saving,
        """
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
            if arg.is_leaf:
                raise RuntimeError(
                    "Only non-leaf tensors can be marked as non-differentiable"
                )
            self._non_differentiable.add(id(arg))

    def mark_shared_storage(self, *args):
        r"""Deprecated: has no effect in this engine."""
        warnings.warn(
            "ctx.mark_shared_storage is deprecated and has no effect",
            stacklevel=2,
        )


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

    @property
    def name(cls):
        return f"{cls.__name__}Backward"


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

        Not yet supported by this engine; provided for API compatibility.
        """
        raise NotImplementedError(
            "You must implement the jvp method for your custom autograd "
            "Function to use it with forward-mode AD. Forward-mode AD is not "
            "supported by this engine yet."
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

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "legacy autograd function with non-static forward method is deprecated. "
            "Please use new-style autograd function with static forward method. "
        )

    @classmethod
    def apply(cls, *args, **kwargs):
        r"""Runs the operation and attaches gradient bookkeeping to outputs.

        flat arguments computes ``needs_input_grad`` and wires next-edges
        BEFORE forward; outputs are marked and attached AFTER
        ``setup_context``.  When the fused C++ helpers are present the hot
        path makes two pybind crossings total (graph setup + output
        attach); otherwise a generic Python fallback runs.
        """
        uses_setup_context = cls.setup_context is not Function.setup_context
        grad_enabled = _autograd.is_grad_enabled()

        flat = not any(
            isinstance(a, (list, tuple, dict)) for a in args)

        # ---- C++ boundary: ONE crossing ----
        if _APPLY_ALL is not None and flat and not kwargs and grad_enabled:
            output, ctx, needs, executable, fn = _APPLY_ALL(
                _Context,
                _NODE_FACTORY,
                cls.forward,
                cls.setup_context if uses_setup_context else None,
                args,
            )
            needs = tuple(needs)
            ctx.needs_input_grad = needs
            if executable:
                ctx.backward_fn = cls.backward
                if not bool(ctx.materialize_grads):
                    fn.set_materialize_grads(False)
                    ctx._engine_materializes = False
                else:
                    ctx._engine_materializes = True
                ctx.backward = _make_backward(ctx, cls)
                return output
            return output

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
            return output

        # choice (possibly set inside setup_context) to the ENGINE, so
        # zero-filling of missing gradient slots happens in C++.
        if fn is not None and hasattr(fn, "set_materialize_grads"):
            fn.set_materialize_grads(bool(ctx.materialize_grads))
            ctx._engine_materializes = bool(ctx.materialize_grads)

        # ---- _wrap_outputs path: mark + attach in one pass ----
        if isinstance(output, tuple):
            ctx._outputs = output
        elif isinstance(output, list):
            ctx._outputs = tuple(output)
        else:
            ctx._outputs = (output,)
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

        ctx.backward = _make_backward(ctx, cls)
        return output


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
