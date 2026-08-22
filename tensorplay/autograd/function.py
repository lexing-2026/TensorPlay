import functools
import types
import warnings

import tensorplay
import tensorplay._C._autograd as _autograd


def _iter_tensors(data):
    """Yields all Tensors contained in a nested structure."""
    if isinstance(data, tensorplay.Tensor):
        yield data
    elif isinstance(data, (list, tuple)):
        for d in data:
            yield from _iter_tensors(d)
    elif isinstance(data, dict):
        for d in data.values():
            yield from _iter_tensors(d)


class _Context:
    """
    Records information needed for computing gradients.
    """

    def __init__(self):
        self._saved_tensors = ()
        self._to_save_for_forward = ()
        self.needs_input_grad = []
        self.materialize_grads = True
        self.dirty_tensors = set()
        self._non_differentiable = set()
        self._output_grad_metas = []
        self.backward_fn = None

    def save_for_backward(self, *tensors):
        r"""Saves given tensors to be accessed via ``ctx.saved_tensors`` in backward."""
        for t in tensors:
            if not isinstance(t, tensorplay.Tensor):
                raise TypeError("save_for_backward only accepts Tensors")
        self._saved_tensors = tensors
        # Record versions for in-place-modification detection (torch parity)
        self._saved_versions = tuple(t._version for t in tensors)

    @property
    def saved_tensors(self):
        r"""Returns saved tensors.

        Raises if any saved tensor was modified in-place since saving,
        mirroring :class:`torch.autograd.Function` version-counter checks.
        """
        versions = getattr(self, "_saved_versions", ())
        for t, v in zip(self._saved_tensors, versions):
            if t._version != v:
                raise RuntimeError(
                    "one of the variables needed for gradient computation has "
                    "been modified by an inplace operation: "
                    f"[Tensor (version {t._version})] is at version "
                    f"{t._version}; expected version {v} instead."
                )
        return tuple(self._saved_tensors)

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
        r"""Marks given tensors as modified in an in-place operation."""
        for arg in args:
            if not isinstance(arg, tensorplay.Tensor):
                raise RuntimeError("mark_dirty only accepts Tensor arguments")
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


class Function:
    r"""Records operation history and defines formulas for differentiating ops.

    Supports two styles, mirroring torch.autograd.Function:

    1. Legacy style: ``forward(ctx, ...)`` / ``backward(ctx, ...)``
       (forward receives a context object).
    2. Combined-forward style: define ``forward(*args, **kwargs)``,
       ``setup_context(ctx, inputs, output)`` and use
       ``save_for_backward``/``save_for_forward`` inside ``setup_context``
       instead of receiving a ``ctx`` argument in ``forward``.
    """

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

        Not yet supported by this engine; provided for API parity.
        """
        raise NotImplementedError(
            "You must implement the jvp method for your custom autograd "
            "Function to use it with forward-mode AD. Forward-mode AD is not "
            "supported by this engine yet."
        )

    @staticmethod
    def vmap(info, in_dims, *args):
        r"""Defines a formula for vectorizing the operation.

        Not yet supported by this engine; provided for API parity.
        """
        raise NotImplementedError(
            "You must implement the vmap method for your custom autograd "
            "Function to use it with functorch-style vmap. vmap is not "
            "supported by this engine yet."
        )

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "legacy autograd function with non-static forward method is deprecated. "
            "Please use new-style autograd function with static forward method. "
            "(Example: https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd)"
        )

    @classmethod
    def apply(cls, *args, **kwargs):
        r"""Runs the operation and attaches gradient bookkeeping to outputs."""
        uses_setup_context = cls.setup_context is not Function.setup_context
        grad_enabled = _autograd.is_grad_enabled()

        flat_args = list(_iter_tensors(args))
        needs_grad = False
        if grad_enabled:
            needs_grad = any(a.requires_grad for a in flat_args)
        else:
            for a in flat_args:
                if a.requires_grad:
                    warnings.warn(
                        "An output of the user-provided Function seems to not "
                        "require grad while at least one input requires grad. "
                        "The autograd engine will not track this op.",
                        stacklevel=2,
                    )
                    break

        def compute_needs(args_):
            out = []
            for a in args_:
                if isinstance(a, tensorplay.Tensor):
                    out.append(bool(a.requires_grad))
                elif isinstance(a, dict):
                    out.extend(compute_needs(tuple(a.values())))
                elif isinstance(a, (list, tuple)):
                    out.extend(compute_needs(tuple(a)))
                else:
                    out.append(False)
            return out

        ctx = _Context()
        ctx.needs_input_grad = tuple(compute_needs(args))

        # Run forward with grad disabled (like the autograd engine does)
        _autograd.set_grad_enabled(False)
        try:
            if uses_setup_context:
                output = cls.forward(*args, **kwargs)
            else:
                output = cls.forward(ctx, *args, **kwargs)
        finally:
            _autograd.set_grad_enabled(grad_enabled)

        if uses_setup_context:
            cls.setup_context(ctx, args, output)

        if not grad_enabled or not needs_grad:
            return output

        # Create PyNode and connect the graph
        ctx.backward_fn = cls.backward

        def backward_wrapper(self, *grads):
            if getattr(self, "materialize_grads", True):
                metas = self._output_grad_metas
                materialized = []
                for i, g in enumerate(grads):
                    if g is None and i < len(metas):
                        shape, dtype, device = metas[i]
                        materialized.append(
                            tensorplay.zeros(shape, dtype=dtype, device=device)
                        )
                    else:
                        materialized.append(g)
                return self.backward_fn(self, *materialized)
            return self.backward_fn(self, *grads)

        ctx.backward = types.MethodType(backward_wrapper, ctx)

        fn = _autograd.PyNode(ctx)

        # Connect inputs
        for arg in args:
            if isinstance(arg, tensorplay.Tensor):
                if arg.requires_grad:
                    edges = _autograd.collect_next_edges(arg)
                    if edges:
                        for edge in edges:
                            # edge is (Node, int) pair
                            fn.add_next_edge(edge[0], edge[1])
                    else:
                        fn.add_next_edge(None)
                else:
                    fn.add_next_edge(None)
            else:
                # Non-tensor arg
                fn.add_next_edge(None)

        # Set grad_fn for outputs
        def attach(out, idx):
            if id(out) in ctx._non_differentiable:
                return
            out.requires_grad = True
            out._set_grad_fn(fn, idx)
            ctx._output_grad_metas.append(
                (tuple(out.shape), out.dtype, out.device)
            )

        if isinstance(output, tensorplay.Tensor):
            attach(output, 0)
        elif isinstance(output, (list, tuple)):
            for i, out in enumerate(output):
                if isinstance(out, tensorplay.Tensor):
                    attach(out, i)

        return output
