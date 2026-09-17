# mypy: allow-untyped-defs

import warnings
from typing import Any

import tensorplay as tp


__all__ = [
    "MaskedTensor",
    "is_masked_tensor",
]


def is_masked_tensor(obj: Any, /) -> bool:
    """Return True if the input is a MaskedTensor, else False.

    Args:
        obj: any input

    Examples:

        >>> # xdoctest: +SKIP
        >>> from tensorplay.masked import MaskedTensor
        >>> data = tensorplay.arange(6).reshape(2, 3)
        >>> mask = tp.tensor([[True, False, False], [True, True, False]])
        >>> mt = MaskedTensor(data, mask)
        >>> is_masked_tensor(mt)
        True
    """
    return isinstance(obj, MaskedTensor)


def _tensors_match(a, b, exact=True, rtol=1e-05, atol=1e-08):
    """Compare two plain tensors elementwise, recursing into the
    coordinate components when the layout is sparse."""
    if is_masked_tensor(a) or is_masked_tensor(b):
        raise ValueError("Neither `a` nor `b` can be a MaskedTensor.")
    if a.layout != b.layout:
        raise ValueError(
            f"`a` and `b` must have the same layout. Got {a.layout} and {b.layout}"
        )

    if a.dtype != b.dtype:
        b = b.type(a.dtype)
    if a.layout == b.layout == tp.sparse_coo:
        return _tensors_match(a.values(), b.values(), exact) and _tensors_match(
            a.indices(), b.indices(), exact
        )
    elif a.layout == b.layout == tp.sparse_csr:
        return (
            _tensors_match(a.crow_indices(), b.crow_indices(), exact)
            and _tensors_match(a.col_indices(), b.col_indices(), exact)
            and _tensors_match(a.values(), b.values(), exact)
        )
    if exact:
        return (a.dim() == b.dim()) and tp.eq(a, b).all().item()
    return (a.dim() == b.dim()) and tp.allclose(a, b, rtol=rtol, atol=atol)


def _masks_match(a, b):
    if is_masked_tensor(a) and is_masked_tensor(b):
        mask_a = a.get_mask()
        mask_b = b.get_mask()
        return _tensors_match(mask_a, mask_b, exact=True)
    return True


def _map_mt_args_kwargs(args, kwargs, map_fn):
    def _helper(a, map_fn):
        if is_masked_tensor(a):
            return map_fn(a)
        elif tp.is_tensor(a):
            return a
        elif isinstance(a, list):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            return a_impl
        elif isinstance(a, tuple):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            return tuple(a_impl)
        else:
            return a

    if kwargs is None:
        kwargs = {}
    impl_args = []
    for a in args:
        impl_args.append(_helper(a, map_fn))
    impl_kwargs = {}
    for k in kwargs:
        impl_kwargs[k] = _helper(a, map_fn)
    return impl_args, impl_kwargs


def _wrap_result(result_data, result_mask):
    if isinstance(result_data, list):
        return [_wrap_result(r, m) for (r, m) in zip(result_data, result_mask)]
    if isinstance(result_data, tuple):
        return tuple(_wrap_result(r, m) for (r, m) in zip(result_data, result_mask))
    if tp.is_tensor(result_data):
        return MaskedTensor(result_data, result_mask)
    # Expect result_data and result_mask to be Tensors only
    return NotImplemented


def _masked_tensor_str(data, mask, formatter):
    if data.layout in {tp.sparse_coo, tp.sparse_csr}:
        data = data.to_dense()
        mask = mask.to_dense()
    if data.dim() == 1:
        formatted_elements = [
            formatter.format(d.item()) if isinstance(d.item(), float) else str(d.item())
            for d in data
        ]
        max_len = max(8 if x[1] else len(x[0]) for x in zip(formatted_elements, ~mask))
        return (
            "["
            + ", ".join(
                [
                    "--".rjust(max_len) if m else e
                    for (e, m) in zip(formatted_elements, ~mask)
                ]
            )
            + "]"
        )
    sub_strings = [_masked_tensor_str(d, m, formatter) for (d, m) in zip(data, mask)]
    sub_strings = ["\n".join(["  " + si for si in s.split("\n")]) for s in sub_strings]
    return "[\n" + ",\n".join(sub_strings) + "\n]"


def _get_data(a):
    if is_masked_tensor(a):
        return a._masked_data
    return a


def _maybe_get_mask(a):
    if is_masked_tensor(a):
        return a.get_mask()
    return None


class MaskedTensor:
    """A pair of plain tensors ``data`` and ``mask`` presented as a single
    value.

    ``mask`` is boolean and has the same shape as ``data``. An element of
    ``data`` participates in computations only where the corresponding
    element of ``mask`` is True; positions where the mask is False are
    rendered as ``--`` in the string representation and are replaced by a
    caller-supplied fill value by :meth:`to_tensor`.

    The mask is always the source of truth for validity: masked-out entries
    of ``data`` may hold arbitrary values and are never read semantically.

    This class is not a tensor subclass and does not hook into a dispatcher.
    Operations are applied through explicit methods (elementwise ops,
    reductions and structural ops) or through the masking-aware functions in
    ``tensorplay.masked``, which accept both plain tensors and MaskedTensor
    inputs.
    """

    def __init__(self, data, mask, requires_grad=False):
        if is_masked_tensor(data) or not tp.is_tensor(data):
            raise TypeError("data must be a Tensor")
        if is_masked_tensor(mask) or not tp.is_tensor(mask):
            raise TypeError("mask must be a Tensor")
        warnings.warn(
            (
                "The MaskedTensor API is in prototype stage and will change "
                "in the near future. Please open an issue for feature requests "
                "and see the documentation of the tensorplay.masked module for "
                "further information about the project."
            ),
            UserWarning,
            stacklevel=2,
        )
        if data.requires_grad:
            warnings.warn(
                "It is not recommended to create a MaskedTensor with a tensor that requires_grad. "
                "To avoid this, you can use data.detach().clone()",
                UserWarning,
                stacklevel=2,
            )
        self._requires_grad = requires_grad
        self._preprocess_data(data, mask)
        self._validate_members()

    def _preprocess_data(self, data, mask):
        from .._ops import _sparse_coo_where, _sparse_csr_where

        if data.layout != mask.layout:
            raise TypeError("data and mask must have the same layout.")
        if data.layout == tp.sparse_coo:
            data = data.coalesce()
            mask = mask.coalesce()
            if data._nnz() != mask._nnz():
                data = _sparse_coo_where(mask, data, tp.tensor(0))
        elif data.layout == tp.sparse_csr:
            if data._nnz() != mask._nnz():
                data = _sparse_csr_where(mask, data, tp.tensor(0))

        # Have to pick awkward names to not conflict with existing fields such as data
        self._masked_data = data.clone()
        self._masked_mask = mask.clone()

    def _validate_members(self):
        data = self._masked_data
        mask = self.get_mask()
        if type(data) is not type(mask):
            raise TypeError(
                f"data and mask must have the same type. Got {type(data)} and {type(mask)}"
            )
        if data.layout not in {tp.strided, tp.sparse_coo, tp.sparse_csr}:
            raise TypeError(f"data layout of {data.layout} is not supported.")
        if data.layout == tp.sparse_coo:
            if not _tensors_match(data.indices(), mask.indices(), exact=True):
                raise ValueError(
                    "data and mask are both sparse COO tensors but do not have the same indices."
                )
        elif data.layout == tp.sparse_csr:
            if not _tensors_match(
                data.crow_indices(), mask.crow_indices(), exact=True
            ) or not _tensors_match(data.col_indices(), mask.col_indices(), exact=True):
                raise ValueError(
                    "data and mask are both sparse CSR tensors but do not share either crow or col indices."
                )
        if mask.dtype != tp.bool:
            raise TypeError("mask must have dtype bool.")
        if not (
            data.dtype == tp.float16
            or data.dtype == tp.float32
            or data.dtype == tp.float64
            or data.dtype == tp.bool
            or data.dtype == tp.int8
            or data.dtype == tp.int16
            or data.dtype == tp.int32
            or data.dtype == tp.int64
        ):
            raise TypeError(f"{data.dtype} is not supported in MaskedTensor.")
        if data.dim() != mask.dim():
            raise ValueError("data.dim() must equal mask.dim()")
        if data.size() != mask.size():
            raise ValueError("data.size() must equal mask.size()")

    @staticmethod
    def _from_values(data, mask):
        """Differentiable constructor for MaskedTensor"""

        class Constructor(tp.autograd.Function):
            @staticmethod
            def forward(ctx, data, mask):
                return MaskedTensor(data, mask)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None

        result = Constructor.apply(data, mask)
        return result

    def _set_data_mask(self, data, mask):
        self._masked_data = data
        self._masked_mask = mask
        self._validate_members()

    # Shape, dtype and device metadata live on the stored data tensor.
    @property
    def shape(self):
        return self._masked_data.shape

    @property
    def ndim(self):
        return self._masked_data.ndim

    def dim(self):
        return self._masked_data.dim()

    def size(self, *args):
        return self._masked_data.size(*args)

    @property
    def dtype(self):
        return self._masked_data.dtype

    @property
    def device(self):
        return self._masked_data.device

    @property
    def layout(self):
        return self._masked_data.layout

    @property
    def requires_grad(self):
        return self._masked_data.requires_grad

    def is_floating_point(self):
        return self._masked_data.is_floating_point()

    def is_complex(self):
        return self._masked_data.is_complex()

    def is_signed(self):
        return self._masked_data.is_signed()

    def numel(self):
        return self._masked_data.numel()

    def __repr__(self):  # type: ignore[override]
        formatter = "{0:8.4f}"
        if self.dim() == 0:
            scalar_data = self.get_data().item()
            data_formatted = (
                formatter.format(scalar_data)
                if isinstance(scalar_data, float)
                else str(scalar_data)
            )
            if not self.get_mask().item():
                data_formatted = "--"
            return (
                "MaskedTensor("
                + data_formatted
                + ", "
                + str(self.get_mask().item())
                + ")"
            )
        s = _masked_tensor_str(self.get_data(), self.get_mask(), formatter)
        s = "\n".join("  " + si for si in s.split("\n"))
        return "MaskedTensor(\n" + s + "\n)"

    @classmethod
    def unary(cls, fn, data, mask):
        return MaskedTensor(fn(data), mask)

    def __lt__(self, other):
        if is_masked_tensor(other):
            return MaskedTensor(self.get_data() < _get_data(other), self.get_mask())
        return MaskedTensor(self.get_data() < other, self.get_mask())

    def to_tensor(self, value):
        if self.layout in {tp.sparse_coo, tp.sparse_csr}:
            # Dense fallback: this backend does not provide a boolean
            # inversion for sparse masks.
            return self.get_data().to_dense().masked_fill(
                ~self.get_mask().to_dense(), value
            )
        return self.get_data().masked_fill(~self.get_mask(), value)

    def get_data(self):
        class GetData(tp.autograd.Function):
            @staticmethod
            def forward(ctx, self):
                return self._masked_data.detach()

            @staticmethod
            def backward(ctx, grad_output):
                if is_masked_tensor(grad_output):
                    return grad_output
                return MaskedTensor(grad_output, self.get_mask())

        return GetData.apply(self)

    def get_mask(self):
        return self._masked_mask

    @classmethod
    def _apply_functional(cls, fn, *args, **kwargs):
        """Apply a masking-aware implementation of ``fn`` to the arguments.

        This is the explicit entry point that stands in for interpreter-level
        dispatch on tensor subclasses: the callable ``fn`` (a module function
        or a tensor method of this package) is looked up in the operation
        tables of ``_ops_refs`` and executed with MaskedTensor semantics.
        The arguments are passed in the same order the operation expects,
        which for most operations puts the MaskedTensor first.
        """
        from ._ops_refs import _MASKEDTENSOR_DISPATCH_TABLE, _MASKEDTENSOR_FUNCTION_TABLE

        if fn in _MASKEDTENSOR_FUNCTION_TABLE:
            return _MASKEDTENSOR_FUNCTION_TABLE[fn](*args, **kwargs)
        if fn in _MASKEDTENSOR_DISPATCH_TABLE:
            return _MASKEDTENSOR_DISPATCH_TABLE[fn](*args, **kwargs)
        raise TypeError(
            f"{fn!r} is not implemented for MaskedTensor. If you would like "
            "this operation to be supported, please propose its semantics "
            "together with a minimal reproducible snippet."
        )

    # Operator dunders route through the explicit operation tables. They are
    # lazily imported to avoid a circular import at module load time.

    def __add__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.add, self, other)

    def __radd__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.add, other, self)

    def __sub__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.sub, self, other)

    def __rsub__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.sub, other, self)

    def __mul__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.mul, self, other)

    def __rmul__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.mul, other, self)

    def __truediv__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.div, self, other)

    def __rtruediv__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.div, other, self)

    def __neg__(self):
        from .unary import _apply_native_unary

        return _apply_native_unary(tp.neg, self)

    def __abs__(self):
        from .unary import _apply_native_unary

        return _apply_native_unary(tp.abs, self)

    def __eq__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.eq, self, other)

    def __ne__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.ne, self, other)

    def __le__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.le, self, other)

    def __ge__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.ge, self, other)

    def __gt__(self, other):
        from .binary import _apply_native_binary

        return _apply_native_binary(tp.gt, self, other)

    # Identity hashing keeps MaskedTensor usable in sets and dicts despite
    # defining elementwise __eq__.
    def __hash__(self):
        return id(self)

    def is_sparse_coo(self):
        return self.layout == tp.sparse_coo

    def is_sparse_csr(self):  # type: ignore[override]
        return self.layout == tp.sparse_csr

    # Update later to support more sparse layouts
    @property
    def is_sparse(self):  # type: ignore[override]
        return self.is_sparse_coo() or self.is_sparse_csr()
