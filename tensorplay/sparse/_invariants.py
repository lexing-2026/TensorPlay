"""Sparse tensor invariant checking and the flag that governs it.

Construction of a sparse tensor from raw index/value buffers can silently
produce an unusable tensor when the buffers disagree (out-of-range
coordinates, a values block whose trailing shape contradicts the declared
size, a compressed row pointer that does not end at ``nnz``).  Checking is
off by default because it costs a device synchronisation; it is turned on
either globally through :class:`check_sparse_tensor_invariants` or locally
through the ``check_invariants`` argument of the constructors.
"""
import tensorplay

__all__ = ["check_sparse_tensor_invariants"]

#: Module-global switch consulted by the sparse constructors.
_checks_enabled = False


def _is_enabled() -> bool:
    return _checks_enabled


def _set_enabled(state: bool) -> None:
    global _checks_enabled
    _checks_enabled = bool(state)


def _validate_coo(indices, values, size) -> None:
    """Raises ``RuntimeError`` unless ``(indices, values, size)`` form a COO tensor."""
    if indices.dtype not in (tensorplay.int32, tensorplay.int64):
        raise RuntimeError("`indices` must have int32 or int64 dtype")
    if indices.device != values.device:
        raise RuntimeError("`indices` and `values` must be on the same device")
    if indices.dim() != 2:
        raise RuntimeError(
            f"`indices` must be 2-D (sparse_dim x nnz), got {indices.dim()}-D"
        )
    sparse_dim, nnz = int(indices.shape[0]), int(indices.shape[1])
    if values.dim() < 1:
        raise RuntimeError("`values` must have at least one dimension (nnz first)")
    if int(values.shape[0]) != nnz:
        raise RuntimeError(
            f"`values.shape[0] == indices.shape[1]` is not satisfied: "
            f"{int(values.shape[0])} != {nnz}"
        )
    dense_dim = values.dim() - 1
    if any(int(value) < 0 for value in size):
        raise RuntimeError("`size` entries must be non-negative")
    if len(size) != sparse_dim + dense_dim:
        raise RuntimeError(
            f"`len(size) == sparse_dim + dense_dim` is not satisfied: "
            f"{len(size)} != {sparse_dim} + {dense_dim}"
        )
    for i in range(dense_dim):
        if int(values.shape[1 + i]) != int(size[sparse_dim + i]):
            raise RuntimeError(
                f"`values.shape[{1 + i}] == size[{sparse_dim + i}]` is not satisfied: "
                f"{int(values.shape[1 + i])} != {int(size[sparse_dim + i])}"
            )
    if nnz == 0:
        return
    lo = indices.min().item()
    if lo < 0:
        raise RuntimeError(f"`indices >= 0` is not satisfied: min index is {lo}")
    for d in range(sparse_dim):
        hi = indices[d].max().item()
        if hi >= int(size[d]):
            raise RuntimeError(
                f"`indices[{d}] < size[{d}]` is not satisfied: {hi} >= {int(size[d])}"
            )


def _validate_csr(crow_indices, col_indices, values, size) -> None:
    """Validate CSR buffers through the shared native invariant worker."""
    _validate_compressed(
        crow_indices,
        col_indices,
        values,
        size,
        tensorplay.sparse_csr,
    )


def _validate_compressed(
    compressed_indices, plain_indices, values, size, layout
) -> None:
    """Validate any CSR/CSC/BSR/BSC component tuple.

    The native worker is deliberately used here as well as by the C++ safe
    constructors.  That keeps the global ``check_sparse_tensor_invariants``
    switch byte-for-byte consistent with constructor validation, including
    batched pointers, block dimensions, dense value tails, and index strides.
    """
    layout = int(layout)
    if layout == int(tensorplay.sparse_csr):
        tensorplay._C._validate_sparse_csr_tensor_args(
            compressed_indices, plain_indices, values, list(size)
        )
    elif layout == int(tensorplay.sparse_csc):
        tensorplay._C._validate_sparse_csc_tensor_args(
            compressed_indices, plain_indices, values, list(size)
        )
    elif layout == int(tensorplay.sparse_bsr):
        tensorplay._C._validate_sparse_bsr_tensor_args(
            compressed_indices, plain_indices, values, list(size)
        )
    elif layout == int(tensorplay.sparse_bsc):
        tensorplay._C._validate_sparse_bsc_tensor_args(
            compressed_indices, plain_indices, values, list(size)
        )
    else:
        tensorplay._C._validate_sparse_compressed_tensor_args(
            compressed_indices,
            plain_indices,
            values,
            list(size),
            layout,
        )


def validate(tensor) -> None:
    """Validates an already-built sparse tensor against its layout invariants."""
    if not tensor.is_sparse:
        raise RuntimeError("expected a sparse tensor")
    size = list(tensor.shape)
    layout = int(tensor.layout)
    if layout == int(tensorplay.sparse_coo):
        _validate_coo(tensor._indices(), tensor.values(), size)
    elif layout == int(tensorplay.sparse_csr):
        _validate_compressed(
            tensor.crow_indices(), tensor.col_indices(), tensor.values(), size, layout
        )
    elif layout == int(tensorplay.sparse_csc):
        _validate_compressed(
            tensor.ccol_indices(), tensor.row_indices(), tensor.values(), size, layout
        )
    elif layout == int(tensorplay.sparse_bsr):
        _validate_compressed(
            tensor.crow_indices(), tensor.col_indices(), tensor.values(), size, layout
        )
    elif layout == int(tensorplay.sparse_bsc):
        _validate_compressed(
            tensor.ccol_indices(), tensor.row_indices(), tensor.values(), size, layout
        )
    else:
        raise RuntimeError(f"unsupported sparse layout: {layout}")


class check_sparse_tensor_invariants:
    """Controls whether sparse tensor construction validates its inputs.

    Four ways to use it:

    1. As a context manager::

           with tensorplay.sparse.check_sparse_tensor_invariants():
               run_my_model()

    2. Procedurally::

           prev = tensorplay.sparse.check_sparse_tensor_invariants.is_enabled()
           tensorplay.sparse.check_sparse_tensor_invariants.enable()
           run_my_model()
           if not prev:
               tensorplay.sparse.check_sparse_tensor_invariants.disable()

    3. As a decorator::

           @tensorplay.sparse.check_sparse_tensor_invariants()
           def run_my_model():
               ...

    4. Per call site, through the ``check_invariants`` keyword argument of
       :func:`tensorplay.sparse.sparse_coo_tensor`, which overrides the
       global flag for that one construction.
    """

    @staticmethod
    def is_enabled():
        """Returns True when sparse tensor invariant checking is enabled."""
        return _is_enabled()

    @staticmethod
    def enable():
        """Enables invariant checking in the sparse tensor constructors.

        Checking is off by default; it can still be overridden per call site
        by the constructors' ``check_invariants`` argument.
        """
        _set_enabled(True)

    @staticmethod
    def disable():
        """Disables invariant checking in the sparse tensor constructors."""
        _set_enabled(False)

    # context manager support
    def __init__(self, enable=True):
        self.state = enable
        self.saved_state = None

    def __enter__(self):
        if self.saved_state is not None:
            raise RuntimeError(
                "This context manager instance is already activated."
                " Use a different context manager instance for context nesting."
            )
        self.saved_state = self.is_enabled()
        _set_enabled(self.state)
        return self

    def __exit__(self, type, value, traceback):
        if self.saved_state is None:
            raise AssertionError("saved_state should not be None on exit")
        _set_enabled(self.saved_state)
        self.saved_state = None

    # decorator support
    def __call__(self, mth):
        def test_mth(*args, **kwargs):
            with type(self)(self.state):
                return mth(*args, **kwargs)

        return test_mth
