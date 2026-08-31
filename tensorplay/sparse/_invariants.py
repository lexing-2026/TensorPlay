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
    """Raises ``RuntimeError`` unless the compressed buffers form a CSR tensor."""
    if crow_indices.dim() != 1:
        raise RuntimeError(
            f"`crow_indices` must be 1-D, got {crow_indices.dim()}-D"
        )
    if col_indices.dim() != 1:
        raise RuntimeError(f"`col_indices` must be 1-D, got {col_indices.dim()}-D")
    nrows = int(size[0])
    if int(crow_indices.shape[0]) != nrows + 1:
        raise RuntimeError(
            f"`len(crow_indices) == size[0] + 1` is not satisfied: "
            f"{int(crow_indices.shape[0])} != {nrows + 1}"
        )
    nnz = int(col_indices.shape[0])
    if int(values.shape[0]) != nnz:
        raise RuntimeError(
            f"`len(values) == len(col_indices)` is not satisfied: "
            f"{int(values.shape[0])} != {nnz}"
        )
    if int(crow_indices[0].item()) != 0:
        raise RuntimeError(
            f"`crow_indices[0] == 0` is not satisfied: {int(crow_indices[0].item())}"
        )
    last = int(crow_indices[nrows].item())
    if last != nnz:
        raise RuntimeError(f"`crow_indices[..., -1] == nnz` is not satisfied: {last} != {nnz}")
    diffs = crow_indices[1:] - crow_indices[:-1]
    if nrows and bool((diffs < 0).any().item()):
        raise RuntimeError("`crow_indices` must be non-decreasing")
    if nnz == 0:
        return
    lo = col_indices.min().item()
    hi = col_indices.max().item()
    if lo < 0 or hi >= int(size[1]):
        raise RuntimeError(
            f"`0 <= col_indices < size[1]` is not satisfied: range [{lo}, {hi}] "
            f"against {int(size[1])} columns"
        )


def validate(tensor) -> None:
    """Validates an already-built sparse tensor against its layout invariants."""
    if not tensor.is_sparse:
        raise RuntimeError("expected a sparse tensor")
    size = list(tensor.shape)
    if tensor.layout == tensorplay.sparse_csr:
        _validate_csr(
            tensor.crow_indices(), tensor.col_indices(), tensor.values(), size
        )
    else:
        _validate_coo(tensor._indices(), tensor.values(), size)


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
