"""Arithmetic, reductions and linear algebra over sparse tensors."""
import tensorplay

from ._construction import sparse_coo_tensor

__all__ = ["add", "addmm", "log_softmax", "mm", "mul", "softmax", "solve", "sum"]


def add(self, other):
    """Adds two sparse COO tensors with the same shape and dtype."""
    return tensorplay.sparse_add(self, other)


def mul(self, other):
    """Multiplies two sparse COO tensors elementwise on shared coordinates."""
    return tensorplay.sparse_mul(self, other)


def mm(sparse, dense):
    """Matrix product of a supported sparse tensor with a dense matrix."""
    if (
        isinstance(sparse, tensorplay.Tensor)
        and type(sparse) is not tensorplay.Tensor
        and callable(getattr(sparse, "__tensorplay_dispatch__", None))
    ):
        return tensorplay.mm(sparse, dense)
    return tensorplay.sparse_mm(sparse, dense)


def addmm(mat, mat1, mat2, *, beta=1.0, alpha=1.0):
    """Computes ``beta * mat + alpha * (mat1 @ mat2)`` with sparse ``mat1``.

    ``mat1`` is a 2-D sparse COO (``sparse_dim == 2``) or CSR matrix and
    ``mat2`` a dense matrix; ``mat`` is dense and broadcasts against the
    product.  A dense ``mat1`` is accepted too and takes the dense path.

    During backward the gradient of a sparse ``mat1`` is itself sparse, laid
    out on ``mat1``'s coordinates.
    """
    if any(
        isinstance(value, tensorplay.Tensor)
        and type(value) is not tensorplay.Tensor
        and callable(getattr(value, "__tensorplay_dispatch__", None))
        for value in (mat1, mat2)
    ):
        return tensorplay.addmm(mat, mat1, mat2, beta=beta, alpha=alpha)
    product = (
        tensorplay.sparse_mm(mat1, mat2)
        if mat1.is_sparse
        else tensorplay.mm(mat1, mat2)
    )
    if alpha != 1.0:
        product = product * alpha
    if beta == 0.0:
        return product
    return (mat * beta if beta != 1.0 else mat) + product


def sum(input, dim=None, dtype=None):
    """Sum of ``input``'s values over ``dim``.

    With no ``dim`` the result is a dense 0-dim tensor; reducing every sparse
    dim yields a dense tensor; a partial reduction returns a coalesced sparse
    COO tensor over the remaining dims with duplicate coordinates folded.
    ``dtype`` converts the input first, acting as the accumulation type.

    During backward only the ``nnz`` locations of ``input`` receive gradient,
    and the gradient is coalesced.
    """
    return tensorplay.sparse_sum(input, dim=dim, dtype=dtype)


def _grouped_softmax(input, dim, dtype, log):
    """Softmax over the specified entries of a sparse COO tensor.

    Unspecified entries are treated as absent rather than as zeros: the
    normalization runs over the stored values of each slice only, and the
    result keeps ``input``'s coordinates.  ``dim`` may address a sparse
    dimension (the slice is then the set of stored entries agreeing on every
    other sparse coordinate) or a dense dimension (the reduction is the
    ordinary one, inside each stored value block).
    """
    if not input.is_sparse:
        raise RuntimeError("expected a sparse tensor")
    if input.layout != tensorplay.sparse_coo:
        raise NotImplementedError("softmax is implemented for the COO layout")
    from tensorplay.nn.functional import log_softmax as _dense_log_softmax
    from tensorplay.nn.functional import softmax as _dense_softmax

    input = input.coalesce()
    if dtype is not None:
        input = sparse_coo_tensor(
            input._indices(),
            input.values().to(dtype),
            list(input.shape),
            is_coalesced=True,
        )

    ndim = input.dim()
    sparse_dim = input.sparse_dim()
    dim = dim % ndim if dim < 0 else dim
    if not 0 <= dim < ndim:
        raise IndexError(f"dimension {dim} out of range for {ndim}-D input")

    indices = input._indices()
    values = input.values()
    sizes = list(input.shape)

    if dim >= sparse_dim:
        # Dense dimension: the reduction never crosses stored entries, so the
        # value block can go through the ordinary kernel directly.
        block_dim = dim - sparse_dim + 1
        fn = _dense_log_softmax if log else _dense_softmax
        new_values = fn(values, dim=block_dim)
        return sparse_coo_tensor(indices, new_values, sizes, is_coalesced=True)

    nnz = int(values.shape[0])
    dense_shape = list(values.shape[1:])
    if nnz == 0:
        return sparse_coo_tensor(indices, values, sizes, is_coalesced=True)

    # Slice key: the mixed-radix encoding of every sparse coordinate but `dim`.
    keys = tensorplay.zeros([nnz], dtype=indices.dtype, device=indices.device)
    for d in range(sparse_dim):
        if d == dim:
            continue
        keys = keys * int(sizes[d]) + indices[d]
    _, inverse, _ = tensorplay.unique(keys, sorted=True, return_inverse=True)
    groups = int(inverse.max().item()) + 1

    width = 1
    for s in dense_shape:
        width *= int(s)
    flat = values.reshape([nnz, width])
    scatter_index = inverse.reshape([nnz, 1]).expand([nnz, width])

    empty = tensorplay.zeros([groups, width], dtype=flat.dtype, device=flat.device)
    slice_max = empty.scatter_reduce(
        0, scatter_index, flat, "amax", include_self=False
    )
    shifted = flat - slice_max.index_select(0, inverse)
    exponent = shifted.exp()
    total = tensorplay.zeros(
        [groups, width], dtype=flat.dtype, device=flat.device
    ).index_add(0, inverse, exponent)
    if log:
        out = shifted - total.log().index_select(0, inverse)
    else:
        out = exponent / total.index_select(0, inverse)

    new_values = out.reshape([nnz] + dense_shape)
    return sparse_coo_tensor(indices, new_values, sizes, is_coalesced=True)


def softmax(input, dim, *, dtype=None):
    r"""Applies a softmax over the stored entries of a sparse tensor.

    :math:`\text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}` where
    ``i`` and ``j`` run over the stored coordinates of one slice.  Entries
    that are not stored are ignored, which is the same as giving them the
    value :math:`-\infty` so that :math:`\exp(x_k) = 0`; the output keeps the
    input's sparsity pattern.

    Args:
        input (Tensor): the input sparse tensor
        dim (int): the dimension the softmax is computed along
        dtype (dtype, optional): result dtype; the input is cast before the
            operation, which is the way to keep the accumulation in higher
            precision than the input.
    """
    return _grouped_softmax(input, dim, dtype, log=False)


def log_softmax(input, dim, *, dtype=None):
    """Applies ``log(softmax(input, dim))`` over the stored entries.

    Shares :func:`softmax`'s treatment of unspecified entries and avoids the
    intermediate exponential in the same way the dense kernel does.
    """
    return _grouped_softmax(input, dim, dtype, log=True)


def solve(input, other, *, left=True):
    """Solves the linear system ``input @ X = other`` for a sparse ``input``.

    Args:
        input (Tensor): square sparse COO/CSR matrix of shape ``(n, n)``.
        other (Tensor): dense right-hand side, ``(n,)`` or ``(n, k)``.
        left (bool): when ``False``, solves ``X @ input = other`` instead.

    The factorization runs on the materialized matrix, so the peak memory is
    that of the dense ``(n, n)`` operand rather than of ``nnz``.
    """
    if not input.is_sparse:
        raise RuntimeError("expected `input` to be a sparse tensor")
    dense = tensorplay.to_dense(input)
    if not left:
        return tensorplay.linalg.solve(dense.transpose(-2, -1), other.transpose(-2, -1)).transpose(-2, -1)
    return tensorplay.linalg.solve(dense, other)
