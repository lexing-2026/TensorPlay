"""Shared validation and amount-conversion helpers for pruning.

Every pruning method interprets a pruning request ("amount") through the
helpers in this module so the rules stay uniform: a float amount is a
fraction of the available units, an int amount is an absolute count, and
structured methods reduce a tensor to one importance value per channel
before ranking.
"""

from __future__ import annotations

import numbers

from tensorplay import Tensor, linalg

__all__ = ["validate_pruning_amount", "compute_nparams_to_prune"]


def _validate_pruning_amount_init(amount: int | float) -> None:
    """Check the validity of ``amount`` at pruning-method construction time.

    Args:
        amount: quantity of units to prune. A float must lie in the closed
            interval ``[0, 1]`` and denotes a fraction of the available
            units; an int denotes an absolute count.

    Raises:
        TypeError: if ``amount`` is neither an int nor a float.
        ValueError: if ``amount`` is a float outside ``[0, 1]`` or a
            negative integer.

    Note:
        The number of units actually available is only known when a mask is
        computed, so it cannot be checked here.
    """
    if not isinstance(amount, numbers.Real):
        raise TypeError(f"Invalid type for amount: {amount}. Must be int or float.")

    if (isinstance(amount, numbers.Integral) and amount < 0) or (
        not isinstance(amount, numbers.Integral)  # so it's a float
        and (float(amount) > 1.0 or float(amount) < 0.0)
    ):
        raise ValueError(
            f"amount={amount} should either be a float in the range [0, 1] or a non-negative integer"
        )


def validate_pruning_amount(amount: int, tensor_size: int) -> None:
    """Check that an absolute pruning count fits the data being pruned.

    Args:
        amount: absolute number of units to prune.
        tensor_size: number of units available for pruning.

    Raises:
        ValueError: if ``amount`` exceeds ``tensor_size``.
    """
    if isinstance(amount, numbers.Integral) and amount > tensor_size:
        raise ValueError(
            f"amount={amount} should be smaller than the number of parameters to prune={tensor_size}"
        )


def compute_nparams_to_prune(amount: int | float, tensor_size: int) -> int:
    """Convert a pruning amount into an absolute unit count.

    An integer ``amount`` is returned unchanged; a float ``amount`` is read
    as a fraction of ``tensor_size`` and rounded to the nearest integer.

    Args:
        amount: pruning quantity (absolute int, or fractional float in
            ``[0, 1]``).
        tensor_size: number of units available for pruning.

    Returns:
        The number of units to prune.
    """
    # the type of amount has already been checked by _validate_pruning_amount_init
    if isinstance(amount, numbers.Integral):
        return amount
    return round(amount * tensor_size)


def _validate_structured_pruning(t: Tensor) -> None:
    """Check that the tensor to prune has at least two dimensions.

    Structured pruning removes whole channels, which is only meaningful for
    multidimensional tensors.

    Args:
        t: tensor whose channels would be pruned.

    Raises:
        ValueError: if ``t`` has fewer than two dimensions.
    """
    shape = t.shape
    if len(shape) <= 1:
        raise ValueError(
            "Structured pruning can only be applied to "
            "multidimensional tensors. Found tensor of shape "
            f"{shape} with {len(shape)} dims"
        )


def _validate_pruning_dim(t: Tensor, dim: int) -> None:
    """Check that ``dim`` is a valid axis index for ``t``.

    Args:
        t: tensor whose channels would be pruned.
        dim: axis along which channels are defined.

    Raises:
        IndexError: if ``dim`` is out of bounds.
    """
    if dim >= t.dim():
        raise IndexError(f"Invalid index {dim} for tensor of size {t.shape}")


def _compute_norm(t: Tensor, n: int | float | str, dim: int) -> Tensor:
    """Reduce ``t`` to one importance value per channel along ``dim``.

    The reduction aggregates every axis except ``dim``. For instance, for a
    tensor of shape ``(3, 2, 4)`` with ``dim=2`` the result has shape ``(4,)``,
    and each entry aggregates the ``3 x 2 = 6`` entries of its channel.

    Args:
        t: tensor whose per-channel norms are wanted.
        n: norm order. ``'nuc'`` and (for a matrix reduction) ``'fro'``
            select matrix norms; numeric values select the corresponding
            p-norm, and ``'fro'`` elsewhere falls back to the 2-norm.
        dim: axis that identifies the channels.

    Returns:
        A tensor holding one norm per channel along ``dim``.
    """
    # dims = all axes, except for the one identified by `dim`
    dims = list(range(t.dim()))
    # convert negative indexing
    if dim < 0:
        dim = dims[dim]
    dims.remove(dim)

    # 'nuc' and 'fro' over a matrix reduction are matrix norms; every other
    # case (numeric n, 'fro' elsewhere) is the flat p-norm.
    is_matrix_dim = len(dims) == 2
    if n == "nuc" or (n == "fro" and is_matrix_dim):
        return linalg.matrix_norm(t, ord=n, dim=dims if is_matrix_dim else (-2, -1))
    return linalg.vector_norm(t, ord=2 if n == "fro" else n, dim=dims)
