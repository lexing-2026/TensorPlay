from collections.abc import Callable, Sequence
from functools import update_wrapper
from typing import Any, Final, Generic, overload, TypeVar

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import SymInt, Tensor
from tensorplay.overrides import is_tensor_like
from tensorplay.distributions._types import _dtype, _Number, Device, Number


euler_constant: Final[float] = 0.57721566490153286060  # Euler Mascheroni Constant

__all__ = [
    "broadcast_all",
    "logits_to_probs",
    "clamp_probs",
    "probs_to_logits",
    "lazy_property",
    "tril_matrix_to_vec",
    "vec_to_tril_matrix",
]


# FIXME: Use (*values: *Ts) -> tuple[Tensor for T in Ts] if Mapping-Type is ever added.
#   See https://github.com/python/typing/issues/1216#issuecomment-2126153831
def broadcast_all(*values: Tensor | Number) -> tuple[Tensor, ...]:
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:

    - `.*Tensor` instances are broadcasted as per :ref:`broadcasting-semantics`.
    - Number instances (scalars) are upcast to tensors having
      the same size and type as the first tensor passed to `values`.  If all the
      values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `Number`, `.*Tensor` or objects implementing __tensorplay_function__)

    Raises:
        ValueError: if any of the values is not a `Number` instance,
            a `.*Tensor` instance, or an instance implementing __tensorplay_function__
    """
    if not all(is_tensor_like(v) or isinstance(v, _Number) for v in values):
        raise ValueError(
            "Input arguments must all be instances of Number, "
            "tensorplay.Tensor or objects implementing __tensorplay_function__."
        )
    if not all(is_tensor_like(v) for v in values):
        options: dict[str, Any] = dict(dtype=tensorplay.get_default_dtype())
        for value in values:
            if isinstance(value, tensorplay.Tensor):
                options = dict(dtype=value.dtype, device=value.device)
                break
        new_values = [
            v if is_tensor_like(v) else tensorplay.tensor(v, **options) for v in values
        ]
        return tensorplay.broadcast_tensors(*new_values)
    return tensorplay.broadcast_tensors(*values)


def _standard_normal(
    shape: Sequence[int | SymInt],
    dtype: _dtype | None,
    device: Device | None,
) -> Tensor:
    if _get_tracing_state():
        # [JIT WORKAROUND] lack of support for .normal_()
        return tensorplay.normal(
            tensorplay.zeros(shape, dtype=dtype, device=device),
            tensorplay.ones(shape, dtype=dtype, device=device),
        )
    return tensorplay.empty(shape, dtype=dtype, device=device).normal_()


def _sum_rightmost(value: Tensor, dim: int) -> Tensor:
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def logits_to_probs(logits: Tensor, is_binary: bool = False) -> Tensor:
    r"""
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return tensorplay.sigmoid(logits)
    return F.softmax(logits, dim=-1)


def clamp_probs(probs: Tensor) -> Tensor:
    """Clamps the probabilities to be in the open interval `(0, 1)`.

    The probabilities would be clamped between `eps` and `1 - eps`,
    and `eps` would be the smallest representable positive number for the input data type.

    Args:
        probs (Tensor): A tensor of probabilities.

    Returns:
        Tensor: The clamped probabilities.

    Examples:
        >>> probs = tensorplay.tensor([0.0, 0.5, 1.0])
        >>> clamp_probs(probs)
        tensor([1.1921e-07, 5.0000e-01, 1.0000e+00])

        >>> probs = tensorplay.tensor([0.0, 0.5, 1.0], dtype=tensorplay.float64)
        >>> clamp_probs(probs)
        tensor([2.2204e-16, 5.0000e-01, 1.0000e+00], dtype=tensorplay.float64)

    """
    eps = tensorplay.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)


def probs_to_logits(probs: Tensor, is_binary: bool = False) -> Tensor:
    r"""
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return tensorplay.log(ps_clamped) - tensorplay.log1p(-ps_clamped)
    return tensorplay.log(ps_clamped)


T = TypeVar("T", contravariant=True)
R = TypeVar("R", covariant=True)


class lazy_property(Generic[T, R]):
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """

    def __init__(self, wrapped: Callable[[T], R]) -> None:
        self.wrapped: Callable[[T], R] = wrapped
        update_wrapper(self, wrapped)  # type:ignore[arg-type]

    @overload
    def __get__(
        self, instance: None, obj_type: Any = None
    ) -> "_lazy_property_and_property[T, R]": ...

    @overload
    def __get__(self, instance: T, obj_type: Any = None) -> R: ...

    def __get__(
        self, instance: T | None, obj_type: Any = None
    ) -> "R | _lazy_property_and_property[T, R]":
        if instance is None:
            return _lazy_property_and_property(self.wrapped)
        with tensorplay.enable_grad():
            value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


class _lazy_property_and_property(lazy_property[T, R], property):
    """We want lazy properties to look like multiple things.

    * property when Sphinx autodoc looks
    * lazy_property when Distribution validate_args looks
    """

    def __init__(self, wrapped: Callable[[T], R]) -> None:
        property.__init__(self, wrapped)


def tril_matrix_to_vec(mat: Tensor, diag: int = 0) -> Tensor:
    r"""
    Convert a `D x D` matrix or a batch of matrices into a (batched) vector
    which comprises of lower triangular elements from the matrix in row order.
    """
    n = mat.shape[-1]
    if not _get_tracing_state() and (diag < -n or diag >= n):
        raise ValueError(f"diag ({diag}) provided is outside [{-n}, {n - 1}].")
    arange = tensorplay.arange(n, device=mat.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    vec = mat[..., tril_mask]
    return vec


def vec_to_tril_matrix(vec: Tensor, diag: int = 0) -> Tensor:
    r"""
    Convert a vector or a batch of vectors into a batched `D x D`
    lower triangular matrix containing elements from the vector in row order.
    """
    # +ve root of D**2 + (1+2*diag)*D - |diag| * (diag+1) - 2*vec.shape[-1] = 0
    n = (
        -(1 + 2 * diag)
        + ((1 + 2 * diag) ** 2 + 8 * vec.shape[-1] + 4 * abs(diag) * (diag + 1)) ** 0.5
    ) / 2
    eps = tensorplay.finfo(vec.dtype).eps
    if not _get_tracing_state() and (round(n) - n > eps):
        raise ValueError(
            f"The size of last dimension is {vec.shape[-1]} which cannot be expressed as "
            + "the lower triangular part of a square D x D matrix."
        )
    n = round(n.item()) if isinstance(n, tensorplay.Tensor) else round(n)
    mat = vec.new_zeros(vec.shape[:-1] + tensorplay.Size((n, n)))
    arange = tensorplay.arange(n, device=vec.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    mat[..., tril_mask] = vec
    return mat


def _get_tracing_state():
    """No tracing machinery exists yet; always report off-trace."""
    return None


def _is_all_true(t):
    """Fast-path validity check used by argument validation."""
    return bool(t.all().item())
