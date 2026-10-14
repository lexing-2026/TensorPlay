"""Data manipulation helpers shared by metrics."""

from collections.abc import Sequence
from typing import Any, List, Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.utilities.imports import _XLA_AVAILABLE

METRIC_EPS = 1e-6


def apply_to_collection(
    data: Any,
    dtype: type,
    function: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Recursively apply ``function`` to every element of ``data`` of type ``dtype``.

    Lists, tuples and dicts are traversed in place; other types are returned
    unchanged.
    """
    if isinstance(data, dtype):
        return function(data, *args, **kwargs)
    if isinstance(data, (list, tuple)):
        return type(data)(apply_to_collection(d, dtype, function, *args, **kwargs) for d in data)
    if isinstance(data, dict):
        return {k: apply_to_collection(v, dtype, function, *args, **kwargs) for k, v in data.items()}
    return data


def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """Concatenate tensors along the zero dimension."""
    if isinstance(x, tensorplay.Tensor):
        return x
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return tensorplay.cat(x, dim=0)


def dim_zero_sum(x: Tensor) -> Tensor:
    """Sum along the zero dimension."""
    return tensorplay.sum(x, dim=0)


def dim_zero_mean(x: Tensor) -> Tensor:
    """Mean along the zero dimension."""
    return tensorplay.mean(x, dim=0)


def dim_zero_max(x: Tensor) -> Tensor:
    """Max along the zero dimension."""
    return tensorplay.max(x, dim=0).values


def dim_zero_min(x: Tensor) -> Tensor:
    """Min along the zero dimension."""
    return tensorplay.min(x, dim=0).values


def _flatten(x: Sequence) -> list:
    """Flatten a list of lists into a single list."""
    return [item for sublist in x for item in sublist]


def _flatten_dict(x: dict) -> tuple[dict, bool]:
    """Flatten a dict of dicts into a single dict, checking for duplicate keys."""
    new_dict = {}
    duplicates = False
    for key, value in x.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if k in new_dict:
                    duplicates = True
                new_dict[k] = v
        else:
            if key in new_dict:
                duplicates = True
            new_dict[key] = value
    return new_dict, duplicates


def to_onehot(
    label_tensor: Tensor,
    num_classes: Optional[int] = None,
) -> Tensor:
    """Convert a dense label tensor to one-hot format.

    Args:
        label_tensor: dense label tensor of shape ``[N, d1, d2, ...]``
        num_classes: number of classes ``C``

    Returns:
        A one-hot tensor of shape ``[N, C, d1, d2, ...]``

    """
    if num_classes is None:
        num_classes = int(label_tensor.max().detach().item() + 1)

    tensor_onehot = tensorplay.zeros(
        label_tensor.shape[0],
        num_classes,
        *label_tensor.shape[1:],
        dtype=label_tensor.dtype,
        device=label_tensor.device,
    )
    index = label_tensor.long().unsqueeze(1).expand_as(tensor_onehot)
    return tensor_onehot.scatter_(1, index, 1.0)


def _top_k_with_half_precision_support(x: Tensor, k: int = 1, dim: int = 1) -> Tensor:
    """Return the indices of the top-k entries.

    Half precision tensors on CPU are sorted instead of using ``topk`` because the
    native reduction does not support that dtype/device combination.
    """
    if x.dtype == tensorplay.half and not x.is_cuda:
        idx = tensorplay.argsort(x, dim=dim, stable=True)
        idx = tensorplay.flip(idx, dim)
        return tensorplay.narrow(idx, dim, 0, k)
    return tensorplay.topk(x, k=k, dim=dim).indices


def select_topk(prob_tensor: Tensor, topk: int = 1, dim: int = 1) -> Tensor:
    """Convert a probability tensor to a binary tensor selecting the top-k entries.

    Args:
        prob_tensor: dense tensor of shape ``[..., C, ...]`` where ``C`` sits at ``dim``
        topk: number of highest entries to turn into 1s
        dim: dimension on which to compare entries

    Returns:
        A binary tensor of the same shape as the input, with ``int`` dtype

    """
    topk_tensor = tensorplay.zeros_like(prob_tensor, dtype=tensorplay.int)
    if topk == 1:  # argmax has better performance than topk
        tensorplay.scatter_(topk_tensor, dim, tensorplay.argmax(prob_tensor, dim=dim, keepdim=True), 1.0)
    else:
        tensorplay.scatter_(topk_tensor, dim, _top_k_with_half_precision_support(prob_tensor, k=topk, dim=dim), 1.0)
    return topk_tensor.int()


def to_categorical(x: Tensor, argmax_dim: int = 1) -> Tensor:
    """Convert a tensor of probabilities to a dense label tensor."""
    return tensorplay.argmax(x, dim=argmax_dim)


def _squeeze_scalar_element_tensor(x: Tensor) -> Tensor:
    return tensorplay.squeeze(x) if x.numel() == 1 else x


def _squeeze_if_scalar(data: Any) -> Any:
    return apply_to_collection(data, Tensor, _squeeze_scalar_element_tensor)


def _bincount(x: Tensor, minlength: Optional[int] = None) -> Tensor:
    """Count occurrences of each value in ``x``.

    Falls back to an explicit equality count when deterministic algorithms are
    enabled, since the native ``bincount`` may be unavailable in that mode.
    """
    if minlength is None:
        minlength = len(tensorplay.unique(x))

    if tensorplay.are_deterministic_algorithms_enabled() or _XLA_AVAILABLE:
        mesh = tensorplay.arange(minlength, device=x.device).repeat(len(x), 1)
        return tensorplay.eq(x.reshape(-1, 1), mesh).sum(dim=0)

    return tensorplay.bincount(x, minlength=minlength)


def _flexible_bincount(x: Tensor) -> Tensor:
    """Like ``_bincount`` but works with tensors containing non-contiguous values."""
    unique_x, inverse_indices = tensorplay.unique(x, return_inverse=True)
    return _bincount(inverse_indices, minlength=len(unique_x))


def allclose(tensor1: Tensor, tensor2: Tensor) -> bool:
    """Wrap ``allclose`` to be robust towards dtype differences."""
    if tensor1.dtype != tensor2.dtype:
        tensor2 = tensor2.to(dtype=tensor1.dtype)
    return tensorplay.allclose(tensor1, tensor2)


def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Piecewise linear interpolation comparable to ``numpy.interp``.

    Args:
        x: x-coordinates where to evaluate the interpolated values
        xp: x-coordinates of the data points
        fp: y-coordinates of the data points

    """
    sorted_indices = tensorplay.argsort(xp)
    xp = xp[sorted_indices]
    fp = fp[sorted_indices]

    slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])

    indices = tensorplay.searchsorted(xp, x) - 1
    indices = tensorplay.clamp(indices, 0, len(slopes) - 1)

    return fp[indices] + slopes[indices] * (x - xp[indices])