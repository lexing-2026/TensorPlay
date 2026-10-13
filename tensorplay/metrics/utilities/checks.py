"""Input validation helpers for metrics."""

import multiprocessing
import os
import sys
from collections.abc import Mapping, Sequence
from functools import partial
from time import perf_counter
from typing import Any, Callable, Optional, no_type_check
from unittest.mock import Mock

import tensorplay as tensorplay
from tensorplay import Tensor

from tensorplay.metrics.metric import Metric

_DOCTEST_DOWNLOAD_TIMEOUT = int(os.environ.get("DOCTEST_DOWNLOAD_TIMEOUT", 120))
_SKIP_SLOW_DOCTEST = bool(os.environ.get("SKIP_SLOW_DOCTEST", 0))


def _check_for_empty_tensors(preds: Tensor, target: Tensor) -> bool:
    return preds.numel() == target.numel() == 0


def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(
            f"Predictions and targets are expected to have the same shape, but got {preds.shape} and {target.shape}."
        )


def _allclose_recursive(res1: Any, res2: Any, atol: float = 1e-6) -> bool:
    """Recursively assert that two results are within a certain tolerance."""
    if isinstance(res1, Tensor):
        return tensorplay.allclose(res1, res2, atol=atol)
    if isinstance(res1, str):
        return res1 == res2
    if isinstance(res1, Sequence):
        return all(_allclose_recursive(r1, r2) for r1, r2 in zip(res1, res2))
    if isinstance(res1, Mapping):
        return all(_allclose_recursive(res1[k], res2[k]) for k in res1)
    return res1 == res2


@no_type_check
def check_forward_full_state_property(
    metric_class: Metric,
    init_args: Optional[dict[str, Any]] = None,
    input_args: Optional[dict[str, Any]] = None,
    num_update_to_compare: Sequence[int] = [10, 100, 1000],
    reps: int = 5,
) -> None:
    """Check if the ``full_state_update`` property can safely be set to ``False``.

    Args:
        metric_class: metric class object that should be checked
        init_args: dict containing arguments for initializing the metric class
        input_args: dict containing arguments to pass to ``forward``
        num_update_to_compare: list of integers for how many steps to compare over
        reps: number of repetitions of the speedup test

    """
    init_args = init_args or {}
    input_args = input_args or {}

    class FullState(metric_class):
        full_state_update = True

    class PartState(metric_class):
        full_state_update = False

    fullstate = FullState(**init_args)
    partstate = PartState(**init_args)

    equal = True
    try:
        for _ in range(num_update_to_compare[0]):
            equal = equal & _allclose_recursive(fullstate(**input_args), partstate(**input_args))
    except RuntimeError:
        equal = False
    res1 = fullstate.compute()
    try:
        res2 = partstate.compute()
    except RuntimeError:
        equal = False
    equal = equal & _allclose_recursive(res1, res2)

    if not equal:
        print("Recommended setting `full_state_update=True`")
        return

    # Do timings
    res = tensorplay.zeros(2, len(num_update_to_compare), reps)
    for i, metric in enumerate([fullstate, partstate]):
        for j, t in enumerate(num_update_to_compare):
            for r in range(reps):
                start = perf_counter()
                for _ in range(t):
                    _ = metric(**input_args)
                end = perf_counter()
                res[i, j, r] = end - start
                metric.reset()

    mean = tensorplay.mean(res, -1)
    std = tensorplay.std(res, -1)

    for t in range(len(num_update_to_compare)):
        print(f"Full state for {num_update_to_compare[t]} steps took: {mean[0, t]}+-{std[0, t]:0.3f}")
        print(f"Partial state for {num_update_to_compare[t]} steps took: {mean[1, t]:0.3f}+-{std[1, t]:0.3f}")

    faster = (mean[1, -1] < mean[0, -1]).item()
    print(f"Recommended setting `full_state_update={not faster}`")
    return


def is_overridden(method_name: str, instance: object, parent: object) -> bool:
    """Check if a method has been overridden by an instance compared to its parent class."""
    instance_attr = getattr(instance, method_name, None)
    if instance_attr is None:
        return False
    if hasattr(instance_attr, "__wrapped__"):
        instance_attr = instance_attr.__wrapped__
    if isinstance(instance_attr, Mock):
        instance_attr = instance_attr._mock_wraps
    elif isinstance(instance_attr, partial):
        instance_attr = instance_attr.func
    if instance_attr is None:
        return False

    parent_attr = getattr(parent, method_name, None)
    if parent_attr is None:
        raise ValueError("The parent should define the method")

    return instance_attr.__code__ != parent_attr.__code__

def _check_retrieval_functional_inputs(
    preds: Tensor,
    target: Tensor,
    allow_non_binary_target: bool = False,
) -> tuple[Tensor, Tensor]:
    """Check that ``preds`` and ``target`` are of the same shape and of the correct data type.

    Args:
        preds: tensor with scores/logits
        target: tensor with ground truth labels
        allow_non_binary_target: whether to allow non-binary values in ``target``

    Returns:
        ``preds`` as float and ``target`` as long (or float when floating point), both flattened.
    """
    if preds.shape != target.shape:
        raise ValueError("`preds` and `target` must be of the same shape")

    if not preds.numel() or not preds.size():
        raise ValueError("`preds` and `target` must be non-empty and non-scalar tensors")

    return _check_retrieval_target_and_prediction_types(preds, target, allow_non_binary_target=allow_non_binary_target)


def _check_retrieval_target_and_prediction_types(
    preds: Tensor,
    target: Tensor,
    allow_non_binary_target: bool = False,
) -> tuple[Tensor, Tensor]:
    """Check ``preds`` and ``target`` tensors for correct data types."""
    if target.dtype not in (tensorplay.bool, tensorplay.long, tensorplay.int) and not tensorplay.is_floating_point(target):
        raise ValueError("`target` must be a tensor of booleans, integers or floats")

    if not preds.is_floating_point():
        raise ValueError("`preds` must be a tensor of floats")

    if not allow_non_binary_target and (target.max() > 1 or target.min() < 0):
        raise ValueError("`target` must contain `binary` values")

    target = target.float() if target.is_floating_point() else target.long()
    preds = preds.float()

    return preds.flatten(), target.flatten()


def _check_retrieval_inputs(
    indexes: Tensor,
    preds: Tensor,
    target: Tensor,
    allow_non_binary_target: bool = False,
    ignore_index: Optional[int] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Check ``indexes``, ``preds`` and ``target`` tensors for correct shapes and data types.

    Args:
        indexes: tensor with query indexes
        preds: tensor with scores/logits
        target: tensor with ground truth labels
        allow_non_binary_target: whether to allow non-binary values in ``target``
        ignore_index: ignore predictions where targets equal this value

    Returns:
        ``indexes`` as long, ``preds`` as float and ``target`` as long.
    """
    if indexes.shape != preds.shape or preds.shape != target.shape:
        raise ValueError("`indexes`, `preds` and `target` must be of the same shape")

    if indexes.dtype != tensorplay.long:
        raise ValueError("`indexes` must be a tensor of long integers")

    if target.dtype not in (tensorplay.bool, tensorplay.long, tensorplay.int) and not tensorplay.is_floating_point(target):
        raise ValueError("`target` must be a tensor of booleans, integers or floats")

    if not preds.is_floating_point():
        raise ValueError("`preds` must be a tensor of floats")

    if not allow_non_binary_target and (target.max() > 1 or target.min() < 0):
        raise ValueError("`target` must contain `binary` values")

    if not ignore_index:
        return indexes.long(), preds.float(), target.long()
    return indexes.long(), preds.float(), target.long()


def _try_proceed_with_timeout(fn: Callable, timeout: int = _DOCTEST_DOWNLOAD_TIMEOUT) -> bool:
    """Check whether a certain function finishes within a timeout.

    The function is only executed when running inside a doctest context.

    Returns:
        whether the function finished within the specified timeout.
    """
    if multiprocessing.current_process().daemon:
        # Skip timeout check in daemon processes as they cannot spawn child processes.
        return True

    proc = multiprocessing.Process(target=fn)

    print(f"Trying to run `{fn.__name__}` for {timeout}s...", file=sys.stderr)
    proc.start()
    proc.join(timeout)
    if not proc.is_alive():
        return False
    proc.terminate()
    return True
