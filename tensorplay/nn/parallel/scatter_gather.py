# Ported from torch/nn/parallel/scatter_gather.py.
#
# The multi-device scatter/gather paths are single-process utilities; tp
# keeps the recursive container walk and delegates tensor movement to .to().
import itertools
from typing import Any, TypeVar

import tensorplay as tp

__all__ = ["scatter", "scatter_kwargs", "gather"]

T = TypeVar("T", dict, list, tuple)


def _is_namedtuple(obj: Any) -> bool:
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and distributes them across given GPUs.

    Duplicates references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, tp.Tensor):
            return [chunk.to(target_gpu)
                    for chunk, target_gpu in zip(_chunk_tensor(obj, len(target_gpus), dim), target_gpus)]
        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for _ in range(len(target_gpus))]

    # After scatter_map is called, a scatter_map cell will be a list of
    # tuples of length len(target_gpus) (one per GPU).
    return list(itertools.chain.from_iterable(scatter_map(inputs)))


def _chunk_tensor(tensor: tp.Tensor, chunks: int, dim: int):
    if chunks == 1:
        return [tensor]
    size = tensor.size(dim)
    if size < chunks:
        # torch pads by repeating the last slice; keep it simple with chunk()
        return list(tp.chunk(tensor, chunks, dim=dim))[:chunks] or [tensor]
    return list(tp.chunk(tensor, chunks, dim=dim))


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary."""
    scattered_inputs = scatter(inputs, target_gpus, dim) if inputs else []
    scattered_kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(scattered_inputs) < len(scattered_kwargs):
        scattered_inputs.extend(
            [() for _ in range(len(scattered_kwargs) - len(scattered_inputs))]
        )
    elif len(scattered_kwargs) < len(scattered_inputs):
        scattered_kwargs.extend(
            [{} for _ in range(len(scattered_inputs) - len(scattered_kwargs))]
        )
    return tuple(
        zip(scattered_inputs, scattered_kwargs)
    )


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device.

    ``target_device`` may be an int (device index).
    """
    device = f"cuda:{target_device}" if isinstance(target_device, int) else str(target_device)

    def gather_map(outputs):
        out = outputs[0]
        if _is_namedtuple(out):
            return type(out)._make(map(gather_map, zip(*outputs)))
        if isinstance(out, dict) and len(out) > 0:
            return type(out)((k, gather_map([d[k] for d in outputs]))
                             for k in out)
        if isinstance(out, list) and len(out) > 0:
            return type(out)(map(gather_map, zip(*outputs)))
        if isinstance(out, tuple) and len(out) > 0:
            return type(out)(map(gather_map, zip(*outputs)))
        if isinstance(out, tp.Tensor) and any(o is not None for o in outputs[1:]):
            return tp.cat(list(outputs), dim=dim).to(device)
        return out

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None  # type: ignore[assignment]
