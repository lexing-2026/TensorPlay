"""Nested (ragged) tensors.

A nested tensor stores a variable-length batch of dense constituents in one
flat buffer plus per-constituent shape metadata (one size row per
constituent).  The strided layout is the supported representation:
``dim()`` reports ``R + 1`` for rank-``R`` constituents, ``size(0)`` is the
batch and deeper dimensions report the largest extent across constituents;
``numel()`` counts only the stored elements.

Construction and padding run through dispatcher-visible primitives, so both
operations work on every device that implements the dense kernels.
"""

import tensorplay as tp
from tensorplay._C import (
    _nested_tensor_from_tensor_list,
    _nested_view_from_buffer,
    nested_to_padded_tensor,
)

__all__ = [
    "as_nested_tensor",
    "nested_tensor",
    "to_padded_tensor",
]

# Layout tag of the strided (dense) representation; the only layout the
# construction entry points below produce.
_STRIDED = 0


def _contiguous_strides_offsets(nested_sizes):
    """Row-major strides per size row plus the packed start offset of each row.

    The last axis of every constituent advances by one element and each
    earlier axis advances by its extent times the stride of the axis after
    it; packed rows place constituent ``i`` at the total volume of the rows
    before it.
    """
    rows = nested_sizes.tolist()
    strides, offsets = [], []
    running = 0
    for row in rows:
        stride = [1] * len(row)
        for j in range(len(row) - 2, -1, -1):
            stride[j] = stride[j + 1] * row[j + 1]
        strides.append(stride)
        offsets.append(running)
        volume = 1
        for extent in row:
            volume *= extent
        running += volume
    device = nested_sizes.device
    return (
        tp.tensor(strides, dtype=tp.int64, device=device),
        tp.tensor(offsets, dtype=tp.int64, device=device),
    )


def as_nested_tensor(ts, dtype=None, device=None, layout=None):
    """Constructs a nested tensor preserving autograd history from a tensor
    or a list / tuple of tensors.

    If a nested tensor is passed, it is returned directly unless the device
    / dtype differ.  A dense tensor of rank two or more is treated as a batch
    of constituents of consistent size; its storage is shared when no device
    / dtype conversion is requested.  Constituents given as a list / tuple
    are always copied into one packed buffer.

    Args:
        ts (Tensor or List[Tensor] or Tuple[Tensor]): a tensor to treat as a
            nested tensor or a list / tuple of tensors with the same rank.

    Keyword args:
        dtype (tensorplay.dtype, optional): desired dtype of the result.
            Default: the dtype of the first input.
        device (tensorplay.device, optional): desired device of the result.
            Default: the device of the first input.
        layout: only the strided layout (``None`` or ``0``) is supported.

    Returns:
        Tensor: the nested tensor.
    """
    if isinstance(ts, tp.Tensor):
        if ts.is_nested():
            raise RuntimeError(
                "as_nested_tensor(): converting between nested tensor layouts "
                "is not supported"
            )
        if ts.dim() < 2:
            raise RuntimeError(
                "as_nested_tensor(): expected the tensor argument to have "
                "dim() > 1, got " + str(ts.dim())
            )
        if layout is not None and layout != _STRIDED:
            raise RuntimeError(
                "as_nested_tensor(): specified layout is unsupported for "
                "nested tensors: " + str(layout)
            )
        buffer = ts.reshape(-1)
        if dtype is not None:
            buffer = buffer.to(dtype)
        if device is not None:
            buffer = buffer.to(device)
        batch = ts.size(0)
        row_shape = list(ts.shape)[1:]
        nested_sizes = tp.tensor(
            [row_shape] * batch, dtype=tp.int64, device=buffer.device
        )
        strides, offsets = _contiguous_strides_offsets(nested_sizes)
        return _nested_view_from_buffer(buffer, nested_sizes, strides, offsets)
    if isinstance(ts, (list, tuple)) and all(isinstance(t, tp.Tensor) for t in ts):
        if layout is not None and layout != _STRIDED:
            raise RuntimeError(
                "as_nested_tensor(): specified layout is unsupported for "
                "nested tensors: " + str(layout)
            )
        return _nested_tensor_from_tensor_list(
            list(ts), dtype=dtype, layout=layout, device=device, pin_memory=None
        )
    raise TypeError(
        "as_nested_tensor(): expected the first argument to be a tensor or a "
        "list / tuple of tensors"
    )


def nested_tensor(ts, *, dtype=None, device=None, requires_grad=False,
                  layout=None, pin_memory=False):
    """Constructs a nested tensor from a list / tuple of tensors.

    Constituents must share the same rank; they are always copied into one
    packed buffer sized to the total element count.

    Args:
        ts (List[Tensor] or Tuple[Tensor]): tensors of the same rank.

    Keyword args:
        dtype (tensorplay.dtype, optional): desired dtype of the result.
        device (tensorplay.device, optional): desired device of the result.
        requires_grad (bool, optional): whether the result tracks gradients.
        layout: only the strided layout (``None`` or ``0``) is supported.
        pin_memory (bool, optional): whether the packed buffer is pinned.

    Returns:
        Tensor: the nested tensor.
    """
    if not isinstance(ts, (list, tuple)) or not all(
        isinstance(t, tp.Tensor) for t in ts
    ):
        raise TypeError(
            "nested_tensor(): expected the first argument to be a list / "
            "tuple of tensors"
        )
    if layout is not None and layout != _STRIDED:
        raise RuntimeError(
            "nested_tensor(): specified layout is unsupported for nested "
            "tensors: " + str(layout)
        )
    nt = _nested_tensor_from_tensor_list(
        list(ts), dtype=dtype, layout=layout, device=device,
        pin_memory=pin_memory,
    )
    if requires_grad:
        nt.requires_grad_()
    return nt


def to_padded_tensor(input, padding, output_size=None):
    """Pads a nested tensor into a regular dense tensor and returns it.

    The leading entries of each output slice carry the nested data while the
    trailing entries are filled with ``padding``.  Padding always copies the
    underlying data, since the nested and dense representations differ in
    memory layout.

    Args:
        input (Tensor): the nested tensor.
        padding (float): the fill value for the trailing entries.

    Keyword args:
        output_size (Tuple[int], optional): the size of the output.  If
            given, it must be large enough to contain all nested data;
            otherwise the maximum extent of the constituents along each
            dimension is inferred.

    Returns:
        Tensor: the dense padded tensor.
    """
    return nested_to_padded_tensor(input, padding, output_size)
