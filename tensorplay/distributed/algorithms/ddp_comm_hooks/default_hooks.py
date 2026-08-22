"""DDP communication hooks — ported from
``torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py``.
"""

from collections.abc import Callable
from typing import Any

import tensorplay as tp
import tensorplay.distributed as dist

__all__ = [
    "allreduce_hook",
    "fp16_compress_hook",
    "bf16_compress_hook",
    "fp16_compress_wrapper",
    "bf16_compress_wrapper",
]


def _allreduce_fut(
    process_group, tensor: tp.Tensor,
):
    """Average the input gradient tensor by allreduce and returns a future."""
    group_to_use = process_group if process_group is not None else dist.GroupMember.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )


def allreduce_hook(process_group, bucket: dist.GradBucket):
    """
    Call ``allreduce`` using ``GradBucket`` tensors.

    Once gradient tensors are aggregated across all workers, its ``then``
    callback takes the mean and returns the result.

    If user registers this DDP communication hook,
    DDP results is expected to be same as the case where no hook was registered.
    Hence, this won't change behavior of DDP and user can use this as a reference
    or modify this hook to log useful information or any other purposes while
    unaffecting DDP behavior.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, allreduce_hook)
    """
    return _allreduce_fut(process_group, bucket.buffer())


def _compress_hook(dtype, process_group, bucket):
    group_to_use = process_group if process_group is not None else dist.GroupMember.WORLD
    world_size = group_to_use.size()

    buffer = (
        tuple(bucket)[0]
        if isinstance(bucket, tuple)
        else bucket.buffer()
    )
    compressed_tensor = buffer.to(dtype).div_(world_size)

    def decompress(fut_or_tensor):
        decompressed_tensor = buffer
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        value = fut_or_tensor.value() \
            if hasattr(fut_or_tensor, "value") else fut_or_tensor
        decompressed_tensor.copy_(value)
        return decompressed_tensor

    work = dist.all_reduce(compressed_tensor, group=group_to_use,
                           async_op=True)
    return work.get_future().then(decompress)


def fp16_compress_hook(process_group, bucket):
    """
    Compress by casting ``GradBucket`` to float16 divided by process group size.

    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``tensorplay.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    return _compress_hook(tp.float16, process_group, bucket)


def bf16_compress_hook(process_group, bucket):
    """
    Warning: This API is experimental, and it requires NCCL version later than 2.9.6.

    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision
    `Brain floating point format <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_ (``tensorplay.bfloat16``)
    and then divides it by the process group size.
    It allreduces those ``bfloat16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, bf16_compress_hook)
    """
    return _compress_hook(tp.bfloat16, process_group, bucket)


def fp16_compress_wrapper(hook: Callable[[Any], Any]):
    """
    Cast input tensor to ``float16``, cast result of hook back to input dtype.

    This wrapper casts the input gradient tensor of a given DDP communication hook to half-precision
    floating point format (``float16``), and casts the resulting tensor of the given hook back to
    the input data type, such as ``float32``.
    Therefore, ``fp16_compress_hook`` is equivalent to ``fp16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, fp16_compress_wrapper(powerSGD_hook))
    """

    def fp16_compress_wrapper_hook(hook_state, bucket) :
        # Cast bucket tensor to FP16.
        bucket.set_buffer(bucket.buffer().to(tp.float16))

        fut = hook(hook_state, bucket)

        def decompress(fut):
            decompressed_tensor = bucket.buffer()
            # Decompress in place to reduce the peak memory.
            # See: https://github.com/pytorch/pytorch/issues/45968
            decompressed_tensor.copy_(fut.value())
            return decompressed_tensor

        # Decompress after hook has run.
        return fut.then(decompress)

    return fp16_compress_wrapper_hook


def bf16_compress_wrapper(hook: Callable[[Any], Any]):
    """
    Warning: This API is experimental, and it requires NCCL version later than 2.9.6.

    This wrapper casts the input gradient tensor of a given DDP communication hook to half-precision
    `Brain floating point format <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_  (``bfloat16``)
    and casts the resulting tensor of the given hook back to
    the input data type.

    Therefore, ``bf16_compress_hook`` is equivalent to ``bf16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, bf16_compress_wrapper(powerSGD_hook))
    """

    def bf16_compress_wrapper_hook(hook_state, bucket):
        # Cast bucket tensor to BF16.
        bucket.set_buffer(bucket.buffer().to(tp.bfloat16))

        fut = hook(hook_state, bucket)

        def decompress(fut):
            decompressed_tensor = bucket.buffer()
            # Decompress in place to reduce the peak memory.
            # See: https://github.com/pytorch/pytorch/issues/45968
            decompressed_tensor.copy_(fut.value())
            return decompressed_tensor

        # Decompress after hook has run.
        return fut.then(decompress)

    return bf16_compress_wrapper_hook
