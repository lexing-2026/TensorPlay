# mypy: allow-untyped-defs
"""CUDA backend property helpers.

Exposes ``tensorplay.backends.cuda.matmul.allow_tf32``, backed by the same
global state as :func:`tensorplay.set_float32_matmul_precision`, plus the
scaled-dot-product attention backend controls: the :class:`SDPBackend` flag
set, the per-backend enable flags consumed by
``tensorplay.nn.attention.sdpa_kernel``, and the ``can_use_*`` eligibility
gates that decide whether a fused kernel may run a given call.
"""

import contextlib
from typing import Any
from typing_extensions import deprecated

import tensorplay


__all__ = [
    "is_built",
    "cuBLASModule",
    "is_ck_sdpa_available",
    "matmul",
    "SDPAParams",
    "enable_cudnn_sdp",
    "cudnn_sdp_enabled",
    "enable_flash_sdp",
    "flash_sdp_enabled",
    "enable_mem_efficient_sdp",
    "mem_efficient_sdp_enabled",
    "math_sdp_enabled",
    "enable_math_sdp",
    "allow_fp16_bf16_reduction_math_sdp",
    "fp16_bf16_reduction_math_sdp_allowed",
    "is_flash_attention_available",
    "can_use_flash_attention",
    "can_use_efficient_attention",
    "can_use_cudnn_attention",
    "sdp_kernel",
    "is_available",
    "get_name",
]


def is_built() -> bool:
    r"""Returns whether TensorPlay is built with CUDA support. Note that this
    doesn't mean CUDA is available; just that if TensorPlay is built for the machine."""
    return bool(tensorplay._C.is_cuda_available())


def is_available() -> bool:
    r"""Returns a bool indicating if CUDA is currently available."""
    return bool(tensorplay.cuda.is_available())


def get_name() -> str:
    r"""Returns the CUDA device name."""
    return tensorplay.cuda.get_device_name()


class cuBLASModule:
    def __getattr__(self, name):
        if name == "allow_tf32":
            return tensorplay._C._get_cublas_allow_tf32()
        raise AttributeError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            return tensorplay._C._set_cublas_allow_tf32(value)
        raise AttributeError("Unknown attribute " + name)


from tensorplay._C import _SDPAParams as SDPAParams, _SDPBackend as SDPBackend  # noqa: E402


# Set the __module__ attribute
SDPAParams.__module__ = "tensorplay.backends.cuda"
SDPAParams.__name__ = "SDPAParams"


def is_ck_sdpa_available() -> bool:
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether composable_kernel may be used as the backend for
    scaled-dot-product-attention.
    """
    return tensorplay._C._is_ck_sdpa_available()


def flash_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether flash scaled dot product attention is enabled or not.
    """
    return tensorplay._C._get_flash_sdp_enabled()


def enable_flash_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables flash scaled dot product attention.
    """
    tensorplay._C._set_sdp_use_flash(enabled)


def mem_efficient_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether memory efficient scaled dot product attention is enabled or not.
    """
    return tensorplay._C._get_mem_efficient_sdp_enabled()


def enable_mem_efficient_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables memory efficient scaled dot product attention.
    """
    tensorplay._C._set_sdp_use_mem_efficient(enabled)


def math_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether math scaled dot product attention is enabled or not.
    """
    return tensorplay._C._get_math_sdp_enabled()


def enable_math_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables math scaled dot product attention.
    """
    tensorplay._C._set_sdp_use_math(enabled)


def allow_fp16_bf16_reduction_math_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables fp16/bf16 reduction in math scaled dot product attention.
    """
    tensorplay._C._set_math_sdp_allow_fp16_bf16_reduction(enabled)


def fp16_bf16_reduction_math_sdp_allowed():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether fp16/bf16 reduction in math scaled dot product attention is enabled or not.
    """
    return tensorplay._C._get_math_sdp_allow_fp16_bf16_reduction()


def is_flash_attention_available() -> bool:
    r"""Check if TensorPlay was built with FlashAttention for scaled_dot_product_attention.

    Returns:
        True if FlashAttention is built and available; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of TensorPlay. It will return False
        in non-CUDA environments.
    """
    return tensorplay._C._is_flash_attention_available()


def can_use_flash_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if FlashAttention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn debug information as to why FlashAttention could not be run.
            Defaults to False.

    Returns:
        True if FlashAttention can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of TensorPlay. It will return False
        in non-CUDA environments.
    """
    return tensorplay._C._can_use_flash_attention(params, debug)


def can_use_efficient_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if efficient_attention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn debug information as to why efficient_attention could not be run.
            Defaults to False.

    Returns:
        True if efficient_attention can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of TensorPlay. It will return False
        in non-CUDA environments.
    """
    return tensorplay._C._can_use_mem_efficient_attention(params, debug)


def can_use_cudnn_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if cudnn_attention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn debug information as to why cuDNN attention could not be run.
            Defaults to False.

    Returns:
        True if cuDNN can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of TensorPlay. It will return False
        in non-CUDA environments.
    """
    return tensorplay._C._can_use_cudnn_attention(params, debug)


def cudnn_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether cuDNN scaled dot product attention is enabled or not.
    """
    return tensorplay._C._get_cudnn_sdp_enabled()


def enable_cudnn_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables cuDNN scaled dot product attention.
    """
    tensorplay._C._set_sdp_use_cudnn(enabled)


@contextlib.contextmanager
@deprecated(
    (
        "`tensorplay.backends.cuda.sdp_kernel()` is deprecated. "
        "In the future, this context manager will be removed. "
        "Please see `tensorplay.nn.attention.sdpa_kernel()` for the new context manager, "
        "with updated signature."
    ),
    category=FutureWarning,
)
def sdp_kernel(
    enable_flash: bool = True,
    enable_math: bool = True,
    enable_mem_efficient: bool = True,
    enable_cudnn: bool = True,
):
    r"""
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """
    from tensorplay.nn.attention import sdpa_kernel

    backend_list = []
    if enable_flash:
        backend_list.append(SDPBackend.FLASH_ATTENTION)
    if enable_mem_efficient:
        backend_list.append(SDPBackend.EFFICIENT_ATTENTION)
    if enable_math:
        backend_list.append(SDPBackend.MATH)
    if enable_cudnn:
        backend_list.append(SDPBackend.CUDNN_ATTENTION)

    with sdpa_kernel(backend_list) as context:
        try:
            yield context
        finally:
            pass


matmul = cuBLASModule()
