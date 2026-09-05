# mypy: allow-untyped-defs
"""CUDA backend property helpers.

Exposes ``tensorplay.backends.cuda.matmul.allow_tf32``, backed by the same
global state as :func:`tensorplay.set_float32_matmul_precision`, the
``cufft_plan_cache`` manager that inspects and bounds the per-device cuFFT
plan caches, the ``preferred_linalg_library``/``preferred_blas_library``
GPU library selection overrides, plus the scaled-dot-product attention
backend controls: the :class:`SDPBackend` flag set, the per-backend enable
flags consumed by ``tensorplay.nn.attention.sdpa_kernel``, and the
``can_use_*`` eligibility gates that decide whether a fused kernel may run
a given call.
"""

import contextlib
from typing import Any, Optional, Union

from typing_extensions import deprecated

import tensorplay


__all__ = [
    "is_built",
    "cuBLASModule",
    "cuFFTPlanCache",
    "cuFFTPlanCacheAttrContextProp",
    "cuFFTPlanCacheManager",
    "cufft_plan_cache",
    "preferred_linalg_library",
    "preferred_blas_library",
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


class cuFFTPlanCacheAttrContextProp:
    # Like a regular property, but the `.device_index` attribute of the
    # calling object is passed as the first argument to getter and setter.
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter(obj.device_index)

    def __set__(self, obj, val):
        if isinstance(self.setter, str):
            raise RuntimeError(self.setter)
        self.setter(obj.device_index, val)


class cuFFTPlanCache:
    r"""
    Represent a specific plan cache for a specific `device_index`.

    The attributes `size` and `max_size`, and method `clear`, can fetch and/or
    change properties of the C++ cuFFT plan cache.
    """

    def __init__(self, device_index):
        self.device_index = device_index

    size = cuFFTPlanCacheAttrContextProp(
        tensorplay._C._cufft_get_plan_cache_size,
        ".size is a read-only property showing the number of plans currently in the "
        "cache. To change the cache capacity, set cufft_plan_cache.max_size.",
    )

    max_size = cuFFTPlanCacheAttrContextProp(
        tensorplay._C._cufft_get_plan_cache_max_size,
        tensorplay._C._cufft_set_plan_cache_max_size,
    )

    def clear(self):
        return tensorplay._C._cufft_clear_plan_cache(self.device_index)


class cuFFTPlanCacheManager:
    r"""
    Represent all cuFFT plan caches, return the cuFFTPlanCache for a given device when indexed.

    Finally, this object, when used directly as a `cuFFTPlanCache` object (e.g.,
    setting the `.max_size`) attribute, the current device's cuFFT plan cache is
    used.
    """

    __initialized = False

    def __init__(self):
        self.caches = []
        self.__initialized = True

    def __getitem__(self, device):
        from tensorplay.cuda._utils import _get_device_index

        index = _get_device_index(device, optional=True)
        if index < 0 or index >= tensorplay.cuda.device_count():
            raise RuntimeError(
                f"cufft_plan_cache: expected 0 <= device index < {tensorplay.cuda.device_count()}, but got "
                f"device with index {index}"
            )
        if len(self.caches) == 0:
            self.caches.extend(
                cuFFTPlanCache(index) for index in range(tensorplay.cuda.device_count())
            )
        return self.caches[index]

    def __getattr__(self, name):
        return getattr(self[tensorplay.cuda.current_device()], name)

    def __setattr__(self, name, value):
        if self.__initialized:
            return setattr(self[tensorplay.cuda.current_device()], name, value)
        else:
            return super().__setattr__(name, value)


_LinalgBackends = {
    "default": tensorplay._C._LinalgBackend.Default,
    "cusolver": tensorplay._C._LinalgBackend.Cusolver,
    "magma": tensorplay._C._LinalgBackend.Magma,
}
_LinalgBackends_str = ", ".join(_LinalgBackends.keys())


def preferred_linalg_library(
    backend: None | str | tensorplay._C._LinalgBackend = None,
) -> tensorplay._C._LinalgBackend:
    r"""Override the heuristic TensorPlay uses to choose the GPU library for
    dense linear algebra operations.

    .. warning:: This flag is experimental and subject to change.

    When a CUDA linear algebra operation runs it is executed by a native
    GPU library; when several are available a heuristic decides which one
    to use. This flag (a :class:`str`) allows overriding those heuristics.

    * If ``"cusolver"`` is set then cuSOLVER will be used wherever possible.
    * If ``"magma"`` is requested an error is raised: this build ships no
      MAGMA backend.
    * If ``"default"`` (the default) is set then heuristics pick the library.
    * When no input is given, this function returns the currently preferred library.

    Note: When a library is preferred other libraries may still be used if the
    preferred library doesn't implement the operation(s) called.
    This flag may achieve better performance if the heuristic library
    selection is incorrect for your application's inputs.

    Currently supported linalg operators:

    * :func:`tensorplay.linalg.inv`
    * :func:`tensorplay.linalg.cholesky`
    * :func:`tensorplay.linalg.lu_factor`
    * :func:`tensorplay.linalg.qr`
    * :func:`tensorplay.linalg.eigh`
    * :func:`tensorplay.linalg.svd`
    * :func:`tensorplay.linalg.svdvals`
    """
    if backend is None:
        pass
    elif isinstance(backend, str):
        if backend not in _LinalgBackends:
            raise RuntimeError(
                f"Unknown input value. Choose from: {_LinalgBackends_str}."
            )
        tensorplay._C._set_linalg_preferred_backend(_LinalgBackends[backend])
    elif isinstance(backend, tensorplay._C._LinalgBackend):
        tensorplay._C._set_linalg_preferred_backend(backend)
    else:
        raise RuntimeError("Unknown input value type.")

    return tensorplay._C._get_linalg_preferred_backend()


_BlasBackends = {
    "default": tensorplay._C._BlasBackend.Default,
    "cublas": tensorplay._C._BlasBackend.Cublas,
    "hipblas": tensorplay._C._BlasBackend.Cublas,  # alias
    "cublaslt": tensorplay._C._BlasBackend.Cublaslt,
    "hipblaslt": tensorplay._C._BlasBackend.Cublaslt,  # alias
}
_BlasBackends_str = ", ".join(_BlasBackends.keys())


def preferred_blas_library(
    backend: None | str | tensorplay._C._BlasBackend = None,
) -> tensorplay._C._BlasBackend:
    r"""Override the library used for GPU BLAS operations. Choose between
    cuBLAS and cuBLASLt.

    .. warning:: This flag is experimental and subject to change.

    BLAS operations default to a heuristic per call even though both
    cuBLAS and cuBLASLt are usually available. This flag (a :class:`str`)
    allows overriding which BLAS library to use.

    * If ``"cublas"`` is set then the classic cuBLAS API will be used wherever possible.
    * If ``"cublaslt"`` is set then cuBLASLt will be used wherever possible.
    * If ``"default"`` (the default) is set then heuristics will be used to pick between the other options.
    * When no input is given, this function returns the currently preferred library.

    Note: When a library is preferred other libraries may still be used if the
    preferred library doesn't implement the operation(s) called; the fused
    bias epilogue always runs on cuBLASLt and complex matrix products always
    run on the classic cuBLAS API.
    """
    if backend is None:
        pass
    elif isinstance(backend, str):
        if backend not in _BlasBackends:
            raise RuntimeError(
                f"Unknown input value. Choose from: {_BlasBackends_str}."
            )
        tensorplay._C._set_blas_preferred_backend(_BlasBackends[backend])
    elif isinstance(backend, tensorplay._C._BlasBackend):
        tensorplay._C._set_blas_preferred_backend(backend)
    else:
        raise RuntimeError("Unknown input value type.")

    return tensorplay._C._get_blas_preferred_backend()


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
cufft_plan_cache = cuFFTPlanCacheManager()
