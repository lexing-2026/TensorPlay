"""Backend selection controls for scaled dot product attention.

The public entry point is :func:`sdpa_kernel`, a context manager that restricts
which attention backends :func:`tensorplay.nn.functional.scaled_dot_product_attention`
may route to while the context is active. Backends are identified by members of
the :class:`SDPBackend` flag enum.
"""

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntFlag
from typing import Optional, Union
from warnings import warn

import tensorplay
from tensorplay import Tensor

__all__ = [
    "SDPBackend",
    "SDPParams",
    "sdpa_kernel",
    "WARN_FOR_UNFUSED_KERNELS",
    "_sdpa_kernel_variadic",
    "_backend_from_string",
    "_cur_sdpa_kernel_backends",
    "can_use_flash_attention",
    "can_use_efficient_attention",
]


class SDPBackend(IntFlag):
    r"""Backends available to scaled dot product attention.

    - ``ERROR``: backend selection failed.
    - ``MATH``: composed reference implementation.
    - ``FLASH_ATTENTION``: fused flash-attention kernel.
    - ``EFFICIENT_ATTENTION``: memory-efficient fused kernel.
    - ``CUDNN_ATTENTION``: cuDNN fused kernel.
    - ``OVERRIDEABLE``: reserved for external overrides.
    """

    ERROR = -1
    MATH = 0
    FLASH_ATTENTION = 1
    EFFICIENT_ATTENTION = 2
    CUDNN_ATTENTION = 3
    OVERRIDEABLE = 4


@dataclass
class SDPParams:
    """Parameter bundle describing one scaled dot product attention call.

    Passed to the ``can_use_*`` predicates so they can judge kernel
    eligibility without re-plumbing every argument.
    """

    query: Tensor
    key: Tensor
    value: Tensor
    attn_mask: Optional[Tensor]
    dropout: float
    is_causal: bool
    need_attn_weights: bool = False


# When True, a call that silently falls back to a non-fused path emits a
# warning describing why each fused backend was rejected.
WARN_FOR_UNFUSED_KERNELS = False

# Everything is routable until a sdpa_kernel context narrows the set.
_enabled_backends: set = {
    SDPBackend.MATH,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
}

# Higher entries are tried first when the caller lets the library route.
_priority_order: list = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

_backend_names = {
    "flash": "FLASH_ATTENTION",
    "mem_efficient": "EFFICIENT_ATTENTION",
    "math": "MATH",
    "cudnn": "CUDNN_ATTENTION",
    "overrideable": "OVERRIDEABLE",
}

_name_to_string_name = {v: k for k, v in _backend_names.items()}

_sdpa_float_dtypes = (
    tensorplay.DType.float16,
    tensorplay.DType.bfloat16,
    tensorplay.DType.float32,
    tensorplay.DType.float64,
)


def _backend_from_string(name: str):
    return getattr(SDPBackend, name)


def _cur_sdpa_kernel_backends(with_priority: bool = False):
    backends = [b for b in _priority_order if b in _enabled_backends]
    if with_priority:
        return list(backends)
    return sorted(backends, key=lambda b: int(b))


def _sdpa_kernel(backends: Iterable, set_priority: bool = False) -> None:
    global _enabled_backends, _priority_order
    backends = list(backends)
    _enabled_backends = set(backends)
    if set_priority:
        user_priority = [b for b in backends]
        for b in _priority_order:
            if b not in user_priority:
                user_priority.append(b)
        _priority_order = user_priority


@contextlib.contextmanager
def sdpa_kernel(
    backends: Union[list, SDPBackend], set_priority: bool = False
):
    r"""Restrict the backends usable by scaled dot product attention.

    Args:
        backends: a single :class:`SDPBackend` or a list of them. With
            ``set_priority=True`` the list order is interpreted as the
            routing priority order.
    """
    if not isinstance(backends, (list, SDPBackend)):
        raise AssertionError(
            f"Backend must be an instance of SDPBackend or a list of "
            f"SDPBackend instances, got {type(backends).__name__}"
        )
    if isinstance(backends, SDPBackend):
        backends = [backends]
    backends = list(dict.fromkeys(backends))

    previous_backends = _cur_sdpa_kernel_backends(with_priority=set_priority)
    try:
        _sdpa_kernel(backends, set_priority)
        yield {}
    finally:
        _sdpa_kernel(previous_backends, set_priority)


@contextlib.contextmanager
def _sdpa_kernel_variadic(*backends: SDPBackend):
    with sdpa_kernel(list(backends)):
        yield


def _check_flash_eligibility(params: SDPParams, debug: bool = False):
    q, k, v = params.query, params.key, params.value
    if params.attn_mask is not None:
        return False, "attn_mask is not supported by the fused kernel"
    if params.dropout != 0.0:
        return False, "dropout is not supported by the fused kernel"
    if q.dim() != 4:
        return False, "query/key/value must be 4D [B, H, T, D]"
    if k.dim() != 4 or v.dim() != 4:
        return False, "query/key/value must be 4D [B, H, T, D]"
    if q.dtype not in _sdpa_float_dtypes:
        return False, f"unsupported dtype {q.dtype}"
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return False, "query/key/value must share one dtype"
    if k.shape != v.shape or k.shape[:-1] != q.shape[:-1]:
        return False, "query/key/value leading shapes must match"
    if params.need_attn_weights:
        return False, "attention weights are not produced by the fused kernel"
    return True, ""


def can_use_flash_attention(params: SDPParams, debug: bool = False) -> bool:
    r"""Whether the fused flash-attention kernel can run this call."""
    ok, reason = _check_flash_eligibility(params)
    if not ok and debug and reason:
        warn(f"Flash attention can't be used because: {reason}", stacklevel=2)
    return ok


def can_use_efficient_attention(params: SDPParams, debug: bool = False) -> bool:
    r"""Whether a memory-efficient fused kernel can run this call.

    This build ships no memory-efficient kernel, so the answer is always
    False; the debug flag reports the reason.
    """
    if debug:
        warn(
            "Efficient attention can't be used because: no memory-efficient "
            "kernel is available in this build",
            stacklevel=2,
        )
    return False


def _raise_kernel_warnings(params: SDPParams) -> None:
    if WARN_FOR_UNFUSED_KERNELS:
        if not can_use_efficient_attention(params):
            warn("Efficient attention can't be used because:", stacklevel=2)
            can_use_efficient_attention(params, True)
        if not can_use_flash_attention(params):
            warn("Flash attention can't be used because:", stacklevel=2)
            can_use_flash_attention(params, True)
