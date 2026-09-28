"""Attention, and the ways it can be written for a device.

There is more than one way to write attention here, and they are not versions of
one thing.  What differs is how a mask is read: a mask is a predicate over
positions, a device that can compare several at once should be handed the ranges
rather than the comparisons, and a device that cannot should be handed the
comparisons one lane at a time.  Which is right is a fact about the device, not
a tuning decision -- so each way is its own module, and a caller that named none
is given the one that suits the device it is on.

  omni_attention   the two programs captured, the seventeen things a mask
                   carries, and the entry that decides which way to write
  omni_flash       the same computation for a device that can read a mask
                   several positions at a time
  omni_decoding    a forward with the query axis short, for one question at a
                   time
  omni_cpu         the same computation as a program the processor runs, with
                   the mask read while it runs

The modules here answer to one file each, and the split is the split: a device
that cannot be written for is not in this list at all, because a kernel for a
device with no blocks is not a slower answer but a wrong one.

Registering a lowering is done by being imported, and a lowering that is not
imported has not been said -- so these are pulled in here, where anything that
has attention has also got the ways of writing it.
"""

from __future__ import annotations

from .omni_attention import (
    OMNI_ATTENTION,
    OMNI_ATTENTION_BACKWARD,
    SubgraphResults,
    check_embedding_is_wide_enough,
    check_flash_supported_scalar_captures,
    create_omni_attention_kernel,
    get_float32_precision,
    guard_kernel_options,
    heads_are_grouped,
    lower_omni_attention,
    raise_omni_kernel_options_error,
    sanitize_kernel_options_for_triton,
    unpack_block_mask,
)
from .omni_cpu import (
    check_cpu_supported,
    lower_omni_attention_cpu,
)
from .omni_decoding import (
    OMNI_DECODING,
    create_omni_decoding_kernel,
    get_split_k,
    raise_omni_decoding_kernel_options_error,
    use_omni_decoding,
)
from .omni_flash_attention import (
    OmniFlashConfig,
    create_omni_flash_attention_backward_kernel,
    create_omni_flash_attention_kernel,
    _use_omni_flash_attention,
    _use_omni_flash_attention_backward,
)

__all__ = [
    "OMNI_ATTENTION",
    "OMNI_ATTENTION_BACKWARD",
    "OMNI_DECODING",
    "OmniFlashConfig",
    "SubgraphResults",
    "check_cpu_supported",
    "check_embedding_is_wide_enough",
    "check_flash_supported_scalar_captures",
    "create_omni_attention_kernel",
    "create_omni_decoding_kernel",
    "create_omni_flash_attention_backward_kernel",
    "create_omni_flash_attention_kernel",
    "get_float32_precision",
    "get_split_k",
    "guard_kernel_options",
    "heads_are_grouped",
    "lower_omni_attention",
    "lower_omni_attention_cpu",
    "raise_omni_decoding_kernel_options_error",
    "raise_omni_kernel_options_error",
    "sanitize_kernel_options_for_triton",
    "unpack_block_mask",
    "use_omni_decoding",
    "_use_omni_flash_attention",
    "_use_omni_flash_attention_backward",
]
