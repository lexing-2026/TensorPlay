# mypy: allow-untyped-defs
"""Caches for autoregressive decoding.

Two cache families are provided:

* :class:`DynamicCache` stores two tensors per layer (key and value), growing
  along the sequence dimension as tokens are processed. Attention variants that
  compress their key/value state, such as latent attention, store the
  compressed pair in the same slots.
* :class:`LinearStateCache` stores the bounded state of recurrent token mixers
  (linear attention family): a rolling convolution window and a fixed-shape
  recurrent matrix per layer.

Both caches grow lazily: nothing is allocated until a layer writes to them.
"""

from __future__ import annotations

from typing import NamedTuple

import tensorplay
from tensorplay import Tensor

__all__ = ["DynamicCache", "LinearStateCache"]


class _KVPair(NamedTuple):
    key: Tensor
    value: Tensor


class DynamicCache:
    """Append-only per-layer store of (key, value) tensor pairs.

    Tensors follow the attention convention ``[batch, heads, seq, head_dim]``;
    a variant may store whatever pair it consumes on the next call as long as
    both tensors share their leading dimensions. The cached sequence length is
    read from the first populated layer.
    """

    def __init__(self) -> None:
        self.key_cache: list[Tensor] = []
        self.value_cache: list[Tensor] = []

    def __len__(self) -> int:
        return len(self.key_cache)

    def __iter__(self):
        return iter(
            _KVPair(k, v) for k, v in zip(self.key_cache, self.value_cache, strict=True)
        )

    def update(self, key: Tensor, value: Tensor, layer_idx: int = 0) -> tuple[Tensor, Tensor]:
        """Append ``key``/``value`` for ``layer_idx`` and return the full state.

        The first call for a layer stores the tensors directly; later calls
        concatenate along the sequence dimension (dim 2).
        """
        if layer_idx >= len(self.key_cache):
            self.key_cache.extend([None] * (layer_idx + 1 - len(self.key_cache)))
            self.value_cache.extend([None] * (layer_idx + 1 - len(self.value_cache)))
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key
            self.value_cache[layer_idx] = value
        else:
            self.key_cache[layer_idx] = tensor_cat(self.key_cache[layer_idx], key)
            self.value_cache[layer_idx] = tensor_cat(self.value_cache[layer_idx], value)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Cached sequence length, 0 when the layer has not been written."""
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def reset(self) -> None:
        self.key_cache.clear()
        self.value_cache.clear()


def tensor_cat(a: Tensor, b: Tensor) -> Tensor:
    return tensorplay.cat([a, b], dim=2)


class LinearStateCache:
    """Bounded per-layer state for recurrent token mixers.

    ``conv_states[i]`` keeps the last ``kernel_size - 1`` inputs of layer ``i``'s
    causal convolution, shape ``[batch, conv_dim, kernel_size - 1]``.
    ``recurrent_states[i]`` keeps the mixer's recurrent matrix, shape
    ``[batch, heads, key_dim, value_dim]``. Layers opt in by writing with
    :meth:`update_conv_state` / :meth:`update_recurrent_state`; a layer absent
    from both maps has no previous state.
    """

    def __init__(self) -> None:
        self.conv_states: dict[int, Tensor] = {}
        self.recurrent_states: dict[int, Tensor] = {}

    def has_previous_state(self, layer_idx: int = 0) -> bool:
        return layer_idx in self.conv_states or layer_idx in self.recurrent_states

    def update_conv_state(self, layer_idx: int, conv_state: Tensor, kernel_size: int) -> Tensor:
        """Roll the convolution window forward and return the padded inputs.

        ``conv_state`` holds the new inputs ``[batch, conv_dim, seq]``; the
        stored window keeps the trailing ``kernel_size - 1`` columns so a
        single-token decode step can reconstruct its full receptive field.
        """
        previous = self.conv_states.get(layer_idx)
        if previous is not None and previous.shape[-1] == kernel_size - 1:
            conv_state = tensorplay.cat([previous, conv_state], dim=-1)
        self.conv_states[layer_idx] = conv_state[..., -(kernel_size - 1):]
        return conv_state

    def update_recurrent_state(self, layer_idx: int, state: Tensor) -> Tensor:
        self.recurrent_states[layer_idx] = state
        return state

    def reset(self) -> None:
        self.conv_states.clear()
        self.recurrent_states.clear()
