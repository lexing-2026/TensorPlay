# mypy: allow-untyped-defs
"""Caches for autoregressive decoding.

All caches share one protocol — ``update(key, value, layer_idx)`` returning the
full per-layer state, ``get_seq_length(layer_idx)``, and ``reset()`` — so the
generation loop and the attention modules are agnostic to which concrete cache
a model runs with.

Available families:

* :class:`DynamicCache` — append-only lists; unlimited length, grows every step.
* :class:`StaticCache` — one preallocated buffer per layer with a write cursor;
  memory is fixed at creation and exceeding it raises.
* :class:`SlidingWindowCache` — ring buffer keeping the most recent
  ``window_size`` positions; older entries are overwritten in place.
* :class:`PagedCache` — fixed-size blocks with a per-row block table, the
  layout paged-attention kernels consume; the Python path gathers a contiguous
  view on read.
* :class:`QuantizedCache` — wraps another cache and stores the key/value
  tensors as int8 with per-position scales, dequantizing on read.

Recurrent token mixers (the linear attention family) keep a different kind of
state: a rolling convolution window and a fixed-shape recurrent matrix. That
lives in :class:`LinearStateCache`, which follows the same protocol surface
where it applies.
"""

from __future__ import annotations

from typing import NamedTuple

import tensorplay
from tensorplay import Tensor

__all__ = [
    "Cache",
    "DynamicCache",
    "StaticCache",
    "SlidingWindowCache",
    "PagedCache",
    "QuantizedCache",
    "LinearStateCache",
]


class _KVPair(NamedTuple):
    key: Tensor
    value: Tensor


class Cache:
    """Protocol surface shared by every key/value cache."""

    def update(self, key: Tensor, value: Tensor, layer_idx: int = 0) -> tuple[Tensor, Tensor]:
        """Store the step's key/value for ``layer_idx`` and return the state
        the layer should attend over."""
        raise NotImplementedError

    def get_seq_length(self, layer_idx: int = 0) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


def tensor_cat(a: Tensor, b: Tensor) -> Tensor:
    return tensorplay.cat([a, b], dim=2)


class DynamicCache(Cache):
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
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
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


class StaticCache(Cache):
    """Preallocated per-layer buffers with a write cursor.

    Buffers are allocated lazily on the first ``update`` from the incoming
    tensor's shape, so callers only need to know ``max_seq_len``. Writes go to
    ``buffer[:, :, cursor:cursor + seq]``; reads return the populated prefix.
    Exceeding ``max_seq_len`` raises instead of growing.
    """

    def __init__(self, max_seq_len: int) -> None:
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.max_seq_len = max_seq_len
        self.key_cache: list[Tensor | None] = []
        self.value_cache: list[Tensor | None] = []
        self._cursor: list[int] = []

    def __len__(self) -> int:
        return len(self.key_cache)

    def update(self, key: Tensor, value: Tensor, layer_idx: int = 0) -> tuple[Tensor, Tensor]:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self._cursor.append(0)
        batch, heads = key.shape[0], key.shape[1]
        if self.key_cache[layer_idx] is None:
            head_dim = key.shape[3]
            v_head_dim = value.shape[3]
            self.key_cache[layer_idx] = tensorplay.zeros(
                batch, heads, self.max_seq_len, head_dim, dtype=key.dtype, device=key.device
            )
            self.value_cache[layer_idx] = tensorplay.zeros(
                batch, heads, self.max_seq_len, v_head_dim, dtype=value.dtype, device=value.device
            )
        seq = key.shape[2]
        start = self._cursor[layer_idx]
        if start + seq > self.max_seq_len:
            raise RuntimeError(
                f"StaticCache overflow: cursor {start} + {seq} exceeds max_seq_len {self.max_seq_len}"
            )
        self.key_cache[layer_idx][:, :, start : start + seq].copy_(key)
        self.value_cache[layer_idx][:, :, start : start + seq].copy_(value)
        self._cursor[layer_idx] = start + seq
        return (
            self.key_cache[layer_idx][:, :, : start + seq],
            self.value_cache[layer_idx][:, :, : start + seq],
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._cursor):
            return 0
        return self._cursor[layer_idx]

    def reset(self) -> None:
        self.key_cache.clear()
        self.value_cache.clear()
        self._cursor.clear()


class SlidingWindowCache(Cache):
    """Ring buffer retaining only the most recent ``window_size`` positions.

    Reads return the retained entries oldest-first; :attr:`positions` (per
    layer) gives their absolute positions so an attention module can build the
    exact within-window mask. A decode step (single query against a full
    window) needs no extra masking because every retained key is within the
    window of the newest query.
    """

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.window_size = window_size
        self.key_cache: list[Tensor | None] = []
        self.value_cache: list[Tensor | None] = []
        self.positions: list[Tensor | None] = []
        self._seen: list[int] = []

    def __len__(self) -> int:
        return len(self.key_cache)

    def update(self, key: Tensor, value: Tensor, layer_idx: int = 0) -> tuple[Tensor, Tensor]:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self.positions.append(None)
            self._seen.append(0)
        seq = key.shape[2]
        first_new = self._seen[layer_idx]
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key
            self.value_cache[layer_idx] = value
            self.positions[layer_idx] = tensorplay.arange(
                first_new, first_new + seq, device=key.device
            ).expand(key.shape[0], -1)
        else:
            self.key_cache[layer_idx] = tensor_cat(self.key_cache[layer_idx], key)[
                :, :, -self.window_size :
            ]
            self.value_cache[layer_idx] = tensor_cat(self.value_cache[layer_idx], value)[
                :, :, -self.window_size :
            ]
            new_positions = tensorplay.arange(
                first_new, first_new + seq, device=key.device
            ).expand(key.shape[0], -1)
            self.positions[layer_idx] = tensorplay.cat(
                [self.positions[layer_idx], new_positions], dim=1
            )[:, -self.window_size :]
        self._seen[layer_idx] = first_new + seq
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Number of tokens ever seen; the buffer itself is capped at
        ``window_size`` — use ``positions`` to locate the stored ones."""
        if layer_idx >= len(self._seen):
            return 0
        return self._seen[layer_idx]

    def get_window_length(self, layer_idx: int = 0) -> int:
        """Positions currently retained for ``layer_idx``, 0 before the first write."""
        if layer_idx >= len(self.positions) or self.positions[layer_idx] is None:
            return 0
        return int(self.positions[layer_idx].shape[1])

    def reset(self) -> None:
        self.key_cache.clear()
        self.value_cache.clear()
        self.positions.clear()
        self._seen.clear()


class PagedCache(Cache):
    """Block-paged storage with per-row block tables.

    Keys/values live in preallocated ``[rows * blocks_per_row, block_size,
    heads, head_dim]`` pools; each row's logical sequence is described by its
    block table. Rows never share blocks — row ``b`` owns the contiguous
    ``blocks_per_row`` range starting at ``b * blocks_per_row``. ``update``
    returns a gathered contiguous view so plain attention can consume it; the
    pools and :attr:`block_tables` are also exposed for paged-attention
    kernels, which can read them without the gather. The batch size is fixed
    by the first ``update`` per layer; capacity (blocks per row) grows on
    demand up to ``num_blocks``.
    """

    def __init__(self, block_size: int = 16, num_blocks: int | None = None) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.key_pools: list[Tensor | None] = []
        self.value_pools: list[Tensor | None] = []
        self.block_tables: list[Tensor | None] = []
        self._seq_len: list[int] = []
        self._blocks_per_row: list[int] = []

    def __len__(self) -> int:
        return len(self.key_pools)

    def _ensure_capacity(self, layer_idx: int, batch: int, needed: int, key: Tensor, value: Tensor) -> None:
        have = self._blocks_per_row[layer_idx] if layer_idx < len(self._blocks_per_row) else 0
        if self.num_blocks is not None and needed > self.num_blocks:
            raise RuntimeError(
                f"PagedCache overflow: {needed} blocks per row needed, pool holds {self.num_blocks}"
            )
        if needed <= have:
            return
        head_dim, v_head_dim = key.shape[3], value.shape[3]
        heads = key.shape[1]
        dtype, device = key.dtype, key.device
        new_k = tensorplay.zeros(
            batch * needed, self.block_size, heads, head_dim, dtype=dtype, device=device
        )
        new_v = tensorplay.zeros(
            batch * needed, self.block_size, heads, v_head_dim, dtype=dtype, device=device
        )
        if have > 0:
            # Migrate each row's blocks into its new, wider range.
            old_k, old_v = self.key_pools[layer_idx], self.value_pools[layer_idx]
            for b in range(batch):
                new_k[b * needed : b * needed + have].copy_(old_k[b * have : b * have + have])
                new_v[b * needed : b * needed + have].copy_(old_v[b * have : b * have + have])
        self.key_pools[layer_idx] = new_k
        self.value_pools[layer_idx] = new_v
        self._blocks_per_row[layer_idx] = needed
        table = tensorplay.arange(needed, device=device, dtype=tensorplay.int64)
        self.block_tables[layer_idx] = (
            table.unsqueeze(0) + tensorplay.arange(batch, device=device, dtype=tensorplay.int64).unsqueeze(1) * needed
        )

    def update(self, key: Tensor, value: Tensor, layer_idx: int = 0) -> tuple[Tensor, Tensor]:
        while len(self.key_pools) <= layer_idx:
            self.key_pools.append(None)
            self.value_pools.append(None)
            self.block_tables.append(None)
            self._seq_len.append(0)
            self._blocks_per_row.append(0)
        batch, heads = key.shape[0], key.shape[1]
        seq = key.shape[2]
        start = self._seq_len[layer_idx]
        needed = (start + seq + self.block_size - 1) // self.block_size
        self._ensure_capacity(layer_idx, batch, needed, key, value)
        pool_k, pool_v = self.key_pools[layer_idx], self.value_pools[layer_idx]
        flat_k = pool_k.view(-1, *pool_k.shape[2:])
        flat_v = pool_v.view(-1, *pool_v.shape[2:])
        table = self.block_tables[layer_idx]

        # Scatter the new tokens into their paged slots: logical token t of row
        # b lands in block table[b, t // block_size] at offset t % block_size.
        token_index = tensorplay.arange(start, start + seq, device=key.device)
        slots = table[:, token_index // self.block_size] * self.block_size + token_index % self.block_size
        flat_k[slots.reshape(-1)] = key.transpose(1, 2).reshape(-1, heads, key.shape[3])
        flat_v[slots.reshape(-1)] = value.transpose(1, 2).reshape(-1, heads, value.shape[3])
        self._seq_len[layer_idx] = start + seq

        # Gather a contiguous view: expand each table entry into its
        # block_size consecutive pool slots, then read them in order.
        slot_ids = (
            table.unsqueeze(-1) * self.block_size
            + tensorplay.arange(self.block_size, device=table.device).view(1, 1, -1)
        ).reshape(-1)
        gathered_k = flat_k[slot_ids].reshape(batch, -1, heads, key.shape[3])
        gathered_v = flat_v[slot_ids].reshape(batch, -1, heads, value.shape[3])
        return (
            gathered_k[:, : start + seq].transpose(1, 2),
            gathered_v[:, : start + seq].transpose(1, 2),
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._seq_len):
            return 0
        return self._seq_len[layer_idx]

    def reset(self) -> None:
        self.key_pools.clear()
        self.value_pools.clear()
        self.block_tables.clear()
        self._seq_len.clear()
        self._blocks_per_row.clear()


class QuantizedCache(Cache):
    """Stores key/value tensors as int8 codes with per-position float scales.

    Each ``(batch, head, position)`` vector is symmetrically quantized over its
    ``head_dim`` axis; reads dequantize back to float32 so the returned tensors
    stay attention-ready. Storage shrinks roughly 4x for float32 states. Lossy
    by construction — only use it where the accuracy trade-off is acceptable.
    """

    def __init__(self) -> None:
        self._key_codes: list[list[tuple[Tensor, Tensor]]] = []
        self._value_codes: list[list[tuple[Tensor, Tensor]]] = []

    def __len__(self) -> int:
        return len(self._key_codes)

    def _quantize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        codes = (x / scale).round().clamp(-127, 127).to(tensorplay.int8)
        return codes, scale

    @staticmethod
    def _cat_stored(stored: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        codes = tensorplay.cat([c for c, _ in stored], dim=2)
        scales = tensorplay.cat([s for _, s in stored], dim=2)
        return codes, scales

    def update(self, key: Tensor, value: Tensor, layer_idx: int = 0) -> tuple[Tensor, Tensor]:
        while len(self._key_codes) <= layer_idx:
            self._key_codes.append([])
            self._value_codes.append([])
        self._key_codes[layer_idx].append(self._quantize(key))
        self._value_codes[layer_idx].append(self._quantize(value))
        k_codes, k_scales = self._cat_stored(self._key_codes[layer_idx])
        v_codes, v_scales = self._cat_stored(self._value_codes[layer_idx])
        return (
            k_codes.to(tensorplay.float32) * k_scales,
            v_codes.to(tensorplay.float32) * v_scales,
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._key_codes) or not self._key_codes[layer_idx]:
            return 0
        return sum(codes.shape[2] for codes, _ in self._key_codes[layer_idx])

    def reset(self) -> None:
        self._key_codes.clear()
        self._value_codes.clear()


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
