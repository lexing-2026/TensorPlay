# mypy: allow-untyped-defs
"""Rotary position embedding (RoPE).

The rotation applied to query/key vectors depends only on the *relative*
distance between positions, so scores become translation invariant. Two
dimension layouts are supported:

* ``interleaved=False`` — the first half of the head rotates against the second
  half (the "NeoX" layout).
* ``interleaved=True`` — even dimensions rotate against their odd successors
  (the "GPT-J" layout).

The cos/sin tables are always built in float32 (building them in a reduced
precision loses too much accuracy at long distances) and indexed by caller
supplied ``position_ids``, which keeps the module usable with a KV cache where
each decode step sits at an arbitrary offset.
"""

from __future__ import annotations

import tensorplay
from tensorplay import Tensor

from .module import Module

__all__ = ["RotaryEmbedding", "apply_rotary_emb"]


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return tensorplay.cat((-x2, x1), dim=-1)


def _rotate_pairs(x: Tensor) -> Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return tensorplay.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rotary_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleaved: bool = False,
) -> tuple[Tensor, Tensor]:
    """Rotate the trailing ``2 * cos.shape[-1]`` dimensions of ``q`` and ``k``.

    Args:
        q: ``[batch, heads, seq, head_dim]`` (any leading dims are fine as long
            as the last dimension is the head).
        k: same layout as ``q`` (heads may differ, e.g. a single shared key).
        cos, sin: ``[batch, seq, rotary_dim / 2]`` float tables.
        interleaved: pair layout, matching the tables' source module.

    Returns:
        Rotated ``(q, k)`` with the original dtypes preserved.
    """
    rotate = _rotate_pairs if interleaved else _rotate_half
    # Broadcast the [batch, seq, d] tables across the head dimension and to the
    # full rotary width: tiled halves for the NeoX layout, interleaved pairs
    # for the GPT-J layout.
    if interleaved:
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
    else:
        cos = tensorplay.cat([cos, cos], dim=-1).unsqueeze(1)
        sin = tensorplay.cat([sin, sin], dim=-1).unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_tail = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_tail = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = q_rot * cos + rotate(q_rot) * sin
    k_rot = k_rot * cos + rotate(k_rot) * sin
    return (
        tensorplay.cat([q_rot, q_tail], dim=-1) if rotary_dim < q.shape[-1] else q_rot,
        tensorplay.cat([k_rot, k_tail], dim=-1) if rotary_dim < k.shape[-1] else k_rot,
    )


class RotaryEmbedding(Module):
    """Builds cos/sin tables for rotary position embeddings.

    Args:
        dim: rotation span in dimensions; must be even and is usually
            ``head_dim`` (or the rope slice of it).
        base: geometric progression base of the inverse frequencies.
        interleaved: pair layout for :func:`apply_rotary_emb`.
    """

    def __init__(self, dim: int, base: float = 10000.0, interleaved: bool = False, device=None) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError(f"rotary dim must be even, got {dim}")
        self.dim = dim
        self.base = float(base)
        self.interleaved = interleaved
        inv_freq = 1.0 / (
            self.base ** (tensorplay.arange(0, dim, 2, device=device, dtype=tensorplay.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def _update_cache(self, max_position: int, device, dtype) -> None:
        if (
            max_position <= self._seq_len_cached
            and self._cos_cached is not None
            and self._cos_cached.device == device
            and self._cos_cached.dtype == tensorplay.float32
        ):
            return
        positions = tensorplay.arange(max_position, device=device, dtype=tensorplay.float32)
        freqs = tensorplay.outer(positions, self.inv_freq.to(device))
        self._cos_cached = tensorplay.cos(freqs)
        self._sin_cached = tensorplay.sin(freqs)
        self._seq_len_cached = max_position

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Tables for the given positions.

        Args:
            x: any tensor on the target device; only its dtype and device are
                consulted.
            position_ids: ``[batch, seq]`` integer positions.

        Returns:
            ``(cos, sin)`` of shape ``[batch, seq, dim / 2]`` cast to ``x``'s
            dtype.
        """
        max_position = int(position_ids.max().item()) + 1
        self._update_cache(max_position, x.device, x.dtype)
        cos = self._cos_cached[position_ids]
        sin = self._sin_cached[position_ids]
        return cos.to(x.dtype), sin.to(x.dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, interleaved={self.interleaved}"
