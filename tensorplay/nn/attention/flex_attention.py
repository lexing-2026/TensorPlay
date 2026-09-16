# mypy: allow-untyped-defs
"""Block-sparse attention surface.

The block-sparse scheduling kernels are not part of this build.  The module
exposes the container and entry points so importers can bind the names
unconditionally; invoking the entry points raises until the sparse kernels
land.  The default block size matches the dense tiling the fused attention
kernels use for 128-wide heads.
"""

__all__ = [
    "BlockMask",
    "create_block_mask",
    "flex_attention",
    "_DEFAULT_SPARSE_BLOCK_SIZE",
]

_DEFAULT_SPARSE_BLOCK_SIZE = 128


class BlockMask:
    """Block-sparse attention schedule.

    Holds the per-query-block key-block layout (selected blocks, per-block
    counts, and their fully-masked counterparts) plus the sequence lengths
    and block-size bookkeeping the schedule was built from.  Constructing
    the schedule itself requires the sparse kernels, which this build
    excludes.
    """

    __slots__ = (
        "seq_lengths",
        "kv_indices",
        "kv_num_blocks",
        "full_kv_indices",
        "full_kv_num_blocks",
        "qv_indices",
        "qv_num_blocks",
        "full_qv_indices",
        "full_qv_num_blocks",
        "BLOCK_SIZE",
        "BLOCK_SIZE_CTRL_DEP",
        "MAX_BLOCKS_PER_ROW",
        "kv_block_size",
        "q_block_size",
        "mask_map",
        "seq_idx",
        "q_idx",
        "kv_idx",
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name))

    def __repr__(self):
        filled = [n for n in self.__slots__ if getattr(self, n, None) is not None]
        return f"BlockMask(populated={filled})"


def create_block_mask(mask_mod, B=None, H=None, Q_LEN=None, KV_LEN=None, device=None, **kwargs):
    """Compile ``mask_mod`` into a :class:`BlockMask` schedule."""
    raise NotImplementedError(
        "create_block_mask requires the block-sparse scheduling kernels, "
        "which are not available in this build"
    )


def flex_attention(query, key, value, score_mod=None, block_mask=None, **kwargs):
    """Scaled dot product attention with per-score modifiers and sparsity."""
    raise NotImplementedError(
        "flex_attention requires the block-sparse fused kernels, "
        "which are not available in this build"
    )


class AuxRequest:
    """Side-channel outputs requested alongside a flex attention call."""

    __slots__ = ("request_indices", "get")
