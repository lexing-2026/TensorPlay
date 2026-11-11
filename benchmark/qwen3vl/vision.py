"""Vision tower of Qwen3-VL, built out of the framework's own modules.

The tower turns a flat batch of patches into one embedding per merged patch
group: a three-dimensional patch projection, a bilinear lookup in the learned
position table, 27 transformer blocks whose attention runs separately inside
each image, and a patch merger that folds ``spatial_merge_size ** 2`` tokens
into one.  Three intermediate blocks also feed their merged features forward
as "deepstack" features, which the language tower adds back at its own early
layers.

Attribute names mirror the checkpoint (``blocks.0.attn.qkv.weight``), so a
weight can be filed under the name it carries in the file:

    model.visual.blocks.0.attn.qkv.weight

The position bookkeeping — the bilinear interpolation of the learned table,
the rotary frequency lookup, and the token reordering that puts the merge
groups next to each other — is pure index arithmetic over the grid, so it is
computed with numpy and only the resulting gather is left to the device.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

import tensorplay as tp
from tensorplay import nn

__all__ = [
    "Qwen3VLVisionConfig",
    "Qwen3VLVisionModel",
    "load_vision",
    "vision_checkpoint_name",
]


def _factory(device, dtype):
    return {"device": device, "dtype": dtype}


class Qwen3VLVisionConfig:
    """The vision-side shape read out of ``config.json``."""

    def __init__(self, raw: dict):
        visual = raw.get("vision_config", raw)
        self.hidden_size = int(visual["hidden_size"])
        self.intermediate_size = int(visual["intermediate_size"])
        self.depth = int(visual["depth"])
        self.num_heads = int(visual["num_heads"])
        self.in_channels = int(visual["in_channels"])
        self.out_hidden_size = int(visual["out_hidden_size"])
        self.patch_size = int(visual["patch_size"])
        self.temporal_patch_size = int(visual["temporal_patch_size"])
        self.spatial_merge_size = int(visual["spatial_merge_size"])
        self.num_position_embeddings = int(visual["num_position_embeddings"])
        self.deepstack_visual_indexes = [int(i) for i in visual["deepstack_visual_indexes"]]
        # The feed-forward gate is the hyperbolic-tangent form of the Gaussian
        # error curve, which the block builds in place.
        self.head_dim = self.hidden_size // self.num_heads

    @classmethod
    def from_dir(cls, model_dir: str) -> "Qwen3VLVisionConfig":
        with open(os.path.join(model_dir, "config.json")) as handle:
            return cls(json.load(handle))

    def __repr__(self) -> str:
        return (
            f"Qwen3VLVisionConfig(hidden={self.hidden_size}, depth={self.depth}, "
            f"heads={self.num_heads}, head_dim={self.head_dim}, "
            f"patch={self.temporal_patch_size}x{self.patch_size}x{self.patch_size}, "
            f"merge={self.spatial_merge_size}, deepstack={self.deepstack_visual_indexes})"
        )


def interpolate_position_table(table: np.ndarray, grid_thw, merge_size: int, side: int) -> np.ndarray:
    """Bilinear lookup of the learned table, in the order the blocks consume.

    ``table`` is ``(side * side, hidden)``.  Each (t, h, w) row of the grid
    picks ``h`` and ``w`` sample points spread over the side of the table,
    takes the four surrounding entries and weights them by the fractional
    distance.  The per-frame result is repeated over the frames and then
    reordered so the ``merge_size ** 2`` tokens that share a merged position
    sit next to each other.
    """
    pieces = []
    for frames, height, width in grid_thw:
        rows = np.linspace(0, side - 1, height)
        cols = np.linspace(0, side - 1, width)
        rows_lo = rows.astype(np.int64)
        cols_lo = cols.astype(np.int64)
        rows_hi = np.minimum(rows_lo + 1, side - 1)
        cols_hi = np.minimum(cols_lo + 1, side - 1)
        dh = rows - rows_lo
        dw = cols - cols_lo

        base_lo = rows_lo * side
        base_hi = rows_hi * side
        corners = [
            (base_lo[:, None] + cols_lo[None, :]).reshape(-1),
            (base_lo[:, None] + cols_hi[None, :]).reshape(-1),
            (base_hi[:, None] + cols_lo[None, :]).reshape(-1),
            (base_hi[:, None] + cols_hi[None, :]).reshape(-1),
        ]
        weights = [
            ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
            ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
            (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
            (dh[:, None] * dw[None, :]).reshape(-1),
        ]
        pos = np.zeros_like(table[:1], dtype=np.float64)
        pos = table[corners[0]].astype(np.float64) * weights[0][:, None]
        for idx, weight in zip(corners[1:], weights[1:]):
            pos += table[idx].astype(np.float64) * weight[:, None]

        pos = np.tile(pos, (frames, 1))
        pos = pos.reshape(
            frames, height // merge_size, merge_size, width // merge_size, merge_size, -1
        )
        pieces.append(np.transpose(pos, (0, 1, 3, 2, 4, 5)).reshape(-1, table.shape[1]))
    return np.concatenate(pieces, axis=0)


def vision_rotary_frequencies(
    grid_thw, merge_size: int, head_dim: int, theta: float = 10000.0
) -> np.ndarray:
    """Rotary frequencies per patch: half of the head repeated twice.

    Every patch is addressed by a fused (row, column) index across all images;
    the frequency table is indexed by the larger of the two grid sides, so one
    table covers every sample point.  Frames of the same image share the
    spatial positions, which is what repeating the coordinate block does.
    """
    rotary_dim = head_dim // 2
    inverse = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
    side = max(max(h, w) for _, h, w in grid_thw)
    table = np.outer(np.arange(side, dtype=np.float32), inverse)

    tokens = []
    for frames, height, width in grid_thw:
        merged_h, merged_w = height // merge_size, width // merge_size
        blocks = np.arange(merged_h, dtype=np.int64)[:, None] * merge_size + np.arange(
            merge_size, dtype=np.int64
        )[None, :]
        columns = np.arange(merged_w, dtype=np.int64)[:, None] * merge_size + np.arange(
            merge_size, dtype=np.int64
        )[None, :]
        rows = np.broadcast_to(blocks[:, None, :, None], (merged_h, merged_w, merge_size, merge_size))
        cols = np.broadcast_to(columns[None, :, None, :], (merged_h, merged_w, merge_size, merge_size))
        coords = np.stack([rows.reshape(-1), cols.reshape(-1)], axis=-1)
        if frames > 1:
            coords = np.tile(coords, (frames, 1))
        tokens.append(table[coords].reshape(coords.shape[0], -1))
    freqs = np.concatenate(tokens, axis=0)
    return np.concatenate([freqs, freqs], axis=-1)


def patch_segments(grid_thw) -> np.ndarray:
    """Cumulative token counts: attention never crosses an image boundary."""
    lengths = [frames * height * width for frames, height, width in grid_thw]
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


class Qwen3VLVisionPatchEmbed(nn.Module):
    """Patch projection: a convolution whose kernel and stride are the patch."""

    def __init__(self, config: Qwen3VLVisionConfig, device=None, dtype=None):
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_size
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(
            config.in_channels, config.hidden_size, kernel_size=kernel, stride=kernel,
            bias=True, **_factory(device, dtype),
        )

    def forward(self, hidden_states: tp.Tensor) -> tp.Tensor:
        patches = hidden_states.reshape(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(patches).reshape(-1, self.embed_dim)


class Qwen3VLVisionMLP(nn.Module):
    """Feed-forward with the tanh approximation of the Gaussian error curve."""

    def __init__(self, config: Qwen3VLVisionConfig, device=None, dtype=None):
        super().__init__()
        self.linear_fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True, **_factory(device, dtype)
        )
        self.linear_fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True, **_factory(device, dtype)
        )
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, hidden_states: tp.Tensor) -> tp.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class Qwen3VLVisionAttention(nn.Module):
    """Full attention within each image, with rotary embedding on queries and keys."""

    def __init__(self, config: Qwen3VLVisionConfig, device=None, dtype=None):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scaling = config.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True, **_factory(device, dtype))
        self.proj = nn.Linear(self.dim, self.dim, bias=True, **_factory(device, dtype))

    @staticmethod
    def _rotate_half(x: tp.Tensor) -> tp.Tensor:
        half = x.shape[-1] // 2
        return tp.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, hidden_states, segments, position_embeddings):
        seq_len = hidden_states.shape[0]
        packed = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = packed.permute(1, 0, 2, 3).unbind(0)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        # The rotation is applied in single precision, as the reference does.
        q = query.float()
        k = key.float()
        query = ((q * cos) + (self._rotate_half(q) * sin)).to(query.dtype)
        key = ((k * cos) + (self._rotate_half(k) * sin)).to(key.dtype)

        outputs = []
        for start, end in zip(segments[:-1], segments[1:]):
            qs = query[int(start) : int(end)].transpose(0, 1).unsqueeze(0)
            ks = key[int(start) : int(end)].transpose(0, 1).unsqueeze(0)
            vs = value[int(start) : int(end)].transpose(0, 1).unsqueeze(0)
            weights = tp.softmax((qs @ ks.transpose(-1, -2)) * self.scaling, dim=-1)
            outputs.append((weights @ vs).squeeze(0).transpose(0, 1))
        attended = outputs[0] if len(outputs) == 1 else tp.cat(outputs, dim=0)
        return self.proj(attended.reshape(seq_len, -1).contiguous())


class Qwen3VLVisionBlock(nn.Module):
    """One pre-norm residual block: attention, then feed-forward."""

    def __init__(self, config: Qwen3VLVisionConfig, device=None, dtype=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            config.hidden_size, eps=1e-6, elementwise_affine=True, **_factory(device, dtype)
        )
        self.norm2 = nn.LayerNorm(
            config.hidden_size, eps=1e-6, elementwise_affine=True, **_factory(device, dtype)
        )
        self.attn = Qwen3VLVisionAttention(config, device=device, dtype=dtype)
        self.mlp = Qwen3VLVisionMLP(config, device=device, dtype=dtype)

    def forward(self, hidden_states, segments, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), segments, position_embeddings
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3VLVisionPatchMerger(nn.Module):
    """Fold ``spatial_merge_size ** 2`` neighbouring patches into one token.

    The plain merger normalizes each patch before folding; the deepstack
    variant folds first and normalizes the merged vector.
    """

    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False,
                 device=None, dtype=None):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size,
            eps=1e-6, elementwise_affine=True, **_factory(device, dtype),
        )
        self.linear_fc1 = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True, **_factory(device, dtype)
        )
        self.linear_fc2 = nn.Linear(
            self.hidden_size, config.out_hidden_size, bias=True, **_factory(device, dtype)
        )
        self.act_fn = nn.GELU()

    def forward(self, x: tp.Tensor) -> tp.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.hidden_size))
        else:
            x = self.norm(x)
        x = x.reshape(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VLVisionModel(nn.Module):
    """The vision tower: patch projection, 27 blocks, merger and deepstack mergers."""

    def __init__(self, config: Qwen3VLVisionConfig, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        self.patch_embed = Qwen3VLVisionPatchEmbed(config, device=device, dtype=dtype)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size, **_factory(device, dtype)
        )
        self.blocks = nn.ModuleList(
            [Qwen3VLVisionBlock(config, device=device, dtype=dtype) for _ in range(config.depth)]
        )
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False,
                                               device=device, dtype=dtype)
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True,
                                         device=device, dtype=dtype)
                for _ in config.deepstack_visual_indexes
            ]
        )

    def forward(self, hidden_states: tp.Tensor, grid_thw):
        """Run the tower over a flat patch batch.

        ``hidden_states`` is ``(patches, channels * t * p * p)`` and
        ``grid_thw`` is one ``(t, h, w)`` row per image, where ``h`` and ``w``
        count the tokens one frame is cut into.
        """
        grid_thw = [(int(t), int(h), int(w)) for t, h, w in grid_thw]
        hidden_states = self.patch_embed(hidden_states)

        table = self.pos_embed.weight.detach().cpu().numpy()
        positions = interpolate_position_table(
            table, grid_thw, self.spatial_merge_size, self.num_grid_per_side
        )
        hidden_states = hidden_states + tp.from_numpy(positions).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        frequencies = vision_rotary_frequencies(
            grid_thw, self.spatial_merge_size, self.config.head_dim
        )
        cos = tp.from_numpy(np.cos(frequencies)).to(device=hidden_states.device, dtype=tp.float32)
        sin = tp.from_numpy(np.sin(frequencies)).to(device=hidden_states.device, dtype=tp.float32)
        segments = patch_segments(grid_thw)

        deepstack = []
        for index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, segments, (cos, sin))
            if index in self.config.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.config.deepstack_visual_indexes.index(index)]
                deepstack.append(merger(hidden_states))

        merged = self.merger(hidden_states)
        return hidden_states, merged, deepstack


def vision_checkpoint_name(parameter_name: str) -> str:
    """Map an attribute path of the tower onto its name in the checkpoint.

    The tower answers to its own attribute path when it stands alone and to
    the ``visual`` attribute when it is held by the language model, so both
    spellings land on the same checkpoint name.
    """
    if parameter_name.startswith("visual."):
        return "model." + parameter_name
    return "model.visual." + parameter_name


def load_vision(model_dir: str, config: Qwen3VLVisionConfig, dtype=tp.float16,
                device="cpu", progress=None):
    """Build the tower and fill it from the checkpoint's dense tensors."""
    from .ct_weights import Checkpoint

    model = Qwen3VLVisionModel(config, device=device, dtype=dtype)
    checkpoint = Checkpoint(model_dir)
    missing = []
    with tp.no_grad():
        for name, parameter in model.named_parameters():
            key = vision_checkpoint_name(name)
            if not checkpoint.has(key):
                missing.append(key)
                continue
            dense = checkpoint.float32(key)
            if tuple(dense.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"{key}: checkpoint {dense.shape} vs tower {tuple(parameter.shape)}"
                )
            parameter.copy_(tp.from_numpy(dense).to(dtype=dtype, device=parameter.device))
    checkpoint.close()
    if missing:
        raise RuntimeError(f"{len(missing)} tower tensors absent: {missing[:4]}")
    model.eval()
    if progress is not None:
        progress(f"vision tower loaded from {model_dir}")
    return model
