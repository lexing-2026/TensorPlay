"""Qwen3-VL built out of the framework's own language-model building blocks.

The model holds both towers: the language side (embedding table, decoder stack
of RMS-normalized pre-norm blocks with grouped-query attention, final
normalization, output projection) and the vision side (patch projection, the
vision blocks and the patch mergers).  Attribute names mirror the checkpoint,
so a weight can be filed under the name it carries in the file:

    model.language_model.layers.3.self_attn.q_proj.weight
    model.visual.blocks.0.attn.qkv.weight

Two ways of holding the language projections are offered:

* ``weight_only=True`` — the checkpoint's 4-bit groups are repacked into the
  framework's weight-only linear, which keeps four bits per element resident
  and recovers the values inside the graph on every forward.  This is the
  native quantized path.
* ``weight_only=False`` — the groups are widened to the model's element type
  at load time and plain dense projections are used.

An image is spliced in where the token stream asks for it: the tower's merged
patches take the place of the image placeholder tokens, its three deepstack
feature maps are added to the same positions after the 8th, 16th and 24th
decoder blocks, and the position streams become three-dimensional so a patch
keeps both of its image coordinates while the surrounding text keeps a running
count.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn.modules.rotary import apply_rotary_emb

from .vision import Qwen3VLVisionConfig, Qwen3VLVisionModel, vision_checkpoint_name

__all__ = [
    "WeightOnlyInt4Linear",
    "Qwen3VLTextConfig",
    "Qwen3VLForCausalLM",
    "checkpoint_name",
    "load_qwen3vl",
    "multimodal_position_ids",
]

QUANT_GROUP_SIZE = 32
QUANT_BITS = 4


def _factory(device, dtype):
    return {"device": device, "dtype": dtype}


def multimodal_position_ids(token_ids, grid_thw, merge_size: int, image_token_id: int):
    """The three position streams of a token sequence with images spliced in.

    Every plain token advances all three streams together.  A run of image
    placeholders is replaced by the vision grid's own addressing: the temporal
    coordinate indexes the frames, the height and width coordinates walk the
    merged grid, and the whole image then advances the streams by the longer
    of its two sides.  Returns ``(streams, delta)``, where ``delta`` carries
    the image's extra extent into the positions of the tokens that follow.

    The token order inside an image is frame-major, then row, then column,
    which is the order the tower produces its merged patches in.
    """
    tokens = np.asarray(token_ids).reshape(-1)
    streams = np.zeros((3, tokens.shape[0]), dtype=np.int64)
    position = 0
    index = 0
    image = 0
    while index < tokens.shape[0]:
        if tokens[index] != image_token_id:
            streams[:, index] = position
            position += 1
            index += 1
            continue
        end = index
        while end < tokens.shape[0] and tokens[end] == image_token_id:
            end += 1
        frames, height, width = (int(v) for v in grid_thw[image])
        merged_h, merged_w = height // merge_size, width // merge_size
        temporal = np.repeat(np.arange(frames, dtype=np.int64), merged_h * merged_w)
        rows = np.tile(np.repeat(np.arange(merged_h, dtype=np.int64), merged_w), frames)
        columns = np.tile(np.arange(merged_w, dtype=np.int64), frames * merged_h)
        streams[0, index:end] = temporal + position
        streams[1, index:end] = rows + position
        streams[2, index:end] = columns + position
        position += max(merged_h, merged_w)
        index = end
        image += 1
    delta = int(streams.max()) + 1 - tokens.shape[0]
    return streams, delta


# One table serves every layer that shares a head width, base and stream
# layout, so a stack of blocks pays for it a single time.
_ROTARY_TABLES: dict = {}
_ROTARY_SIZE = 32768


class MultimodalRotaryEmbedding(nn.Module):
    """Rotary tables built from one, two or three position streams.

    A single stream is plain rotary embedding.  Three streams — temporal,
    height and width — are folded into one frequency layout: within the
    interleaved region every third slot keeps the temporal stream and the
    others are taken from the height and width streams, and the slots past the
    region stay temporal.  When all three streams are equal this is exactly the
    single-stream layout, so the same module serves text and images.

    The tables live on the device and are gathered by position, so a forward
    is a gather plus, for three streams, a per-slot selection — no work travels
    through the host.  A table covers every position the module can be asked
    for; asking beyond its extent is a programming error rather than something
    silently rounded down.
    """

    def __init__(self, head_dim: int, base: float, mrope_section, device=None, dtype=None):
        super().__init__()
        self.head_dim = head_dim
        self.base = float(base)
        self.mrope_section = [int(v) for v in mrope_section]
        # The rotation pairs the two halves of a head, the layout the tower's
        # frequency table is built for.
        self.interleaved = False
        inverse = 1.0 / (
            self.base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        )
        self.register_buffer("inv_freq", tp.from_numpy(inverse), persistent=False)

    def _tables(self, device):
        key = (str(device), self.head_dim, self.base, tuple(self.mrope_section))
        entry = _ROTARY_TABLES.get(key)
        if entry is None:
            positions = tp.arange(_ROTARY_SIZE, device=device, dtype=tp.float32)
            frequencies = tp.outer(positions, self.inv_freq.to(device))
            entry = (tp.cos(frequencies), tp.sin(frequencies))
            _ROTARY_TABLES[key] = entry
        return entry

    def forward(self, x, position_ids=None, max_position=None):
        if position_ids is None:
            raise ValueError("a rotary table needs positions")
        cos_table, sin_table = self._tables(x.device)
        if position_ids.ndim == 2:
            cos = cos_table[position_ids]
            sin = sin_table[position_ids]
        else:
            table_cos = cos_table[position_ids]
            table_sin = sin_table[position_ids]
            cos = table_cos[0]
            sin = table_sin[0]
            for axis in (1, 2):
                length = self.mrope_section[axis] * 3
                cos[..., axis:length:3] = table_cos[axis][..., axis:length:3]
                sin[..., axis:length:3] = table_sin[axis][..., axis:length:3]
        return cos.to(x.dtype), sin.to(x.dtype)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, base={self.base}, section={self.mrope_section}"



def _projection(in_features, out_features, weight_only, device, dtype):
    """A projection layer: weight-only quantized when asked, dense otherwise."""
    if weight_only:
        return WeightOnlyInt4Linear(in_features, out_features, groupsize=QUANT_GROUP_SIZE)
    return nn.Linear(in_features, out_features, bias=False, **_factory(device, dtype))


class WeightOnlyInt4Linear(nn.Module):
    """Four-bit weight-only projection feeding a reduced-precision product.

    The packed codes and their group factors stay on the device; a forward
    recovers the values, rounds them to the activation's element type and runs
    the product, so the weights are four-bit resident.

    Recovering them is proportional to the weight's size and happens on every
    forward, which for a projection this wide costs far more than the product
    itself.  ``cache_dense_weight`` moves that recovery to a single pass: the
    four-bit groups stay in place as the stored form and the recovered values
    are held beside them, so a forward becomes one plain product.  The extra
    residency is two bytes per element on top of the half byte the codes take.
    """

    def __init__(self, in_features, out_features, groupsize=QUANT_GROUP_SIZE, bits=QUANT_BITS, bias=False):
        super().__init__()
        from tensorplay.ao.quantization.intx_weight_only import WeightOnlyIntxLinear

        self._impl = WeightOnlyIntxLinear(
            in_features, out_features, bits=bits, groupsize=groupsize, bias=bias
        )
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.groupsize = groupsize
        self.bias = None
        self.register_buffer("_dense_cache", None, persistent=False)

    @property
    def weight(self):
        return self._impl.weight

    @property
    def scales_and_zeros(self):
        return self._impl.scales_and_zeros

    def dense_weight(self):
        """The values the four-bit codes stand for, at single precision."""
        from tensorplay.ao.quantization.intx_weight_only import _dequantize_intx

        scale = self.scales_and_zeros[..., 0].t().contiguous()
        zero = self.scales_and_zeros[..., 1].t().contiguous()
        return _dequantize_intx(self.weight, scale, zero, self.bits, self.in_features)

    def cache_dense_weight(self, dtype=None):
        """Hold the recovered values transposed, ready for the product."""
        dense = self.dense_weight()
        if dtype is not None:
            dense = dense.to(dtype)
        self._dense_cache = dense.t().contiguous()

    def forward(self, x):
        if self._dense_cache is not None:
            return x @ self._dense_cache
        dense = self.dense_weight().to(x.dtype)
        return x @ dense.t().contiguous()


def cache_dense_weights(model, dtype=None):
    """Recover every weight-only projection once instead of once per forward."""
    count = 0
    for module in model.modules():
        if isinstance(module, WeightOnlyInt4Linear):
            module.cache_dense_weight(dtype)
            count += 1
    return count



class Qwen3VLTextConfig:
    """The shapes read out of ``config.json``, language and vision side alike."""

    def __init__(self, raw: dict):
        text = raw.get("text_config", raw)
        self.vocab_size = int(text["vocab_size"])
        self.hidden_size = int(text["hidden_size"])
        self.intermediate_size = int(text["intermediate_size"])
        self.num_hidden_layers = int(text["num_hidden_layers"])
        self.num_attention_heads = int(text["num_attention_heads"])
        self.num_key_value_heads = int(text["num_key_value_heads"])
        self.head_dim = int(text["head_dim"])
        self.rms_norm_eps = float(text["rms_norm_eps"])
        self.rope_theta = float(text.get("rope_theta", 10000.0))
        self.tie_word_embeddings = bool(raw.get("tie_word_embeddings", False))
        rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
        self.mrope_section = [int(v) for v in rope.get("mrope_section", [24, 20, 20])]
        self.image_token_id = int(raw.get("image_token_id", -1))
        self.vision = Qwen3VLVisionConfig(raw) if "vision_config" in raw else None

    @classmethod
    def from_dir(cls, model_dir: str) -> "Qwen3VLTextConfig":
        with open(os.path.join(model_dir, "config.json")) as handle:
            return cls(json.load(handle))

    def __repr__(self) -> str:
        return (
            f"Qwen3VLTextConfig(hidden={self.hidden_size}, layers={self.num_hidden_layers}, "
            f"heads={self.num_attention_heads}/{self.num_key_value_heads}, "
            f"head_dim={self.head_dim}, vocab={self.vocab_size}, "
            f"image_token={self.image_token_id})"
        )


class Qwen3VLMLP(nn.Module):
    """SwiGLU feed-forward: the gate scales the up-projection before the down-projection."""

    def __init__(self, config: Qwen3VLTextConfig, weight_only=False, device=None, dtype=None):
        super().__init__()
        self.gate_proj = _projection(
            config.hidden_size, config.intermediate_size, weight_only, device, dtype
        )
        self.up_proj = _projection(
            config.hidden_size, config.intermediate_size, weight_only, device, dtype
        )
        self.down_proj = _projection(
            config.intermediate_size, config.hidden_size, weight_only, device, dtype
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: tp.Tensor) -> tp.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLAttention(nn.Module):
    """Grouped-query attention with per-head normalization on queries and keys."""

    def __init__(self, config, layer_idx, weight_only=False, device=None, dtype=None):
        super().__init__()
        self.mixer = nn.GroupedQueryAttention(
            config.hidden_size,
            config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            bias=False,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            rope_base=config.rope_theta,
            layer_idx=layer_idx,
            **_factory(device, dtype),
        )
        # The tower addresses image tokens by three coordinates, so the plain
        # rotary table is replaced by one that folds three streams together.
        self.mixer.rope = MultimodalRotaryEmbedding(
            config.head_dim, config.rope_theta, config.mrope_section, device=device
        )
        if weight_only:
            # The mixer owns its projections; swap in the quantized ones so
            # the rotation, the head normalization and the cache stay as the
            # mixer implements them.
            kv_width = config.num_key_value_heads * config.head_dim
            self.mixer.q_proj = _projection(
                config.hidden_size, config.num_attention_heads * config.head_dim, True, None, None
            )
            self.mixer.k_proj = _projection(config.hidden_size, kv_width, True, None, None)
            self.mixer.v_proj = _projection(config.hidden_size, kv_width, True, None, None)
            self.mixer.o_proj = _projection(
                config.num_attention_heads * config.head_dim, config.hidden_size, True, None, None
            )

    def forward(self, x, position_ids=None, past_key_values=None):
        mixer = self.mixer
        batch, length, _ = x.shape
        heads, kv_heads, head_dim = mixer.num_heads, mixer.num_kv_heads, mixer.head_dim
        if position_ids is None:
            offset = (
                past_key_values.get_seq_length(mixer.layer_idx)
                if past_key_values is not None
                else 0
            )
            position_ids = (
                tp.arange(offset, offset + length, device=x.device)
                .unsqueeze(0)
                .expand(batch, length)
            )
        q = mixer.q_proj(x).view(batch, length, heads, head_dim).transpose(1, 2)
        k = mixer.k_proj(x).view(batch, length, kv_heads, head_dim).transpose(1, 2)
        v = mixer.v_proj(x).view(batch, length, kv_heads, head_dim).transpose(1, 2)
        if mixer.qk_norm:
            q = mixer.q_norm(q)
            k = mixer.k_norm(k)
        cos, sin = mixer.rope(x, position_ids)
        q, k = apply_rotary_emb(q, k, cos, sin, mixer.rope.interleaved)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, mixer.layer_idx)

        if length == 1:
            # A decode step covers the whole cache with a single query per
            # head.  The arithmetic is written out because the fused call
            # takes a path whose cost grows with the cache width, while these
            # three products stay flat.  The group expansion is explicit,
            # which is a copy of the cache, not of the weights.
            group = heads // kv_heads
            if group == 1:
                keys, values = k, v
            else:
                keys = k.repeat_interleave(group, dim=1)
                values = v.repeat_interleave(group, dim=1)
            scores = (q @ keys.transpose(-1, -2)) * mixer.scaling
            weights = tp.nn.functional.softmax(scores, dim=-1)
            attn = weights @ values
        else:
            attn = tp.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=mixer.dropout if self.training else 0.0,
                is_causal=q.shape[2] == k.shape[2],
                scale=mixer.scaling,
                enable_gqa=heads != kv_heads,
            )
        attn = attn.transpose(1, 2).reshape(batch, length, heads * head_dim)
        return mixer.o_proj(attn)


class Qwen3VLDecoderLayer(nn.Module):
    """One pre-norm residual block: attention, then feed-forward."""

    def __init__(self, config, layer_idx, weight_only=False, device=None, dtype=None):
        super().__init__()
        self.self_attn = Qwen3VLAttention(
            config, layer_idx, weight_only=weight_only, device=device, dtype=dtype
        )
        self.mlp = Qwen3VLMLP(config, weight_only=weight_only, device=device, dtype=dtype)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **_factory(device, dtype)
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **_factory(device, dtype)
        )

    def forward(self, hidden_states, position_ids=None, past_key_values=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_ids=position_ids, past_key_values=past_key_values
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states + residual


class Qwen3VLLanguageModel(nn.Module):
    """The decoder stack: token embeddings, the blocks and the final normalization."""

    def __init__(self, config, weight_only=False, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, **_factory(device, dtype)
        )
        self.layers = nn.ModuleList(
            [
                Qwen3VLDecoderLayer(
                    config, i, weight_only=weight_only, device=device, dtype=dtype
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **_factory(device, dtype)
        )

    @staticmethod
    def _add_visual_features(hidden_states, positions, visual_embeds):
        """Add the vision features back at the positions they were spliced into."""
        width = hidden_states.shape[-1]
        flat = hidden_states.reshape(-1, width)
        flat = flat.index_add(
            0, positions, visual_embeds.to(device=flat.device, dtype=flat.dtype)
        )
        return flat.reshape(hidden_states.shape)

    def forward(self, hidden_states, position_ids=None, past_key_values=None,
                visual_positions=None, deepstack_visual_embeds=None):
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, position_ids=position_ids, past_key_values=past_key_values
            )
            if deepstack_visual_embeds is not None and index < len(deepstack_visual_embeds):
                hidden_states = self._add_visual_features(
                    hidden_states, visual_positions, deepstack_visual_embeds[index]
                )
        return self.norm(hidden_states)


class Qwen3VLForCausalLM(nn.Module):
    """The model as a causal language model: hidden states followed by next-token logits."""

    def __init__(self, config, weight_only=False, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.visual = (
            Qwen3VLVisionModel(config.vision, device=device, dtype=dtype)
            if config.vision is not None
            else None
        )
        self.language_model = Qwen3VLLanguageModel(
            config, weight_only=weight_only, device=device, dtype=dtype
        )
        # The checkpoint keeps the output projection in full precision.
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, **_factory(device, dtype)
        )
        # Set once an image has been seen: it tells the later decode steps how
        # far the vision grid pushed the position streams past the token count.
        self.rope_delta = None
        # Held between :meth:`prepare_image` and the prefill call, because the
        # generation loop forwards tokens only.
        self._pending_image = None

    def prepare_image(self, pixel_values, image_grid_thw):
        """Hand the patch batch to the next forward pass.

        The generation loop calls the model with tokens and positions only, so
        the image travels beside it and is picked up by the prefill call.
        """
        self._pending_image = (pixel_values, image_grid_thw)

    def image_features(self, pixel_values, image_grid_thw):
        """Merged patch features plus the deepstack maps, image by image."""
        _, merged, deepstack = self.visual(pixel_values, image_grid_thw)
        return merged, deepstack

    def _splice_visual(self, inputs_embeds, input_ids, pixel_values, image_grid_thw):
        tokens = input_ids.detach().cpu().numpy().reshape(-1)
        positions = np.nonzero(tokens == self.config.image_token_id)[0]
        features, deepstack = self.image_features(pixel_values, image_grid_thw)
        if positions.shape[0] != features.shape[0]:
            raise RuntimeError(
                f"{positions.shape[0]} image placeholders against "
                f"{features.shape[0]} merged patches"
            )
        width = inputs_embeds.shape[-1]
        flat = inputs_embeds.reshape(-1, width)
        index = tp.from_numpy(positions.astype(np.int64)).to(device=flat.device)
        flat = flat.index_copy(
            0, index, features.to(device=flat.device, dtype=flat.dtype)
        )
        return flat.reshape(inputs_embeds.shape), index, deepstack

    def _resolve_positions(self, position_ids, input_ids, image_grid_thw):
        """Three position streams for the grid, a shifted count for the rest.

        The streams are always handed over three-dimensional, ``[3, batch,
        seq]``; a plain two-dimensional input means "one position every
        token", which the rotary table folds into the same three streams.
        """
        if image_grid_thw is not None and input_ids is not None:
            streams, self.rope_delta = multimodal_position_ids(
                input_ids.detach().cpu().numpy(),
                image_grid_thw,
                self.visual.spatial_merge_size,
                self.config.image_token_id,
            )
            return tp.from_numpy(streams).unsqueeze(1), self.rope_delta
        if self.rope_delta is not None and position_ids is not None:
            shifted = position_ids + self.rope_delta
            return shifted.unsqueeze(0).expand(3, *shifted.shape), self.rope_delta
        return position_ids, self.rope_delta

    def forward(self, input_ids=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, pixel_values=None, image_grid_thw=None):
        if pixel_values is None and self._pending_image is not None:
            pixel_values, image_grid_thw = self._pending_image
            self._pending_image = None
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        visual_positions = None
        deepstack = None
        if pixel_values is not None:
            inputs_embeds, visual_positions, deepstack = self._splice_visual(
                inputs_embeds, input_ids, pixel_values, image_grid_thw
            )

        position_ids, _ = self._resolve_positions(position_ids, input_ids, image_grid_thw)
        hidden = self.language_model(
            inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            visual_positions=visual_positions,
            deepstack_visual_embeds=deepstack,
        )
        return self.lm_head(hidden)

    def create_empty_cache(self):
        from tensorplay.nn.cache import DynamicCache

        return DynamicCache()


def checkpoint_name(parameter_name: str) -> Optional[str]:
    """Map an attribute path onto the name the checkpoint files the tensor under."""
    if parameter_name.startswith("visual."):
        return vision_checkpoint_name(parameter_name)
    if parameter_name.startswith("language_model."):
        prefix = "model.language_model."
        name = parameter_name[len("language_model.") :]
    else:
        prefix = ""
        name = parameter_name
    # The mixer keeps its projections and per-head normalizations as
    # submodules; the checkpoint writes them directly under ``self_attn``.
    name = name.replace(".self_attn.mixer.", ".self_attn.")
    return prefix + name


def load_qwen3vl(
    model_dir: str,
    dtype=tp.float16,
    device="cpu",
    weight_only: bool = True,
    progress=None,
):
    """Build the model and fill it from a checkpoint directory.

    The model is built straight in the requested element type, so no
    single-precision copy of the whole stack is ever resident.  Quantized
    layers keep their four-bit groups and take only the group scales from the
    file; everything else is read as it sits on disk.
    """
    from .ct_weights import Checkpoint, dequantize_linear, weight_only_group

    config = Qwen3VLTextConfig.from_dir(model_dir)
    model = Qwen3VLForCausalLM(
        config, weight_only=weight_only, device="cpu", dtype=dtype
    )
    ckpt = Checkpoint(model_dir)
    quantized, dense, missing = 0, 0, []
    with tp.no_grad():
        for module_path, module in model.named_modules():
            if not isinstance(module, WeightOnlyInt4Linear):
                continue
            base = checkpoint_name(module_path + ".weight")
            try:
                packed, scale, zero = weight_only_group(ckpt, base, QUANT_BITS)
            except KeyError:
                missing.append(base)
                continue
            module.weight.copy_(tp.from_numpy(packed))
            scales = tp.from_numpy(scale.T).contiguous()
            zeros = tp.from_numpy(zero.T).contiguous()
            module.scales_and_zeros.copy_(tp.stack([scales, zeros], dim=-1))
            quantized += 1
        for name, param in model.named_parameters():
            key = checkpoint_name(name)
            try:
                dense_w = dequantize_linear(ckpt, key)
            except KeyError:
                missing.append(key)
                continue
            if tuple(dense_w.shape) != tuple(param.shape):
                raise RuntimeError(
                    f"{key}: checkpoint {dense_w.shape} vs model {tuple(param.shape)}"
                )
            param.copy_(tp.from_numpy(dense_w).to(dtype=dtype, device=param.device))
            dense += 1
    ckpt.close()
    if missing:
        raise RuntimeError(f"{len(missing)} tensors absent from the checkpoint: {missing[:4]}")
    if progress is not None:
        progress(f"filled {quantized} quantized and {dense} dense tensors from {model_dir}")
    model = model.to(device=device)
    model.eval()
    return model, config
