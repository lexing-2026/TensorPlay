"""LLM attention-map extraction and visualization.

These helpers read attention weights directly from a transformer model, so
they work with any model that exposes per-layer attention tensors (load the
model with ``attn_implementation="eager"`` when using the SDPA/flash path,
which does not materialize attention maps). No attribution passes are needed;
the model's own forward pass supplies the maps.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


def _resolve_indices(
    total: int, selection: Optional[Union[int, Sequence[int]]]
) -> List[int]:
    if selection is None:
        return list(range(total))
    if isinstance(selection, int):
        return [selection]
    return list(selection)


def extract_attention(
    model: Any,
    tokenizer: Any,
    text: Union[str, List[str]],
    layers: Optional[Union[int, Sequence[int]]] = None,
    heads: Optional[Union[int, Sequence[int]]] = None,
    use_decoder_attention: bool = True,
) -> Dict[str, Any]:
    """Extract self-attention maps for one or more texts.

    Args:
        model: A transformer model whose forward accepts ``input_ids`` and
            returns an object with an ``attentions`` attribute (a tuple of
            per-layer tensors of shape ``(batch, heads, seq, seq)``).
        tokenizer: Tokenizer used to encode ``text``.
        text: A string or a list of strings.
        layers: Which layer indices to keep; ``None`` keeps all.
        heads: Which head indices to keep; ``None`` keeps all.
        use_decoder_attention: If ``True``, uses ``outputs.attentions``
            (self-attention); if ``False``, falls back to
            ``outputs.encoder_attentions`` when present (cross-attention in
            encoder-decoder models).

    Returns:
        A dict with keys:
        ``attention``: list of ``(batch, selected_heads, seq, seq)`` numpy
        arrays, one per selected layer;
        ``tokens``: list of decoded tokens;
        ``input_ids``: numpy ``(batch, seq)`` token ids;
        ``attention_mask``: numpy mask or ``None``;
        ``layers`` / ``heads``: selected indices.
    """
    # Imported lazily so this module stays importable without a framework.
    import torch

    if isinstance(text, str):
        text = [text]

    tokenizer_kwargs: Dict[str, Any] = {"truncation": True}
    # Padding requires a pad token; single sequences do not need it.
    if len(text) > 1 and getattr(tokenizer, "pad_token", None) is not None:
        tokenizer_kwargs["padding"] = True
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", **tokenizer_kwargs)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    attentions = getattr(outputs, "attentions", None)
    if attentions is None or len(attentions) == 0:
        if not use_decoder_attention:
            attentions = getattr(outputs, "encoder_attentions", None)
        if attentions is None or len(attentions) == 0:
            raise ValueError(
                "The model did not return attention maps. Reload it with "
                "attn_implementation='eager' because fused attention kernels "
                "hide the maps."
            )

    layer_indices = _resolve_indices(len(attentions), layers)
    n_heads = attentions[0].shape[1]
    head_indices = _resolve_indices(n_heads, heads)

    maps = [
        attentions[li].detach().cpu().numpy()[:, head_indices]
        for li in layer_indices
    ]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    return {
        "attention": maps,
        "tokens": tokens,
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": (
            attention_mask.cpu().numpy() if attention_mask is not None else None
        ),
        "layers": layer_indices,
        "heads": head_indices,
    }


def visualize_attention_heads(
    attention: Union[Dict[str, Any], List[np.ndarray], np.ndarray],
    tokens: Optional[List[str]] = None,
    layer: Optional[int] = None,
    heads: Optional[Union[int, Sequence[int]]] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    figsize: Optional[Sequence[float]] = None,
    use_pyplot: bool = False,
) -> Any:
    """Plot attention heatmaps, one subplot per head.

    Args:
        attention: Either the dict returned by :func:`extract_attention`, a
            list of layer maps, or a single ``(heads, seq, seq)`` / ``(seq,
            seq)`` numpy array.
        tokens: Token labels for the axes.
        layer: When ``attention`` holds multiple layers, which layer to plot.
        heads: Which heads to plot; ``None`` plots all.
        cmap: Matplotlib colormap name.
        title: Optional figure title.
        figsize: Optional ``(width, height)`` figure size.
        use_pyplot: If ``True`` calls ``plt.show()`` (only useful in an
            interactive backend).

    Returns:
        ``(fig, axes)`` matplotlib objects.
    """
    import matplotlib.pyplot as plt

    if isinstance(attention, dict):
        maps = attention["attention"]
        tokens = attention.get("tokens", tokens)
        layer = 0 if layer is None else layer
        attention = maps[layer]
    elif isinstance(attention, (list, tuple)):
        layer = 0 if layer is None else layer
        attention = attention[layer]

    arr = np.asarray(attention, dtype=float)
    if arr.ndim == 2:
        arr = arr[None]  # (1, seq, seq)
    elif arr.ndim == 4:
        arr = arr[0]  # (batch, heads, seq, seq) -> (heads, seq, seq)
    if arr.ndim != 3:
        raise ValueError(
            "Attention must be (heads, seq, seq), (seq, seq), or "
            "(batch, heads, seq, seq); got shape %s" % (arr.shape,)
        )

    head_indices = _resolve_indices(arr.shape[0], heads)
    arr = arr[head_indices]
    n = arr.shape[0]
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    if figsize is None:
        figsize = (cols * 3.5, rows * 3.5)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    seq_len = arr.shape[1]
    tick_step = max(1, seq_len // 20)
    tick_positions = list(range(0, seq_len, tick_step))
    tick_labels = None
    if tokens is not None and len(tokens) >= seq_len:
        tick_labels = [tokens[i] for i in tick_positions]

    for ax, hmap, hi in zip(axes, arr, head_indices):
        im = ax.imshow(hmap, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title("head %d" % hi, fontsize=9)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        if tick_labels is not None:
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
            ax.set_yticklabels(tick_labels, fontsize=7)
        ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if use_pyplot:
        plt.show()
    return fig, axes