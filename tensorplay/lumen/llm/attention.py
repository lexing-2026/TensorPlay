"""LLM attention-map extraction and visualization.

Framework-native helpers: they operate directly on models built with
tensorplay primitives and need no third-party runtime. Attention maps are read
either from a model forward that returns ``outputs.attentions`` or, failing
that, by capturing the per-head weights returned by ``nn.MultiheadAttention``
modules during a forward pass.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import tensorplay as tensorplay
from tensorplay import nn


def _resolve_indices(
    total: int, selection: Optional[Union[int, Sequence[int]]]
) -> List[int]:
    if selection is None:
        return list(range(total))
    if isinstance(selection, int):
        return [selection]
    return list(selection)


def _encode(text: str, tokenizer: Any) -> List[int]:
    """Tokenize a single string into a list of integer ids."""
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
        if isinstance(ids, tensorplay.Tensor):
            return [int(i) for i in ids.reshape(-1).tolist()]
        if isinstance(ids, int):
            return [ids]
        return [int(i) for i in ids]
    if callable(tokenizer):
        ids = tokenizer(text)
        if isinstance(ids, tensorplay.Tensor):
            return [int(i) for i in ids.reshape(-1).tolist()]
        if isinstance(ids, int):
            return [ids]
        return [int(i) for i in ids]
    raise TypeError(
        "tokenizer must provide `encode(text)` or be callable, got %r" % type(tokenizer)
    )


def _decode_ids(input_ids: tensorplay.Tensor, tokenizer: Any) -> List[str]:
    ids = input_ids[0].tolist()
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return list(tokenizer.convert_ids_to_tokens(ids))
    if hasattr(tokenizer, "decode"):
        return [str(tokenizer.decode([i])) for i in ids]
    return [str(i) for i in ids]


def _run_with_attention_outputs(model: Any, input_ids: tensorplay.Tensor) -> Any:
    """Try a forward that returns attentions; ``None`` if unsupported."""
    try:
        outputs = model(input_ids=input_ids, output_attentions=True)
    except (TypeError, ValueError):
        return None
    return outputs


def _capture_mha_attentions(model: Any, input_ids: tensorplay.Tensor) -> List[tensorplay.Tensor]:
    """Capture per-head attention weights from MultiheadAttention modules."""
    captured: List[tensorplay.Tensor] = []
    handles: List[Any] = []

    def make_hook(index: int) -> Any:
        def hook(module: Any, args: Any, output: Any) -> None:
            if isinstance(output, tuple) and len(output) >= 2:
                captured.append(output[1].detach())

        return hook

    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(make_hook(len(captured))))
    try:
        with tensorplay.no_grad():
            model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def extract_attention(
    model: Any,
    tokenizer: Any,
    text: Union[str, List[str]],
    layers: Optional[Union[int, Sequence[int]]] = None,
    heads: Optional[Union[int, Sequence[int]]] = None,
) -> Dict[str, Any]:
    """Extract self-attention maps for one or more texts.

    Args:
        model: A tensorplay module whose forward accepts ``input_ids`` (and
            optionally ``output_attentions``) and returns an object with an
            ``attentions`` attribute. If the forward does not expose
            attentions, per-head weights are captured from every
            ``nn.MultiheadAttention`` module instead.
        tokenizer: Object with ``encode(text) -> list[int]`` and
            ``convert_ids_to_tokens(ids) -> list[str]`` (or ``decode``), or a
            callable returning a list of token ids.
        text: A string or a list of strings.
        layers: Which layer indices to keep; ``None`` keeps all.
        heads: Which head indices to keep; ``None`` keeps all.

    Returns:
        A dict with keys:
        ``attention``: list of ``(batch, selected_heads, seq, seq)`` numpy
        arrays, one per selected layer;
        ``tokens``: list of decoded tokens;
        ``input_ids``: numpy ``(batch, seq)`` token ids;
        ``layers`` / ``heads``: selected indices.
    """
    if isinstance(text, str):
        text = [text]
    device = next(model.parameters()).device
    input_ids = tensorplay.as_tensor(
        [_encode(t, tokenizer) for t in text], dtype=tensorplay.int64, device=device
    )

    model.eval()
    outputs = _run_with_attention_outputs(model, input_ids)
    attentions: Optional[List[tensorplay.Tensor]] = None
    if outputs is not None:
        attentions = getattr(outputs, "attentions", None)
    if not attentions:
        attentions = _capture_mha_attentions(model, input_ids)
    if not attentions:
        raise ValueError(
            "The model did not expose attention maps. Use nn.MultiheadAttention "
            "in the model or return outputs.attentions from forward."
        )

    layer_indices = _resolve_indices(len(attentions), layers)
    if attentions[0].ndim == 4:
        n_heads = attentions[0].shape[1]
    else:
        n_heads = 1
    head_indices = _resolve_indices(n_heads, heads)

    maps = []
    for li in layer_indices:
        att = attentions[li].detach().cpu().numpy()
        if att.ndim == 4:
            maps.append(att[:, head_indices])
        else:
            # averaged attention weights with shape (batch, seq, seq)
            maps.append(att[:, None, :, :])

    return {
        "attention": maps,
        "tokens": _decode_ids(input_ids, tokenizer),
        "input_ids": input_ids.cpu().numpy(),
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