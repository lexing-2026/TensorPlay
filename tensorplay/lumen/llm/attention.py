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
    highlight_tokens: Optional[Sequence[str]] = None,
    show_token_labels: bool = True,
    tick_fontsize: int = 8,
    vmax: float = 1.0,
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
        highlight_tokens: Token strings to emphasize (bold, colored) on the
            axes, e.g. a subject token in a causal-tracing study.
        show_token_labels: If ``False``, hides token labels on the axes.
        tick_fontsize: Font size of the token labels.
        vmax: Maximum colorbar value (attention weights are normalized per row
            and usually at most 1.0).
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

    layer = 0 if layer is None else layer
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
        figsize = (cols * 3.6, rows * 3.6)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    seq_len = arr.shape[1]
    max_ticks = 12
    tick_step = max(1, int(np.ceil(seq_len / max_ticks)))
    tick_positions = list(range(0, seq_len, tick_step))
    tick_labels = [
        tokens[i] if tokens is not None and i < len(tokens) else str(i)
        for i in tick_positions
    ]
    highlight = set(highlight_tokens or [])

    # first subplot decides the shared colorbar
    im = None
    for ax, hmap, hi in zip(axes, arr, head_indices):
        im = ax.imshow(
            hmap,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title("Layer %d \u00b7 Head %d" % (layer, hi), fontsize=10)
        if show_token_labels:
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            xtick_labels = []
            for label in tick_labels:
                if label in highlight:
                    xtick_labels.append(
                        plt.Text(
                            0, 0, label, fontsize=tick_fontsize + 1,
                            fontweight="bold", color="#d62728",
                        )
                    )
                else:
                    xtick_labels.append(label)
            ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
            ax.set_yticklabels(tick_labels, fontsize=tick_fontsize)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.tick_params(length=0, labelsize=tick_fontsize)

        # light grid lines between cells for readability
        ax.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.4, alpha=0.25)

    for ax in axes[n:]:
        ax.axis("off")

    if im is not None:
        if n == 1:
            fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        else:
            fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.02)

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
    if n == 1:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=[0, 0, 0.96, 1])
    if use_pyplot:
        plt.show()
    return fig, axes

def _normalize_attention_maps(
    attention: Union[Dict[str, Any], List[np.ndarray], np.ndarray],
    tokens: Optional[List[str]] = None,
    layer: Optional[int] = None,
) -> Any:
    """Normalize attention input to ``(maps, tokens, layers, heads)``.

    ``maps`` is a list of ``(heads, seq, seq)`` arrays; ``layers`` and ``heads``
    are the corresponding index lists (or ``None`` when unknown).
    """
    if isinstance(attention, dict):
        maps = attention["attention"]
        tokens = attention.get("tokens", tokens)
        layers = attention.get("layers")
        heads = attention.get("heads")
        if layer is not None:
            maps = [maps[layer]]
            layers = [layers[layer]]
    elif isinstance(attention, (list, tuple)):
        maps = attention
        layers = list(range(len(maps)))
        heads = None
        if layer is not None:
            maps = [maps[layer]]
            layers = [layer]
    else:
        maps = [attention]
        layers = [0]
        heads = None

    normalized = []
    for m in maps:
        arr = np.asarray(m, dtype=float)
        if arr.ndim == 2:
            arr = arr[None]  # (1, seq, seq)
        elif arr.ndim == 4:
            arr = arr[0]  # (batch, heads, seq, seq) -> (heads, seq, seq)
        normalized.append(arr)
    return normalized, tokens, layers, heads


def visualize_attention_multi_layer(
    attention: Union[Dict[str, Any], List[np.ndarray], np.ndarray],
    tokens: Optional[List[str]] = None,
    query_token: Optional[str] = None,
    layers: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[int]] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    figsize: Optional[Sequence[float]] = None,
    use_pyplot: bool = False,
) -> Any:
    """Plot attention for many layers at once, one subplot per head.

    This is the multi-layer "head view" familiar from interpretability papers:
    rows are layers, columns are heads. When ``query_token`` is given, the
    corresponding row and column of every attention matrix are outlined in red,
    making it easy to see what that token attends to (and which tokens attend to
    it) across the whole stack.

    Returns:
        ``(fig, axes)`` matplotlib objects.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    maps, tokens, all_layers, all_heads = _normalize_attention_maps(attention, tokens)
    if layers is None:
        layers = list(range(len(maps)))[:8]
    if heads is None:
        heads = list(range(maps[0].shape[0]))[:8]
    layers = list(layers)
    heads = list(heads)

    query_pos = None
    if query_token is not None and tokens is not None and query_token in tokens:
        query_pos = tokens.index(query_token)

    rows, cols = len(layers), len(heads)
    if figsize is None:
        figsize = (cols * 2.8, rows * 2.8)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    seq_len = maps[0].shape[1]
    max_ticks = 8
    tick_step = max(1, int(np.ceil(seq_len / max_ticks)))
    tick_positions = list(range(0, seq_len, tick_step))
    tick_labels = [
        tokens[i] if tokens is not None and i < len(tokens) else str(i)
        for i in tick_positions
    ]

    im = None
    for r, li in enumerate(layers):
        layer_map = maps[li]
        for c, hi in enumerate(heads):
            ax = axes[r][c]
            hmap = layer_map[hi]
            im = ax.imshow(
                hmap, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto",
                interpolation="nearest",
            )
            if query_pos is not None:
                ax.add_patch(
                    Rectangle(
                        (query_pos - 0.5, -0.5), 1, seq_len,
                        fill=False, edgecolor="#d62728", linewidth=1.2,
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (-0.5, query_pos - 0.5), seq_len, 1,
                        fill=False, edgecolor="#d62728", linewidth=1.2,
                    )
                )
            if r == rows - 1:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(
                    tick_labels, rotation=45, ha="right", fontsize=7
                )
            else:
                ax.set_xticks([])
            if c == 0:
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels, fontsize=7)
            else:
                ax.set_yticks([])
            ax.tick_params(length=0)
            if r == 0:
                ax.set_title("Head %d" % hi, fontsize=9)
            if c == 0:
                ax.set_ylabel("Layer %d" % li, fontsize=9)
            ax.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.2)

    if im is not None:
        fig.colorbar(im, ax=list(axes.ravel()), fraction=0.025, pad=0.02)
    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.96, 1])
    if use_pyplot:
        plt.show()
    return fig, axes


def token_attention(
    attention: Union[Dict[str, Any], List[np.ndarray], np.ndarray],
    query_token: str,
    tokens: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract what ``query_token`` attends to across layers and heads.

    Returns a dict with ``matrix`` of shape ``(num_layers, num_heads, seq)``
    (attention distribution of the query token at every layer/head), plus
    ``tokens``, ``layers``, ``heads``, ``query_token`` and ``query_position``.
    """
    maps, tokens, layers, heads = _normalize_attention_maps(attention, tokens)
    if tokens is None or query_token not in tokens:
        raise ValueError(
            "query_token %r not found in tokens %r" % (query_token, tokens)
        )
    query_pos = tokens.index(query_token)
    seq = maps[0].shape[1]
    matrix = np.zeros((len(maps), maps[0].shape[0], seq), dtype=float)
    for li, m in enumerate(maps):
        for h in range(m.shape[0]):
            matrix[li, h] = m[h, query_pos, :]
    return {
        "matrix": matrix,
        "tokens": tokens,
        "layers": layers if layers is not None else list(range(len(maps))),
        "heads": heads if heads is not None else list(range(maps[0].shape[0])),
        "query_token": query_token,
        "query_position": query_pos,
    }


def visualize_token_attention(
    token_attn: Dict[str, Any],
    cmap: str = "viridis",
    title: Optional[str] = None,
    figsize: Optional[Sequence[float]] = None,
    use_pyplot: bool = False,
) -> Any:
    """Plot what a single token attends to across every layer and head.

    The heatmap has one row per ``(layer, head)`` and one column per token; the
    cell value is the attention weight from the query token to that column's
    token. The query token's own column is outlined in red.

    Returns:
        ``(fig, ax)`` matplotlib objects.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    matrix = np.asarray(token_attn["matrix"], dtype=float)
    tokens = token_attn["tokens"]
    query_pos = token_attn["query_position"]
    n_layers, n_heads, seq = matrix.shape
    flat = matrix.reshape(n_layers * n_heads, seq)
    y_labels = [
        "L%d\u00b7H%d" % (li, hi)
        for li in range(n_layers)
        for hi in range(n_heads)
    ]

    if figsize is None:
        figsize = (max(6.0, 0.5 * seq + 3.0), max(3.0, 0.35 * n_layers * n_heads + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        flat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest"
    )
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    tick_step = max(1, int(np.ceil(seq / 12)))
    tick_positions = list(range(0, seq, tick_step))
    tick_labels = [tokens[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.tick_params(length=0)
    ax.add_patch(
        Rectangle(
            (query_pos - 0.5, -0.5), 1, n_layers * n_heads,
            fill=False, edgecolor="#d62728", linewidth=1.4,
        )
    )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    if use_pyplot:
        plt.show()
    return fig, ax
