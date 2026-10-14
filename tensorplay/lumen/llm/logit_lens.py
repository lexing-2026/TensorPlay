"""Logit lens: inspect how token predictions develop across layers.

The logit lens decodes each intermediate hidden state through the language
model head (optionally after the final layer norm), showing which token the
model "would have predicted" at every layer. This helps locate where a
prediction emerges in the stack. Framework-native, no third-party runtime.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import tensorplay as tensorplay

from tensorplay.lumen.llm.patching import resolve_layer_modules


def _find_attr(obj: Any, dotted: str) -> Optional[Any]:
    for part in dotted.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return None
    return obj


def _resolve_lm_head(model: Any, lm_head: Optional[Any]) -> Any:
    if lm_head is not None:
        return lm_head
    for name in ("lm_head", "transformer.lm_head", "model.lm_head", "output_layer"):
        found = _find_attr(model, name)
        if found is not None:
            return found
    raise ValueError(
        "Could not locate the language model head on the model; pass lm_head explicitly."
    )


def _resolve_final_norm(model: Any, final_norm: Optional[Any]) -> Optional[Any]:
    if final_norm is not None:
        return final_norm
    for name in ("transformer.ln_f", "model.norm", "model.final_layernorm", "ln_f"):
        found = _find_attr(model, name)
        if found is not None:
            return found
    return None


class LogitLens:
    """Decode hidden states at every layer through the LM head.

    Args:
        model: A tensorplay transformer with per-layer modules (see
            :func:`~tensorplay.lumen.llm.patching.resolve_layer_modules`) and
            a language modeling head.
        tokenizer: Object with ``convert_ids_to_tokens``.
        lm_head: Optional LM head module; auto-detected if ``None``.
        final_norm: Optional final layer norm applied before the head;
            auto-detected if ``None``.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        lm_head: Optional[Any] = None,
        final_norm: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.lm_head = _resolve_lm_head(model, lm_head)
        self.final_norm = _resolve_final_norm(model, final_norm)

    def _decode_layer(self, hidden: tensorplay.Tensor) -> tensorplay.Tensor:
        if self.final_norm is not None:
            hidden = self.final_norm(hidden)
        return self.lm_head(hidden)

    def _capture_hidden(
        self, input_ids: tensorplay.Tensor, layers: Optional[Sequence[int]]
    ) -> Dict[int, tensorplay.Tensor]:
        layer_dict = resolve_layer_modules(self.model, layers)
        captured: Dict[int, tensorplay.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int) -> Any:
            def hook(module: Any, args: Any, output: Any) -> None:
                h = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = h.detach()

            return hook

        for layer_idx, module in layer_dict.items():
            handles.append(module.register_forward_hook(make_hook(layer_idx)))
        try:
            with tensorplay.no_grad():
                self.model(input_ids=input_ids)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    @staticmethod
    def _as_input_ids(input_ids: Any) -> tensorplay.Tensor:
        if isinstance(input_ids, tensorplay.Tensor):
            ids = input_ids
        else:
            ids = tensorplay.as_tensor(np.asarray(input_ids), dtype=tensorplay.int64)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        return ids

    def lens(
        self,
        input_ids: Any,
        layers: Optional[Sequence[int]] = None,
        position: int = -1,
        topk: int = 5,
    ) -> Dict[str, Any]:
        """Return the top token predictions at every layer.

        Args:
            input_ids: Token ids (tensor, list, or numpy array).
            layers: Layer indices to inspect; ``None`` uses all.
            position: Token position of interest (negative indexes from end).
            topk: Number of top predictions per layer.

        Returns:
            A dict with keys ``layers``, ``predictions`` (mapping layer index
            to a list of ``(token, probability)`` pairs), and ``tokens``
            (the input tokens).
        """
        ids = self._as_input_ids(input_ids)
        captured = self._capture_hidden(ids, layers)
        seq = ids.shape[1]
        if position < 0:
            position = seq + position
        if not (0 <= position < seq):
            raise IndexError("position out of range")

        predictions: Dict[int, List[tuple]] = {}
        for layer_idx in sorted(captured):
            hidden = captured[layer_idx]
            logits = self._decode_layer(hidden)[0, position]
            probs = tensorplay.softmax(logits, dim=-1)
            top = tensorplay.topk(probs, min(topk, probs.numel()))
            preds = []
            for tid, p in zip(top.indices.tolist(), top.values.tolist()):
                token = self._token(tid)
                preds.append((token, p))
            predictions[layer_idx] = preds

        token_ids = ids[0].tolist()
        return {
            "layers": sorted(captured),
            "predictions": predictions,
            "tokens": [self._token(t) for t in token_ids],
        }

    def target_probability(
        self,
        input_ids: Any,
        target_token_id: int,
        layers: Optional[Sequence[int]] = None,
    ) -> Dict[int, float]:
        """Probability of a target token at the last position, per layer."""
        ids = self._as_input_ids(input_ids)
        captured = self._capture_hidden(ids, layers)
        result: Dict[int, float] = {}
        for layer_idx in sorted(captured):
            hidden = captured[layer_idx]
            logits = self._decode_layer(hidden)[0, -1]
            probs = tensorplay.softmax(logits, dim=-1)
            result[layer_idx] = float(probs[target_token_id])
        return result

    def _token(self, token_id: int) -> str:
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            try:
                return str(self.tokenizer.convert_ids_to_tokens([token_id])[0])
            except (KeyError, IndexError, ValueError):
                pass
        return str(token_id)

    def visualize(
        self,
        lens_result: Dict[str, Any],
        target_token: Optional[str] = None,
        figsize: Optional[Sequence[float]] = None,
        use_pyplot: bool = False,
    ) -> Any:
        """Plot per-layer probability curves for the top tokens.

        Args:
            lens_result: Output of :meth:`lens`.
            target_token: If given, always plot this token's probability.
            figsize: Optional ``(width, height)``.
            use_pyplot: If ``True`` calls ``plt.show()``.

        Returns:
            ``(fig, ax)`` matplotlib objects.
        """
        import matplotlib.pyplot as plt

        layers = lens_result["layers"]
        predictions = lens_result["predictions"]
        token_counts: Dict[str, int] = {}
        for li in layers:
            for token, _ in predictions[li]:
                token_counts[token] = token_counts.get(token, 0) + 1
        top_tokens = sorted(token_counts, key=token_counts.get, reverse=True)[:5]
        if target_token is not None and target_token not in top_tokens:
            top_tokens.append(target_token)

        fig, ax = plt.subplots(figsize=figsize if figsize is not None else (8, 5))
        for token in top_tokens:
            ys = []
            for li in layers:
                d = dict(predictions[li])
                ys.append(d.get(token, 0.0))
            ax.plot(layers, ys, marker="o", label=token, linewidth=1.6)
        ax.set_xlabel("layer")
        ax.set_ylabel("probability")
        ax.set_xticks(layers)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8, loc="best")
        ax.set_title("Logit lens")
        fig.tight_layout()
        if use_pyplot:
            plt.show()
        return fig, ax