"""Activation patching and causal tracing for transformer language models.

Activation patching is a causal-intervention technique used in interpretability
research to locate which layers and token positions encode a given behavior. A
forward pass is run with the hidden state of one layer partially replaced by a
value taken from another run (a clean run, a corrupted run, or a zero ablation),
and the effect on the target output is measured. This is the diagnostic step
that knowledge-editing pipelines use to decide which weights to modify.

Everything here is framework-native: it operates on models built with
tensorplay primitives, uses tensorplay tensors and hooks, and requires no
third-party runtime. The only structural assumption is that the model exposes
its per-layer modules through a list-like container (``transformer.h`` for
GPT-family, ``model.layers`` for Llama-family, ``encoder.layer`` for BERT).
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorplay as tensorplay


def _resolve_indices(
    total: int, selection: Optional[Union[int, Sequence[int]]]
) -> List[int]:
    if selection is None:
        return list(range(total))
    if isinstance(selection, int):
        return [selection]
    return list(selection)


def _as_input_ids(model: Any, input_ids: Any) -> tensorplay.Tensor:
    """Convert python lists / numpy arrays to a tensorplay int64 tensor."""
    if isinstance(input_ids, tensorplay.Tensor):
        ids = input_ids
    else:
        ids = tensorplay.as_tensor(input_ids, dtype=tensorplay.int64)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    return ids.to(next(model.parameters()).device)


def resolve_layer_modules(
    model: Any, layer_selector: Optional[Union[int, Sequence[int]]] = None
) -> Dict[int, Any]:
    """Locate the per-layer modules of common transformer layouts.

    Returns a dict mapping layer indices to modules. Raises ``ValueError`` if
    no known layer container is found on the model.
    """
    containers = []
    for attr in ("transformer.h", "model.layers", "encoder.layer", "transformer.layer"):
        obj = model
        parts = attr.split(".")
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                obj = None
                break
        if obj is not None and hasattr(obj, "__len__"):
            containers.append((attr, obj))
    if not containers:
        raise ValueError(
            "Could not locate transformer layers on the model. Pass the "
            "layer modules explicitly or use a supported model family."
        )
    _, container = containers[0]
    if layer_selector is None:
        indices = list(range(len(container)))
    else:
        indices = _resolve_indices(len(container), layer_selector)
    return {i: container[i] for i in indices}


class ActivationPatching:
    """Causal intervention on transformer hidden states.

    Example::

        patcher = ActivationPatching(model)
        states = patcher.hidden_states(input_ids)
        logits = patcher.intervene(
            input_ids, layer=2, replacement=states[2], positions=[3]
        )
        trace = patcher.causal_trace(
            clean_input_ids, corrupted_input_ids, target_position=-1
        )
    """

    def __init__(self, model: Any, layers: Optional[Union[int, Sequence[int]]] = None):
        self.model = model
        self.layers = resolve_layer_modules(model, layers)

    # -- helpers ---------------------------------------------------------

    def _record_hidden(
        self, input_ids: Any, **model_kwargs: Any
    ) -> Dict[int, tensorplay.Tensor]:
        input_ids = _as_input_ids(self.model, input_ids)
        recorded: Dict[int, tensorplay.Tensor] = {}
        handles: List[Any] = []

        def make_hook(layer_idx: int) -> Callable:
            def hook(module: Any, args: Tuple[Any, ...], output: Any) -> None:
                h = output[0] if isinstance(output, tuple) else output
                recorded[layer_idx] = h.detach()

            return hook

        for layer_idx, module in self.layers.items():
            handles.append(module.register_forward_hook(make_hook(layer_idx)))
        try:
            with tensorplay.no_grad():
                self.model(input_ids=input_ids, **model_kwargs)
        finally:
            for handle in handles:
                handle.remove()
        return recorded

    # -- public API ------------------------------------------------------

    def hidden_states(
        self, input_ids: Any, **model_kwargs: Any
    ) -> Dict[int, np.ndarray]:
        """Run a forward pass and return the hidden state of each selected
        layer as numpy arrays with shape ``(batch, seq, hidden)``."""
        recorded = self._record_hidden(input_ids, **model_kwargs)
        return {idx: h.cpu().numpy() for idx, h in recorded.items()}

    def intervene(
        self,
        input_ids: Any,
        layer: int,
        replacement: Any,
        positions: Optional[Sequence[int]] = None,
        features: Optional[Sequence[int]] = None,
        mode: str = "replace",
        **model_kwargs: Any,
    ) -> Any:
        """Run a forward pass with a layer's hidden state partially rewritten.

        Args:
            input_ids: Token ids for the forward pass.
            layer: Layer index whose hidden state is modified.
            replacement: Tensor (or numpy array) of shape ``(batch, seq,
                hidden)`` whose values are written into the hidden state; for
                ``mode="ablate"`` this is ignored and the selected entries are
                zeroed.
            positions: Token positions to modify; ``None`` modifies all.
            features: Hidden dimensions to modify; ``None`` modifies all.
            mode: ``"replace"`` writes ``replacement`` values; ``"ablate"``
                zeroes the selected entries.
            **model_kwargs: Extra keyword arguments for the model forward.

        Returns:
            The model output object (``.logits`` on causal LMs).
        """
        input_ids = _as_input_ids(self.model, input_ids)
        if layer not in self.layers:
            raise KeyError(
                "Layer %d is not among the patching layers %s"
                % (layer, sorted(self.layers))
            )
        if mode not in ("replace", "ablate"):
            raise ValueError("mode must be 'replace' or 'ablate', got %r" % mode)

        if mode == "replace":
            if not isinstance(replacement, tensorplay.Tensor):
                replacement = tensorplay.as_tensor(
                    np.asarray(replacement), device=next(self.model.parameters()).device
                )
            if replacement.ndim == 2:
                replacement = replacement.unsqueeze(0)
            replacement = replacement.expand(input_ids.shape[0], -1, -1)

        if positions is None:
            pos_sel = slice(None)
        else:
            pos_sel = list(positions)
        if features is None:
            feat_sel = slice(None)
        else:
            feat_sel = list(features)

        def hook(module: Any, args: Tuple[Any, ...], output: Any) -> Any:
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            if mode == "replace":
                h[:, pos_sel, feat_sel] = replacement[:, pos_sel, feat_sel]
            else:
                h[:, pos_sel, feat_sel] = 0.0
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            with tensorplay.no_grad():
                return self.model(input_ids=input_ids, **model_kwargs)
        finally:
            handle.remove()

    def ablate(
        self,
        input_ids: Any,
        layer: int,
        positions: Optional[Sequence[int]] = None,
        features: Optional[Sequence[int]] = None,
        **model_kwargs: Any,
    ) -> Any:
        """Zero out a layer's hidden state at selected positions / features."""
        return self.intervene(
            input_ids,
            layer,
            replacement=None,
            positions=positions,
            features=features,
            mode="ablate",
            **model_kwargs,
        )

    def causal_trace(
        self,
        clean_input_ids: Any,
        corrupted_input_ids: Any,
        target: Optional[int] = None,
        target_position: int = -1,
        layers: Optional[Sequence[int]] = None,
        positions: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Measure how much restoring the clean hidden state at each
        ``(layer, position)`` recovers the clean prediction from a corrupted
        run.

        This is the single-token causal tracing protocol: the corrupted prompt
        is run forward, and at each candidate layer/position the corrupted
        hidden state is replaced by the corresponding clean hidden state. The
        score is the fraction of the clean-minus-corrupted logit gap that is
        recovered.

        Args:
            clean_input_ids: Token ids of the clean prompt.
            corrupted_input_ids: Token ids of the corrupted prompt (same
                sequence length; shorter of the two is used).
            target: Token id whose logit is tracked; ``None`` uses the token
                predicted by the clean run at ``target_position``.
            target_position: Token position of interest (negative indexes from
                the end).
            layers: Layer indices to evaluate; ``None`` uses all selected
                layers.
            positions: Token positions to evaluate; ``None`` uses all.

        Returns:
            A dict with keys ``scores`` (``(num_layers, num_positions)``
            numpy array), ``layers``, ``positions``, ``target_token_id``,
            ``clean_score`` and ``corrupted_score``.
        """
        clean = _as_input_ids(self.model, clean_input_ids)
        corrupted = _as_input_ids(self.model, corrupted_input_ids)
        seq_len = min(clean.shape[1], corrupted.shape[1])
        clean = clean[:, :seq_len]
        corrupted = corrupted[:, :seq_len]
        if target_position < 0:
            target_position = seq_len + target_position
        if not (0 <= target_position < seq_len):
            raise IndexError("target_position out of range")

        layer_dict = (
            self.layers
            if layers is None
            else {i: self.layers[i] for i in layers if i in self.layers}
        )
        if not layer_dict:
            raise ValueError("No matching layers to trace")

        with tensorplay.no_grad():
            clean_out = self.model(input_ids=clean)
            corrupted_out = self.model(input_ids=corrupted)
        if target is None:
            target = int(clean_out.logits[0, target_position].argmax())
        clean_val = float(clean_out.logits[0, target_position, target].item())
        corrupted_val = float(corrupted_out.logits[0, target_position, target].item())

        clean_states = self._record_hidden(clean)

        layer_list = sorted(layer_dict)
        pos_list = list(range(seq_len)) if positions is None else list(positions)
        scores = np.zeros((len(layer_list), len(pos_list)), dtype=float)

        for r, layer_idx in enumerate(layer_list):
            module = layer_dict[layer_idx]
            clean_h = clean_states[layer_idx]

            def make_hook(clean_h: tensorplay.Tensor, pos: int) -> Callable:
                def hook(module_: Any, args: Tuple[Any, ...], output: Any) -> Any:
                    h = output[0] if isinstance(output, tuple) else output
                    h = h.clone()
                    h[:, pos, :] = clean_h[:, pos, :]
                    if isinstance(output, tuple):
                        return (h,) + output[1:]
                    return h

                return hook

            for c, pos in enumerate(pos_list):
                handle = module.register_forward_hook(make_hook(clean_h, pos))
                try:
                    with tensorplay.no_grad():
                        patched = self.model(input_ids=corrupted)
                finally:
                    handle.remove()
                patched_val = float(
                    patched.logits[0, target_position, target].item()
                )
                if abs(clean_val - corrupted_val) < 1e-12:
                    scores[r, c] = 0.0
                else:
                    scores[r, c] = (patched_val - corrupted_val) / (
                        clean_val - corrupted_val
                    )

        return {
            "scores": scores,
            "layers": layer_list,
            "positions": pos_list,
            "target_token_id": target,
            "clean_score": clean_val,
            "corrupted_score": corrupted_val,
        }