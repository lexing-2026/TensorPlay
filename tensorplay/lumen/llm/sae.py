"""Sparse autoencoder feature learning for transformer hidden states.

Sparse autoencoders (SAEs) learn a dictionary of sparse, interpretable features
from a model's activations: every hidden state is approximately reconstructed
as a sparse non-negative combination of learned feature directions. This is the
dictionary-learning step used in feature research. Everything is framework-native
and runs on tensorplay primitives; no third-party runtime is needed.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorplay as tensorplay
from tensorplay import nn

from tensorplay.lumen.llm.patching import resolve_layer_modules


class SparseAutoencoder(nn.Module):
    """One-layer ReLU sparse autoencoder: ``x ~= decode(relu(encode(x)))``.

    Args:
        input_dim: Dimension of the input activations.
        hidden_dim: Number of learned features (usually larger than
            ``input_dim``).
        tied_decoder: If ``True``, the decoder weight is the transpose of the
            encoder weight (weight tying).
    """

    def __init__(self, input_dim: int, hidden_dim: int, tied_decoder: bool = False) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        if tied_decoder:
            self.decoder.weight = self.encoder.weight.t()  # type: ignore[assignment]
            self.decoder.bias = tensorplay.zeros(input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x: tensorplay.Tensor) -> Tuple[tensorplay.Tensor, tensorplay.Tensor]:
        """Return ``(reconstruction, features)``."""
        features = tensorplay.relu(self.encoder(x))
        return self.decoder(features), features

    def encode(self, x: tensorplay.Tensor) -> tensorplay.Tensor:
        """Return sparse feature activations for ``x``."""
        return tensorplay.relu(self.encoder(x))

    def decode(self, features: tensorplay.Tensor) -> tensorplay.Tensor:
        """Reconstruct input activations from sparse features."""
        return self.decoder(features)


def train_sae(
    sae: SparseAutoencoder,
    dataloader: Any,
    n_epochs: int = 10,
    lr: float = 1e-3,
    l1_coef: float = 1e-3,
    optimizer_cls: Any = None,
    device: Optional[tensorplay.device] = None,
    show_progress: bool = False,
) -> List[float]:
    """Train a sparse autoencoder on hidden states.

    The loss is ``mse(reconstruction, x) + l1_coef * mean(|features|)``, which
    encourages sparse, faithful feature dictionaries.

    Args:
        sae: The autoencoder to train.
        dataloader: Iterable of batches of hidden states.
        n_epochs: Number of passes over the data.
        lr: Learning rate.
        l1_coef: Sparsity penalty coefficient.
        optimizer_cls: Optimizer class (default ``tensorplay.optim.Adam``).
        device: Optional device to move the model to.
        show_progress: If ``True``, prints per-epoch loss.

    Returns:
        List of mean loss values, one per epoch.
    """
    if optimizer_cls is None:
        optimizer_cls = tensorplay.optim.Adam
    if device is not None:
        sae.to(device)
    optimizer = optimizer_cls(sae.parameters(), lr=lr)
    history: List[float] = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            if device is not None:
                x = x.to(device)
            optimizer.zero_grad()
            recon, features = sae(x)
            mse = ((recon - x) ** 2).mean()
            sparsity = features.abs().mean()
            loss = mse + l1_coef * sparsity
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            n_batches += 1
        mean_loss = epoch_loss / max(1, n_batches)
        history.append(mean_loss)
        if show_progress:
            print("epoch %d loss %.6f" % (epoch, mean_loss))
    return history


def collect_hidden_states(
    model: Any,
    layer: int,
    input_ids_list: Sequence[Any],
    batch_size: int = 8,
) -> tensorplay.Tensor:
    """Extract hidden states from a layer for SAE training data.

    Args:
        model: A transformer model with a layer container (see
            :func:`~tensorplay.lumen.llm.patching.resolve_layer_modules`).
        layer: Layer index whose output hidden state is collected.
        input_ids_list: List of token-id sequences (tensors, lists, or numpy
            arrays).
        batch_size: Number of sequences processed per forward pass.

    Returns:
        A tensor of shape ``(total_tokens, hidden_dim)`` with the layer's
        hidden states flattened over batch and sequence positions.
    """
    layers = resolve_layer_modules(model, layer)
    if not layers:
        raise ValueError("Layer %d not found" % layer)
    module = layers[layer]
    captured: List[tensorplay.Tensor] = []
    handle: Any = None

    def hook(mod: Any, args: Any, output: Any) -> None:
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach())

    handle = module.register_forward_hook(hook)
    collected: List[tensorplay.Tensor] = []
    try:
        for start in range(0, len(input_ids_list), batch_size):
            chunk = input_ids_list[start : start + batch_size]
            ids = tensorplay.stack(
                [
                    tensorplay.as_tensor(s, dtype=tensorplay.int64).reshape(-1)
                    if not isinstance(s, tensorplay.Tensor)
                    else s.reshape(-1)
                    for s in chunk
                ]
            )
            with tensorplay.no_grad():
                model(input_ids=ids)
            collected.extend(captured)
            captured.clear()
    finally:
        if handle is not None:
            handle.remove()
    if not collected:
        raise ValueError("No hidden states were collected")
    return tensorplay.cat([h.reshape(-1, h.shape[-1]) for h in collected], dim=0)


def feature_frequency(features: tensorplay.Tensor) -> tensorplay.Tensor:
    """Fraction of tokens on which each feature is active (``> 0``)."""
    return (features > 0).float().mean(dim=0)


def top_activating_indices(
    features: tensorplay.Tensor, k: int = 5
) -> Dict[int, List[int]]:
    """For each feature, the indices of the ``k`` most activating examples.

    Args:
        features: ``(n_examples, n_features)`` activation matrix.
        k: Number of top examples per feature.

    Returns:
        Dict mapping feature index to a list of example indices, sorted by
        activation descending.
    """
    result: Dict[int, List[int]] = {}
    for f in range(features.shape[1]):
        vals = features[:, f]
        top = tensorplay.topk(vals, min(k, vals.numel())).indices.tolist()
        result[f] = top
    return result


def reconstruction_error(
    sae: SparseAutoencoder, x: tensorplay.Tensor
) -> float:
    """Mean squared reconstruction error of the autoencoder on ``x``."""
    with tensorplay.no_grad():
        recon, _ = sae(x)
        return float(((recon - x) ** 2).mean())