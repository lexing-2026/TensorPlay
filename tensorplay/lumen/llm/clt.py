"""Cross-layer transcoders (CLTs) for transformer residual-stream updates.

A cross-layer transcoder is a sparse dictionary model trained to predict the
*update* a transformer component adds to the residual stream, given the residual
stream entering that component. Whereas a sparse autoencoder reconstructs a
single activation, a CLT reconstructs the component's output delta as a sparse
combination of features, decomposing the computation performed by the component
into interpretable pieces. This is framework-native: it runs entirely on
tensorplay primitives.
"""

from typing import Any, List, Optional, Sequence, Tuple

import tensorplay as tensorplay
from tensorplay import nn

from tensorplay.lumen.llm.patching import resolve_layer_modules


class CrossLayerTranscoder(nn.Module):
    """Sparse transcoder: ``delta ~= decode(relu(encode(residual)))``.

    Args:
        input_dim: Dimension of the residual stream.
        hidden_dim: Number of learned features.
        tied_decoder: If ``True``, the decoder weight is the transpose of the
            encoder weight.
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
        """Return ``(predicted_delta, features)``."""
        features = tensorplay.relu(self.encoder(x))
        return self.decoder(features), features

    def encode(self, x: tensorplay.Tensor) -> tensorplay.Tensor:
        """Return sparse feature activations for the incoming residual stream."""
        return tensorplay.relu(self.encoder(x))

    def decode(self, features: tensorplay.Tensor) -> tensorplay.Tensor:
        """Reconstruct a component output delta from sparse features."""
        return self.decoder(features)


def train_clt(
    clt: CrossLayerTranscoder,
    dataloader: Any,
    n_epochs: int = 10,
    lr: float = 1e-3,
    l1_coef: float = 1e-3,
    optimizer_cls: Any = None,
    device: Optional[tensorplay.device] = None,
    show_progress: bool = False,
) -> List[float]:
    """Train a cross-layer transcoder on ``(residual, delta)`` pairs.

    The loss is ``mse(predicted_delta, delta) + l1_coef * mean(|features|)``.

    Args:
        clt: The transcoder to train.
        dataloader: Iterable of batches yielding ``(residual, delta)`` pairs.
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
        clt.to(device)
    optimizer = optimizer_cls(clt.parameters(), lr=lr)
    history: List[float] = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            residual, delta = batch[0], batch[1]
            if device is not None:
                residual = residual.to(device)
                delta = delta.to(device)
            optimizer.zero_grad()
            predicted, features = clt(residual)
            mse = ((predicted - delta) ** 2).mean()
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


def collect_component_updates(
    model: Any,
    layer: int,
    input_ids_list: Sequence[Any],
    batch_size: int = 8,
) -> Tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Collect ``(residual_before, component_delta)`` pairs from a layer.

    The delta is the full block update ``block_out - block_in`` (attention plus
    MLP contributions added to the residual stream). Both tensors are flattened
    over batch and sequence positions.

    Args:
        model: A transformer model with per-layer modules.
        layer: Layer index whose block update is collected.
        input_ids_list: List of token-id sequences.
        batch_size: Number of sequences per forward pass.

    Returns:
        A tuple ``(inputs, deltas)`` of tensors with shape
        ``(total_tokens, hidden_dim)``.
    """
    layers = resolve_layer_modules(model, layer)
    if not layers:
        raise ValueError("Layer %d not found" % layer)
    module = layers[layer]
    captured_in: List[tensorplay.Tensor] = []
    captured_out: List[tensorplay.Tensor] = []
    handle: Any = None

    def hook(mod: Any, args: Any, output: Any) -> None:
        x_in = args[0] if isinstance(args, tuple) else args
        x_out = output[0] if isinstance(output, tuple) else output
        captured_in.append(x_in.detach())
        captured_out.append(x_out.detach())

    handle = module.register_forward_hook(hook)
    inputs: List[tensorplay.Tensor] = []
    deltas: List[tensorplay.Tensor] = []
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
            for x_in, x_out in zip(captured_in, captured_out):
                h_in = x_in.reshape(-1, x_in.shape[-1])
                delta = (x_out - x_in).reshape(-1, x_out.shape[-1])
                inputs.append(h_in)
                deltas.append(delta)
            captured_in.clear()
            captured_out.clear()
    finally:
        if handle is not None:
            handle.remove()
    if not inputs:
        raise ValueError("No component updates were collected")
    return tensorplay.cat(inputs, dim=0), tensorplay.cat(deltas, dim=0)