"""Probabilities, expectation values, and shot-based sampling.

Probabilities are the squared amplitudes of the computational basis; they sum
to one for a normalized state and give the exact measurement distribution.
Sampling draws from that distribution on whichever device holds the state, so a
sampled batch never leaves the accelerator.
"""
from typing import Optional, Sequence

import tensorplay as tp

from . import _native
from ._gates import apply_gate, i_matrix, x_matrix, y_matrix, z_matrix
from ._helpers import check_pauli_string, n_qubits_of, ordered_wires
from ._state import unflatten

__all__ = [
    "apply_pauli_word",
    "counts_from_indices",
    "density_probs",
    "expectation",
    "marginal_probs",
    "measure",
    "probs",
    "sample_indices",
]

_PAULI_BUILDERS = {
    "I": i_matrix,
    "X": x_matrix,
    "Y": y_matrix,
    "Z": z_matrix,
}


def probs(state: tp.Tensor) -> tp.Tensor:
    """The measurement distribution of each state, shaped ``[batch, 2**n]``."""
    direct = _native.native_state_probs(state)
    if direct is not None:
        return direct
    flat = state.reshape(state.shape[0], -1)
    real = flat.real
    imag = flat.imag
    return real * real + imag * imag


def density_probs(density: tp.Tensor) -> tp.Tensor:
    """The measurement distribution read off a density matrix's diagonal."""
    batch = density.shape[0]
    diag = tp.diagonal(density.reshape(batch, density.shape[1], -1), dim1=1, dim2=2)
    return diag.real


def marginal_probs(state: tp.Tensor, wires: Sequence[int]) -> tp.Tensor:
    """The distribution of the listed qubits, marginalized over the rest.

    ``wires`` is ordered: entry ``i`` of the result is the ``i``-th listed
    qubit, most significant first.
    """
    n_qubits = n_qubits_of(state)
    target = ordered_wires(wires, n_qubits)
    missing = [w for w in range(n_qubits) if w not in target]
    # The square is elementwise, so moving the weights after the square sums
    # the same entries as moving the amplitudes before it -- and the weights
    # are the cheaper thing to rearrange.  Qubit w lives in dimension w + 1 of
    # the unfolded weights, batch aside.
    axes = (0, *(w + 1 for w in target), *(w + 1 for w in missing))
    weights = unflatten(probs(state), n_qubits).permute(axes)
    if missing:
        # An empty dim list would fold every dimension instead of none, so the
        # no-op case is spelled out.
        weights = weights.sum(dim=tuple(range(1 + len(target), 1 + n_qubits)))
    return weights.reshape(state.shape[0], 1 << len(target))


def apply_pauli_word(state: tp.Tensor, word: str) -> tp.Tensor:
    """``P|psi>`` for the Pauli word ``word``, applied one factor at a time.

    Each factor is a single-qubit gate, so the product costs one pass over the
    amplitudes per factor instead of the dense ``2**n`` by ``2**n`` matrix.
    """
    word = check_pauli_string(word, n_qubits_of(state))
    out = state
    for wire, letter in enumerate(word):
        if letter != "I":
            out = apply_gate(
                out, _PAULI_BUILDERS[letter](c_dtype=state.dtype), wire
            )
    return out


def expectation(state: tp.Tensor, word: str) -> tp.Tensor:
    """``<psi|P|psi>`` for the Pauli word ``word``, real and per batch entry.

    The state stays in the loop instead of being turned into a density matrix,
    so the cost grows with the state rather than with its square.
    """
    rotated = apply_pauli_word(state, word)
    overlap = (rotated.conj() * state)
    return overlap.sum(dim=tuple(range(1, state.dim()))).real


def sample_indices(distribution: tp.Tensor, shots: int) -> tp.Tensor:
    """Draw ``shots`` basis indices per row by inverse transform sampling.

    Walks the cumulative distribution with one uniform per shot and takes the
    first index whose cumulative weight reaches it -- the same draw the
    multinomial sampler produces, expressed so the whole batch stays on device.
    """
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")
    dim = distribution.shape[-1]
    cumulative = distribution.cumsum(dim=-1)
    # Rounding can leave the last cumulative weight a hair under one; widen the
    # last bucket so a uniform landing in that gap still names a basis entry.
    pad = (1.0 - cumulative[..., -1:]).clamp(min=0.0)
    cumulative = tp.cat([cumulative[..., :-1], cumulative[..., -1:] + pad], dim=-1)
    uniforms = tp.rand(*distribution.shape[:-1], shots, dtype=distribution.dtype,
                       device=distribution.device)
    indices = tp.searchsorted(cumulative.contiguous(), uniforms)
    return indices.clamp(max=dim - 1)


def counts_from_indices(indices: tp.Tensor, n_qubits: int) -> tp.Tensor:
    """Tally sampled basis indices into counts, shaped ``[batch, 2**n_qubits]``."""
    dim = 1 << n_qubits
    batch = indices.shape[0]
    rows = indices.reshape(batch, -1)
    counts = tp.zeros((batch, dim), dtype=tp.float32, device=indices.device)
    weights = tp.ones_like(rows, dtype=tp.float32)
    return counts.scatter_add_(1, rows, weights)


def measure(state: tp.Tensor, shots: int, wires: Optional[Sequence[int]] = None) -> tp.Tensor:
    """Sample ``shots`` measurements per state and tally them.

    Args:
        state: amplitudes shaped ``[batch, 2, 2, ...]``.
        shots: how many independent measurements to draw per state.
        wires: qubits to read out; all of them when omitted.  The result has
            one column per distinct outcome of the listed qubits.

    Returns:
        Counts shaped ``[batch, 2**len(wires)]``, real.
    """
    n_qubits = n_qubits_of(state)
    target = list(range(n_qubits)) if wires is None else ordered_wires(wires, n_qubits)
    marginal = marginal_probs(state, target)
    return counts_from_indices(sample_indices(marginal, shots), len(target))
