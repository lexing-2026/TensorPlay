"""Probabilities, expectation values, and shot-based sampling.

Probabilities are the squared amplitudes of the computational basis; they sum
to one for a normalized state and give the exact measurement distribution.
Sampling draws from that distribution on whichever device holds the state, so a
sampled batch never leaves the accelerator.
"""
from typing import List, Optional, Sequence

import tensorplay as tp

from . import _native
from ._gates import (
    apply_gate,
    h_matrix,
    i_matrix,
    sdg_matrix,
    x_matrix,
    y_matrix,
    z_matrix,
)
from ._helpers import check_pauli_string, n_qubits_of, ordered_wires
from ._state import unflatten

__all__ = [
    "apply_measurement_basis",
    "apply_pauli_word",
    "commuting_groups",
    "counts_from_indices",
    "density_probs",
    "expectation",
    "expectation_from_counts",
    "joint_expectation",
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
    direct = _native.native_marginal_probs(state, target)
    if direct is not None:
        return direct
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
    so the cost grows with the state rather than with its square.  Every row of a
    Pauli matrix holds one entry, so the overlap needs no rotated copy of the
    state: one pass over the amplitudes decides each term's contribution.
    """
    checked = check_pauli_string(word, n_qubits_of(state))
    direct = _native.native_pauli_expectation(
        state, _native.pauli_codes(checked, state.device)
    )
    if direct is not None:
        return direct
    rotated = apply_pauli_word(state, checked)
    overlap = (rotated.conj() * state)
    return overlap.sum(dim=tuple(range(1, state.dim()))).real


def pauli_bit_mask(word: str, wires: Sequence[int]) -> int:
    """The amplitude-index bits a Pauli word reads, as a bit mask.

    Wire ``wires[i]`` is the ``i``-th most significant bit of the index, which is
    the same order :func:`measure` and :func:`marginal_probs` hand their columns
    in.
    """
    mask = 0
    width = len(wires)
    for position, wire in enumerate(wires):
        if word[wire] != "I":
            mask |= 1 << (width - 1 - position)
    return mask


def apply_measurement_basis(state: tp.Tensor, word: str) -> tp.Tensor:
    """Rotates the state so a plain histogram reads the signs of ``word``.

    Each wire the word acts on is turned so that reading it in the Z basis
    reports that factor's eigenvalue: ``X`` needs the Hadamard and ``Y`` needs
    the inverse phase gate followed by it, since ``H S^dagger Y S H = Z``.  The
    factors the word leaves alone are not touched, so a histogram over just its
    wires is enough.
    """
    word = check_pauli_string(word, n_qubits_of(state))
    out = state
    for wire, letter in enumerate(word):
        if letter == "X":
            out = apply_gate(out, h_matrix(c_dtype=state.dtype), [wire])
        elif letter == "Y":
            out = apply_gate(out, sdg_matrix(c_dtype=state.dtype), [wire])
            out = apply_gate(out, h_matrix(c_dtype=state.dtype), [wire])
    return out


def expectation_from_counts(
    counts: tp.Tensor, word: str, wires: Optional[Sequence[int]] = None
) -> tp.Tensor:
    """Estimates ``<P>`` for a Pauli word from a histogram of measurements.

    The estimate is the shot average of ``+1`` where the wires the word touches
    are jointly even and ``-1`` where they are jointly odd -- each of those
    wires contributes its own sign, so the product over them is the reading, and
    the sign of a wire follows its own bit rather than the outcome as a whole.
    The counts have to come from a state prepared by :func:`apply_measurement_basis`,
    which is what makes a ``Y`` wire readable at all.

    Args:
        counts: shots per outcome, shaped ``[batch, 2**len(wires)]``.
        word: the Pauli word, spelled over the whole register.
        wires: which wires the columns cover, in the order they were measured;
            the whole register when omitted.

    Returns:
        One estimate per state, shaped ``[batch]``.
    """
    n_columns = counts.shape[-1]
    if wires is None:
        wires = list(range(n_columns.bit_length() - 1))
    ordered = list(wires)
    if len(ordered) != n_columns.bit_length() - 1:
        raise ValueError(
            f"the counts have {n_columns.bit_length() - 1} column(s), but "
            f"{len(ordered)} wire(s) were named"
        )
    n_qubits = max(len(word), max(ordered) + 1)
    word = check_pauli_string(word, n_qubits)
    mask = pauli_bit_mask(word, ordered)

    outcomes = tp.arange(n_columns, device=counts.device)
    ones_up = tp.zeros_like(outcomes)
    for bit in range(n_columns.bit_length() - 1):
        if (mask >> bit) & 1:
            ones_up = ones_up + ((outcomes >> bit) & 1)
    # An even number of ones reads +1, an odd one reads -1.
    signs = 1 - 2 * (ones_up & 1).to(counts.dtype)
    return (counts * signs).sum(dim=-1) / counts.sum(dim=-1)


def commuting_groups(words: Sequence[str]) -> List[List[str]]:
    """Groups Pauli words that one basis rotation can measure together.

    Two words fit one rotation when they agree on every wire where both of them
    act -- each is then a sub-word of their product, so a single histogram of
    that product reads off all of them.  Words that fit together are also
    guaranteed to commute, which is why the groups come out pairwise commuting.

    The groups are built greedily and the result is not unique: a word joins the
    first group it fits, and the group's product grows to cover it.

    Args:
        words: the Pauli words to group, spelled over the same register.

    Returns:
        One list of words per measurement setting needed.
    """
    groups: List[List[str]] = []
    merged: List[str] = []
    width = 0
    for raw in words:
        word = check_pauli_string(raw)
        width = max(width, len(word))
    for raw in words:
        word = check_pauli_string(raw).ljust(width, "I")
        for index, representative in enumerate(merged):
            if all(
                letter == "I" or other == "I" or letter == other
                for letter, other in zip(word, representative)
            ):
                groups[index].append(word)
                merged[index] = "".join(
                    letter if letter != "I" else other
                    for letter, other in zip(word, representative)
                )
                break
        else:
            groups.append([word])
            merged.append(word)
    return groups


def group_product(words: Sequence[str]) -> str:
    """The product of a group from :func:`commuting_groups`, as a Pauli word.

    The members agree wherever two of them act, so the product is their
    position-wise merge and every member is a sub-word of it.
    """
    width = max(len(word) for word in words)
    product = ["I"] * width
    for word in words:
        padded = check_pauli_string(word).ljust(width, "I")
        for position, letter in enumerate(padded):
            if letter != "I":
                product[position] = letter
    return "".join(product)


def joint_expectation(
    state: tp.Tensor, words: Sequence[str], shots: int
) -> tp.Tensor:
    """Estimates several Pauli words from as few measurement settings as possible.

    The words are grouped by :func:`commuting_groups`; each group needs one
    rotation and one histogram, and every member is read off that same
    histogram.  Grouping is what keeps the noise down: words read together share
    their shots, while measured apart each would carry its own.

    Args:
        state: amplitudes shaped ``[batch, 2, 2, ...]``.
        words: the Pauli words to estimate, spelled over the whole register.
        shots: how many measurements to draw per group.

    Returns:
        One estimate per word, shaped ``[batch, len(words)]``, in the order the
        words were given.
    """
    if not words:
        raise ValueError("at least one Pauli word is needed")
    n_qubits = n_qubits_of(state)
    width = max(len(check_pauli_string(word)) for word in words)
    if width > n_qubits:
        raise ValueError(f"a Pauli word does not fit in {n_qubits} qubit(s)")
    words = [check_pauli_string(word).ljust(width, "I") for word in words]

    estimates: List[Optional[tp.Tensor]] = [None] * len(words)
    for group in commuting_groups(words):
        product = group_product(group)
        wires = [w for w, letter in enumerate(product) if letter != "I"]
        rotated = apply_measurement_basis(state, product)
        counts = measure(rotated, shots, wires)
        for word in group:
            estimates[words.index(word)] = expectation_from_counts(counts, word, wires)

    filled = [item for item in estimates if item is not None]
    dtype = filled[0].dtype if filled else tp.float32
    return tp.stack([item.to(dtype) for item in estimates], dim=-1)


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
