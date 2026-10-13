"""Native fast paths for the state simulation primitives.

Four things in the module have a compiled counterpart: applying a local unitary
to the targeted wires, turning amplitudes into measurement probabilities,
marginalizing that distribution onto a subset of the qubits, and reading a Pauli
expectation straight off the amplitudes.  All of them read the flat layout
``[batch, 2**n]``, where qubit ``w`` owns bit ``n-1-w`` of the amplitude index --
the same indexing a folded view of a ``[batch, 2, 2, ...]`` state produces, so
each becomes a single pass over the amplitudes with no permuted intermediate
copy.

A gate's gradient is carried by the decomposition instead (permute, batched
matrix multiply, permute back), which composes from existing primitives and
stays differentiable in its parameters.  These paths therefore only take over
when no gradient is being recorded; otherwise the decomposition runs.
"""
from typing import Optional, Sequence

import tensorplay as tp

from tensorplay import _C

__all__ = [
    "native_apply_gate",
    "native_marginal_probs",
    "native_pauli_expectation",
    "native_state_probs",
    "records_grad",
]

# Resolved once: the extension is a build product, so a tree built before these
# ops existed simply runs the decomposition instead of failing at import.
_APPLY_GATE = getattr(_C, "_quantum_apply_gate", None)
_STATE_PROBS = getattr(_C, "_quantum_state_probs", None)
_MARGINAL_PROBS = getattr(_C, "_quantum_marginal_probs", None)
_PAULI_EXPECTATION = getattr(_C, "_quantum_pauli_expectation", None)


def records_grad(*tensors: Optional[tp.Tensor]) -> bool:
    """Whether a graph is being recorded for any of these tensors."""
    if not tp.is_grad_enabled():
        return False
    return any(
        isinstance(item, tp.Tensor) and item.requires_grad for item in tensors
    )


def _flat(state: tp.Tensor) -> tp.Tensor:
    """The amplitudes of a state as ``[batch, 2**n]``, contiguous."""
    return state.reshape(state.shape[0], -1).contiguous()


def native_apply_gate(
    state: tp.Tensor, mat: tp.Tensor, wires: Sequence[int]
) -> Optional[tp.Tensor]:
    """Applies a local unitary to a flat state, or returns ``None``.

    Args:
        state: amplitudes shaped ``[batch, 2**n]``.
        mat: the unitary, ``[2**k, 2**k]`` or ``[batch, 2**k, 2**k]``.
        wires: the target qubits, ascending.

    Returns:
        The new amplitudes in the same layout, or ``None`` when the compiled
        path is unavailable or a gradient is being recorded.
    """
    if _APPLY_GATE is None or records_grad(state, mat):
        return None
    return _APPLY_GATE(_flat(state), mat.contiguous(), [int(wire) for wire in wires])


def native_state_probs(state: tp.Tensor) -> Optional[tp.Tensor]:
    """Squares the amplitudes of a flat state, or returns ``None``."""
    if _STATE_PROBS is None or records_grad(state):
        return None
    return _STATE_PROBS(_flat(state))


def native_marginal_probs(
    state: tp.Tensor, wires: Sequence[int]
) -> Optional[tp.Tensor]:
    """The distribution of the listed qubits, or returns ``None``.

    The order given is the column order: entry ``i`` reads the ``i``-th listed
    qubit, most significant first.
    """
    if _MARGINAL_PROBS is None or records_grad(state):
        return None
    return _MARGINAL_PROBS(_flat(state), [int(wire) for wire in wires])


# The compiled path reads a word as one small integer per qubit; the codes are
# short enough that a narrow int carries them comfortably.
_PAULI_CODES = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def pauli_codes(word: str, device) -> tp.Tensor:
    """The per-qubit codes a compiled expectation value reads."""
    codes = [_PAULI_CODES[letter] for letter in word]
    return tp.tensor(codes, dtype=tp.int32, device=device)


def native_pauli_expectation(
    state: tp.Tensor, codes: tp.Tensor
) -> Optional[tp.Tensor]:
    """The overlap of each state with the Pauli word, or returns ``None``.

    Args:
        state: amplitudes shaped ``[batch, 2**n]``.
        codes: one entry per qubit, ``0`` for I, ``1`` for X, ``2`` for Y and
            ``3`` for Z.
    """
    if _PAULI_EXPECTATION is None or records_grad(state):
        return None
    return _PAULI_EXPECTATION(_flat(state), codes.contiguous())