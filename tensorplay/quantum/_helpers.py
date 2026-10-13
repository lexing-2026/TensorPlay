"""Shared spelling for wire lists, layouts, and complex construction.

A state tensor carries the batch in dimension 0 and one dimension of size 2 per
qubit after it, so qubit ``w`` lives in dimension ``w + 1``.  Qubit 0 is the
leading -- most significant -- index: for a flat amplitude index ``i`` the
qubit-``w`` bit is ``(i >> (n_qubits - 1 - w)) & 1``.

Gate matrices are assembled from real pieces (cosines, sines, parameters) so a
parameter that carries a gradient flows through real arithmetic only; the
imaginary halves are attached afterwards with :func:`tensorplay.complex`.
"""
import operator

import tensorplay as tp

__all__ = [
    "DEFAULT_C_DTYPE",
    "check_pauli_string",
    "check_wires",
    "n_qubits_of",
    "ordered_wires",
    "pauli_wires",
    "promoted_c_dtype",
    "state_shape",
]

DEFAULT_C_DTYPE = tp.complex64


def promoted_c_dtype(dtype):
    """The complex dtype that keeps the precision of ``dtype``."""
    return tp.complex128 if dtype in (tp.float64, tp.complex128) else DEFAULT_C_DTYPE


def n_qubits_of(state):
    """The qubit count of a ``[batch, 2, 2, ...]`` state tensor."""
    return state.dim() - 1


def state_shape(n_qubits, batch=1):
    """The shape of an unbent ``|0...0>`` state with ``n_qubits`` qubits."""
    return (batch, *([2] * n_qubits))


def ordered_wires(wires, n_qubits):
    """Normalize a wire spec to distinct qubit indices, keeping the given order.

    Readout keeps the order it was handed -- entry ``i`` of a marginal or a
    histogram is the ``i``-th listed qubit -- while a gate needs ascending
    wires (see :func:`check_wires`).
    """
    if isinstance(wires, int):
        wires = [wires]
    try:
        normalized = list(dict.fromkeys(operator.index(w) for w in wires))
    except TypeError:
        raise TypeError(f"wires must be an int or a sequence of ints, got {wires!r}") from None
    if not normalized:
        raise ValueError("wires must name at least one qubit")
    for wire in normalized:
        if wire < 0 or wire >= n_qubits:
            raise IndexError(
                f"wire {wire} is out of range for a state with {n_qubits} qubit(s)"
            )
    return normalized


def check_wires(wires, n_qubits):
    """Normalize a wire spec to a sorted list of distinct qubit indices.

    The gate application moves the targeted dimensions to the front of the
    state, so their order decides which bit of the gate matrix maps to which
    qubit; requiring ascending order keeps that mapping the same everywhere.
    """
    return sorted(ordered_wires(wires, n_qubits))


def check_pauli_string(pauli, n_qubits=None):
    """Normalize a Pauli word to upper case, one character per qubit."""
    if not isinstance(pauli, str):
        raise TypeError(f"a Pauli word is a string of I/X/Y/Z, got {type(pauli).__name__}")
    word = pauli.upper()
    if not word:
        raise ValueError("a Pauli word must not be empty")
    bad = set(word) - set("IXYZ")
    if bad:
        raise ValueError(f"a Pauli word contains only I, X, Y and Z; found {sorted(bad)}")
    if n_qubits is not None and len(word) != n_qubits:
        raise ValueError(f"Pauli word {word!r} does not span {n_qubits} qubit(s)")
    return word


def pauli_wires(word):
    """The qubit indices a Pauli word acts on, in ascending order."""
    return [i for i, letter in enumerate(word) if letter != "I"]
