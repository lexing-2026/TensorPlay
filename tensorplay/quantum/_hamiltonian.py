"""Pauli-sum Hamiltonians, their energy, and time evolution.

A Hamiltonian is a weighted sum of Pauli words.  Because every Pauli word
squares to the identity, the evolution of a single term is a rotation about
that word with a closed form -- no matrix exponential and no diagonalization --
which is what makes a Trotter step cheap and differentiable.  The dense
exponential is offered alongside for small registers, where it is exact.
"""
from typing import Iterable, List, Optional, Tuple

import tensorplay as tp

from ._circuit import expand_to_register
from ._gates import pauli_matrix, pauli_rotation_matrix
from ._helpers import DEFAULT_C_DTYPE, check_pauli_string, n_qubits_of, pauli_wires
from ._measurement import expectation
from ._state import flatten

__all__ = [
    "Hamiltonian",
    "energy",
    "matrix_exp_evolve",
    "pauli_string_matrix",
    "trotter_evolve",
]


def pauli_string_matrix(word: str, *, device=None, c_dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
    """The dense matrix of a Pauli word.

    Identity factors are dropped first, so the result spans only the wires the
    word acts on -- ``"XIY"`` gives a 4 by 4 matrix, not an 8 by 8 one.
    """
    return pauli_matrix(word, device=device, c_dtype=c_dtype)


class Hamiltonian:
    """A weighted sum of Pauli words.

    Args:
        terms: ``(coefficient, word)`` pairs.
        n_qubits: register width; taken from the longest word when omitted.

    Example:
        >>> h = Hamiltonian([(-1.0523732, "II"), (0.3979374, "IZ"),
        ...                  (-0.3979374, "ZI"), (0.1809312, "XX")])
        >>> energy(state)
    """

    def __init__(self, terms: Iterable[Tuple[float, str]], n_qubits: Optional[int] = None):
        collected: List[Tuple[float, str]] = []
        for term in terms:
            if not isinstance(term, (tuple, list)) or len(term) != 2:
                raise ValueError(f"a term is a (coefficient, word) pair, got {term!r}")
            coeff, word = term
            collected.append((float(coeff), check_pauli_string(str(word))))
        if not collected:
            raise ValueError("a Hamiltonian needs at least one term")
        widths = {len(word) for _, word in collected}
        if len(widths) != 1:
            raise ValueError(
                f"every Pauli word must span the same register, got widths {sorted(widths)}"
            )
        span = widths.pop()
        if n_qubits is not None and n_qubits != span:
            raise ValueError(
                f"the terms span {span} qubit(s), the register holds {n_qubits}"
            )
        self.coeffs: List[float] = [coeff for coeff, _ in collected]
        self.paulis: List[str] = [word for _, word in collected]
        self.n_qubits = span

    @property
    def terms(self) -> List[Tuple[float, str]]:
        """The ``(coefficient, word)`` pairs."""
        return list(zip(self.coeffs, self.paulis))

    def __len__(self) -> int:
        return len(self.paulis)

    def __repr__(self) -> str:
        body = ", ".join(f"({c!r}, {w!r})" for c, w in self.terms)
        return f"Hamiltonian(n_qubits={self.n_qubits}, terms=[{body}])"

    def matrix(self, *, device=None, c_dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
        """The dense matrix of the sum over the whole register.

        Every word spans ``n_qubits`` already, so each term is lifted onto the
        register -- a word of pure identities contributing its coefficient along
        the diagonal -- and the results are added.
        """
        total = None
        for coeff, word in self.terms:
            wires = pauli_wires(word)
            if wires:
                local = pauli_string_matrix(word, device=device, c_dtype=c_dtype)
                wide = expand_to_register(local, wires, self.n_qubits)
            else:
                wide = tp.eye(1 << self.n_qubits, dtype=c_dtype, device=device)
            piece = coeff * wide
            total = piece if total is None else total + piece
        return total

    def register_matrix(self, *, device=None, c_dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
        """The dense matrix on all ``n_qubits`` of the register."""
        return self.matrix(device=device, c_dtype=c_dtype)


def energy(state: tp.Tensor, hamiltonian: Hamiltonian) -> tp.Tensor:
    """``<psi|H|psi>``, real and per batch entry.

    Each word is applied to the state one factor at a time, so the cost grows
    with the state rather than with its square.
    """
    n_qubits = n_qubits_of(state)
    if hamiltonian.n_qubits > n_qubits:
        raise ValueError(
            f"the Hamiltonian spans {hamiltonian.n_qubits} qubit(s), the state has {n_qubits}"
        )
    total = None
    for coeff, word in hamiltonian.terms:
        padded = word.ljust(n_qubits, "I")
        part = coeff * expectation(state, padded)
        total = part if total is None else total + part
    return total


def trotter_evolve(state: tp.Tensor, hamiltonian: Hamiltonian, time: float, order: int = 1) -> tp.Tensor:
    """Advance the state by ``exp(-i * time * H)`` with a Trotter product.

    Each term's evolution is exact -- a Pauli word squares to the identity, so
    ``exp(-i tau P)`` is the rotation ``cos(tau) I - i sin(tau) P``.  The
    approximation is only in dropping non-commuting terms of the product, which
    a larger ``order`` shrinks at a proportional cost.

    Args:
        state: amplitudes shaped ``[batch, 2, 2, ...]``.
        hamiltonian: the generator.
        time: the evolution time.
        order: how many times the exponential factors are split.

    Returns:
        The evolved state.
    """
    if order < 1:
        raise ValueError(f"a Trotter order is at least 1, got {order}")
    n_qubits = n_qubits_of(state)
    for coeff, word in hamiltonian.terms:
        if len(word) > n_qubits:
            raise ValueError(f"Pauli word {word!r} does not fit in {n_qubits} qubit(s)")

    from ._gates import apply_gate

    evolved = state
    for _ in range(order):
        for coeff, word in hamiltonian.terms:
            wires = pauli_wires(word)
            padded = word.ljust(n_qubits, "I")
            angle = 2.0 * time * coeff / order
            rotation = pauli_rotation_matrix(padded, angle, c_dtype=state.dtype)
            # An identity word owns no wires; routing its 1x1 rotation through
            # wire 0 keeps the global phase it carries.
            evolved = apply_gate(evolved, rotation.to(device=state.device), wires or [0])
    return evolved


def matrix_exp_evolve(state: tp.Tensor, hamiltonian: Hamiltonian, time: float) -> tp.Tensor:
    """Advance the state by the exact ``exp(-i * time * H)``.

    Dense and therefore restricted to small registers: the exponential is
    computed over the full ``2**n`` by ``2**n`` matrix.
    """
    n_qubits = n_qubits_of(state)
    if hamiltonian.n_qubits > n_qubits:
        raise ValueError(
            f"the Hamiltonian spans {hamiltonian.n_qubits} qubit(s), the state has {n_qubits}"
        )
    generator = hamiltonian.register_matrix(device=state.device, c_dtype=state.dtype)
    propagator = tp.matrix_exp(-1j * time * generator)
    flat = flatten(state).to(propagator.dtype)
    return (propagator @ flat.transpose(-1, -2)).transpose(-1, -2).reshape(state.shape)
