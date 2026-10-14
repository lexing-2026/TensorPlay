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
from ._measurement import expectation, expectation_from_density
from ._state import _density_width, flatten

__all__ = [
    "Hamiltonian",
    "energy",
    "energy_from_density",
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


def energy_from_density(density: tp.Tensor, hamiltonian: Hamiltonian) -> tp.Tensor:
    """``tr(H rho)`` for a mixed state, real and per batch entry.

    The companion of :func:`energy`: an open-system run answers with a density
    matrix, and the energy of a mixed state is the trace of its product rather
    than an overlap.
    """
    n_qubits = _density_width(density)
    if hamiltonian.n_qubits > n_qubits:
        raise ValueError(
            f"the Hamiltonian spans {hamiltonian.n_qubits} qubit(s), the density "
            f"matrix has {n_qubits}"
        )
    total = None
    for coeff, word in hamiltonian.terms:
        part = coeff * expectation_from_density(density, word.ljust(n_qubits, "I"))
        total = part if total is None else total + part
    return total


def _trotter_stages(order: int) -> List[Tuple[float, int]]:
    """The products that make up a Trotter product, as weights and repeats.

    Each entry is a weight on the time and how many times each term is applied
    -- once for a product run forwards only, twice for one run backwards after
    being run forwards.  The weight is on the angle, so a term applied twice
    carries half its time in each pass.

    Running a split forwards only is first-order however many times it is
    repeated, because the product it forms is not symmetric -- so a second-order
    scheme has to be built by undoing the first one, and a fourth-order one by
    composing that with uneven weights, which cancel the leading error rather
    than merely making the steps smaller.
    """
    if order == 1:
        return [(1.0, 1)]
    if order == 2:
        # Forwards then backwards: the palindromic product, and the second-order
        # scheme every other one is built from.
        return [(0.5, 2)]
    if order == 4:
        # That same symmetric product three times, the middle one with a weight
        # that is negative and larger than the others.  The weights are the
        # usual ones halved, because a term applied twice carries half its time
        # in each pass; they are chosen so the leading error terms of the three
        # cancel, which is what lifts the scheme from second order to fourth.
        first = 0.5 / (2.0 - 2.0 ** (1.0 / 3.0))
        middle = 0.5 - 2.0 * first
        return [(first, 2), (middle, 2), (first, 2)]
    raise ValueError(f"a Trotter order of 1, 2 or 4 is available, got {order}")


def trotter_evolve(state: tp.Tensor, hamiltonian: Hamiltonian, time: float, order: int = 1) -> tp.Tensor:
    """Advance the state by ``exp(-i * time * H)`` with a Trotter product.

    Each term's evolution is exact -- a Pauli word squares to the identity, so
    ``exp(-i tau P)`` is the rotation ``cos(tau) I - i sin(tau) P``.  The
    approximation is only in dropping the pieces of the terms that do not
    commute, and the order says how carefully that is done: a first-order
    product has an error that falls with the step, a second-order one has an
    error that falls with the square of it, and a fourth-order one with the
    fourth power, at a cost that grows the same way.

    Args:
        state: amplitudes shaped ``[batch, 2, 2, ...]``.
        hamiltonian: the generator.
        time: the evolution time.
        order: ``1``, ``2`` or ``4``; see :func:`_trotter_stages`.

    Returns:
        The evolved state.
    """
    stages = _trotter_stages(order)
    n_qubits = n_qubits_of(state)
    for coeff, word in hamiltonian.terms:
        if len(word) > n_qubits:
            raise ValueError(f"Pauli word {word!r} does not fit in {n_qubits} qubit(s)")

    from ._gates import apply_gate

    evolved = state
    for weight, repeats in stages:
        terms = list(hamiltonian.terms)
        if repeats > 1:
            # The same split run backwards after running forwards, which is what
            # makes the product symmetric; the time each term is given is split
            # with it, so the span is still the one asked for.
            terms = terms + terms[::-1]
        for coeff, word in terms:
            wires = pauli_wires(word)
            padded = word.ljust(n_qubits, "I")
            angle = 2.0 * time * coeff * weight
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
