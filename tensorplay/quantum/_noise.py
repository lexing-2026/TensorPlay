"""Noise channels on a density matrix.

A channel is given as its Kraus operators: the map it induces on a density
matrix is ``rho -> sum_r K_r rho K_r^dagger``, applied on the wires it acts
upon.  Every Kraus operator is a local unitary of the same width, so each term
reuses the single gate primitive -- no dense channel superoperator is ever
built, and the cost stays linear in the register size.
"""
from typing import Sequence

import tensorplay as tp

from ._circuit import expand_to_register
from ._gates import (
    apply_gate,
    dagger,
    i_matrix,
    pauli_matrix,
    x_matrix,
    y_matrix,
    z_matrix,
)
from ._helpers import DEFAULT_C_DTYPE, check_pauli_string, check_wires

__all__ = [
    "amplitude_damping_kraus",
    "apply_amplitude_damping",
    "apply_bit_flip",
    "apply_channel",
    "apply_depolarizing",
    "apply_phase_flip",
    "bit_flip_kraus",
    "depolarizing_kraus",
    "pauli_kraus",
    "phase_flip_kraus",
]

# A Pauli channel over k qubits carries one operator per word, so naming every
# word on four wires is a quarter of a thousand matrices of sixteen by sixteen.
# Past that the cost of reading them starts to dwarf the cost of the noise.
_MAX_CHANNEL_WIRES = 4


def _check_rate(value, name: str) -> None:
    """Range check for a channel strength given as a plain number."""
    if isinstance(value, (int, float)) and not 0.0 <= value <= 1.0:
        raise ValueError(f"a channel {name} lies in [0, 1], got {value}")


def _root(value, device, c_dtype):
    """``sqrt(value)`` as a complex scalar, keeping any gradient on ``value``.

    A plain number is read in double precision: a channel strength carries no
    precision of its own to preserve, and reading it in single precision would
    leave every operator built from it a few digits short of the register it
    will act on.
    """
    tensor = (
        value
        if isinstance(value, tp.Tensor)
        else tp.as_tensor(value, dtype=tp.float64, device=device)
    )
    return tp.sqrt(tensor).to(dtype=c_dtype, device=device)


def bit_flip_kraus(p: float, *, device=None, c_dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
    """Kraus operators of the bit-flip channel: X flips the qubit with chance ``p``."""
    _check_rate(p, "probability")
    stay = _root(1.0 - p, device, c_dtype)
    flip = _root(p, device, c_dtype)
    identity = i_matrix(device=device, c_dtype=c_dtype)
    x = x_matrix(device=device, c_dtype=c_dtype)
    return tp.stack([stay * identity, flip * x])


def phase_flip_kraus(p: float, *, device=None, c_dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
    """Kraus operators of the phase-flip channel: Z rotates the qubit with chance ``p``."""
    _check_rate(p, "probability")
    stay = _root(1.0 - p, device, c_dtype)
    flip = _root(p, device, c_dtype)
    identity = i_matrix(device=device, c_dtype=c_dtype)
    z = z_matrix(device=device, c_dtype=c_dtype)
    return tp.stack([stay * identity, flip * z])


def amplitude_damping_kraus(gamma: float, *, device=None,
                            c_dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
    """Kraus operators of amplitude damping: the qubit relaxes to 0 with rate ``gamma``."""
    _check_rate(gamma, "damping rate")
    survive = _root(1.0 - gamma, device, c_dtype)
    decay = _root(gamma, device, c_dtype)
    zero = tp.zeros((), dtype=c_dtype, device=device)
    one = tp.ones((), dtype=c_dtype, device=device)
    keep = tp.stack([tp.stack([one, zero]), tp.stack([zero, survive])])
    jump = tp.stack([tp.stack([zero, decay]), tp.stack([zero, zero])])
    return tp.stack([keep, jump])


def depolarizing_kraus(p: float, *, device=None, c_dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
    """Kraus operators of the depolarizing channel."""
    _check_rate(p, "probability")
    stay = _root(1.0 - p, device, c_dtype)
    share = _root(p / 3.0, device, c_dtype)
    return tp.stack(
        [
            stay * i_matrix(device=device, c_dtype=c_dtype),
            share * x_matrix(device=device, c_dtype=c_dtype),
            share * y_matrix(device=device, c_dtype=c_dtype),
            share * z_matrix(device=device, c_dtype=c_dtype),
        ]
    )


def pauli_kraus(errors, *, device=None, c_dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
    """Kraus operators of a Pauli channel: named errors with their chances.

    A Pauli channel applies one of a set of Pauli words, and names which with
    what chance; whatever is left over is the chance of no error at all, which
    is the identity operator's share.  That is how a noise model is usually
    written down -- ``{"XX": 0.02, "ZI": 0.01}`` says a two-qubit register gets
    two per-cent X-on-both errors and one per-cent Z-on-the-second, and nothing
    else -- and building the operators from it by hand is the fiddly part.

    Args:
        errors: the named errors as ``(word, chance)`` pairs, in a mapping or a
            sequence.  Every word must span the same number of qubits.
        device: where to build them.
        c_dtype: the precision to build them in.

    Returns:
        The operators, stacked as ``[r, 2**k, 2**k]``.

    Raises:
        ValueError: if a word is not a Pauli word, if the words are of different
            lengths, or if the chances run past one.
    """
    items = list(errors.items() if hasattr(errors, "items") else errors)
    if not items:
        raise ValueError("a channel needs at least one error, or an empty chance")
    words = []
    total = 0.0
    for entry in items:
        try:
            word, chance = entry
        except (TypeError, ValueError):
            raise ValueError(
                "an error is a (word, chance) pair, got "
                f"{type(entry).__name__}"
            ) from None
        if not isinstance(word, str):
            raise ValueError(f"an error is named by a Pauli word, got {type(word).__name__}")
        words.append(check_pauli_string(word))
        _check_rate(chance, f"chance of {word!r}")
        if not isinstance(chance, (int, float)):
            raise ValueError(f"a chance is a number, got {type(chance).__name__}")
        total += float(chance)
    if total > 1.0:
        raise ValueError(f"the named errors come to {total}, which is more than one")
    widths = {len(word) for word in words}
    if len(widths) != 1:
        raise ValueError(f"the named errors are of different lengths, {sorted(widths)}")
    width = widths.pop()
    if width > _MAX_CHANNEL_WIRES:
        raise ValueError(
            f"a channel over {width} qubits would carry a "
            f"{1 << width} by {1 << width} operator; this reads at most "
            f"{_MAX_CHANNEL_WIRES} qubits at a time"
        )
    quiet = _root(1.0 - total, device, c_dtype)
    operators = [quiet * tp.eye(1 << width, dtype=c_dtype, device=device)]
    for word, (_, chance) in zip(words, items):
        if float(chance):
            # pauli_matrix drops the identity factors, so the word is lifted
            # back onto the whole register: an error named on one wire of two
            # has to arrive at apply_channel as an operator on both.
            operator = expand_to_register(
                pauli_matrix(word, device=device, c_dtype=c_dtype),
                [index for index, letter in enumerate(word) if letter != "I"],
                width,
            )
            operators.append(_root(float(chance), device, c_dtype) * operator)
    return tp.stack(operators)


def apply_channel(density: tp.Tensor, kraus: tp.Tensor, wires: Sequence[int]) -> tp.Tensor:
    """Apply a local channel to the wires of a density matrix.

    Args:
        density: shaped ``[batch, 2**n, 2**n]``.
        kraus: the channel's operators, shaped ``[r, 2**k, 2**k]``.
        wires: the qubits the channel acts on, ascending.

    Returns:
        The new density matrix, in the same layout.
    """
    if density.dim() != 3 or density.shape[-1] != density.shape[-2]:
        raise ValueError(
            f"a density matrix is [batch, 2**n, 2**n], got {tuple(density.shape)}"
        )
    if kraus.dim() != 3:
        raise ValueError(f"Kraus operators are stacked as [r, 2**k, 2**k], got {tuple(kraus.shape)}")
    if kraus.shape[0] == 0:
        raise ValueError("a channel needs at least one Kraus operator")

    batch, dim, _ = density.shape
    n_qubits = dim.bit_length() - 1 if dim > 0 and (dim & (dim - 1)) == 0 else -1
    if n_qubits < 0:
        raise ValueError(f"{dim} does not describe a whole number of qubits")
    target = check_wires(wires, n_qubits)
    bra = tuple(w + n_qubits for w in target)

    reshaped = density.reshape((batch, *([2] * (2 * n_qubits))))
    total = None
    for index in range(kraus.shape[0]):
        operator = kraus[index].to(reshaped.dtype)
        term = apply_gate(reshaped, operator, target)
        # A targeted index always transforms the same way -- the matrix picks
        # the new index on its rows -- so the bra side takes the conjugate of
        # the operator: that is what turns the step into rho K^dagger.
        term = apply_gate(term, operator.conj(), bra)
        total = term if total is None else total + term
    return total.reshape((batch, dim, dim))


def _apply_kraus(density, kraus, wire):
    return apply_channel(density, kraus, [wire])


def apply_bit_flip(density: tp.Tensor, p: float, wire: int) -> tp.Tensor:
    """Apply bit-flip noise to one wire of a density matrix."""
    return _apply_kraus(density, bit_flip_kraus(p, device=density.device, c_dtype=density.dtype), wire)


def apply_phase_flip(density: tp.Tensor, p: float, wire: int) -> tp.Tensor:
    """Apply phase-flip noise to one wire of a density matrix."""
    return _apply_kraus(density, phase_flip_kraus(p, device=density.device, c_dtype=density.dtype), wire)


def apply_amplitude_damping(density: tp.Tensor, gamma: float, wire: int) -> tp.Tensor:
    """Apply amplitude damping to one wire of a density matrix."""
    kraus = amplitude_damping_kraus(gamma, device=density.device, c_dtype=density.dtype)
    return _apply_kraus(density, kraus, wire)


def apply_depolarizing(density: tp.Tensor, p: float, wire: int) -> tp.Tensor:
    """Apply depolarizing noise to one wire of a density matrix."""
    kraus = depolarizing_kraus(p, device=density.device, c_dtype=density.dtype)
    return _apply_kraus(density, kraus, wire)
