"""State construction, normalization, and density-matrix views.

A state tensor is ``[batch, 2, 2, ...]``: one dimension of size 2 per qubit
after the batch, qubit 0 the leading one.  Flattening it therefore makes qubit
0 the most significant bit of the amplitude index.  Density matrices use the
flat form ``[batch, 2**n, 2**n]``, where the same bit order applies to the row
and to the column.
"""
import math
from typing import Optional, Sequence

import tensorplay as tp

from ._helpers import DEFAULT_C_DTYPE, promoted_c_dtype, state_shape

__all__ = [
    "basis_state",
    "density_to_state",
    "fidelity",
    "flatten",
    "global_phase",
    "inner",
    "norm",
    "normalize",
    "partial_trace",
    "plus_state",
    "product_state",
    "state_from_bases",
    "to_density",
    "unflatten",
    "zeros_state",
]


def zeros_state(n_qubits: int, batch: int = 1, *, device=None, dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
    """The all-zero state ``|0...0>``: amplitude 1 on the first basis entry."""
    dim = 1 << n_qubits
    lead = tp.ones((batch, 1), dtype=dtype, device=device)
    tail = tp.zeros((batch, dim - 1), dtype=dtype, device=device)
    return tp.cat([lead, tail], dim=-1).reshape(state_shape(n_qubits, batch))


def plus_state(n_qubits: int, batch: int = 1, *, device=None, dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
    """The equal superposition over every basis entry of ``n_qubits``."""
    dim = 1 << n_qubits
    amplitude = 1.0 / math.sqrt(dim)
    return tp.full(state_shape(n_qubits, batch), amplitude, dtype=dtype, device=device)


def basis_state(index: int, n_qubits: int, *, device=None,
                dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
    """The computational basis state ``|index>``, shaped ``[1, 2, 2, ...]``."""
    dim = 1 << n_qubits
    if index < 0 or index >= dim:
        raise IndexError(f"basis index {index} is out of range for {n_qubits} qubit(s)")
    entries = tp.arange(dim, device=device).reshape(state_shape(n_qubits))
    return (entries == index).to(dtype)


def state_from_bases(indices: Sequence[int], n_qubits: int, *, device=None,
                     dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
    """The equal superposition of the listed basis states."""
    dim = 1 << n_qubits
    entries = tp.arange(dim, device=device)
    wanted = tp.zeros(dim, dtype=dtype, device=device)
    for index in indices:
        if index < 0 or index >= dim:
            raise IndexError(f"basis index {index} is out of range for {n_qubits} qubit(s)")
        wanted = wanted + (entries == index).to(dtype)
    return normalize(wanted.reshape((1, *([2] * n_qubits))))


def product_state(states: Sequence[tp.Tensor], *, device=None, dtype=DEFAULT_C_DTYPE) -> tp.Tensor:
    """The tensor product of single-qubit states, in the given qubit order."""
    result = None
    for state in states:
        if state.dim() != 1:
            raise ValueError(f"a factor of a product state is one-qubit, got {state.dim()}-D")
        result = state if result is None else tp.kron(result, state)
    if result is None:
        raise ValueError("a product state needs at least one factor")
    return result.to(dtype)


def flatten(state: tp.Tensor) -> tp.Tensor:
    """``[batch, 2, 2, ...]`` to ``[batch, 2**n]``."""
    batch = state.shape[0]
    return state.reshape(batch, -1)


def unflatten(state: tp.Tensor, n_qubits: Optional[int] = None) -> tp.Tensor:
    """``[batch, 2**n]`` to ``[batch, 2, 2, ...]``; ``n`` is read off the length if unset."""
    batch, dim = state.shape
    if n_qubits is None:
        n_qubits = round(math.log2(dim)) if dim > 0 else 0
        if 1 << n_qubits != dim:
            raise ValueError(f"{dim} amplitudes do not describe a whole number of qubits")
    return state.reshape((batch, *([2] * n_qubits)))


def norm(state: tp.Tensor) -> tp.Tensor:
    """The Euclidean norm of each state in the batch."""
    return flatten(state).abs().pow(2).sum(dim=-1).sqrt()


def normalize(state: tp.Tensor, eps: float = 0.0) -> tp.Tensor:
    """Rescale each state in the batch to unit length."""
    lengths = norm(state)
    if eps:
        lengths = tp.clamp(lengths, min=eps)
    return state / lengths.reshape((-1, *([1] * (state.dim() - 1))))


def inner(state_a: tp.Tensor, state_b: tp.Tensor) -> tp.Tensor:
    """``<a|b>`` for each pair of states in the batch."""
    return (flatten(state_a).conj() * flatten(state_b)).sum(dim=-1)


def fidelity(state_a: tp.Tensor, state_b: tp.Tensor) -> tp.Tensor:
    """``|<a|b>|**2`` for each pair of states in the batch."""
    return inner(state_a, state_b).abs().pow(2)


def global_phase(
    first: tp.Tensor, second: tp.Tensor, *, atol: float = 1e-6
) -> complex:
    """The single unit factor that turns one state or gate into the other.

    Two operators that agree up to a phase are the same operator as far as any
    measurement is concerned, which is the usual answer when a fused kernel is
    checked against a dense product: the two differ in exactly one factor and
    nothing else.

    The factor is read off the whole overlap rather than off the first entry big
    enough to divide, so one small entry cannot decide it and rounding noise
    averages out instead of picking the answer.

    Args:
        first: the reference, a state or a gate matrix.
        second: the operator to compare against it; same shape and dtype.  The
            whole tensor is compared as one operator, so a batch of states is
            read as a single one -- compare them one at a time.
        atol: how far the largest entry may sit from the phase times its
            counterpart, relative to the largest entry in ``first``.

    Returns:
        The complex factor ``p`` with ``second == p * first``.

    Raises:
        ValueError: if the shapes differ, if ``first`` is all zeros -- which has
            no phase -- or if the two are not the same operator up to a phase.
    """
    if tuple(first.shape) != tuple(second.shape):
        raise ValueError(
            f"comparing {tuple(first.shape)} against {tuple(second.shape)}: "
            "the two have to have the same shape"
        )
    c_dtype = promoted_c_dtype(tp.promote_types(first.dtype, second.dtype))
    left = first.to(dtype=c_dtype)
    right = second.to(dtype=c_dtype)
    if not tp.is_complex(left):
        left = tp.complex(left, tp.zeros_like(left))
    if not tp.is_complex(right):
        right = tp.complex(right, tp.zeros_like(right))

    reference = left.abs().max()
    if float(reference) == 0.0:
        raise ValueError("the zero operator carries no phase")
    overlap = (left.conj() * right).sum()
    if abs(complex(overlap)) == 0.0:
        raise ValueError("the two are orthogonal, so no phase turns one into the other")
    phase = complex(overlap) / (left.conj() * left).sum().real.item()

    residual = float((right - phase * left).abs().max()) / float(reference)
    if residual > atol:
        raise ValueError(
            f"the two differ by more than {atol} in their largest entry, which is "
            "more than a global phase"
        )
    return phase


def to_density(state: tp.Tensor) -> tp.Tensor:
    """The density matrix ``|psi><psi|`` of each state in the batch."""
    flat = flatten(state)
    return flat.unsqueeze(-1) * flat.conj().unsqueeze(-2)


def density_to_state(density: tp.Tensor, n_qubits: Optional[int] = None) -> tp.Tensor:
    """The pure state whose density matrix is ``density``, when it has one.

    The dominant eigenpair carries the state; a mixed density matrix has no
    such state, and its largest eigenvalue is reported instead.
    """
    batch, dim, _ = density.shape
    eigenvalues, eigenvectors = tp.linalg.eigh(density)
    # A column is picked with a one-hot product rather than a gather, which
    # keeps this to arithmetic every dtype accepts.
    position = tp.arange(dim, device=density.device).reshape(1, dim)
    index = eigenvalues.abs().argmax(dim=-1, keepdim=True)
    selector = (position == index).to(eigenvectors.dtype)
    flat = (selector * eigenvectors).sum(dim=-1)
    return normalize(unflatten(flat, n_qubits))


def partial_trace(density: tp.Tensor, keep: Sequence[int], n_qubits: Optional[int] = None) -> tp.Tensor:
    """Trace ``density`` down to the qubits in ``keep``, in the given order."""
    if n_qubits is None:
        n_qubits = (density.dim() - 1) // 2
    kept = [w for w in keep]
    for wire in kept:
        if wire < 0 or wire >= n_qubits:
            raise IndexError(f"wire {wire} is out of range for a state with {n_qubits} qubit(s)")
    if len(set(kept)) != len(kept):
        raise ValueError("a partial trace keeps each qubit at most once")
    dropped = [w for w in range(n_qubits) if w not in kept]
    batch = density.shape[0]

    reshaped = density.reshape((batch, *([2] * (2 * n_qubits))))
    order = (
        [0]
        + [w + 1 for w in kept]
        + [w + 1 for w in dropped]
        + [n_qubits + w + 1 for w in kept]
        + [n_qubits + w + 1 for w in dropped]
    )
    moved = reshaped.permute(order).contiguous()
    k, d = len(kept), len(dropped)
    # The moved axes run kept-ket, dropped-ket, kept-bra, dropped-bra.
    moved = moved.reshape((batch, 1 << k, 1 << d, 1 << k, 1 << d))
    # Tracing pairs the two dropped indices, which is a diagonal followed by a
    # sum over what the diagonal left behind.
    return tp.diagonal(moved, dim1=2, dim2=4).sum(dim=-1)
