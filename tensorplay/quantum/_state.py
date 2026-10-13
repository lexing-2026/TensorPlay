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

from ._gates import _as_complex, pauli_matrix
from ._helpers import (
    DEFAULT_C_DTYPE,
    check_wires,
    n_qubits_of,
    promoted_c_dtype,
    state_shape,
)

__all__ = [
    "apply_reset",
    "basis_state",
    "bloch2dens",
    "bloch2sv",
    "dens2bloch",
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
    "random_state",
    "state_from_bases",
    "sv2bloch",
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


def random_state(
    n_qubits: int, batch: int = 1, *, device=None, dtype=DEFAULT_C_DTYPE
) -> tp.Tensor:
    """A state drawn uniformly from the unit sphere of an ``n_qubits`` register.

    A complex normal rescaled to unit length is exactly uniform on the sphere,
    which is the same distribution the last column of a Haar-random unitary
    follows, so a state drawn this way is what a uniformly random operator would
    have prepared.

    Args:
        n_qubits: register width.
        batch: how many independent states to draw.
        device: where to draw and where to put the result.
        dtype: the states' complex type.

    Returns:
        Amplitudes shaped ``[batch, 2, 2, ...]``, unit norm each.
    """
    if n_qubits < 1:
        raise ValueError(f"a register is at least one qubit wide, got {n_qubits}")
    real_dtype = tp.float64 if dtype == tp.complex128 else tp.float32
    side = 1 << n_qubits
    gauss = tp.complex(
        tp.randn(batch, side, device=device, dtype=real_dtype),
        tp.randn(batch, side, device=device, dtype=real_dtype),
    ) * (0.5 ** 0.5)
    flat = gauss / gauss.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    return unflatten(flat.to(dtype), n_qubits)


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


def apply_reset(state: tp.Tensor, wire: int) -> tp.Tensor:
    """Project ``wire`` onto |0> and renormalize the register.

    Resetting a qubit is the non-unitary cousin of measuring it: every branch
    where the wire reads 1 is discarded, and the rest is scaled back up.  The
    kept branches carry the probability of the 0 outcome, so the projection
    changes the norm and ``normalize`` hands back a unit state.
    """
    n_qubits = n_qubits_of(state)
    target = check_wires(wire, n_qubits)
    w = target[0]
    dim = 1 << n_qubits
    bit = (tp.arange(dim, device=state.device) >> (n_qubits - 1 - w)) & 1
    keep = (bit == 0).to(state.dtype)
    keep = keep.reshape(1, *([2] * n_qubits))
    return normalize(state * keep)


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


# ---------------------------------------------------------------------------
# One qubit as a point in the ball
#
# A single qubit is a point of the unit ball, and reading a simulation off in
# those coordinates is what makes a comparison readable: the distance between
# two states is the angle between their vectors, a mixed state is inside rather
# than on, and a tomography fit lands somewhere you can look at.
# ---------------------------------------------------------------------------


def _one_qubit(state: tp.Tensor, what: str) -> tp.Tensor:
    """The two amplitudes of a one-qubit state, or say why there are not two."""
    if state.dim() != 2 or state.shape[1] != 2:
        raise ValueError(
            f"{what} is one qubit, [batch, 2], got {tuple(state.shape)}"
        )
    return state


def _three(vector: tp.Tensor, what: str) -> tp.Tensor:
    """Three components on the last axis, or say why there are not three."""
    if vector.dim() < 1 or vector.shape[-1] != 3:
        raise ValueError(
            f"{what} is [batch, 3], got {tuple(vector.shape)}"
        )
    return vector


def sv2bloch(state: tp.Tensor) -> tp.Tensor:
    """The Bloch vector ``(x, y, z)`` of a one-qubit state.

    A normalized state is on the surface of the ball, so its vector is of unit
    length; the reading is unique up to a global phase, which the ball cannot
    see.  Batched, and real whatever the state's precision.

    The three components are the readings of the three Pauli operators --
    ``x = Tr(rho X)`` and so on -- so a state and its density matrix agree here
    exactly, and a rotation of ``pi/2`` about z takes ``x`` to ``y``.
    """
    amplitudes = _one_qubit(state, "a state")
    top = amplitudes[:, 0]
    bottom = amplitudes[:, 1]
    overlap = top.conj() * bottom
    components = [
        2.0 * tp.real(overlap),
        2.0 * tp.imag(overlap),
        tp.real(top.conj() * top - bottom.conj() * bottom),
    ]
    return tp.stack(components, dim=-1).to(dtype=_real_dtype(state))


def dens2bloch(density: tp.Tensor) -> tp.Tensor:
    """The Bloch vector of a one-qubit density matrix.

    A state that is not pure reads as a point inside the ball, and where it sits
    says how mixed it is: the length is twice the larger eigenvalue less one, so
    a fully mixed qubit is the origin, a pure one is on the surface, and a state
    that is three-quarters ``|0><0|`` is a quarter of the way along z.
    """
    if density.dim() != 3 or density.shape[-1] != 2 or density.shape[-2] != 2:
        raise ValueError(
            f"a density matrix is [batch, 2, 2] for one qubit, got {tuple(density.shape)}"
        )
    coherence = density[:, 0, 1]
    return tp.stack(
        [
            2.0 * tp.real(coherence),
            -2.0 * tp.imag(coherence),
            tp.real(density[:, 0, 0] - density[:, 1, 1]),
        ],
        dim=-1,
    ).to(dtype=_real_dtype(density))


def bloch2sv(vector: tp.Tensor) -> tp.Tensor:
    """The state a Bloch vector names, the way ``sv2bloch`` reads one.

    The state is the one with a non-negative real amplitude on ``|0>``, which
    fixes the global phase the ball cannot see.  A vector pointing at
    ``-z`` lands on ``|1>``, since the formula divides by a length that has
    gone to zero there.
    """
    parts = _three(vector, "a Bloch vector")
    c_dtype = _complex_dtype(_real_dtype(vector))
    real = parts.to(dtype=_real_dtype(vector))
    top = tp.sqrt(tp.clamp((1.0 + real[:, 2]) / 2.0, min=0.0)).reshape(-1, 1)
    # rho_01 = (x - i y) / 2 is the state overlapped with itself one qubit over,
    # so the second amplitude is that overlap divided by the first.  At the south
    # pole the first is zero and the formula divides by it: there the state is
    # |1>, whose second amplitude is the whole of it.
    coherence = _as_complex(real[:, 0], -real[:, 1], c_dtype).reshape(-1, 1)
    bottom = tp.where(
        top > 1e-12, coherence / (2.0 * top + 1e-12), tp.ones_like(top)
    )
    amplitudes = tp.stack([top.to(c_dtype), bottom], dim=-1)
    return amplitudes.squeeze(-2)


def bloch2dens(vector: tp.Tensor) -> tp.Tensor:
    """The density matrix a Bloch vector names, ``(I + r.sigma)/2``.

    Inside the ball this is a mixed state of its own; on the surface it is the
    pure state ``sv2bloch`` reads back, and the round trip is exact there.
    """
    parts = _three(vector, "a Bloch vector")
    c_dtype = _complex_dtype(_real_dtype(vector))
    real = parts.to(dtype=_real_dtype(parts))
    identity = tp.eye(2, dtype=c_dtype, device=parts.device)
    paulis = [
        pauli_matrix(letter, device=parts.device, c_dtype=c_dtype)
        for letter in "XYZ"
    ]
    weighted = [
        real[:, index].to(c_dtype).reshape(-1, 1, 1) * matrix
        for index, matrix in enumerate(paulis)
    ]
    return 0.5 * (identity.unsqueeze(0) + weighted[0] + weighted[1] + weighted[2])


def _real_dtype(tensor: tp.Tensor):
    """The real precision beside a tensor's own."""
    return tp.float64 if tensor.dtype in (tp.complex128, tp.float64) else tp.float32


def _complex_dtype(real: tp.Tensor):
    """The complex precision that goes with a real one."""
    return tp.complex128 if real == tp.float64 else tp.complex64


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


def _density_width(density: tp.Tensor) -> int:
    """How many qubits a density matrix describes, read off its side.

    A density matrix is flat -- ``[batch, 2**n, 2**n]`` -- so the width is the
    side's base-two logarithm.  Reading it any other way makes a two-qubit matrix
    indistinguishable from a one-qubit one, which is the one thing the flat form
    cannot say by itself.
    """
    if density.dim() != 3 or density.shape[-1] != density.shape[-2]:
        raise ValueError(
            f"a density matrix is [batch, 2**n, 2**n], got {tuple(density.shape)}"
        )
    side = density.shape[-1]
    if side < 1 or side & (side - 1):
        raise ValueError(f"a density matrix's side is a power of two, got {side}")
    return side.bit_length() - 1


def partial_trace(density: tp.Tensor, keep: Sequence[int], n_qubits: Optional[int] = None) -> tp.Tensor:
    """Trace ``density`` down to the qubits in ``keep``, in the given order.

    Args:
        density: shaped ``[batch, 2**n, 2**n]`` -- the flat form every other
            function here reads a density matrix in, so the width is read off the
            side.
        keep: which qubits survive, in the order they come out.  Anything not
            named is traced out.
        n_qubits: the register width, when the side cannot say it.

    Returns:
        The reduced density matrix, ``[batch, 2**len(keep), 2**len(keep)]``.
    """
    if n_qubits is None:
        n_qubits = _density_width(density)
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
