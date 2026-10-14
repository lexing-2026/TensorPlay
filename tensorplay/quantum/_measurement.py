"""Probabilities, expectation values, shot-based sampling, and readout error.

Probabilities are the squared amplitudes of the computational basis; they sum
to one for a normalized state and give the exact measurement distribution.
Sampling draws from that distribution on whichever device holds the state, so a
sampled batch never leaves the accelerator.  A readout matrix is the one step
that happens after the state is measured: each shot's reading is decided on its
own, so the histogram it produces carries the error a device would really add.
"""
from typing import List, Optional, Sequence

import tensorplay as tp

from tensorplay.autograd import Function

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
from ._helpers import (
    DEFAULT_C_DTYPE,
    check_pauli_string,
    n_qubits_of,
    ordered_wires,
)
from ._state import unflatten

__all__ = [
    "apply_measurement_basis",
    "apply_pauli_word",
    "apply_readout_error",
    "commuting_groups",
    "counts_from_indices",
    "density_from_counts",
    "density_probs",
    "expectation",
    "expectation_from_counts",
    "gen_bitstrings",
    "joint_expectation",
    "marginal_probs",
    "measure",
    "probs",
    "readout_error_matrix",
    "sample_energy",
    "sample_indices",
    "state_from_counts",
]

_PAULI_BUILDERS = {
    "I": i_matrix,
    "X": x_matrix,
    "Y": y_matrix,
    "Z": z_matrix,
}


# The compiled reads below hand back a number and nothing else, so a graph that
# is being recorded has to be given a backward by hand.  All three of them are
# linear in |psi|**2, and that is what makes the backward short: the gradient of
# a weight with respect to the amplitude it came from is the amplitude itself, so
# the backward reuses whatever the forward already computed instead of walking
# the state a second time.  Every one of them is a real output read against the
# conjugate of the input, and that reading counts both halves -- d|psi|**2 comes
# back as 2 psi here, not psi -- so each backward carries a factor of two.


def _batch_scale(value: tp.Tensor, state: tp.Tensor) -> tp.Tensor:
    """``value`` per batch entry, spread back over a state's own dimensions."""
    return value.reshape(value.shape[0], *([1] * (state.dim() - 1)))


def gen_bitstrings(n_qubits: int) -> List[str]:
    """The computational basis labels of ``n_qubits``, in flat-index order.

    Entry ``i`` of the returned list is the label flat amplitude index ``i``
    reads, with qubit 0 the leading bit: ``gen_bitstrings(2)`` is
    ``["00", "01", "10", "11"]``.  The strings are the column names of a counts
    histogram, which is how ``measure`` and ``counts_from_indices`` tally.
    """
    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError(f"a register has a whole number of qubits, got {n_qubits!r}")
    return ["{:0{}b}".format(index, n_qubits) for index in range(1 << n_qubits)]


class _CompiledStatePros(Function):
    """``probs`` with a gradient: each weight's own amplitude comes back.

    The weights line up with the amplitudes one for one, so the incoming
    gradient needs no reshaping beyond the state layout it came from.
    """

    @staticmethod
    def forward(ctx, state):
        with tp.no_grad():
            weights = _native.compiled_state_probs(state)
        ctx.save_for_backward(state)
        return weights

    @staticmethod
    def backward(ctx, grad_weights):
        (state,) = ctx.saved_tensors
        state, grad_weights = state.detach(), grad_weights.detach()
        with tp.no_grad():
            scale = grad_weights.reshape(state.shape)
            grad = 2 * scale * state
        return grad


class _CompiledMarginalPros(Function):
    """``marginal_probs`` with a gradient.

    Every amplitude in a column shares that column's weight, so the backward is
    the forward's own weights spread back over the unlisted wires and multiplied
    into the state.  Nothing is walked a second time.
    """

    @staticmethod
    def forward(ctx, state, wires):
        # The wires arrive as a small tensor rather than a list: a list argument
        # is read as one input per entry, and the backward answers one input per
        # argument.
        listed = [int(wire) for wire in wires]
        with tp.no_grad():
            weights = _native.compiled_marginal_probs(state, listed)
        ctx.save_for_backward(state)
        ctx.wires = listed
        return weights

    @staticmethod
    def backward(ctx, grad_weights):
        (state,) = ctx.saved_tensors
        # Detached: the backward runs inside a traversal, and a graph built here
        # would ask to be traversed again.
        state, grad_weights = state.detach(), grad_weights.detach()
        with tp.no_grad():
            n_qubits = state.dim() - 1
            batch = state.shape[0]
            listed = ctx.wires
            # The incoming gradient is paired with the forward's columns, so it
            # is walked back the same way the columns were: each axis of the
            # folded weight sits at its wire's own position in the amplitude
            # index, which for wires named out of order means moving them first.
            folded = unflatten(grad_weights.reshape(batch, -1), len(listed))
            ascending = sorted(range(len(listed)), key=lambda j: listed[j])
            if ascending != list(range(len(listed))):
                folded = folded.permute([0, *(axis + 1 for axis in ascending)])
            spread = folded.reshape(
                batch,
                *[
                    2 if wire in listed else 1
                    for wire in range(n_qubits)
                ],
            )
            # The unfolded amplitude index has one bit per wire, so every axis of
            # the spread weight is two long: the listed ones hold the column, the
            # rest are stretched back over the amplitudes that share it.
            full = tp.broadcast_to(spread, [batch, *([2] * n_qubits)])
            grad = 2 * full.reshape(batch, -1) * state.reshape(batch, -1)
        return grad, None


class _CompiledPauliExpectation(Function):
    """``expectation`` with a gradient: the word applied once, by hand.

    ``d/dpsi Re<psi|P|psi>`` is ``P|psi>`` -- the word is Hermitian, so the two
    halves of the conjugate derivative agree and the real part halves them into
    exactly the operator itself.  That is one pass per non-identity factor, which
    is what the forward was avoiding by not rotating the state at all.
    """

    @staticmethod
    def forward(ctx, state, word):
        with tp.no_grad():
            codes = _native.pauli_codes(word, state.device)
            value = _native.compiled_pauli_expectation(state, codes)
        ctx.save_for_backward(state)
        ctx.word = word
        return value

    @staticmethod
    def backward(ctx, grad_value):
        (state,) = ctx.saved_tensors
        state, grad_value = state.detach(), grad_value.detach()
        with tp.no_grad():
            turned = apply_pauli_word(state, ctx.word)
            grad = _batch_scale(2 * grad_value, state) * turned
        return grad, None


def probs(state: tp.Tensor) -> tp.Tensor:
    """The measurement distribution of each state, one probability per basis entry."""
    direct = _native.native_state_probs(state)
    if direct is not None:
        return direct
    if _native.has_state_probs() and _native.records_grad(state):
        return _CompiledStatePros.apply(state)
    return _probs_by_squaring(state)


def _probs_by_squaring(state: tp.Tensor) -> tp.Tensor:
    """The measurement distribution from the amplitudes themselves.

    Square and add, which is the one step every other distribution here is built
    out of; the state carries the gradient on this path.
    """
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
    if _native.has_marginal_probs() and _native.records_grad(state):
        wire_tensor = tp.tensor(target, dtype=tp.int64, device=state.device)
        return _CompiledMarginalPros.apply(state, wire_tensor)
    return _marginal_probs_by_folding(state, target)


def _marginal_probs_by_folding(
    state: tp.Tensor, target: Sequence[int]
) -> tp.Tensor:
    """The marginal by moving the listed qubits to the front and summing."""
    n_qubits = n_qubits_of(state)
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
    """Apply the Pauli word ``word`` to each state, one factor at a time.

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
    """The expectation value of the Pauli word ``word`` for each state, real and per batch entry.

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
    if _native.has_pauli_expectation() and _native.records_grad(state):
        return _CompiledPauliExpectation.apply(state, checked)
    return _expectation_by_rotation(state, checked)


def expectation_from_density(density: tp.Tensor, word: str) -> tp.Tensor:
    """``tr(P rho)`` for the Pauli word ``word``, real and per batch entry.

    The companion of :func:`expectation` for a mixed state: an open-system run
    hands back a density matrix rather than amplitudes, and the observable is
    read off the trace rather than an overlap.  A Pauli matrix holds one entry
    per row, so the trace is one pass over the entries -- no rotated copy of the
    state, and nothing that grows with anything but the matrix itself.
    """
    if density.dim() != 3 or density.shape[-1] != density.shape[-2]:
        raise ValueError(
            f"a density matrix is [batch, 2**n, 2**n], got {tuple(density.shape)}"
        )
    side = density.shape[-1]
    if side < 1 or side & (side - 1):
        raise ValueError(f"a density matrix's side is a power of two, got {side}")
    checked = check_pauli_string(word, side.bit_length() - 1)
    matrix = _PAULI_BUILDERS[checked[0]](c_dtype=density.dtype, device=density.device)
    for letter in checked[1:]:
        matrix = tp.kron(matrix, _PAULI_BUILDERS[letter](c_dtype=density.dtype))
    return (matrix.conj() * density).sum(dim=(-2, -1)).real


def _expectation_by_rotation(state: tp.Tensor, word: str) -> tp.Tensor:
    """The overlap of a state with its Pauli-rotated self, by rotation.

    The state itself carries the gradient here, one single-qubit gate per
    non-identity factor, and it stays in the loop rather than being turned into a
    density matrix, so the cost grows with the state rather than with its square.
    """
    rotated = apply_pauli_word(state, word)
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
    state: tp.Tensor, words: Sequence[str], shots: int, *, readout=None
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
        readout: an optional readout matrix, as :func:`measure` takes it: each
            shot's outcome is then reported as the hardware would report it, so
            the estimate carries the error a device would add.

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
        counts = measure(rotated, shots, wires, readout=readout)
        for word in group:
            estimates[words.index(word)] = expectation_from_counts(counts, word, wires)

    filled = [item for item in estimates if item is not None]
    dtype = filled[0].dtype if filled else tp.float32
    return tp.stack([item.to(dtype) for item in estimates], dim=-1)


def sample_energy(
    state: tp.Tensor,
    hamiltonian,
    shots: int,
    *,
    readout=None,
) -> tp.Tensor:
    """``<H>`` estimated from a finite number of measurements.

    What a device would report, and what a variational routine has to optimise
    against: the terms are grouped so that one rotation and one histogram read
    as many of them as commute, each term keeps its own sign in the sum, and a
    readout matrix can be folded in so the estimate carries the error the
    hardware would add.

    The result is a sample, not a number: it is exact in expectation and carries
    the noise of ``shots`` draws.  :func:`~tensorplay.quantum.param_shift` is the
    way to differentiate it.

    Args:
        state: amplitudes shaped ``[batch, 2, 2, ...]``.
        hamiltonian: the :class:`~tensorplay.quantum.Hamiltonian` to read off.
        shots: how many measurements to draw per measurement setting.
        readout: an optional readout matrix, as :func:`measure` takes it.

    Returns:
        One estimate per state, shaped ``[batch]``, real.
    """
    terms = hamiltonian.terms
    estimates = joint_expectation(
        state, [word for _, word in terms], shots, readout=readout
    )
    total = None
    for index, (coeff, _) in enumerate(terms):
        piece = coeff * estimates[:, index]
        total = piece if total is None else total + piece
    return total


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


def _readout_matrices(matrix, n_wires: int, device, dtype) -> tp.Tensor:
    """Normalize a readout spec into one ``[2, 2]`` matrix per measured wire.

    The spec is either a single matrix, used for every wire, or one matrix per
    wire in the order the wires were measured.  Rows are what a prepared bit
    reads back as, so every row is a distribution over the reported bit.
    """
    tensor = matrix if isinstance(matrix, tp.Tensor) else tp.as_tensor(matrix)
    tensor = tensor.to(device=device, dtype=tp.float32)
    if tensor.dim() == 2:
        tensor = tensor.reshape(1, 2, 2)
    if tensor.dim() != 3 or tensor.shape[1] != 2 or tensor.shape[2] != 2:
        raise ValueError(
            f"a readout matrix is [2, 2], or one per wire as [k, 2, 2], got "
            f"{tuple(tensor.shape)}"
        )
    if tensor.shape[0] == 1:
        tensor = tensor.expand(n_wires, 2, 2)
    if tensor.shape[0] != n_wires:
        raise ValueError(
            f"got {tensor.shape[0]} readout matrix(es) for {n_wires} measured wire(s)"
        )
    if bool(((tensor < -1e-6) | (tensor > 1.0 + 1e-6)).any()):
        raise ValueError("readout probabilities lie in [0, 1]")
    if not bool(tp.allclose(tensor.sum(dim=-1), tp.ones_like(tensor[:, :, 0]), atol=1e-5, rtol=1e-5)):
        raise ValueError("every row of a readout matrix sums to one")
    return tensor


def readout_error_matrix(
    p_one_as_zero: float,
    p_zero_as_one: Optional[float] = None,
    *,
    device=None,
    dtype=tp.float32,
) -> tp.Tensor:
    """The two outcomes one wire reports for each bit it was handed.

    Row ``p`` is the distribution over the reported bit for a prepared ``p``, so
    the entry at ``[1][0]`` is the chance a 1 is read as a 0 and the entry at
    ``[0][1]`` the chance a 0 is read as a 1 -- the two numbers a calibration
    reports separately, since they differ on a real device.  One rate covers both
    directions when the error is symmetric.

    Args:
        p_one_as_zero: chance a prepared 1 is reported as 0.
        p_zero_as_one: chance a prepared 0 is reported as 1; the same as
            ``p_one_as_zero`` when omitted.
        device: where to place the matrix.
        dtype: its floating point type.

    Returns:
        A ``[2, 2]`` matrix, row prepared and column reported.
    """
    for name, value in (("misfire chance", p_one_as_zero), ("misfire chance", p_zero_as_one)):
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"a readout {name} lies in [0, 1], got {value}")
    reverse = p_one_as_zero if p_zero_as_one is None else p_zero_as_one
    return tp.tensor(
        [[1.0 - reverse, reverse], [p_one_as_zero, 1.0 - p_one_as_zero]],
        dtype=dtype,
        device=device,
    )


def apply_readout_error(counts: tp.Tensor, matrix) -> tp.Tensor:
    """Reports a histogram the way a noisy readout would have.

    The map is linear, so it applies just as well to exact probabilities as to
    tallied shots, and it keeps the total weight: each row of a readout matrix is
    a distribution, so nothing is created or lost.  Errors on different wires are
    taken to be independent, which is what a per-wire calibration measures.

    Args:
        counts: a distribution over outcomes, shaped ``[..., 2**n]``; the
            leading column is the highest bit.
        matrix: a ``[2, 2]`` readout matrix applied to every wire, or one per
            wire as ``[n, 2, 2]``, from :func:`readout_error_matrix`.

    Returns:
        The reported distribution, in the same shape.
    """
    n_columns = counts.shape[-1]
    n_wires = n_columns.bit_length() - 1
    if n_columns <= 0 or 1 << n_wires != n_columns:
        raise ValueError(f"{n_columns} columns do not describe a whole number of qubits")
    matrices = _readout_matrices(matrix, n_wires, counts.device, counts.dtype)

    # One wire at a time: the bit axis is pulled to the end, contracted against
    # its matrix, and put back where it came from, so the caller's column order
    # and every leading dimension survive untouched.
    leading = counts.dim() - 1
    folded = counts.reshape(*counts.shape[:-1], *([2] * n_wires))
    for position in range(n_wires):
        axis = leading + position
        moved = tp.movedim(folded, axis, -1)
        moved = moved @ matrices[position]
        folded = tp.movedim(moved, -1, axis)
    return folded.reshape(counts.shape)


def counts_from_indices(indices: tp.Tensor, n_qubits: int) -> tp.Tensor:
    """Tally sampled basis indices into per-basis counts for each state in the batch."""
    dim = 1 << n_qubits
    batch = indices.shape[0]
    rows = indices.reshape(batch, -1)
    counts = tp.zeros((batch, dim), dtype=tp.float32, device=indices.device)
    weights = tp.ones_like(rows, dtype=tp.float32)
    return counts.scatter_add_(1, rows, weights)


def density_from_counts(
    counts: tp.Tensor, *, pseudocount: float = 0.0, c_dtype=DEFAULT_C_DTYPE
) -> tp.Tensor:
    """The density matrix a histogram of basis measurements describes.

    A histogram in the computational basis is the diagonal of a density matrix
    and nothing else: every off-diagonal term is a phase, and a phase is exactly
    what a basis measurement throws away.  So what comes back here is the state
    the measurements cannot rule out, and reading it as the answer to what the
    system is would be wrong in the one place it matters.

    Args:
        counts: shots per outcome, shaped ``[batch, 2**n_qubits]``.
        pseudocount: a floor added to every outcome before normalizing, in the
            same units as the counts.  It makes the matrix full rank, which a
            tomography fit wants; it also makes the answer wrong by that much,
            so it is nothing but what its name says.
        c_dtype: the precision to build the matrix in.

    Returns:
        The matrix, shaped ``[batch, 2**n_qubits, 2**n_qubits]``.

    Raises:
        ValueError: if a row holds no shots at all, or if a count is negative.
    """
    _check_histogram(counts, pseudocount)
    n_columns = counts.shape[-1]
    weights = counts.to(dtype=tp.float64) + float(pseudocount)
    probabilities = weights / weights.sum(dim=-1, keepdim=True)
    return tp.diag_embed(probabilities).to(dtype=c_dtype)


def state_from_counts(
    counts: tp.Tensor, *, pseudocount: float = 0.0, dtype=DEFAULT_C_DTYPE
) -> tp.Tensor:
    """The state whose squared amplitudes are the histogram's shares.

    A histogram has no phases to read, so it cannot name a state -- it names a
    whole family of them, one per phase per amplitude.  What it can name is the
    member of that family the square roots give: every amplitude real and
    non-negative, which is the pure state overlapping the measured one the most.
    Comparing that state with a theory is a comparison of populations; comparing
    its density matrix with the one measured in a rotated basis is what says
    anything about the phases.

    Args:
        counts: shots per outcome, shaped ``[batch, 2**n_qubits]``.
        pseudocount: a floor added to every outcome, as in
            :func:`density_from_counts`.  Without one an outcome that never came
            out stays out of the state, which is the honest reading of a
            finite number of shots.
        dtype: the precision to build the state in.

    Returns:
        The state, shaped ``[batch, 2**n_qubits]``.

    Raises:
        ValueError: if a row holds no shots at all, or if a count is negative.
    """
    _check_histogram(counts, pseudocount)
    n_columns = counts.shape[-1]
    weights = counts.reshape(-1, n_columns).to(dtype=tp.float64) + float(pseudocount)
    # The square roots are already the canonical reading: non-negative, so the
    # one phase a histogram leaves free is fixed on every amplitude at once, and
    # two states that share a histogram are the same state up to that phase.
    amplitudes = tp.sqrt(weights / weights.sum(dim=-1, keepdim=True))
    flat = amplitudes.reshape(*counts.shape[:-1], n_columns).to(dtype=dtype)
    # The module's own shape for a state: one axis of size 2 per qubit after the
    # batch, whatever else the histogram had in front of it.
    n_qubits = n_columns.bit_length() - 1
    return flat.reshape(*counts.shape[:-1], *([2] * n_qubits))


def _check_histogram(counts: tp.Tensor, pseudocount: float) -> None:
    """Say what a histogram has to carry before it is read as a state."""
    if counts.dim() < 1 or counts.shape[-1] < 2:
        raise ValueError(
            "a histogram is [batch, 2**n_qubits] over at least two outcomes, got "
            f"{tuple(counts.shape)}"
        )
    if float(counts.min()) < 0.0:
        raise ValueError("a histogram cannot hold a negative count")
    if pseudocount < 0.0:
        raise ValueError(f"a pseudocount is not negative, got {pseudocount}")
    if counts.to(dtype=tp.float64).sum(dim=-1).min().item() <= 0.0:
        raise ValueError("a histogram of no shots describes no state")


def _read_out(indices: tp.Tensor, matrices: tp.Tensor) -> tp.Tensor:
    """Draw the bit each wire reports for a batch of sampled outcomes.

    Every shot is decided on its own, the way a real readout decides each shot on
    its own: the outcome is sampled first and the reported bit is then a fresh
    draw from that wire's matrix, so a shot that is misread is lost rather than
    averaged into its neighbours.
    """
    n_wires = matrices.shape[0]
    if n_wires == 0:
        return indices
    # One draw per wire: sharing a single draw would tie the wires together and
    # make the errors on different wires perfectly correlated.
    draws = tp.rand((n_wires, *indices.shape), dtype=matrices.dtype, device=indices.device)
    flips = tp.zeros_like(indices)
    for position in range(n_wires):
        # The leading column of an outcome is the highest bit, so the wire listed
        # first owns the shift that leaves it.
        bit = n_wires - 1 - position
        prepared = tp.bitwise_and(tp.bitwise_right_shift(indices, bit), 1)
        # The entry at column zero is the chance of being told 0; comparing the
        # draw against it settles the reported bit in one step.  The two
        # thresholds are picked by a select rather than gathered, so the whole
        # step stays elementwise on whatever device holds the state.
        stay_zero = tp.where(
            prepared == 1, matrices[position][1, 0], matrices[position][0, 0]
        )
        reported = tp.where(draws[position] < stay_zero, 0, 1).to(indices.dtype)
        # A wire moves only where the two readings disagree, so the mask carries
        # the difference rather than the reading itself.
        moved = tp.bitwise_xor(reported, prepared)
        flips = tp.bitwise_or(flips, tp.bitwise_left_shift(moved, bit))
    return tp.bitwise_xor(indices, flips)


def measure(
    state: tp.Tensor,
    shots: int,
    wires: Optional[Sequence[int]] = None,
    readout=None,
) -> tp.Tensor:
    """Sample ``shots`` measurements per state and tally them.

    Args:
        state: amplitudes shaped ``[batch, 2, 2, ...]``.
        shots: how many independent measurements to draw per state.
        wires: qubits to read out; all of them when omitted.  The result has
            one column per distinct outcome of the listed qubits.
        readout: an optional readout matrix, shaped ``[2, 2]`` for every wire or
            ``[len(wires), 2, 2]`` for one per wire; each shot's outcome is
            reported through it.

    Returns:
        Counts shaped ``[batch, 2**len(wires)]``, real.
    """
    n_qubits = n_qubits_of(state)
    target = list(range(n_qubits)) if wires is None else ordered_wires(wires, n_qubits)
    marginal = marginal_probs(state, target)
    indices = sample_indices(marginal, shots)
    if readout is not None:
        indices = _read_out(
            indices,
            _readout_matrices(readout, len(target), indices.device, marginal.dtype),
        )
    return counts_from_indices(indices, len(target))
