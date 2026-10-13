"""Unitary application and the standard gate set.

:func:`apply_gate` is the one primitive every gate goes through.  It moves the
targeted dimensions of the state to the front, folds them into a single axis of
length ``2**k``, and contracts that axis with the gate matrix -- the layout a
batched matrix multiply wants, so no gather table and no interleaved copy of
the whole state is built.  The move is a permutation, so it is differentiable
and every gate defined on top of it stays differentiable in its parameters.

Parameter-dependent matrices are assembled from real pieces (cosines, sines,
raw parameters) and only then given an imaginary half, so a parameter that
carries a gradient is differentiated through real arithmetic alone.
"""
import math
from typing import List, Sequence

import tensorplay as tp

from . import _native
from ._helpers import (
    DEFAULT_C_DTYPE,
    MAX_GATE_WIRES,
    check_pauli_string,
    check_wires,
    n_qubits_of,
    ordered_wires,
    pauli_wires,
    promoted_c_dtype,
)

__all__ = [
    "apply_ccx",
    "apply_ccz",
    "apply_chadamard",
    "apply_cnot",
    "apply_controlled",
    "apply_cp",
    "apply_crx",
    "apply_cry",
    "apply_crz",
    "apply_cs",
    "apply_csdg",
    "apply_cswap",
    "apply_csx",
    "apply_cx",
    "apply_cy",
    "apply_cz",
    "apply_dcx",
    "apply_ecr",
    "apply_gate",
    "apply_global_phase",
    "apply_h",
    "apply_i",
    "apply_iswap",
    "apply_mcx",
    "apply_multirz",
    "apply_p",
    "apply_rc3x",
    "apply_rccx",
    "apply_rot",
    "apply_rx",
    "apply_rxx",
    "apply_ry",
    "apply_ryy",
    "apply_rz",
    "apply_rzx",
    "apply_rzz",
    "apply_s",
    "apply_sdg",
    "apply_single_excitation",
    "apply_swap",
    "apply_sx",
    "apply_t",
    "apply_tdg",
    "apply_toffoli",
    "apply_u2",
    "apply_u3",
    "apply_x",
    "apply_xx_minus_yy",
    "apply_xx_plus_yy",
    "apply_y",
    "apply_z",
    "controlled_matrix",
    "dagger",
    "dcx_matrix",
    "ecr_matrix",
    "globalphase_matrix",
    "h_matrix",
    "i_matrix",
    "iswap_matrix",
    "mcx_matrix",
    "multirz_eigvals",
    "multirz_matrix",
    "p_matrix",
    "pauli_rotation_matrix",
    "phase_matrix",
    "qft_matrix",
    "random_unitary",
    "rc3x_matrix",
    "rccx_matrix",
    "rot_matrix",
    "rot_x_matrix",
    "rot_y_matrix",
    "rot_z_matrix",
    "rxx_matrix",
    "ryy_matrix",
    "rzx_matrix",
    "rzz_matrix",
    "s_matrix",
    "sdg_matrix",
    "single_excitation_matrix",
    "swap_matrix",
    "sx_matrix",
    "t_matrix",
    "tdg_matrix",
    "u2_matrix",
    "u3_matrix",
    "x_matrix",
    "xx_minus_yy_matrix",
    "xx_plus_yy_matrix",
    "y_matrix",
    "z_matrix",
]


# ---------------------------------------------------------------------------
# Core primitive
# ---------------------------------------------------------------------------


def apply_gate(state: tp.Tensor, mat: tp.Tensor, wires) -> tp.Tensor:
    """Apply the unitary ``mat`` to the qubits named by ``wires``.

    Every targeted index transforms the same way: the new index reads off the
    row of ``mat`` and the old index off its column.  That makes the call
    ``psi -> M psi`` on a state and ``rho -> rho M^T`` on a bra axis of a
    density matrix, so a channel step passes the conjugate of its operator on
    the bra side to obtain ``rho K^dagger``.

    Args:
        state: amplitudes shaped ``[batch, 2, 2, ...]`` -- one dimension of
            size 2 per qubit, qubit 0 leading.
        mat: the unitary, shaped ``[2**k, 2**k]``, or ``[batch, 2**k, 2**k]``
            when it carries one matrix per batch entry.
        wires: the target qubits, in the order the matrix names them: the first
            index bit of ``mat`` belongs to the first wire listed.  Any order
            works; ``k = len(wires)``.

    Returns:
        The new state, in the layout of the input.
    """
    n_qubits = n_qubits_of(state)
    named = ordered_wires(wires, n_qubits)
    target = sorted(named)
    k = len(target)

    if mat.dim() not in (2, 3):
        raise ValueError(f"a gate matrix is 2-D or batched 3-D, got {mat.dim()}-D")

    c_dtype = state.dtype
    if state.dtype not in (tp.complex64, tp.complex128) or mat.dtype not in (
        tp.complex64,
        tp.complex128,
    ):
        raise TypeError(f"gates act on complex state, got {state.dtype} and {mat.dtype}")

    side = 1 << k
    if mat.shape[-1] != side or mat.shape[-2] != side:
        # A 1x1 operator is the scalar it is, and on one qubit that scales the
        # whole state -- the form an identity Pauli term arrives in.
        if k == 1 and mat.shape[-1] == 1 and mat.shape[-2] == 1:
            mat = mat * tp.eye(2, dtype=c_dtype, device=mat.device)
        else:
            raise ValueError(
                f"a gate on {k} qubit(s) needs a {side}x{side} matrix, got "
                f"{tuple(mat.shape[-2:])}"
            )

    batch = state.shape[0]
    dims = [w + 1 for w in target]
    rest = [d for d in range(1, state.dim()) if d not in dims]
    perm = [0, *dims, *rest]

    # The gate follows the state: a matrix built for another device or in a
    # different precision is finished off here.  The register sets the precision
    # rather than meeting the matrix's -- a circuit written in single precision
    # should stay in single precision, and one written in double should not be
    # pulled down by a matrix built from a plain number.
    mat = mat.to(device=state.device, dtype=c_dtype)

    # The matrix spells its index bits in the order the wires were named, while
    # the gather below walks the targeted dimensions in ascending order.  The
    # two agree only when the wires already came in ascending, so the matrix
    # axes are permuted onto that order whenever they differ -- otherwise a
    # control named on a higher wire would land on the lower one.
    order = [named.index(wire) for wire in target]
    if any(position != source for position, source in enumerate(order)):
        leading = mat.dim() - 2
        axes = list(range(leading))
        axes += [leading + order[bit] for bit in range(k)]
        axes += [leading + k + order[bit] for bit in range(k)]
        shape = mat.shape
        mat = mat.reshape(*([1] * leading), *([2] * (2 * k))).permute(axes).reshape(shape)

    # Outside a recorded graph the compiled gather does the whole gate in one
    # pass; inside one, the decomposition below carries the gradient.
    folded = state.reshape(batch, -1)
    direct = _native.native_apply_gate(folded, mat, target)
    if direct is not None:
        return direct.reshape(state.shape)

    moved = state.permute(perm).contiguous().reshape(batch, side, -1)
    if mat.dim() == 3:
        if mat.shape[0] != batch:
            raise ValueError(
                f"a batched gate matrix carries {mat.shape[0]} entries, the state "
                f"carries {batch}"
            )
    else:
        mat = mat.unsqueeze(0).expand(batch, side, side)

    # The contraction is spelled as the operator rather than as the batched
    # product: the operator's backward takes the adjoint of the other operand,
    # which for a complex state is the conjugate transpose -- a plain batched
    # product differentiates against a plain transpose and loses the phase a
    # state carries, so every angle downstream of a second gate would come out
    # wrong.
    out = tp.matmul(mat, moved)
    # The contracted block goes back to one dimension per targeted qubit; the
    # untouched ones already sit where they belong.
    back = [0]
    for axis in range(1, state.dim()):
        if axis in dims:
            back.append(1 + target.index(axis - 1))
        else:
            back.append(1 + len(target) + rest.index(axis))
    shape = [batch, *(state.shape[d] for d in dims), *(state.shape[d] for d in rest)]
    return out.reshape(shape).permute(back)


def dagger(mat: tp.Tensor) -> tp.Tensor:
    """The conjugate transpose of a gate matrix, batched over leading dims."""
    return mat.conj().transpose(-1, -2)


def random_unitary(
    n_qubits: int, *, device=None, c_dtype=DEFAULT_C_DTYPE
) -> tp.Tensor:
    """A unitary drawn from the Haar measure on an ``n_qubits`` register.

    The distribution is uniform over the unitaries rather than over their
    entries, which is what a random operator has to be for an average over it to
    mean anything.  It comes from a factorization of a matrix of complex
    normals, with each column turned to line up with that factor's diagonal --
    without the turn the factorization is uniform only up to a phase per column,
    which leaves the entries biased even though the matrix is unitary.

    Args:
        n_qubits: register width; the matrix is ``2**n_qubits`` square.
        device: where to draw and where to put the result.
        c_dtype: the matrix's complex type.

    Returns:
        A ``[2**n, 2**n]`` unitary.
    """
    if n_qubits < 1:
        raise ValueError(f"a register is at least one qubit wide, got {n_qubits}")
    if n_qubits > MAX_GATE_WIRES:
        raise ValueError(
            f"a gate may span at most {MAX_GATE_WIRES} wires, got {n_qubits}; "
            f"the unitary would be {1 << n_qubits} square"
        )
    side = 1 << n_qubits
    real_dtype = tp.float64 if c_dtype == tp.complex128 else tp.float32
    # Two real normals make one complex normal; the square root keeps the two
    # halves from each carrying a whole variance.
    gauss = tp.complex(
        tp.randn(side, side, device=device, dtype=real_dtype),
        tp.randn(side, side, device=device, dtype=real_dtype),
    ) * (0.5 ** 0.5)
    unitary, upper = tp.linalg.qr(gauss)
    turns = tp.diagonal(upper)
    return unitary * (turns / tp.abs(turns)).to(c_dtype)


# ---------------------------------------------------------------------------
# Constant gate matrices
# ---------------------------------------------------------------------------


def _constant(rows, *, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    return tp.tensor(rows, dtype=c_dtype, device=device)


def i_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The identity gate."""
    return _constant([[1, 0], [0, 1]], device=device, c_dtype=c_dtype)


def x_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The Pauli-X gate."""
    return _constant([[0, 1], [1, 0]], device=device, c_dtype=c_dtype)


def y_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The Pauli-Y gate."""
    return _constant([[0, -1j], [1j, 0]], device=device, c_dtype=c_dtype)


def z_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The Pauli-Z gate."""
    return _constant([[1, 0], [0, -1]], device=device, c_dtype=c_dtype)


def h_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The Hadamard gate."""
    half = 2.0**-0.5
    return _constant([[half, half], [half, -half]], device=device, c_dtype=c_dtype)


def s_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The phase gate: a quarter turn around Z."""
    return _constant([[1, 0], [0, 1j]], device=device, c_dtype=c_dtype)


def sdg_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The inverse of the phase gate."""
    return _constant([[1, 0], [0, -1j]], device=device, c_dtype=c_dtype)


def t_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The T gate: an eighth turn around Z."""
    half = 2.0**-0.5
    return _constant([[1, 0], [0, half + half * 1j]], device=device, c_dtype=c_dtype)


def tdg_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The inverse of the T gate."""
    half = 2.0**-0.5
    return _constant([[1, 0], [0, half - half * 1j]], device=device, c_dtype=c_dtype)


def sx_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The sqrt(X) gate: a quarter turn around the X axis."""
    return _constant(
        [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], device=device, c_dtype=c_dtype
    )


def swap_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The two-qubit swap gate on ``[first wire, second wire]``."""
    return _constant(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        device=device,
        c_dtype=c_dtype,
    )


def iswap_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The iswap gate on ``[first wire, second wire]``."""
    return _constant(
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
        device=device,
        c_dtype=c_dtype,
    )


# ---------------------------------------------------------------------------
# Parameterized gate matrices
# ---------------------------------------------------------------------------


def _as_complex(real_part: tp.Tensor, imag_part: tp.Tensor, c_dtype) -> tp.Tensor:
    """``real_part + i*imag_part``, given the two halves as real tensors.

    The halves keep their own precision: the pair decides the complex dtype,
    and ``c_dtype`` only widens or narrows the finished tensor.
    """
    halves = tp.promote_types(real_part.dtype, imag_part.dtype)
    return tp.complex(real_part.to(halves), imag_part.to(halves)).to(c_dtype)


def _polar(cos_part: tp.Tensor, sin_part: tp.Tensor, c_dtype) -> tp.Tensor:
    """``cos_part + i*sin_part`` as a complex tensor of ``c_dtype``."""
    return _as_complex(cos_part, sin_part, c_dtype)


def _param_c_dtype(*params, given=None):
    if given is not None:
        return given
    dtype = params[0].dtype
    for other in params[1:]:
        dtype = tp.promote_types(dtype, other.dtype)
    return promoted_c_dtype(dtype)


def _angle(value, name: str = "angle") -> tp.Tensor:
    """A parameter read as an angle, with a trailing entry of one dropped.

    A parameter block for a circuit of ``k`` angles is naturally
    ``[batch, k]``, and a one-angle block is ``[batch, 1]``.  Left as it is, the
    trailing axis would be built into the gate matrix as an axis of the matrix
    itself -- giving a matrix the gate path cannot use, or, where the matrix is
    assembled by arithmetic, one that is quietly wrong.  So a trailing axis of
    length one is dropped, which puts every parameter block in the shape the gate
    path already reads: one angle per batch entry.

    A plain number is read in double precision.  An angle written as a literal
    carries no precision of its own to preserve, and reading it in single
    precision would quietly put the whole gate -- and with it the state, since
    the amplitudes are only as good as the matrix that made them -- one digit
    short of what a double-precision state is asking for.
    """
    if isinstance(value, tp.Tensor):
        tensor = value
    else:
        tensor = tp.as_tensor(value, dtype=tp.float64)
    while tensor.dim() > 1 and tensor.shape[-1] == 1:
        tensor = tensor.reshape(tensor.shape[:-1])
    return tensor


def rot_x_matrix(theta, *, device=None, c_dtype=None) -> tp.Tensor:
    """A rotation of ``theta`` radians around the X axis."""
    theta = _angle(theta, "theta")
    c_dtype = _param_c_dtype(theta, given=c_dtype)
    half = theta / 2
    cos_half = tp.cos(half)
    neg_i_sin = _as_complex(tp.zeros_like(cos_half), -tp.sin(half), c_dtype)
    cos_c = cos_half.to(c_dtype)
    return tp.stack(
        [tp.stack([cos_c, neg_i_sin], dim=-1), tp.stack([neg_i_sin, cos_c], dim=-1)],
        dim=-2,
    )


def rot_y_matrix(theta, *, device=None, c_dtype=None) -> tp.Tensor:
    """A rotation of ``theta`` radians around the Y axis."""
    theta = _angle(theta, "theta")
    c_dtype = _param_c_dtype(theta, given=c_dtype)
    half = theta / 2
    cos_c = tp.cos(half).to(c_dtype)
    neg_sin = (-tp.sin(half)).to(c_dtype)
    sin_c = tp.sin(half).to(c_dtype)
    return tp.stack(
        [tp.stack([cos_c, neg_sin], dim=-1), tp.stack([sin_c, cos_c], dim=-1)],
        dim=-2,
    )


def rot_z_matrix(theta, *, device=None, c_dtype=None) -> tp.Tensor:
    """A rotation of ``theta`` radians around the Z axis."""
    theta = _angle(theta, "theta")
    c_dtype = _param_c_dtype(theta, given=c_dtype)
    half = theta / 2
    lower = _polar(tp.cos(half), tp.sin(half), c_dtype)
    upper = _polar(tp.cos(half), -tp.sin(half), c_dtype)
    zero = tp.zeros_like(lower)
    return tp.stack(
        [tp.stack([upper, zero], dim=-1), tp.stack([zero, lower], dim=-1)],
        dim=-2,
    )


def phase_matrix(theta, *, device=None, c_dtype=None) -> tp.Tensor:
    """The phase gate family: ``diag(1, e^{i theta})``."""
    theta = _angle(theta, "theta")
    c_dtype = _param_c_dtype(theta, given=c_dtype)
    one = tp.ones_like(theta).to(c_dtype)
    lower = _polar(tp.cos(theta), tp.sin(theta), c_dtype)
    zero = tp.zeros_like(lower)
    return tp.stack(
        [tp.stack([one, zero], dim=-1), tp.stack([zero, lower], dim=-1)],
        dim=-2,
    )


def p_matrix(theta, *, device=None, c_dtype=None) -> tp.Tensor:
    """The phase gate ``P(theta)``; a shorter spelling of :func:`phase_matrix`."""
    return phase_matrix(theta, device=device, c_dtype=c_dtype)


def u2_matrix(phi, lam, *, device=None, c_dtype=None) -> tp.Tensor:
    """The two-axis gate ``U2(phi, lam)``."""
    phi = _angle(phi, "phi")
    lam = _angle(lam, "lam")
    c_dtype = _param_c_dtype(phi, lam, given=c_dtype)
    top_left = _polar(tp.ones_like(tp.cos(phi)), tp.zeros_like(tp.sin(lam)), c_dtype)
    top_right = _polar(tp.zeros_like(tp.sin(lam)), -tp.sin(lam), c_dtype)
    bottom_left = _polar(tp.cos(phi), tp.sin(phi), c_dtype)
    bottom_right = _polar(tp.cos(phi + lam), tp.sin(phi + lam), c_dtype)
    return tp.stack(
        [tp.stack([top_left, top_right], dim=-1), tp.stack([bottom_left, bottom_right], dim=-1)],
        dim=-2,
    )


def u3_matrix(theta, phi, lam, *, device=None, c_dtype=None) -> tp.Tensor:
    """The three-parameter gate ``U3(theta, phi, lam)`` -- every one-qubit unitary."""
    theta = _angle(theta, "theta")
    phi = _angle(phi, "phi")
    lam = _angle(lam, "lam")
    c_dtype = _param_c_dtype(theta, phi, lam, given=c_dtype)
    half = theta / 2
    cos_half = tp.cos(half)
    top_left = _polar(cos_half, tp.zeros_like(cos_half), c_dtype)
    top_right = _polar(tp.zeros_like(tp.sin(lam)), -tp.sin(lam), c_dtype)
    bottom_left = _polar(tp.cos(phi), tp.sin(phi), c_dtype)
    bottom_right = _polar(tp.cos(phi + lam), tp.sin(phi + lam), c_dtype)
    return tp.stack(
        [tp.stack([top_left, top_right], dim=-1), tp.stack([bottom_left, bottom_right], dim=-1)],
        dim=-2,
    )


# ---------------------------------------------------------------------------
# Multi-qubit and controlled gates
# ---------------------------------------------------------------------------


def _control_values(control_values, n_controls: int) -> tuple:
    """Normalize a control pattern to ``n_controls`` entries of 0 or 1.

    A bare integer asks for the same reading on every control.
    """
    if control_values is None:
        return (1,) * n_controls
    if isinstance(control_values, int):
        return (control_values,) * n_controls
    try:
        values = tuple(int(value) for value in control_values)
    except TypeError:
        raise TypeError(
            f"control_values must be an int or a sequence of ints, got {control_values!r}"
        ) from None
    if len(values) != n_controls:
        raise ValueError(
            f"{n_controls} control(s) need {n_controls} control values, got {len(values)}"
        )
    for value in values:
        if value not in (0, 1):
            raise ValueError(f"a control reads 0 or 1, got {value}")
    return values


def controlled_matrix(
    mat: tp.Tensor, n_controls: int = 1, *, control_values=None
) -> tp.Tensor:
    """``mat`` applied only when the control qubits read ``control_values``.

    Controls are the leading qubits of the result, so the matrix is block
    diagonal in the control index: every control pattern but the one asked for
    carries the identity on the target, and the matching one carries ``mat``.
    Built by concatenation, so a gradient carried by ``mat`` survives.

    Args:
        mat: the operator the controls guard, ``[d, d]``, or ``[batch, d, d]``
            when it carries one operator per batch entry.
        n_controls: how many control qubits lead the result.
        control_values: what the controls have to read, one entry each and
            either 0 or 1; all ones when omitted.
    """
    if n_controls < 1:
        raise ValueError(f"a controlled gate needs at least one control, got {n_controls}")
    wanted = _control_values(control_values, n_controls)
    block = 1 << n_controls
    d = mat.shape[-1]
    # A guarded operator with an angle per batch entry keeps that entry
    # throughout, so the blocks either side of it are laid out the same way.
    batch = tuple(mat.shape[:-2])
    eye = tp.eye(d, dtype=mat.dtype, device=mat.device)
    zero = tp.zeros((d, d), dtype=mat.dtype, device=mat.device)
    if batch:
        eye = eye.expand(*batch, d, d)
        zero = tp.zeros((*batch, d, d), dtype=mat.dtype, device=mat.device)
    rows = []
    for pattern in range(block):
        matched = all(
            ((pattern >> (n_controls - 1 - i)) & 1) == wanted[i]
            for i in range(n_controls)
        )
        columns = [
            mat if q == pattern and matched else (eye if q == pattern else zero)
            for q in range(block)
        ]
        rows.append(tp.cat(columns, dim=-1))
    return tp.cat(rows, dim=-2)


def mcx_matrix(
    n_controls: int, *, control_values=None, device=None, c_dtype=tp.complex64
) -> tp.Tensor:
    """The multi-controlled X gate: X on the last wire, identity elsewhere.

    Controls come first, so the only basis pair that swaps is the one whose
    control bits read ``control_values`` -- all ones unless the caller asks for
    another reading.  ``n_controls`` counts the controls, so the matrix covers
    ``n_controls + 1`` qubits.
    """
    if n_controls < 1:
        raise ValueError(f"a controlled X needs at least one control, got {n_controls}")
    wanted = _control_values(control_values, n_controls)
    dim = 1 << (n_controls + 1)
    mat = tp.eye(dim, dtype=c_dtype, device=device)
    # The target owns bit 0 of the index; control i sits at wire position i, so
    # it owns bit n_controls - i and the first control is the leading one.
    low = sum((1 << (n_controls - i)) for i, value in enumerate(wanted) if value)
    high = low | 1
    mat[low, low] = 0
    mat[high, high] = 0
    mat[low, high] = 1
    mat[high, low] = 1
    return mat


def rccx_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The relative-phase Toffoli, on ``[control a, control b, target]``.

    The wire flip is the plain one: two controls at 1 turn the target, and the
    branch is entered through the phases ``i`` and ``-i`` rather than the whole
    space a decomposed Toffoli would walk.  Off that branch exactly one state
    changes -- control a at 1 with control b clear and the target set -- and it
    picks up a sign.  It is its own inverse.
    """
    mat = tp.eye(8, dtype=c_dtype, device=device)
    # Two controls at 1: |110> and |111> trade places through those phases.
    mat[6, 6] = 0
    mat[7, 7] = 0
    mat[6, 7] = -1j
    mat[7, 6] = 1j
    # Index 5 is control a = 1, control b = 0, target = 1.
    mat[5, 5] = -1
    return mat


def dcx_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The two-wire controlled X written the way its matrix spells it.

    The leading index bit of this matrix is the *target*: it flips when the
    other wire reads 1, so the second wire named drives the first.  That is the
    mirror of :func:`controlled_matrix` over :func:`x_matrix`, which reads its
    control off the leading bit -- same operator, opposite convention, which is
    why the gate is worth carrying under a name of its own.
    """
    return _constant(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        device=device,
        c_dtype=c_dtype,
    )


def ecr_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The echoed cross-resonance gate, the entangling native of one family.

    It swaps the single-excitation subspace with a quarter turn of phase on each
    leg, which is what makes it entangle from a product state where a plain
    controlled rotation would not.
    """
    scale = 1.0 / math.sqrt(2.0)
    return _polar(
        tp.tensor(
            [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=tp.float64,
            device=device,
        )
        * scale,
        tp.tensor(
            [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]],
            dtype=tp.float64,
            device=device,
        )
        * scale,
        c_dtype,
    )


def rc3x_matrix(*, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The four-wire relative-phase gate on ``[a, b, c, target]``.

    It is diagonal: the identity everywhere except on the states where all three
    controls read 1, where it puts ``i``, ``-i``, ``1`` and ``-1`` on ``|1100>``,
    ``|1101>``, ``|1110>`` and ``|1111>`` in turn.  Two of those are the sign a
    decomposed gate would have paid, which is what makes it cheaper; the target
    itself is never flipped.
    """
    mat = tp.eye(16, dtype=c_dtype, device=device)
    for offset, phase in enumerate((1j, -1j, 1.0, -1.0)):
        row = 12 + offset
        mat[row, row] = phase
    return mat


def globalphase_matrix(angle, *, device=None, c_dtype=None) -> tp.Tensor:
    """The scalar ``e^{i angle}`` as a 1 by 1 operator.

    Applying it to one qubit multiplies the whole state by the phase, so a
    circuit can carry a tracked phase without it touching the amplitudes'
    relative phases.
    """
    angle = _angle(angle, "angle")
    c_dtype = _param_c_dtype(angle, given=c_dtype)
    value = _polar(tp.cos(angle), tp.sin(angle), c_dtype)
    return value.reshape((1, 1))


def _sum_axis_counts(n_qubits: int, *, device=None):
    """``n_qubits - 2 * popcount`` for every basis index, as a real tensor.

    The quantity is the eigenvalue of ``Z`` on every wire summed over the
    register, which is what the multi-wire rotation turns into a phase.
    """
    idx = tp.arange(1 << n_qubits, dtype=tp.int64, device=device)
    bits = tp.arange(n_qubits, dtype=tp.int64, device=device)
    ones = ((idx.unsqueeze(-1) >> bits) & 1).sum(-1)
    return (n_qubits - 2 * ones).to(tp.float32)


def _assemble(rows):
    """A 2-D tensor from a list of rows of scalars, keeping any gradients.

    An entry that is itself a batched parameter lands the batch in front, since
    that is the layout a gate matrix has to come out in for the whole batch to
    be applied at once.
    """
    return tp.stack([tp.stack(list(row), dim=-1) for row in rows], dim=-2)


def _pair_mixer(theta, beta, *, on_odd, device, c_dtype):
    """The four-by-four matrix behind the number-conserving pair gates.

    ``on_odd`` picks the subspace the gate turns: the states carrying exactly
    one excitation for the sum gate, the states carrying an even number for the
    difference gate.  Either way the two wires start and end on the same value,
    so the excitation count is untouched.
    """
    theta = _angle(theta, "theta")
    beta = _angle(beta, "beta")
    c_dtype = _param_c_dtype(theta, beta, given=c_dtype)
    cos_half = tp.cos(theta / 2)
    sin_half = tp.sin(theta / 2)
    # -i sin(theta/2) e^{+i beta} carries sin*sin(beta) real and -sin*cos(beta)
    # imaginary; the conjugate turning is the other sign of the real part.
    beta_sin = sin_half * tp.sin(beta)
    beta_cos = sin_half * tp.cos(beta)
    zero = tp.zeros_like(cos_half)
    one = tp.ones_like(cos_half)

    if on_odd:
        real = [[one, zero, zero, zero],
                [zero, cos_half, beta_sin, zero],
                [zero, -beta_sin, cos_half, zero],
                [zero, zero, zero, one]]
        imag = [[zero, zero, zero, zero],
                [zero, zero, -beta_cos, zero],
                [zero, -beta_cos, zero, zero],
                [zero, zero, zero, zero]]
    else:
        real = [[cos_half, zero, zero, -beta_sin],
                [zero, one, zero, zero],
                [zero, zero, one, zero],
                [beta_sin, zero, zero, cos_half]]
        imag = [[zero, zero, zero, -beta_cos],
                [zero, zero, zero, zero],
                [zero, zero, zero, zero],
                [-beta_cos, zero, zero, zero]]
    return _as_complex(_assemble(real), _assemble(imag), c_dtype)


def xx_plus_yy_matrix(theta, beta=0.0, *, device=None, c_dtype=None) -> tp.Tensor:
    """The pair mixer that trades one excitation for the other.

    The states that already carry an even number of excitations are left alone
    and the pair with exactly one is turned, so the register's excitation count
    never changes -- what chemistry ansätze need from an entangler.  ``beta``
    tilts the turning towards ``X`` at 0 and towards ``Y`` at pi/2.

    Args:
        theta: how far to turn the pair.
        beta: where the turning starts from; 0 for ``XX``, pi/2 for ``YY``.
    """
    return _pair_mixer(theta, beta, on_odd=True, device=device, c_dtype=c_dtype)


def xx_minus_yy_matrix(theta, beta=0.0, *, device=None, c_dtype=None) -> tp.Tensor:
    """The pair mixer that does not touch a state of one excitation.

    The mirror image of :func:`xx_plus_yy_matrix`: same turning, opposite
    subspace, so together the two cover the ``XX`` and ``YY`` generators over the
    same two wires.
    """
    return _pair_mixer(theta, beta, on_odd=False, device=device, c_dtype=c_dtype)


def single_excitation_matrix(theta, *, device=None, c_dtype=None) -> tp.Tensor:
    """A real rotation of the pair that carries exactly one excitation.

    It starts and ends inside that pair, so no phases are needed at all -- the
    cheapest way to mix two states -- and ``-theta`` undoes it.
    """
    theta = _angle(theta, "theta")
    c_dtype = _param_c_dtype(theta, given=c_dtype)
    cos_half = tp.cos(theta / 2)
    sin_half = tp.sin(theta / 2)
    zero = tp.zeros_like(cos_half)
    one = tp.ones_like(cos_half)
    rows = [[one, zero, zero, zero],
            [zero, cos_half, -sin_half, zero],
            [zero, sin_half, cos_half, zero],
            [zero, zero, zero, one]]
    return _as_complex(_assemble(rows), _assemble([[zero] * 4 for _ in range(4)]), c_dtype)


# The Pauli products two wires can rotate about, as plain real matrices.  All
# four of them are real -- Y and Y together multiply two imaginary halves into
# a real one -- so the whole family shares one construction below.
_PAULI_PRODUCTS = {
    "XX": [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
    "YY": [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
    "ZZ": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    "ZX": [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]],
}


def _two_body_rotation(theta, product: str, *, c_dtype) -> tp.Tensor:
    """``exp(-i theta P / 2)`` for a two-wire Pauli product ``P``.

    ``cos(theta/2) * I - i * sin(theta/2) * P`` is the whole of it, and ``P`` is
    real for every product in the table, so the angle only ever goes through
    real arithmetic before the imaginary halves are attached.
    """
    theta = _angle(theta, "theta")
    c_dtype = _param_c_dtype(theta, given=c_dtype)
    cos_half = tp.cos(theta / 2)
    sin_half = tp.sin(theta / 2)
    zero = tp.zeros_like(cos_half)
    identity = [[cos_half if row == column else zero for column in range(4)]
                for row in range(4)]
    turn = _assemble([[entry * sin_half for entry in row]
                      for row in _PAULI_PRODUCTS[product]])
    return _polar(_assemble(identity), -turn, c_dtype)


def rxx_matrix(theta, *, device=None, c_dtype=None) -> tp.Tensor:
    """A rotation of ``theta`` radians generated by ``X`` on both wires."""
    return _two_body_rotation(theta, "XX", c_dtype=c_dtype)


def ryy_matrix(theta, *, device=None, c_dtype=None) -> tp.Tensor:
    """A rotation of ``theta`` radians generated by ``Y`` on both wires."""
    return _two_body_rotation(theta, "YY", c_dtype=c_dtype)


def rzz_matrix(theta, *, device=None, c_dtype=None) -> tp.Tensor:
    """A rotation of ``theta`` radians generated by ``Z`` on both wires.

    Diagonal, so it adds a phase to each basis state and never moves one: the
    gate an Ising Hamiltonian is written with.
    """
    return _two_body_rotation(theta, "ZZ", c_dtype=c_dtype)


def rzx_matrix(theta, *, device=None, c_dtype=None) -> tp.Tensor:
    """A rotation generated by ``Z`` on the leading wire and ``X`` on the other."""
    return _two_body_rotation(theta, "ZX", c_dtype=c_dtype)


def multirz_matrix(angle, n_qubits: int, *, device=None, c_dtype=None) -> tp.Tensor:
    """A rotation of ``angle`` radians about ``Z`` on every one of ``n_qubits``.

    The same angle on ``n`` wires is ``exp(-i * angle * Z^{tensor n} / 2)``, which
    is diagonal: each basis state picks up ``e^{-i angle (n - 2k) / 2}`` with
    ``k`` the number of ones in its index.
    """
    if n_qubits < 1:
        raise ValueError(f"a multi-wire rotation covers 1 or more qubits, got {n_qubits}")
    angle = _angle(angle, "angle")
    c_dtype = _param_c_dtype(angle, given=c_dtype)
    return tp.diag(_polar(*_multirz_phase(angle, n_qubits), c_dtype))


def multirz_eigvals(angle, n_qubits: int, *, device=None, c_dtype=None) -> tp.Tensor:
    """The eigenvalues :func:`multirz_matrix` is built from, as a vector.

    Measuring the rotation needs only its readings, not the matrix: they are the
    phase of each basis state.
    """
    if n_qubits < 1:
        raise ValueError(f"a multi-wire rotation covers 1 or more qubits, got {n_qubits}")
    angle = _angle(angle, "angle")
    c_dtype = _param_c_dtype(angle, given=c_dtype)
    return _polar(*_multirz_phase(angle, n_qubits), c_dtype)


def _multirz_phase(angle, n_qubits: int):
    """The cosine and sine halves of each basis state's phase.

    A rotation by ``angle`` about Z on every wire turns into the factor
    ``e^{-i angle (n_qubits - 2 k) / 2}`` on the basis state with ``k`` ones, so
    the sine half carries the minus.
    """
    counts = _sum_axis_counts(n_qubits, device=angle.device).to(angle.dtype)
    half = angle * counts / 2
    return tp.cos(half), -tp.sin(half)


def rot_matrix(axis, phi, *, device=None, c_dtype=None) -> tp.Tensor:
    """A rotation of ``phi`` radians about any axis.

    ``cos(phi/2) I - i sin(phi/2) (n_x X + n_y Y + n_z Z)`` for the unit vector
    ``axis``, so the three coordinate rotations are the three basis cases of
    this one expression.

    Args:
        axis: direction to turn about, three real components; it is normalized,
            and a zero vector is rejected.
        phi: how far to turn.
    """
    direction = tp.as_tensor(axis)
    if direction.dim() != 1 or direction.shape[0] != 3:
        raise ValueError(f"an axis is three components, got {tuple(direction.shape)}")
    phi = _angle(phi, "phi")
    c_dtype = _param_c_dtype(direction, phi, given=c_dtype)
    dtype = tp.promote_types(direction.dtype, phi.dtype)
    direction = direction.to(dtype)
    phi = phi.to(dtype)
    norm = tp.sqrt((direction * direction).sum())
    if float(norm) == 0.0:
        raise ValueError("a rotation axis has to point somewhere")
    unit = direction / norm
    nx, ny, nz = unit[0], unit[1], unit[2]
    half = phi / 2
    cosine = tp.cos(half)
    sine = tp.sin(half)
    # The real and imaginary halves are kept apart so a gradient carried by phi
    # runs through real arithmetic alone.
    entries = (
        (cosine, -sine * nz),
        (-sine * ny, -sine * nx),
        (sine * ny, -sine * nx),
        (cosine, sine * nz),
    )
    cells = [_as_complex(real, imag, c_dtype) for real, imag in entries]
    return tp.stack(
        [tp.stack(cells[0:2], dim=-1), tp.stack(cells[2:4], dim=-1)], dim=-2
    )


def pauli_rotation_matrix(word: str, angle, *, device=None, c_dtype=None) -> tp.Tensor:
    """``cos(angle/2) * I - i * sin(angle/2) * P`` for the Pauli word ``word``.

    Every Pauli factor squares to the identity, so a rotation about ``P`` has
    this closed form and needs no matrix exponential.  The word may stop after
    its last non-identity factor -- apply the result to the wires it touches.
    """
    word = check_pauli_string(word)
    # The angle decides the precision unless the caller asked for one: the
    # matrix is no wider than the parameter it was built from.
    angle = _angle(angle, "angle")
    c_dtype = _param_c_dtype(angle, given=c_dtype)
    pauli = _pauli_product(word, c_dtype=c_dtype)

    half = angle / 2
    cosine = tp.cos(half).to(c_dtype)
    # -i * sin(angle/2) as a complex scalar, so the rotation is assembled from
    # a real angle alone and a parameter gradient stays in real arithmetic.
    neg_i_sine = _as_complex(tp.zeros_like(tp.sin(half)), -tp.sin(half), c_dtype)
    identity = tp.eye(pauli.shape[-1], dtype=c_dtype, device=pauli.device)
    if angle.dim():
        # One angle per batch entry is one matrix per entry, so each half is
        # given an axis pair of its own to broadcast along rather than being
        # multiplied across the matrix.
        shape = [-1] + [1] * pauli.dim()
        cosine = cosine.reshape(shape)
        neg_i_sine = neg_i_sine.reshape(shape)
    return cosine * identity + neg_i_sine * pauli


def qft_matrix(n_qubits: int, *, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The dense unitary of the quantum Fourier transform on ``n_qubits``.

    ``F[j, k] = exp(2j*pi*j*k / 2**n) / sqrt(2**n)`` -- the matrix-product form,
    kept as the reference a gate sequence is checked against.
    """
    if n_qubits < 0:
        raise ValueError(f"a Fourier transform covers 0 or more qubits, got {n_qubits}")
    dim = 1 << n_qubits
    rows = tp.arange(dim, dtype=tp.float64, device=device).unsqueeze(-1)
    cols = tp.arange(dim, dtype=tp.float64, device=device).unsqueeze(0)
    phase = 2.0 * math.pi * rows * cols / dim
    scale = 1.0 / math.sqrt(dim)
    return _polar(tp.cos(phase) * scale, tp.sin(phase) * scale, c_dtype)


# ---------------------------------------------------------------------------
# Named applications, all funnels into apply_gate
# ---------------------------------------------------------------------------


def _on(state, mat, wire):
    return apply_gate(state, mat.to(device=state.device), wire)


def _const(state, builder, *args):
    return apply_gate(state, builder(device=state.device, c_dtype=state.dtype), *args)


def apply_i(state, wire):
    """Apply the identity on ``wire``."""
    return _const(state, i_matrix, wire)


def apply_x(state, wire):
    """Apply X on ``wire``."""
    return _const(state, x_matrix, wire)


def apply_y(state, wire):
    """Apply Y on ``wire``."""
    return _const(state, y_matrix, wire)


def apply_z(state, wire):
    """Apply Z on ``wire``."""
    return _const(state, z_matrix, wire)


def apply_h(state, wire):
    """Apply H on ``wire``."""
    return _const(state, h_matrix, wire)


def apply_s(state, wire, inverse: bool = False):
    """Apply S, or its inverse, on ``wire``."""
    return _const(state, sdg_matrix if inverse else s_matrix, wire)


def apply_t(state, wire, inverse: bool = False):
    """Apply T, or its inverse, on ``wire``."""
    return _const(state, tdg_matrix if inverse else t_matrix, wire)


def apply_sdg(state, wire):
    """Apply the inverse of S on ``wire``."""
    return _const(state, sdg_matrix, wire)


def apply_tdg(state, wire):
    """Apply the inverse of T on ``wire``."""
    return _const(state, tdg_matrix, wire)


def apply_sx(state, wire):
    """Apply sqrt(X) on ``wire``."""
    return _const(state, sx_matrix, wire)


def apply_rx(state, wire, theta):
    """Apply the X-axis rotation ``RX(theta)`` on ``wire``."""
    return _on(state, rot_x_matrix(theta), wire)


def apply_ry(state, wire, theta):
    """Apply the Y-axis rotation ``RY(theta)`` on ``wire``."""
    return _on(state, rot_y_matrix(theta), wire)


def apply_rz(state, wire, theta):
    """Apply the Z-axis rotation ``RZ(theta)`` on ``wire``."""
    return _on(state, rot_z_matrix(theta), wire)


def apply_p(state, wire, theta):
    """Apply the phase ``P(theta) = diag(1, e^{i theta})`` on ``wire``."""
    return _on(state, phase_matrix(theta), wire)


def apply_u2(state, wire, phi, lam):
    """Apply ``U2(phi, lam)`` on ``wire``."""
    return _on(state, u2_matrix(phi, lam), wire)


def apply_u3(state, wire, theta, phi, lam):
    """Apply ``U3(theta, phi, lam)`` on ``wire``."""
    return _on(state, u3_matrix(theta, phi, lam), wire)


def apply_rot(state, wire, axis, phi):
    """Apply a rotation of ``phi`` about ``axis`` on ``wire``."""
    return _on(state, rot_matrix(axis, phi), wire)


def apply_global_phase(state, angle, wire: int = 0):
    """Multiply the whole state by ``e^{i angle}``.

    The phase is carried by a 1 by 1 operator, which one wire scales the state
    by; any wire does, so the argument is only there to name one.
    """
    return _on(state, globalphase_matrix(angle), wire)


def apply_rccx(state, control_a, control_b, target):
    """Apply the relative-phase Toffoli on ``(control_a, control_b, target)``."""
    return _const(state, rccx_matrix, [control_a, control_b, target])


def apply_dcx(state, wire_a, wire_b):
    """Apply :func:`dcx_matrix` on ``(wire_a, wire_b)``, in that order.

    The matrix drives its first wire from its second, so this is
    ``apply_cnot(state, wire_b, wire_a)``.  The separate name is here because
    the gate is usually reached for as a block whose wires are written the
    other way round.
    """
    return _const(state, dcx_matrix, [wire_a, wire_b])


def apply_ecr(state, wire_a, wire_b):
    """Apply the echoed cross-resonance gate on ``(wire_a, wire_b)``."""
    return _const(state, ecr_matrix, [wire_a, wire_b])


def apply_rc3x(state, control_a, control_b, control_c, target):
    """Apply the four-wire relative-phase gate; the wires are given in order."""
    return _const(state, rc3x_matrix, [control_a, control_b, control_c, target])


def apply_xx_plus_yy(state, wires, theta, beta=0.0):
    """Mix the pair of states that carries exactly one excitation.

    Args:
        state: amplitudes shaped ``[batch, 2, 2, ...]``.
        wires: the two wires to mix.
        theta: how far to turn the pair.
        beta: where the turning starts from; 0 for ``XX``, pi/2 for ``YY``.
    """
    return apply_gate(
        state,
        xx_plus_yy_matrix(theta, beta, device=state.device),
        list(wires),
    )


def apply_xx_minus_yy(state, wires, theta, beta=0.0):
    """Mix the pair of states that carries an even number of excitations.

    The mirror of :func:`apply_xx_plus_yy`: the same turning, landing on the
    other subspace.
    """
    return apply_gate(
        state,
        xx_minus_yy_matrix(theta, beta, device=state.device),
        list(wires),
    )


def apply_single_excitation(state, wires, theta):
    """Rotate the pair that carries exactly one excitation, with no phases."""
    return apply_gate(
        state,
        single_excitation_matrix(theta, device=state.device),
        list(wires),
    )


def _apply_two_body(state, wires, theta, product: str):
    return apply_gate(
        state,
        _two_body_rotation(theta, product, c_dtype=state.dtype).to(device=state.device),
        list(wires),
    )


def apply_rxx(state, wires, theta):
    """Rotate ``theta`` radians about ``X`` on both listed wires."""
    return _apply_two_body(state, wires, theta, "XX")


def apply_ryy(state, wires, theta):
    """Rotate ``theta`` radians about ``Y`` on both listed wires."""
    return _apply_two_body(state, wires, theta, "YY")


def apply_rzz(state, wires, theta):
    """Rotate ``theta`` radians about ``Z`` on both listed wires.

    The pair is listed the same way for every two-body rotation; only
    :func:`apply_rzx` cares which wire comes first.
    """
    return _apply_two_body(state, wires, theta, "ZZ")


def apply_rzx(state, wires, theta):
    """Rotate about ``Z`` on the leading listed wire and ``X`` on the other."""
    return _apply_two_body(state, wires, theta, "ZX")


def apply_multirz(state, angle, wires=None):
    """Apply the same ``angle`` rotation about Z on every listed wire."""
    n_qubits = n_qubits_of(state)
    target = list(range(n_qubits)) if wires is None else check_wires(wires, n_qubits)
    return apply_gate(
        state,
        multirz_matrix(angle, len(target)).to(device=state.device, dtype=state.dtype),
        target,
    )


def apply_chadamard(state, control, target):
    """Apply Hadamard on ``target``, controlled by ``control``."""
    return apply_gate(
        state,
        controlled_matrix(h_matrix(device=state.device, c_dtype=state.dtype)),
        [control, target],
    )


def apply_cnot(state, control, target):
    """Apply CNOT from ``control`` to ``target``."""
    return apply_gate(
        state,
        controlled_matrix(x_matrix(device=state.device, c_dtype=state.dtype)),
        [control, target],
    )


def apply_cx(state, control, target):
    """Apply X on ``target``, controlled by ``control``."""
    return apply_cnot(state, control, target)


def apply_cz(state, control, target):
    """Apply Z on ``target``, controlled by ``control``."""
    return apply_gate(
        state,
        controlled_matrix(z_matrix(device=state.device, c_dtype=state.dtype)),
        [control, target],
    )


def apply_controlled(state, mat, wires, control_values=None):
    """Apply ``mat`` on the last wires, guarded by the leading ones.

    ``wires`` reads controls first and the target last, and ``control_values``
    says what those controls have to hold -- all ones when omitted.
    """
    return apply_gate(
        state,
        controlled_matrix(
            mat.to(device=state.device), len(wires) - 1, control_values=control_values
        ),
        wires,
    )


def apply_cy(state, control, target):
    """Apply Y on ``target``, controlled by ``control``."""
    return apply_gate(
        state,
        controlled_matrix(y_matrix(device=state.device, c_dtype=state.dtype)),
        [control, target],
    )


def apply_swap(state, wire_a, wire_b):
    """Apply a plain swap between two qubits."""
    return _const(state, swap_matrix, [wire_a, wire_b])


def apply_iswap(state, wire_a, wire_b):
    """Apply iswap between two qubits."""
    return _const(state, iswap_matrix, [wire_a, wire_b])


def apply_cswap(state, control, wire_a, wire_b):
    """Apply swap between two qubits, controlled by a third."""
    return apply_gate(
        state,
        controlled_matrix(swap_matrix(device=state.device, c_dtype=state.dtype)),
        [control, wire_a, wire_b],
    )


def apply_toffoli(state, control_a, control_b, target, control_values=None):
    """Apply CCX with two controls."""
    return apply_gate(
        state,
        mcx_matrix(
            2,
            control_values=control_values,
            device=state.device,
            c_dtype=state.dtype,
        ),
        [control_a, control_b, target],
    )


def apply_ccx(state, control_a, control_b, target):
    """Apply CCX with two controls; a shorter spelling of :func:`apply_toffoli`."""
    return apply_toffoli(state, control_a, control_b, target)


def apply_rccx(state, control_a, control_b, target):
    """Apply the relative-phase Toffoli on ``(control_a, control_b, target)``."""
    return _const(state, rccx_matrix, [control_a, control_b, target])


def apply_mcx(state, wires, control_values=None):
    """Apply a multi-controlled X; the last wire is the target.

    Every control reads 1 unless ``control_values`` asks for another reading.
    """
    wires = ordered_wires(wires, n_qubits_of(state))
    return apply_gate(
        state,
        mcx_matrix(
            len(wires) - 1,
            control_values=control_values,
            device=state.device,
            c_dtype=state.dtype,
        ),
        wires,
    )


def _apply_controlled_rotation(state, wires, angle, builder):
    """Apply ``exp(-i angle P / 2)`` on ``target``, guarded by the controls."""
    return apply_gate(
        state,
        controlled_matrix(
            builder(angle).to(device=state.device), len(wires) - 1
        ),
        wires,
    )


def _apply_controlled_unitary(state, control, target, builder):
    """Apply a fixed single-qubit unitary, guarded by one control wire."""
    return apply_gate(
        state,
        controlled_matrix(builder(device=state.device, c_dtype=state.dtype)),
        [control, target],
    )


def apply_cs(state, control, target):
    """Apply ``S`` on ``target``, controlled by ``control``."""
    return _apply_controlled_unitary(state, control, target, s_matrix)


def apply_csdg(state, control, target):
    """Apply ``S^dagger`` on ``target``, controlled by ``control``."""
    return _apply_controlled_unitary(state, control, target, sdg_matrix)


def apply_csx(state, control, target):
    """Apply the square root of ``X`` on ``target``, controlled by ``control``."""
    return _apply_controlled_unitary(state, control, target, sx_matrix)


def apply_ccz(state, control_a, control_b, target):
    """Turn the one state with all three wires set over by a sign.

    Two controls and a phase: nothing moves, but ``|...1 1 1>`` picks up a
    minus, which is the gate a phase-preserving ansatz reaches for.
    """
    return apply_controlled(
        state,
        z_matrix(device=state.device, c_dtype=state.dtype),
        [control_a, control_b, target],
    )


def apply_cp(state, control, target, theta):
    """Apply ``P(theta)`` on ``target``, controlled by ``control``."""
    return _apply_controlled_rotation(state, [control, target], theta, phase_matrix)


def apply_crx(state, control, target, theta):
    """Apply ``RX(theta)`` on ``target``, controlled by ``control``."""
    return _apply_controlled_rotation(state, [control, target], theta, rot_x_matrix)


def apply_cry(state, control, target, theta):
    """Apply ``RY(theta)`` on ``target``, controlled by ``control``."""
    return _apply_controlled_rotation(state, [control, target], theta, rot_y_matrix)


def apply_crz(state, control, target, theta):
    """Apply ``RZ(theta)`` on ``target``, controlled by ``control``."""
    return _apply_controlled_rotation(state, [control, target], theta, rot_z_matrix)


def apply_pauli_rotation(state, word: str, angle):
    """Rotate the state by ``exp(-i * angle * P / 2)`` for the Pauli word ``word``.

    The half is the usual one -- an angle is what a half turn costs, so a full
    turn is ``2 * pi`` -- which is what makes the parameter-shift rule come out
    in :mod:`tensorplay.quantum._gradients`.

    The word is padded with identities on the right, so ``"XY"`` acts on wires
    0 and 1 of a longer register.  Identity factors carry no rotation of their
    own, so the gate lands on the wires the word actually touches.
    """
    n_qubits = n_qubits_of(state)
    word = check_pauli_string(word)
    if len(word) > n_qubits:
        raise ValueError(f"Pauli word {word!r} does not fit in {n_qubits} qubit(s)")
    wires = pauli_wires(word)
    rotation = pauli_rotation_matrix(word, angle).to(device=state.device)
    # An identity word owns no wires; routing its 1x1 rotation through wire 0
    # keeps the global phase it carries.
    return apply_gate(state, rotation, wires or [0])


def pauli_matrix(word: str, *, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The dense matrix of a Pauli word.

    Identity factors are dropped, so ``"IXZ"`` gives a 4 by 4 matrix over the
    wires the word touches; use ``expand_to_register`` to lift it back.  A word
    of pure identities is the empty product, a 1 by 1 matrix holding 1 -- which
    broadcasts onto any register.
    """
    word = check_pauli_string(word)
    return _pauli_product(word, device=device, c_dtype=c_dtype)


def _pauli_product(word: str, *, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The Kronecker product of a Pauli word's non-identity factors."""
    factors = {
        "I": i_matrix(c_dtype=c_dtype),
        "X": x_matrix(c_dtype=c_dtype),
        "Y": y_matrix(c_dtype=c_dtype),
        "Z": z_matrix(c_dtype=c_dtype),
    }
    letters = [letter for letter in check_pauli_string(word) if letter != "I"]
    if not letters:
        return tp.ones((1, 1), dtype=c_dtype, device=device)
    mat = factors[letters[0]].to(device=device)
    for letter in letters[1:]:
        mat = tp.kron(mat, factors[letter])
    return mat


def pauli_rotation_wires(word: str) -> List[int]:
    """The wires a Pauli word acts on."""
    return pauli_wires(word)
