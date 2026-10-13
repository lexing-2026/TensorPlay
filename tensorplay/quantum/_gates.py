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
from ._helpers import check_pauli_string, check_wires, n_qubits_of, pauli_wires, promoted_c_dtype

__all__ = [
    "apply_cnot",
    "apply_cswap",
    "apply_cx",
    "apply_cz",
    "apply_gate",
    "apply_h",
    "apply_iswap",
    "apply_mcx",
    "apply_p",
    "apply_rx",
    "apply_ry",
    "apply_rz",
    "apply_s",
    "apply_swap",
    "apply_sx",
    "apply_t",
    "apply_toffoli",
    "apply_u3",
    "apply_x",
    "apply_y",
    "apply_z",
    "controlled_matrix",
    "dagger",
    "h_matrix",
    "i_matrix",
    "iswap_matrix",
    "mcx_matrix",
    "p_matrix",
    "pauli_rotation_matrix",
    "phase_matrix",
    "qft_matrix",
    "rot_x_matrix",
    "rot_y_matrix",
    "rot_z_matrix",
    "s_matrix",
    "sdg_matrix",
    "swap_matrix",
    "sx_matrix",
    "t_matrix",
    "tdg_matrix",
    "u2_matrix",
    "u3_matrix",
    "x_matrix",
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
        wires: the target qubits, ascending; ``k = len(wires)``.

    Returns:
        The new state, in the layout of the input.
    """
    n_qubits = n_qubits_of(state)
    target = check_wires(wires, n_qubits)
    k = len(target)

    if mat.dim() not in (2, 3):
        raise ValueError(f"a gate matrix is 2-D or batched 3-D, got {mat.dim()}-D")

    c_dtype = promoted_c_dtype(tp.promote_types(state.dtype, mat.dtype))
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

    # The gate follows the state: a matrix built for another device or without
    # an imaginary half is finished off here.  The compiled path reads both
    # operands as the result dtype, so the narrower one is converted rather
    # than reinterpreted.
    mat = mat.to(device=state.device, dtype=c_dtype)
    if state.dtype != c_dtype:
        state = state.to(c_dtype)

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

    out = mat.bmm(moved)
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


def rot_x_matrix(theta, *, device=None, c_dtype=None) -> tp.Tensor:
    """A rotation of ``theta`` radians around the X axis."""
    theta = tp.as_tensor(theta)
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
    theta = tp.as_tensor(theta)
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
    theta = tp.as_tensor(theta)
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
    theta = tp.as_tensor(theta)
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
    phi = tp.as_tensor(phi)
    lam = tp.as_tensor(lam)
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
    theta = tp.as_tensor(theta)
    phi = tp.as_tensor(phi)
    lam = tp.as_tensor(lam)
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


def controlled_matrix(mat: tp.Tensor, n_controls: int = 1) -> tp.Tensor:
    """``mat`` applied only when every control qubit is 1.

    Controls are the leading qubits of the result, so the matrix is block
    diagonal in the control index: identity blocks for every control pattern
    except the all-ones one, which carries ``mat``.  Built by concatenation, so
    a gradient carried by ``mat`` survives.
    """
    if n_controls < 1:
        raise ValueError(f"a controlled gate needs at least one control, got {n_controls}")
    block = 1 << n_controls
    d = mat.shape[-1]
    side = block * d
    top = side - d
    identity = tp.eye(top, dtype=mat.dtype, device=mat.device)
    zeros_right = tp.zeros((top, d), dtype=mat.dtype, device=mat.device)
    zeros_left = tp.zeros((d, top), dtype=mat.dtype, device=mat.device)
    return tp.cat(
        [tp.cat([identity, zeros_right], dim=-1), tp.cat([zeros_left, mat], dim=-1)],
        dim=-2,
    )


def mcx_matrix(n_controls: int, *, device=None, c_dtype=tp.complex64) -> tp.Tensor:
    """The multi-controlled X gate: X on the last wire, identity elsewhere.

    Controls come first, so the only basis pair that swaps is the one where
    every control bit is 1.
    """
    if n_controls < 1:
        raise ValueError(f"a controlled X needs at least one control, got {n_controls}")
    dim = 1 << (n_controls + 1)
    mat = tp.eye(dim, dtype=c_dtype, device=device)
    low = dim - 2  # controls all 1, target 0
    high = dim - 1  # controls all 1, target 1
    mat[low, low] = 0
    mat[high, high] = 0
    mat[low, high] = 1
    mat[high, low] = 1
    return mat


def pauli_rotation_matrix(word: str, angle, *, device=None, c_dtype=None) -> tp.Tensor:
    """``cos(angle/2) * I - i * sin(angle/2) * P`` for the Pauli word ``word``.

    Every Pauli factor squares to the identity, so a rotation about ``P`` has
    this closed form and needs no matrix exponential.  The word may stop after
    its last non-identity factor -- apply the result to the wires it touches.
    """
    word = check_pauli_string(word)
    c_dtype = c_dtype or tp.complex64
    pauli = _pauli_product(word, c_dtype=c_dtype)

    angle = tp.as_tensor(angle)
    c_dtype = _param_c_dtype(angle, given=c_dtype)
    half = angle / 2
    cosine = tp.cos(half).to(c_dtype)
    # -i * sin(angle/2) as a complex scalar, so the rotation is assembled from
    # a real angle alone and a parameter gradient stays in real arithmetic.
    neg_i_sine = _as_complex(tp.zeros_like(tp.sin(half)), -tp.sin(half), c_dtype)
    identity = tp.eye(pauli.shape[-1], dtype=c_dtype, device=pauli.device)
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


def apply_u3(state, wire, theta, phi, lam):
    """Apply ``U3(theta, phi, lam)`` on ``wire``."""
    return _on(state, u3_matrix(theta, phi, lam), wire)


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


def apply_toffoli(state, control_a, control_b, target):
    """Apply CCX with two controls."""
    return apply_gate(
        state,
        mcx_matrix(2, device=state.device, c_dtype=state.dtype),
        [control_a, control_b, target],
    )


def apply_mcx(state, wires):
    """Apply a multi-controlled X; the last wire is the target."""
    wires = check_wires(wires, n_qubits_of(state))
    return apply_gate(
        state,
        mcx_matrix(len(wires) - 1, device=state.device, c_dtype=state.dtype),
        wires,
    )


def apply_pauli_rotation(state, word: str, angle):
    """Apply ``exp(-i * angle * P)`` for the Pauli word ``word``.

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
