"""Gate semantics: known amplitudes, dense agreement, wire ordering."""
import math

import pytest

import tensorplay as tp
from tensorplay import quantum as tq


def _flat(state):
    return state.reshape(1, -1)


def test_hadamard_on_zero_state():
    state = tq.zeros_state(1)
    out = tq.apply_h(state, 0)
    tp.testing.assert_close(_flat(out), tp.tensor([[0.70710678, 0.70710678]], dtype=tp.complex64))


def test_x_flips_the_only_qubit():
    state = tq.zeros_state(1)
    out = tq.apply_x(state, 0)
    tp.testing.assert_close(_flat(out), tp.tensor([[0.0, 1.0]], dtype=tp.complex64))


def test_z_is_diagonal():
    state = tq.zeros_state(1)
    tp.testing.assert_close(_flat(tq.apply_z(state, 0)), _flat(state))


def test_s_gives_quarter_phase():
    state = tq.basis_state(1, 1)
    out = tq.apply_s(state, 0)
    tp.testing.assert_close(_flat(out), tp.tensor([[0.0, 1.0j]], dtype=tp.complex64))


def test_s_inverse_undoes_s():
    state = tq.zeros_state(2)
    rotated = tq.apply_s(tq.apply_x(tq.apply_h(state, 1), 1), 1)
    restored = tq.apply_s(rotated, 1, inverse=True)
    tp.testing.assert_close(_flat(restored), _flat(tq.apply_x(tq.apply_h(state, 1), 1)))


def test_t_matrix_is_an_eighth_turn():
    mat = tq.t_matrix()
    expected = tp.tensor([[1.0, 0.0], [0.0, 0.70710678 + 0.70710678j]], dtype=tp.complex64)
    tp.testing.assert_close(mat, expected, atol=1e-6, rtol=1e-6)


def test_cnot_bell_pair():
    state = tq.zeros_state(2)
    out = tq.apply_cnot(tq.apply_h(state, 0), 0, 1)
    expected = tp.tensor([[0.70710678, 0.0, 0.0, 0.70710678]], dtype=tp.complex64)
    tp.testing.assert_close(_flat(out), expected)


def test_cnot_does_nothing_when_control_is_zero():
    state = tq.basis_state(1, 2)  # |01>: wire 0 is 0
    tp.testing.assert_close(_flat(tq.apply_cnot(state, 0, 1)), _flat(state))


def test_wire_zero_is_the_most_significant_bit():
    state = tq.zeros_state(2)
    out = tq.apply_x(state, 0)  # |00> -> |10>, index 2 under big-endian
    tp.testing.assert_close(_flat(out), tp.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=tp.complex64))


def test_swap_exchanges_amplitudes():
    state = tq.basis_state(1, 2)  # |01>
    out = tq.apply_swap(state, 0, 1)
    tp.testing.assert_close(_flat(out), tp.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=tp.complex64))


def test_iswap_moves_amplitude_with_a_phase():
    state = tq.basis_state(1, 2)
    out = tq.apply_iswap(state, 0, 1)
    tp.testing.assert_close(_flat(out), tp.tensor([[0.0, 0.0, 1.0j, 0.0]], dtype=tp.complex64), atol=1e-6, rtol=0)


def test_toffoli_flips_only_when_both_controls_are_set():
    state = tq.basis_state(6, 3)  # |110>
    out = tq.apply_toffoli(state, 0, 1, 2)
    tp.testing.assert_close(
        _flat(out),
        tp.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=tp.complex64),
    )


def test_toffoli_is_inert_on_a_quiet_control():
    state = tq.basis_state(2, 3)  # |010>
    tp.testing.assert_close(_flat(tq.apply_toffoli(state, 0, 1, 2)), _flat(state))


def test_mcx_with_three_controls():
    state = tq.basis_state(14, 4)  # |1110>
    out = tq.apply_mcx(state, [0, 1, 2, 3])
    tp.testing.assert_close(
        _flat(out), tp.tensor([[0.0] * 15 + [1.0]], dtype=tp.complex64)
    )


def test_rotations_match_their_matrices():
    theta = 0.7
    for apply, builder in (
        (tq.apply_rx, tq.rot_x_matrix),
        (tq.apply_ry, tq.rot_y_matrix),
        (tq.apply_rz, tq.rot_z_matrix),
    ):
        state = tq.zeros_state(1)
        by_gate = _flat(apply(state, 0, theta))
        by_matrix = (builder(theta).to(state.dtype) @ _flat(state).transpose(-1, -2)).transpose(-1, -2)
        tp.testing.assert_close(by_gate, by_matrix, atol=1e-6, rtol=1e-6)


def test_ry_by_a_half_turn_moves_to_one():
    state = tq.zeros_state(1)
    out = tq.apply_ry(state, 0, math.pi)
    # RY(pi) = [[0, -1], [1, 0]]: |0> picks up the +1 in the first column.
    tp.testing.assert_close(_flat(out), tp.tensor([[0.0, 1.0]], dtype=tp.complex64), atol=1e-6, rtol=0)


def test_rotation_is_unitarily_inverted_by_the_negative_angle():
    state = tq.zeros_state(2)
    forward = tq.apply_ry(tq.apply_x(tq.apply_h(state, 0), 0), 0, 0.3)
    back = tq.apply_ry(forward, 0, -0.3)
    tp.testing.assert_close(_flat(back), _flat(tq.apply_x(tq.apply_h(state, 0), 0)), atol=1e-6, rtol=0)


def test_apply_gate_agrees_with_the_register_unary():
    state = tq.plus_state(3)
    mat = tq.rot_y_matrix(0.42).to(state.dtype)
    by_gate = tq.apply_gate(state, mat, [1])
    wide = tq.expand_to_register(mat, [1], 3)
    by_matrix = (wide @ _flat(state).transpose(-1, -2)).transpose(-1, -2)
    tp.testing.assert_close(_flat(by_gate), by_matrix, atol=1e-6, rtol=1e-6)


def test_apply_gate_on_several_wires():
    state = tq.basis_state(6, 3)  # |110>: both controls set
    mat = tq.mcx_matrix(2, c_dtype=state.dtype)
    out = tq.apply_gate(state, mat, [0, 1, 2])
    expected = tp.zeros((1, 8), dtype=tp.complex64)
    expected[0, 7] = 1.0
    tp.testing.assert_close(_flat(out), expected)


def test_batched_gate_matrix_applies_one_matrix_per_state():
    theta = tp.tensor([0.0, math.pi])
    states = tq.zeros_state(1, batch=2)
    out = tq.apply_gate(states, tq.rot_y_matrix(theta), [0])
    flat = out.reshape(2, -1)
    tp.testing.assert_close(flat[0], tp.tensor([1.0, 0.0], dtype=tp.complex64))
    tp.testing.assert_close(flat[1], tp.tensor([0.0, 1.0], dtype=tp.complex64), atol=1e-6, rtol=0)


def test_dagger_inverts_a_unitary():
    mat = tq.rot_x_matrix(0.31)
    product = tq.dagger(mat) @ mat
    tp.testing.assert_close(product, tp.eye(2, dtype=product.dtype), atol=1e-6, rtol=1e-6)


def test_gate_rejects_a_matrix_of_the_wrong_width():
    state = tq.zeros_state(2)
    with pytest.raises(ValueError, match="2x2"):
        tq.apply_gate(state, tp.eye(4, dtype=tp.complex64), [0])


def test_gate_rejects_an_out_of_range_wire():
    state = tq.zeros_state(2)
    with pytest.raises(IndexError):
        tq.apply_h(state, 5)


def test_apply_gate_keeps_the_norm():
    state = tq.plus_state(4)
    out = tq.apply_h(state, 2)
    tp.testing.assert_close(tq.norm(out), tp.tensor([1.0]), atol=1e-6, rtol=1e-6)


def test_pauli_rotation_is_a_unitary():
    word = "XY"
    rotation = tq.pauli_rotation_matrix(word, 0.6)
    product = tq.dagger(rotation) @ rotation
    tp.testing.assert_close(product, tp.eye(4, dtype=product.dtype), atol=1e-6, rtol=1e-6)


def test_pauli_rotation_matches_its_own_exponential():
    pauli = tq.pauli_matrix("XZ")
    time = 0.35
    from_exp = tp.matrix_exp(-1j * time * pauli)
    from_rotation = tq.pauli_rotation_matrix("XZ", 2.0 * time)
    tp.testing.assert_close(from_rotation, from_exp, atol=1e-6, rtol=1e-6)
