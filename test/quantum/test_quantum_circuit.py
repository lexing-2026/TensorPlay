"""Circuit composition, dense unitaries, inversion, and the Fourier transform."""

import pytest

import tensorplay as tp
from tensorplay import quantum as tq


def _flat(state):
    return state.reshape(state.shape[0], -1)


def test_bell_circuit_prepares_the_bell_state():
    circuit = tq.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    expected = tp.tensor([[0.70710678, 0.0, 0.0, 0.70710678]], dtype=tp.complex64)
    tp.testing.assert_close(_flat(circuit()), expected, atol=1e-6, rtol=1e-6)


def test_circuit_starts_from_the_all_zero_state():
    circuit = tq.Circuit(3)
    tp.testing.assert_close(_flat(circuit()), tp.tensor([[1.0] + [0.0] * 7], dtype=tp.complex64))


def test_circuit_accepts_a_start_state():
    circuit = tq.Circuit(1)
    circuit.x(0)
    tp.testing.assert_close(_flat(circuit(tq.zeros_state(1))), tp.tensor([[0.0, 1.0]], dtype=tp.complex64))


def test_circuit_records_its_gates():
    circuit = tq.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    assert len(circuit) == 2
    assert [name for name, _, _ in circuit.gates()] == ["h", "cx"]
    assert circuit.gates()[1][1] == (0, 1)


def test_barrier_is_a_marker_not_a_gate():
    circuit = tq.Circuit(2)
    circuit.h(0)
    circuit.barrier()
    circuit.cx(0, 1)
    assert len(circuit) == 3
    tp.testing.assert_close(_flat(circuit()), _flat(tq.apply_cnot(tq.apply_h(tq.zeros_state(2), 0), 0, 1)))


def test_circuit_rejects_an_unknown_gate():
    with pytest.raises(KeyError):
        tq.Circuit(2).add("teleport", 0)


def test_circuit_rejects_a_wrong_parameter_count():
    with pytest.raises(ValueError):
        tq.Circuit(2).add("rx", 0)


def test_circuit_rejects_the_wrong_wire_count():
    with pytest.raises(ValueError):
        tq.Circuit(3).add("cx", 0)


def test_circuit_rejects_a_wire_outside_the_register():
    with pytest.raises(IndexError):
        tq.Circuit(2).add("h", 4)


def test_unitary_matches_the_gate_sequence():
    circuit = tq.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.t(1)
    unitary = circuit.unitary()
    assert unitary.shape == (4, 4)
    # |00> becomes (|00> + e^{i pi/4}|11>)/sqrt(2): the T gate is on the target.
    expected = tp.tensor(
        [[0.70710678, 0.0, 0.0, 0.5 + 0.5j]], dtype=tp.complex64
    )
    tp.testing.assert_close(_flat(circuit()), expected, atol=1e-6, rtol=1e-6)
    tp.testing.assert_close(
        unitary[:, :1].reshape(1, 4), expected, atol=1e-6, rtol=1e-6
    )


def test_unitary_of_the_bell_circuit_maps_zero_to_the_bell_state():
    circuit = tq.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    unitary = circuit.unitary()
    expected = tp.tensor(
        [[0.70710678, 0.0, 0.0, 0.70710678]], dtype=tp.complex64
    )
    tp.testing.assert_close(unitary[:, :1].reshape(1, 4), expected, atol=1e-6, rtol=1e-6)


def test_unitary_is_a_unitary_matrix():
    circuit = tq.Circuit(2)
    circuit.h(0)
    circuit.rx(1, 0.3)
    unitary = circuit.unitary()
    product = tq.dagger(unitary) @ unitary
    tp.testing.assert_close(product, tp.eye(4, dtype=product.dtype), atol=1e-6, rtol=1e-6)


def test_inverse_undoes_the_circuit():
    circuit = tq.Circuit(2)
    circuit.h(0)
    circuit.rx(1, 0.4)
    circuit.cx(0, 1)
    start = tq.plus_state(2)
    restored = circuit.inverse()(circuit.apply(start))
    tp.testing.assert_close(_flat(restored), _flat(start), atol=1e-6, rtol=1e-6)


def test_inverse_negates_rotation_angles():
    circuit = tq.Circuit(1)
    circuit.rx(0, 0.8)
    inverse = circuit.inverse()
    names = [name for name, _, _ in inverse.gates()]
    assert names == ["rx"]
    tp.testing.assert_close(float(inverse.gates()[0][2][0]), -0.8, atol=1e-6, rtol=0)


def test_inverse_of_self_inverse_gates():
    circuit = tq.Circuit(1)
    circuit.s(0)
    circuit.t(0)
    restored = circuit.inverse()(circuit(tq.zeros_state(1)))
    tp.testing.assert_close(_flat(restored), _flat(tq.zeros_state(1)), atol=1e-6, rtol=1e-6)


def test_circuit_expectation_reads_the_prepared_state():
    circuit = tq.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    tp.testing.assert_close(circuit.expectation("ZZ"), tp.tensor([1.0]), atol=1e-6, rtol=1e-6)


def test_circuit_batch():
    circuit = tq.Circuit(2)
    circuit.h(0)
    state = circuit(batch=5)
    assert state.shape == (5, 2, 2)
    tp.testing.assert_close(tq.probs(state)[:, 0], tp.full((5,), 0.5), atol=1e-6, rtol=1e-6)


def test_expand_to_register_places_the_gate_on_its_wires():
    # Wire 1 of a two-qubit register is the low index bit, so X on it swaps
    # the pairs (0, 1) and (2, 3).
    mat = tq.x_matrix()
    wide = tq.expand_to_register(mat, [1], 2)
    expected = tp.tensor(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=tp.complex64
    )
    tp.testing.assert_close(wide, expected)


def test_expand_to_register_keeps_a_full_width_gate():
    mat = tq.swap_matrix()
    tp.testing.assert_close(tq.expand_to_register(mat, [0, 1], 2), mat)


def test_expand_to_register_rejects_a_mismatched_matrix():
    with pytest.raises(ValueError):
        tq.expand_to_register(tq.swap_matrix(), [0], 2)


def test_qft_unitary_matches_the_dense_matrix():
    circuit = tq.qft_circuit(3)
    tp.testing.assert_close(circuit.unitary(), tq.qft_matrix(3), atol=1e-5, rtol=1e-5)


def test_inverse_qft_undoes_the_transform():
    circuit = tq.qft_circuit(3)
    state = tq.basis_state(5, 3)
    tp.testing.assert_close(
        _flat(circuit.inverse()(circuit(state))), _flat(state), atol=1e-5, rtol=1e-5
    )


def test_qft_unitary_squares_to_the_bit_reversal():
    # The transform is symmetric and unitary, so its square pairs every basis
    # index with its mirror, and its fourth power is the identity.
    circuit = tq.qft_circuit(2)
    unitary = circuit.unitary()
    reversal = tp.tensor(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=tp.complex64
    )
    squared = unitary @ unitary
    tp.testing.assert_close(squared, reversal, atol=1e-5, rtol=1e-5)
    tp.testing.assert_close(
        squared @ squared, tp.eye(4, dtype=unitary.dtype), atol=1e-5, rtol=1e-5
    )


def test_circuit_repr_lists_the_gates():
    circuit = tq.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    assert "h[0]" in repr(circuit)
    assert "cx[0, 1]" in repr(circuit)
