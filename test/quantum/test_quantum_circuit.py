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


# ---------------------------------------------------------------------------
# Layer templates
# ---------------------------------------------------------------------------


def _names(circuit):
    return [(name, wires) for name, wires, _ in circuit.gates()]


def test_entanglement_pairs_walk_the_register():
    assert tq.entanglement_pairs(3, "linear") == [(0, 1), (1, 2)]
    assert tq.entanglement_pairs(3, "reverse_linear") == [(1, 0), (2, 1)]


def test_the_circular_pattern_closes_the_ring():
    assert tq.entanglement_pairs(3, "circular") == [(0, 1), (1, 2), (2, 0)]
    # A single wire has no edge to close, and a two-wire ring has two directions
    # through the one edge it has.
    assert tq.entanglement_pairs(1, "circular") == []
    assert tq.entanglement_pairs(2, "circular") == [(0, 1), (1, 0)]


def test_the_full_pattern_takes_every_pair():
    assert tq.entanglement_pairs(3, "full") == [(0, 1), (0, 2), (1, 2)]


def test_an_unknown_entanglement_pattern_is_rejected():
    with pytest.raises(ValueError):
        tq.entanglement_pairs(3, "spiral")


def test_real_amplitudes_repeats_rotations_and_pairs():
    names = _names(tq.real_amplitudes(3, reps=2))
    assert names == [
        ("ry", (0,)), ("ry", (1,)), ("ry", (2,)),
        ("cx", (1, 0)), ("cx", (2, 1)),
        ("ry", (0,)), ("ry", (1,)), ("ry", (2,)),
        ("cx", (1, 0)), ("cx", (2, 1)),
        ("ry", (0,)), ("ry", (1,)), ("ry", (2,)),
    ]


def test_the_closing_rotation_layer_can_be_skipped():
    names = _names(tq.real_amplitudes(2, reps=1, skip_final_rotation_layer=True))
    assert names == [("ry", (0,)), ("ry", (1,)), ("cx", (1, 0))]


def test_efficient_su2_rotates_about_both_axes():
    names = _names(tq.efficient_su2(2, reps=1))
    assert names == [
        ("ry", (0,)), ("ry", (1,)),
        ("rz", (0,)), ("rz", (1,)),
        ("cx", (1, 0)),
        ("ry", (0,)), ("ry", (1,)),
        ("rz", (0,)), ("rz", (1,)),
    ]


def test_every_rotation_gets_its_own_angle():
    circuit = tq.real_amplitudes(3, reps=2)
    angles = list(circuit.parameters())
    assert len(angles) == 9  # three wires, two blocks, one closing layer
    assert all(angle.shape == (1,) for angle in angles)


def test_the_angles_arrive_within_a_turn_of_each_other():
    for angle in tq.real_amplitudes(4, reps=1).parameters():
        assert float(angle.reshape(())) <= 3.1415927
        assert float(angle.reshape(())) >= -3.1415927


def test_zero_angles_leave_the_start_state_alone():
    circuit = tq.real_amplitudes(3, reps=2)
    for angle in circuit.parameters():
        angle.data = tp.zeros_like(angle)
    probs = tq.probs(circuit.state()).reshape(-1)
    tp.testing.assert_close(probs, tp.tensor([1.0] + [0.0] * 7), atol=1e-6, rtol=1e-6)


def test_a_real_ansatz_keeps_the_state_real():
    state = tq.real_amplitudes(3, reps=2).state()
    assert float(state.imag.abs().max()) == 0.0


def test_the_angles_are_trainable():
    circuit = tq.real_amplitudes(2, reps=1)
    tq.expectation(circuit.state(), "ZI").backward()
    for angle in circuit.parameters():
        assert angle.grad is not None


def test_an_ansatz_repeats_at_least_once():
    with pytest.raises(ValueError):
        tq.real_amplitudes(2, reps=0)


def test_a_custom_rotation_set_drives_the_same_layout():
    names = _names(tq.nlocal_circuit(2, ("ry", "rz", "p"), reps=1, entanglement="linear"))
    # Two rotation layers, two wires each.
    assert [name for name, _ in names].count("p") == 4
    assert ("cx", (0, 1)) in names  # the linear pattern names the lower wire first


def test_the_dense_unitary_agrees_with_the_gate_application_in_both_directions():
    from tensorplay.quantum import basis_state

    for wires in ((0, 1), (1, 0)):
        circuit = tq.Circuit(2)
        circuit.add("cx", wires)
        for index in range(4):
            entry = basis_state(index, 2)
            tp.testing.assert_close(
                circuit.unitary() @ entry.reshape(-1),
                circuit.apply(entry).reshape(-1),
                atol=1e-6,
                rtol=1e-6,
            )


def test_a_multi_controlled_gate_finds_its_wires_in_the_dense_unitary():
    from tensorplay.quantum import basis_state

    circuit = tq.Circuit(3)
    circuit.add("mcx", (0, 2, 1))
    for index in range(8):
        entry = basis_state(index, 3)
        tp.testing.assert_close(
            circuit.unitary() @ entry.reshape(-1),
            circuit.apply(entry).reshape(-1),
            atol=1e-6,
            rtol=1e-6,
        )


def test_a_circuit_can_drive_a_higher_wire_from_a_lower_one():
    circuit = tq.Circuit(2)
    circuit.cx(0, 1)
    assert circuit.gates()[0][1] == (0, 1)
    circuit = tq.Circuit(2)
    circuit.cx(1, 0)
    assert circuit.gates()[0][1] == (1, 0)
    # |01> -> |11> when wire 1 drives wire 0, and stays put the other way round.
    tp.testing.assert_close(
        _flat(circuit.apply(tq.apply_x(tq.zeros_state(2), 1))),
        _flat(tq.basis_state(3, 2)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_the_inverse_of_a_reversed_circuit_still_undoes_it():
    from tensorplay.quantum import basis_state

    circuit = tq.Circuit(3)
    circuit.cx(2, 0)
    circuit.cry(1, 0, 0.3)
    undone = circuit.inverse()(circuit())
    tp.testing.assert_close(
        _flat(undone), _flat(tq.zeros_state(3)), atol=1e-6, rtol=1e-6
    )
    for index in range(8):
        entry = basis_state(index, 3)
        tp.testing.assert_close(
            _flat(circuit.inverse().apply(circuit.apply(entry))),
            _flat(entry),
            atol=1e-6,
            rtol=1e-6,
        )


def test_an_ansatz_trains_down_the_energy_of_a_hamiltonian():
    h2 = tq.Hamiltonian(
        [
            (-1.0523732, "II"),
            (0.3979374, "IZ"),
            (-0.3979374, "ZI"),
            (-0.1809312, "XX"),
        ]
    )
    circuit = tq.real_amplitudes(2, reps=3)
    optimizer = tp.optim.Adam(circuit.parameters(), lr=0.1)
    start = float(tq.energy(circuit.state(), h2).sum())
    for _ in range(400):
        energy = tq.energy(circuit.state(), h2).sum()
        optimizer.zero_grad()
        energy.backward()
        optimizer.step()
    end = float(energy)
    assert end < start
    # The real-amplitude ansatz reaches the ground state of this Hamiltonian
    # exactly, so it has little left to find.
    tp.testing.assert_close(
        tp.tensor(end), tp.tensor(float(tq.energy(tq.basis_state(1, 2), h2).sum())),
        atol=0.05,
        rtol=0.05,
    )


def test_the_fourier_transform_reads_its_phases_from_the_lower_wire():
    for n_qubits in (2, 3, 4):
        circuit = tq.qft_circuit(n_qubits)
        tp.testing.assert_close(
            circuit.unitary(), tq.qft_matrix(n_qubits), atol=1e-5, rtol=1e-5
        )
        tp.testing.assert_close(
            circuit.unitary() @ circuit.inverse().unitary(),
            tp.eye(1 << n_qubits, dtype=tq.qft_matrix(n_qubits).dtype),
            atol=1e-5,
            rtol=1e-5,
        )
