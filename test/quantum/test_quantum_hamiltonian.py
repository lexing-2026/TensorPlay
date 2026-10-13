"""Pauli-sum Hamiltonians: energy, dense matrices, and time evolution."""

import pytest

import tensorplay as tp
from tensorplay import quantum as tq

H2 = [(-1.0523732, "II"), (0.3979374, "IZ"), (-0.3979374, "ZI"), (-0.1809312, "XX")]


def _flat(state):
    return state.reshape(state.shape[0], -1)


def test_hamiltonian_keeps_its_terms():
    hamiltonian = tq.Hamiltonian([(1.0, "ZI"), (-2.0, "IX")])
    assert len(hamiltonian) == 2
    assert hamiltonian.n_qubits == 2
    assert hamiltonian.coeffs == [1.0, -2.0]
    assert hamiltonian.paulis == ["ZI", "IX"]


def test_hamiltonian_rejects_an_empty_term_list():
    with pytest.raises(ValueError):
        tq.Hamiltonian([])


def test_hamiltonian_rejects_words_of_different_width():
    with pytest.raises(ValueError):
        tq.Hamiltonian([(1.0, "Z"), (1.0, "ZI")])


def test_hamiltonian_rejects_a_word_that_is_not_pauli():
    with pytest.raises(ValueError):
        tq.Hamiltonian([(1.0, "ZQ")])


def test_hamiltonian_rejects_a_mismatched_register_width():
    with pytest.raises(ValueError):
        tq.Hamiltonian([(1.0, "ZI")], n_qubits=3)


def test_hamiltonian_rejects_a_malformed_term():
    with pytest.raises(ValueError):
        tq.Hamiltonian([(1.0,)])


def test_energy_of_the_ground_state_of_z_on_zero():
    energy = tq.energy(tq.zeros_state(1), tq.Hamiltonian([(1.0, "Z")]))
    tp.testing.assert_close(energy, tp.tensor([1.0]), atol=1e-6, rtol=1e-6)


def test_energy_of_z_on_one_is_minus_the_coefficient():
    energy = tq.energy(tq.basis_state(1, 1), tq.Hamiltonian([(2.5, "Z")]))
    tp.testing.assert_close(energy, tp.tensor([-2.5]), atol=1e-6, rtol=1e-6)


def test_energy_sums_the_terms():
    state = tq.zeros_state(2)
    energy = tq.energy(state, tq.Hamiltonian([(1.0, "ZI"), (2.0, "IZ"), (-0.5, "ZZ")]))
    tp.testing.assert_close(energy, tp.tensor([2.5]), atol=1e-6, rtol=1e-6)


def test_energy_matches_the_dense_matrix():
    state = tq.plus_state(2)
    hamiltonian = tq.Hamiltonian([(0.3, "XI"), (0.7, "IZ"), (-1.1, "ZZ")])
    dense = hamiltonian.register_matrix(device=state.device, c_dtype=state.dtype)
    flat = _flat(state)
    expected = (
        flat.conj() * (dense @ flat.transpose(-1, -2)).transpose(-1, -2)
    ).sum().real
    tp.testing.assert_close(
        tq.energy(state, hamiltonian).reshape(()), expected, atol=1e-6, rtol=1e-6
    )


def test_energy_of_the_dense_ground_state_matches_its_eigenvalue():
    # Reading the energy off amplitudes has to agree with the dense matrix it
    # is the expansion of, eigenvectors and all.
    hamiltonian = tq.Hamiltonian(H2)
    matrix = hamiltonian.register_matrix(c_dtype=tp.complex64)
    eigenvalues, eigenvectors = tp.linalg.eigh(matrix)
    ground = eigenvectors[:, int(eigenvalues.argmin())]
    amplitudes = tp.polar(ground.abs(), ground.angle()).unsqueeze(0)
    state = tq.unflatten(amplitudes, 2)
    tp.testing.assert_close(
        tq.energy(state, hamiltonian).reshape(()),
        eigenvalues.min(),
        atol=1e-5,
        rtol=1e-5,
    )


def test_energy_rejects_a_hamiltonian_wider_than_the_state():
    with pytest.raises(ValueError):
        tq.energy(tq.zeros_state(1), tq.Hamiltonian([(1.0, "ZI")]))


def test_pauli_string_matrix_drops_identity_factors():
    tp.testing.assert_close(
        tq.pauli_string_matrix("XZ"),
        tp.kron(tq.x_matrix(), tq.z_matrix()),
        atol=1e-6,
        rtol=0,
    )


def test_hamiltonian_matrix_is_the_weighted_sum():
    hamiltonian = tq.Hamiltonian([(2.0, "Z"), (-1.0, "X")])
    expected = 2.0 * tq.z_matrix() - tq.x_matrix()
    tp.testing.assert_close(hamiltonian.matrix(), expected, atol=1e-6, rtol=1e-6)


def test_single_term_trotter_matches_the_exact_evolution():
    state = tq.zeros_state(1)
    hamiltonian = tq.Hamiltonian([(0.8, "Z")])
    time = 0.37
    exact = tq.matrix_exp_evolve(state, hamiltonian, time)
    stepped = tq.trotter_evolve(state, hamiltonian, time)
    tp.testing.assert_close(_flat(stepped), _flat(exact), atol=1e-5, rtol=1e-5)


def test_commuting_terms_are_exact_at_any_order():
    state = tq.zeros_state(2)
    hamiltonian = tq.Hamiltonian([(0.5, "ZI"), (-0.25, "IZ")])
    time = 0.5
    exact = tq.matrix_exp_evolve(state, hamiltonian, time)
    stepped = tq.trotter_evolve(state, hamiltonian, time, order=1)
    tp.testing.assert_close(_flat(stepped), _flat(exact), atol=1e-5, rtol=1e-5)


def test_trotter_order_improves_a_non_commuting_split():
    # X on wire 0 and X on wire 1 do not commute, so splitting the product
    # costs something an order can shrink.
    state = tq.zeros_state(2)
    hamiltonian = tq.Hamiltonian([(1.0, "XI"), (1.0, "ZI")])
    time = 0.4
    exact = _flat(tq.matrix_exp_evolve(state, hamiltonian, time))
    errors = []
    for order in (1, 2, 8, 32):
        stepped = _flat(tq.trotter_evolve(state, hamiltonian, time, order=order))
        errors.append(float((stepped - exact).abs().max()))
    # The error of a product rule falls like the inverse square of the order.
    assert errors[-1] < errors[0] / 20.0
    assert errors[-1] < 0.01


def test_trotter_keeps_the_norm():
    state = tq.plus_state(3)
    hamiltonian = tq.Hamiltonian([(0.7, "XIZ"), (-0.4, "ZZX"), (0.2, "YYY")])
    evolved = tq.trotter_evolve(state, hamiltonian, 0.6, order=4)
    tp.testing.assert_close(tq.norm(evolved), tp.tensor([1.0]), atol=1e-5, rtol=1e-5)


def test_trotter_preserves_the_energy_of_a_fixed_hamiltonian():
    # An eigenvector of one term sees only that term's rotation, so the
    # energy it carries is the same before and after the step -- up to the
    # phase every Pauli rotation adds.
    state = tq.state_from_bases([5, 7], 3)  # ZXZ swaps |101> and |111>
    hamiltonian = tq.Hamiltonian([(0.6, "ZXZ"), (0.3, "III")])
    evolved = tq.trotter_evolve(state, hamiltonian, 0.25, order=8)
    before = tq.energy(state, hamiltonian)
    after = tq.energy(evolved, hamiltonian)
    tp.testing.assert_close(before, tp.tensor([0.9]), atol=1e-5, rtol=1e-5)
    tp.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)


def test_trotter_rejects_a_non_positive_order():
    with pytest.raises(ValueError):
        tq.trotter_evolve(tq.zeros_state(1), tq.Hamiltonian([(1.0, "Z")]), 0.1, order=0)


def test_trotter_carries_the_global_phase_of_an_identity_term():
    # An identity term contributes a phase and nothing else; dropping it would
    # change the overlap with any later operation.
    hamiltonian = tq.Hamiltonian([(0.5, "Z"), (0.25, "I")])
    # The Z term turns the amplitude by -t*c and the identity term adds its
    # own -t*c on top, which any later gate would feel.
    for start, phases in (
        (tq.zeros_state(1), (-0.375, 0.0)),
        (tq.basis_state(1, 1), (0.0, 0.125)),
    ):
        evolved = tq.trotter_evolve(start, hamiltonian, 0.5)
        weights = tp.tensor([1.0, 0.0]) if phases[1] == 0.0 else tp.tensor([0.0, 1.0])
        expected = tp.polar(weights, tp.tensor(list(phases))).reshape(1, 2)
        tp.testing.assert_close(
            _flat(evolved), expected.to(start.dtype), atol=1e-5, rtol=1e-5
        )


def test_matrix_exp_evolve_reproduces_the_dense_product():
    state = tq.zeros_state(1)
    hamiltonian = tq.Hamiltonian([(0.9, "Y")])
    time = 0.6
    exact = tq.matrix_exp_evolve(state, hamiltonian, time)
    dense = tp.linalg.matrix_exp(-1j * time * hamiltonian.matrix().to(state.dtype))
    expected = (dense @ _flat(state).transpose(-1, -2)).transpose(-1, -2)
    tp.testing.assert_close(_flat(exact), expected, atol=1e-5, rtol=1e-5)


def test_hamiltonian_repr_lists_the_terms():
    text = repr(tq.Hamiltonian([(1.0, "ZI")]))
    assert "ZI" in text
    assert "n_qubits=2" in text
