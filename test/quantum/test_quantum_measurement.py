"""Probabilities, expectation values, and shot-based sampling."""

import pytest

import tensorplay as tp
from tensorplay import quantum as tq


def _bell():
    return tq.apply_cnot(tq.apply_h(tq.zeros_state(2), 0), 0, 1)


def test_probs_are_the_squared_amplitudes():
    state = tq.apply_h(tq.zeros_state(1), 0)
    tp.testing.assert_close(tq.probs(state), tp.tensor([[0.5, 0.5]]), atol=1e-6, rtol=1e-6)


def test_probs_sum_to_one():
    state = tq.plus_state(3)
    tp.testing.assert_close(tq.probs(state).sum(), tp.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_probs_of_a_bell_pair():
    expected = tp.tensor([[0.5, 0.0, 0.0, 0.5]])
    tp.testing.assert_close(tq.probs(_bell()), expected, atol=1e-6, rtol=1e-6)


def test_density_probs_read_the_diagonal():
    density = tq.to_density(_bell())
    expected = tp.tensor([[0.5, 0.0, 0.0, 0.5]])
    tp.testing.assert_close(tq.density_probs(density), expected, atol=1e-6, rtol=1e-6)


def test_marginal_probs_drops_the_other_qubit():
    state = tq.basis_state(4, 3)  # |100>: wire 0 is 1, wire 2 is 0
    tp.testing.assert_close(tq.marginal_probs(state, [0]), tp.tensor([[0.0, 1.0]]))
    tp.testing.assert_close(tq.marginal_probs(state, [2]), tp.tensor([[1.0, 0.0]]))


def test_marginal_probs_follow_the_given_order():
    # Two qubits still carry four outcomes; the order decides which is which.
    state = tq.basis_state(4, 3)  # |100>: wire 0 = 1, wires 1 and 2 = 0
    tp.testing.assert_close(
        tq.marginal_probs(state, [0, 2]), tp.tensor([[0.0, 0.0, 1.0, 0.0]])
    )
    tp.testing.assert_close(
        tq.marginal_probs(state, [2, 0]), tp.tensor([[0.0, 1.0, 0.0, 0.0]])
    )


def test_marginal_probs_of_a_bell_pair_is_uniform():
    marginal = tq.marginal_probs(_bell(), [0])
    tp.testing.assert_close(marginal, tp.tensor([[0.5, 0.5]]), atol=1e-6, rtol=1e-6)


def test_expectation_of_a_pauli_on_zero_state():
    tp.testing.assert_close(
        tq.expectation(tq.zeros_state(2), "ZI"), tp.tensor([1.0]), atol=1e-6, rtol=1e-6
    )
    tp.testing.assert_close(
        tq.expectation(tq.zeros_state(2), "IZ"), tp.tensor([1.0]), atol=1e-6, rtol=1e-6
    )
    tp.testing.assert_close(
        tq.expectation(tq.zeros_state(2), "XI"), tp.tensor([0.0]), atol=1e-6, rtol=0
    )


def test_expectation_of_z_on_one():
    tp.testing.assert_close(
        tq.expectation(tq.basis_state(1, 1), "Z"), tp.tensor([-1.0]), atol=1e-6, rtol=1e-6
    )


def test_expectation_of_y_on_the_plus_y_state():
    # |+y> = (|0> + i|1>)/sqrt(2): Y maps it to itself.
    state = tp.tensor([[2.0**-0.5, 1.0j * 2.0**-0.5]], dtype=tp.complex64)
    tp.testing.assert_close(
        tq.expectation(state, "Y"), tp.tensor([1.0]), atol=1e-6, rtol=1e-6
    )


def test_expectation_of_the_bell_pair_words():
    bell = _bell()
    tp.testing.assert_close(tq.expectation(bell, "ZZ"), tp.tensor([1.0]), atol=1e-6, rtol=1e-6)
    tp.testing.assert_close(tq.expectation(bell, "XX"), tp.tensor([1.0]), atol=1e-6, rtol=1e-6)
    tp.testing.assert_close(tq.expectation(bell, "YY"), tp.tensor([-1.0]), atol=1e-6, rtol=1e-6)
    tp.testing.assert_close(tq.expectation(bell, "ZI"), tp.tensor([0.0]), atol=1e-6, rtol=0)


def test_apply_pauli_word_matches_the_dense_matrix():
    state = tq.plus_state(2)
    word = "XZ"
    by_word = tq.apply_pauli_word(state, word)
    dense = tq.pauli_matrix(word, c_dtype=state.dtype)
    by_matrix = (dense @ state.reshape(1, -1).transpose(-1, -2)).transpose(-1, -2)
    tp.testing.assert_close(by_word.reshape(1, -1), by_matrix, atol=1e-6, rtol=1e-6)


def test_sample_indices_stays_inside_the_distribution():
    distribution = tp.tensor([[0.0, 0.0, 1.0, 0.0]])
    drawn = tq.sample_indices(distribution, 64)
    assert drawn.shape == (1, 64)
    assert int(drawn.min()) == 2
    assert int(drawn.max()) == 2


def test_sample_indices_rejects_a_non_positive_shot_count():
    with pytest.raises(ValueError):
        tq.sample_indices(tp.tensor([[1.0]]), 0)


def test_counts_tally_the_draws():
    distribution = tp.tensor([[0.5, 0.5]])
    drawn = tp.tensor([[0, 0, 1, 1]])
    counts = tq.counts_from_indices(drawn, 1)
    tp.testing.assert_close(counts, tp.tensor([[2.0, 2.0]]))


def test_measure_on_a_deterministic_state():
    counts = tq.measure(tq.zeros_state(2), shots=128)
    tp.testing.assert_close(counts, tp.tensor([[128.0, 0.0, 0.0, 0.0]]))


def test_measure_on_a_bell_pair_splits_evenly():
    counts = tq.measure(_bell(), shots=4096)
    tp.testing.assert_close(counts.sum(), tp.tensor(4096.0))
    # |00> and |11> each take about half the shots.
    tp.testing.assert_close(counts[0, 0], tp.tensor(2048.0), atol=180.0, rtol=0)
    tp.testing.assert_close(counts[0, 3], tp.tensor(2048.0), atol=180.0, rtol=0)
    tp.testing.assert_close(counts[0, 1], tp.tensor(0.0), atol=2.0, rtol=0)
    tp.testing.assert_close(counts[0, 2], tp.tensor(0.0), atol=2.0, rtol=0)


def test_measure_can_read_a_subset_of_wires():
    counts = tq.measure(_bell(), shots=512, wires=[1])
    tp.testing.assert_close(counts.sum(), tp.tensor(512.0))
    tp.testing.assert_close(counts[0, 0], tp.tensor(256.0), atol=80.0, rtol=0)
    tp.testing.assert_close(counts[0, 1], tp.tensor(256.0), atol=80.0, rtol=0)


def test_measure_handles_a_batch():
    states = tq.zeros_state(2, batch=4)
    counts = tq.measure(states, shots=32)
    assert counts.shape == (4, 4)
    tp.testing.assert_close(counts[:, 0], tp.full((4,), 32.0))
