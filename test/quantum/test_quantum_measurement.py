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


# ---------------------------------------------------------------------------
# Reading expectation values back out of a histogram
# ---------------------------------------------------------------------------


def _exact_counts(state, word, shots=1.0):
    """A histogram carrying the exact distribution of the measurement basis."""
    rotated = tq.apply_measurement_basis(state, word)
    wires = [w for w, letter in enumerate(word) if letter != "I"]
    if not wires:
        return tq.probs(rotated) * shots
    return tq.marginal_probs(rotated, wires) * shots


def test_expectation_from_counts_reads_back_the_exact_value():
    state = tq.apply_cnot(tq.apply_h(tq.zeros_state(2), 0), 0, 1)
    state = tq.apply_ry(state, 1, 0.7)
    for word in ("ZZ", "XI", "IX", "IZ", "XY", "II"):
        wires = [w for w, letter in enumerate(word) if letter != "I"]
        counts = _exact_counts(state, word)
        tp.testing.assert_close(
            tq.expectation_from_counts(counts, word, wires or None),
            tq.expectation(state, word),
            atol=1e-6,
            rtol=1e-6,
        )


def test_a_word_of_one_sign_reads_the_population_odds():
    # Outcomes 00, 01 and 11: the first wire never came up, the second did half
    # the time.
    counts = tp.tensor([[3.0, 1.0, 0.0, 0.0]])
    tp.testing.assert_close(
        tq.expectation_from_counts(counts, "ZI"), tp.tensor([1.0]), atol=1e-6, rtol=1e-6
    )
    tp.testing.assert_close(
        tq.expectation_from_counts(counts, "IZ"), tp.tensor([0.5]), atol=1e-6, rtol=1e-6
    )


def test_a_word_reading_two_wires_counts_them_jointly():
    # Outcomes 00 and 11 only: each wire is up half the time but never alone, so
    # the pair reads 1 where the single-wire readings multiply to a quarter.
    counts = tp.tensor([[3.0, 0.0, 0.0, 1.0]])
    tp.testing.assert_close(
        tq.expectation_from_counts(counts, "ZI"), tp.tensor([0.5]), atol=1e-6, rtol=1e-6
    )
    tp.testing.assert_close(
        tq.expectation_from_counts(counts, "IZ"), tp.tensor([0.5]), atol=1e-6, rtol=1e-6
    )
    tp.testing.assert_close(
        tq.expectation_from_counts(counts, "ZZ"), tp.tensor([1.0]), atol=1e-6, rtol=1e-6
    )
    tp.testing.assert_close(
        tq.expectation_from_counts(counts, "XI"), tp.tensor([0.5]), atol=1e-6, rtol=1e-6
    )
    tp.testing.assert_close(
        tq.expectation_from_counts(counts, "IX"), tp.tensor([0.5]), atol=1e-6, rtol=1e-6
    )
    tp.testing.assert_close(
        tq.expectation_from_counts(counts, "XX"), tp.tensor([1.0]), atol=1e-6, rtol=1e-6
    )


def test_expectation_from_counts_reads_the_whole_register_by_default():
    counts = tp.tensor([[2.0, 0.0, 0.0, 0.0]])
    tp.testing.assert_close(
        tq.expectation_from_counts(counts, "ZI"), tp.tensor([1.0]), atol=1e-6, rtol=1e-6
    )


def test_expectation_from_counts_needs_one_column_per_wire():
    counts = tp.tensor([[1.0, 1.0, 1.0, 1.0]])
    with pytest.raises(ValueError):
        tq.expectation_from_counts(counts, "ZI", [0])


def test_words_that_fit_together_come_out_in_one_group():
    # Every word here is a sub-word of X on the first wire crossed with X on the
    # second, so one rotation reads all of them.
    words = ["XI", "IX", "XX", "II"]
    groups = tq.commuting_groups(words)
    assert len(groups) == 1
    assert sorted(groups[0]) == sorted(words)


def test_words_that_disagree_on_a_wire_need_their_own_setting():
    # X and Z on the same wire cannot come off one histogram.
    groups = tq.commuting_groups(["XI", "ZI"])
    assert len(groups) == 2


def test_grouping_is_deterministic():
    words = ["ZI", "XI", "ZZ", "IX", "IZ"]
    assert tq.commuting_groups(words) == tq.commuting_groups(words)


def test_every_member_of_a_group_is_a_sub_word_of_its_product():
    from tensorplay.quantum._measurement import group_product

    words = ["XI", "IX", "ZI", "ZZ", "IZ", "ZI", "IX"]
    for group in tq.commuting_groups(words):
        product = group_product(group)
        for member in group:
            for letter, product_letter in zip(member, product):
                assert letter == "I" or letter == product_letter


def test_group_product_of_one_word_is_that_word():
    from tensorplay.quantum._measurement import group_product

    assert group_product(["XYZ"]) == "XYZ"
    assert group_product(["XI"]) == "XI"


def test_joint_estimation_lands_on_the_exact_values():
    state = _bell()
    state = tq.apply_ry(state, 0, 0.6)
    words = ["ZZ", "XI", "IX", "IZ", "ZI"]
    estimated = tq.joint_expectation(state, words, shots=100000)
    assert estimated.shape == (1, len(words))
    for column, word in enumerate(words):
        tp.testing.assert_close(
            estimated[0, column].reshape(()),
            tq.expectation(state, word).reshape(()),
            atol=0.05,
            rtol=0.05,
        )


def test_joint_estimation_of_a_word_reads_it_back():
    state = tq.apply_ry(tq.zeros_state(2), 1, 0.8)
    estimated = tq.joint_expectation(state, ["IZ"], shots=20000)
    tp.testing.assert_close(
        estimated[0, 0].reshape(()),
        tq.expectation(state, "IZ").reshape(()),
        atol=0.05,
        rtol=0.05,
    )


def test_joint_estimation_over_a_batch():
    states = tq.apply_cnot(tq.apply_h(tq.zeros_state(2, batch=3), 0), 0, 1)
    estimated = tq.joint_expectation(states, ["ZZ", "XX"], shots=4096)
    assert estimated.shape == (3, 2)
    tp.testing.assert_close(estimated[:, 0], tp.ones(3), atol=0.1, rtol=0)
    tp.testing.assert_close(estimated[:, 1], tp.ones(3), atol=0.1, rtol=0)


def test_joint_estimation_needs_a_word():
    with pytest.raises(ValueError):
        tq.joint_expectation(_bell(), [], shots=16)


def test_joint_estimation_rejects_a_word_that_does_not_fit():
    with pytest.raises(ValueError):
        tq.joint_expectation(tq.zeros_state(1), ["XX"], shots=16)
