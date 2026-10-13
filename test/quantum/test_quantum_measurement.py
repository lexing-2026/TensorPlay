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


# ---------------------------------------------------------------------------
# Readout error
# ---------------------------------------------------------------------------


def test_a_clean_readout_changes_nothing():
    clean = tq.readout_error_matrix(0.0)
    tp.testing.assert_close(clean, tp.eye(2), atol=1e-6, rtol=1e-6)
    state = _bell()
    tp.testing.assert_close(
        tq.apply_readout_error(tq.probs(state), clean),
        tq.probs(state),
        atol=1e-6,
        rtol=1e-6,
    )


def test_the_readout_matrix_separates_the_two_directions():
    matrix = tq.readout_error_matrix(0.1, 0.3)
    expected = tp.tensor([[0.7, 0.3], [0.1, 0.9]])
    tp.testing.assert_close(matrix, expected, atol=1e-6, rtol=1e-6)


def test_one_readout_rate_covers_both_directions():
    matrix = tq.readout_error_matrix(0.2)
    tp.testing.assert_close(matrix, tp.tensor([[0.8, 0.2], [0.2, 0.8]]), atol=1e-6, rtol=1e-6)


def test_a_readout_rate_outside_the_unit_interval_is_rejected():
    for bad in (-0.1, 1.4):
        with pytest.raises(ValueError):
            tq.readout_error_matrix(bad)
        with pytest.raises(ValueError):
            tq.readout_error_matrix(0.1, bad)


def test_a_prepared_one_is_misread_at_the_rate_it_was_given():
    matrix = tq.readout_error_matrix(0.25)
    prepared_one = tp.tensor([[0.0, 1.0]])
    reported = tq.apply_readout_error(prepared_one, matrix)
    tp.testing.assert_close(reported, tp.tensor([[0.25, 0.75]]), atol=1e-6, rtol=1e-6)


def test_a_prepared_zero_is_misread_at_its_own_rate():
    matrix = tq.readout_error_matrix(0.25, 0.5)
    prepared_zero = tp.tensor([[1.0, 0.0]])
    reported = tq.apply_readout_error(prepared_zero, matrix)
    tp.testing.assert_close(reported, tp.tensor([[0.5, 0.5]]), atol=1e-6, rtol=1e-6)


def test_the_two_wires_of_a_register_are_read_independently():
    # Wire 0 set out of two: each wire turns over with chance p, so the four
    # outcomes take the four products.  |10> is index 2, qubit 0 leading.
    p = 0.1
    matrix = tq.readout_error_matrix(p)
    state = tq.basis_state(2, 2)
    reported = tq.apply_readout_error(tq.probs(state), matrix)
    expected = tp.tensor([[
        p * (1 - p),    # |00>: wire 0 turned over, wire 1 did not
        p * p,          # |01>: both turned over
        (1 - p) * (1 - p),  # |10>: neither turned over
        (1 - p) * p,    # |11>: wire 1 turned over
    ]])
    tp.testing.assert_close(reported, expected, atol=1e-6, rtol=1e-6)


def test_readout_error_keeps_the_weight():
    state = tq.plus_state(3)
    reported = tq.apply_readout_error(tq.probs(state), tq.readout_error_matrix(0.3))
    tp.testing.assert_close(reported.sum(), tp.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_readout_error_keeps_the_shot_count():
    counts = tq.measure(_bell(), 512)
    reported = tq.apply_readout_error(counts, tq.readout_error_matrix(0.2))
    tp.testing.assert_close(reported.sum(), tp.tensor(512.0), atol=1e-4, rtol=1e-6)


def test_a_matrix_can_be_given_for_each_wire():
    per_wire = tp.stack([
        tq.readout_error_matrix(0.0),
        tq.readout_error_matrix(1.0),   # this one is always wrong
        tq.readout_error_matrix(0.0),
    ])
    reported = tq.apply_readout_error(tq.probs(tq.zeros_state(3)), per_wire)
    assert abs(float(reported.reshape(-1)[2]) - 1.0) < 1e-6


def test_the_wires_keep_the_order_the_matrix_list_is_given_in():
    per_wire = tp.stack([tq.readout_error_matrix(1.0), tq.readout_error_matrix(0.0)])
    reported = tq.apply_readout_error(tq.probs(tq.zeros_state(2)), per_wire)
    # The leading wire is the one that turns over, so |00> reads as |10>.
    assert abs(float(reported.reshape(-1)[2]) - 1.0) < 1e-6


def test_readout_error_scales_the_shots_the_way_the_exact_map_does():
    state = tq.basis_state(1, 2)
    matrix = tq.readout_error_matrix(0.15)
    exact = tq.apply_readout_error(tq.probs(state), matrix)
    shots = tq.measure(state, 20000, readout=matrix)
    tp.testing.assert_close(shots / shots.sum(), exact, atol=0.02, rtol=0)


def test_a_measured_value_shrinks_by_the_readout_fidelity():
    # A value that sits on the axis being read survives a readout with
    # probability 1 - p on each side, so it comes back scaled by 1 - 2p.
    rotated = tq.apply_measurement_basis(tq.plus_state(1), "X")
    for p in (0.0, 0.2, 0.35):
        counts = tq.measure(rotated, 20000, readout=tq.readout_error_matrix(p))
        estimate = tq.expectation_from_counts(counts, "X")
        tp.testing.assert_close(
            estimate, tp.tensor([1.0 - 2 * p]), atol=0.05, rtol=0
        )


def test_a_clean_readout_leaves_the_shots_where_they_were():
    state = tq.plus_state(2)
    counts = tq.measure(state, 4000, readout=tq.readout_error_matrix(0.0))
    tp.testing.assert_close(
        counts, 4000 * tq.probs(state), atol=250.0, rtol=0
    )


def test_the_readout_only_covers_the_wires_that_were_measured():
    state = tq.apply_x(tq.zeros_state(3), 2)
    # A matrix per wire for a two-wire readout is a shape error; a single one
    # covers both.
    with pytest.raises(ValueError):
        tq.measure(state, 16, wires=[0, 1], readout=tp.stack([
            tq.readout_error_matrix(0.1),
            tq.readout_error_matrix(0.1),
            tq.readout_error_matrix(0.1),
        ]))
    counts = tq.measure(state, 4096, wires=[0, 1], readout=tq.readout_error_matrix(0.05))
    assert counts.shape == (1, 4)
    tp.testing.assert_close(counts.sum(), tp.tensor(4096.0), atol=1e-4, rtol=1e-6)


def test_a_readout_matrix_of_the_wrong_shape_is_rejected():
    state = tq.zeros_state(2)
    with pytest.raises(ValueError):
        tq.measure(state, 16, readout=tp.rand(2, 2, 2)[:, 0])
    with pytest.raises(ValueError):
        tq.apply_readout_error(tq.probs(state), tp.rand(3, 2, 2))


def test_a_readout_row_that_does_not_sum_to_one_is_rejected():
    with pytest.raises(ValueError):
        tq.apply_readout_error(tq.probs(tq.zeros_state(1)), tp.tensor([[0.4, 0.4], [0.5, 0.5]]))


def test_a_readout_entry_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        tq.apply_readout_error(tq.probs(tq.zeros_state(1)), tp.tensor([[1.4, -0.4], [0.1, 0.9]]))


def test_a_column_count_that_is_not_a_whole_register_is_rejected():
    with pytest.raises(ValueError):
        tq.apply_readout_error(tp.zeros(1, 3), tq.readout_error_matrix(0.1))


# ---------------------------------------------------------------------------
# Column order
# ---------------------------------------------------------------------------


def test_the_marginal_columns_follow_the_wires_as_listed():
    # |01> has wire 1 set, so it lands in the column that names wire 1 first.
    state = tq.basis_state(1, 2)
    leading_one = tq.marginal_probs(state, [1, 0])
    trailing_one = tq.marginal_probs(state, [0, 1])
    assert abs(float(leading_one.reshape(-1)[2]) - 1.0) < 1e-6
    assert abs(float(trailing_one.reshape(-1)[1]) - 1.0) < 1e-6


def test_a_marginal_carries_the_same_weights_in_any_order():
    state = tq.apply_h(tq.zeros_state(3), 0)
    ascending = tp.sort(tq.marginal_probs(state, [0, 2]).reshape(-1)).values
    descending = tp.sort(tq.marginal_probs(state, [2, 0]).reshape(-1)).values
    tp.testing.assert_close(ascending, descending, atol=1e-6, rtol=1e-6)


def test_the_sampled_columns_follow_the_wires_as_listed():
    state = tq.basis_state(1, 2)
    counts = tq.measure(state, 256, wires=[1, 0])
    assert abs(float(counts[0, 2]) - 256.0) < 1e-4


def test_a_word_read_in_an_unusual_order_still_reads_its_own_wires():
    # Wire 1 is measured first here, so the histogram column bit that names the
    # word's factor is the leading one.  Reading the value back has to agree
    # with the exact expectation either way round.
    state = tq.apply_h(tq.zeros_state(2), 0)
    rotated = tq.apply_measurement_basis(state, "ZI")
    exact = tq.expectation(state, "IZ")
    for wires in ([0, 1], [1, 0]):
        counts = tq.measure(rotated, 8192, wires=wires)
        estimate = tq.expectation_from_counts(counts, "IZ", wires=wires)
        tp.testing.assert_close(estimate, exact, atol=0.05, rtol=0)


def test_a_sub_word_of_the_wires_read_stays_correct():
    # The word names a wire that is not measured at all, and one that is.
    state = tq.apply_h(tq.zeros_state(2), 0)
    counts = tq.measure(state, 8192, wires=[1])
    estimate = tq.expectation_from_counts(counts, "IZ", wires=[1])
    tp.testing.assert_close(estimate, tp.tensor([1.0]), atol=0.05, rtol=0)


def test_a_marginal_may_span_more_wires_than_a_gate():
    # A gate's matrix is 2**k by 2**k, so it stops well short of a wide
    # register; a histogram carries no matrix and reads all of it at once.
    n_qubits = 10
    state = tq.basis_state(1, n_qubits)
    counts = tq.measure(state, 128)
    assert counts.shape == (1, 1 << n_qubits)
    tp.testing.assert_close(counts.sum(), tp.tensor(128.0), atol=1e-4, rtol=1e-6)
    tp.testing.assert_close(counts[0, 1], tp.tensor(128.0), atol=1e-4, rtol=1e-6)


# ---------------------------------------------------------------------------
# Observables of a mixed state
# ---------------------------------------------------------------------------


def test_a_mixed_state_answers_the_same_way_as_the_mixture_it_is():
    # An open-system run hands back a density matrix, and the observable is a
    # trace there rather than an overlap.  The two routes have to agree entry by
    # entry: the weights of a mixture add, so the expectation of the mixture is
    # the weighted sum of the parts.
    rho = tq.to_density(tq.basis_state(0, 1)) * 0.5 + tq.to_density(tq.basis_state(1, 1)) * 0.5
    for word in ("I", "X", "Y", "Z"):
        mixture = tq.expectation(tq.basis_state(0, 1), word) * 0.5 + tq.expectation(
            tq.basis_state(1, 1), word
        ) * 0.5
        tp.testing.assert_close(
            tq.expectation_from_density(rho, word), mixture, atol=1e-6, rtol=1e-6
        )
    tp.testing.assert_close(
        tq.expectation_from_density(rho, "Z"), tp.tensor([0.0]), atol=1e-6, rtol=0
    )


def test_a_pure_state_read_as_a_density_matches_the_state_route():
    state = tq.apply_cnot(tq.apply_h(tq.zeros_state(2), 0), 0, 1)
    rho = tq.to_density(state)
    for word in ("ZZ", "XX", "ZI", "IZ"):
        tp.testing.assert_close(
            tq.expectation_from_density(rho, word),
            tq.expectation(state, word),
            atol=1e-6,
            rtol=1e-6,
        )
    # And the whole Hamiltonian, summed over its terms, lands where the state's
    # own energy does.
    h2 = tq.Hamiltonian(
        [(-1.0523732, "II"), (0.3979374, "IZ"), (-0.3979374, "ZI"), (-0.1809312, "XX")]
    )
    from_density = sum(
        coeff * tp.reshape(tq.expectation_from_density(rho, word), [-1])[0]
        for coeff, word in h2.terms
    )
    tp.testing.assert_close(
        tp.tensor(from_density).reshape(()),
        tp.reshape(tq.energy(state, h2), [-1])[0].reshape(()),
        atol=1e-6,
        rtol=1e-6,
    )


def test_a_density_observable_is_real_and_wide():
    # The trace of a Hermitian word against a density is real, and it is as wide
    # as the density was -- a single-precision register answers in single
    # precision rather than being quietly widened.
    wide = tq.to_density(tq.plus_state(2, dtype=tp.complex128))
    assert tq.expectation_from_density(wide, "II").dtype == tp.float64
    narrow = tq.to_density(tq.plus_state(2))
    assert tq.expectation_from_density(narrow, "II").dtype == tp.float32


def test_a_density_observable_wants_a_density_and_a_pauli_word():
    state = tq.plus_state(3)
    with pytest.raises(ValueError):
        tq.expectation_from_density(state, "ZZZ")
    with pytest.raises(ValueError):
        tq.expectation_from_density(tq.to_density(tq.plus_state(1)), "XX")
    with pytest.raises(ValueError):
        tq.expectation_from_density(tq.to_density(tq.plus_state(1)), "Q")
    with pytest.raises(ValueError):
        tq.expectation_from_density(tp.ones(1, 3, 3, dtype=tp.complex128), "Z")
