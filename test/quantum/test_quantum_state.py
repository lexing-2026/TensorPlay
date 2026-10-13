"""State layout, normalization, density matrices, and partial trace."""

import math

import pytest

import tensorplay as tp
from tensorplay import quantum as tq


def test_zeros_state_is_the_first_basis_entry():
    state = tq.zeros_state(3)
    assert state.shape == (1, 2, 2, 2)
    flat = state.reshape(1, -1)
    tp.testing.assert_close(flat[0, 0], tp.tensor(1.0 + 0.0j))


def test_plus_state_is_uniform():
    state = tq.plus_state(2)
    tp.testing.assert_close(tq.probs(state), tp.tensor([[0.25, 0.25, 0.25, 0.25]]))


def test_plus_state_sums_to_one():
    tp.testing.assert_close(tq.probs(tq.plus_state(4)).sum(), tp.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_basis_state_selects_the_right_index():
    state = tq.basis_state(5, 3)
    flat = state.reshape(1, -1)
    expected = tp.zeros(8, dtype=tp.complex64)
    expected[5] = 1.0
    tp.testing.assert_close(flat[0], expected)


def test_basis_state_rejects_an_index_past_the_register():
    with pytest.raises(IndexError):
        tq.basis_state(8, 3)


def test_state_from_bases_sums_the_listed_entries():
    state = tq.state_from_bases([0, 3], 2)
    flat = state.reshape(1, -1)
    tp.testing.assert_close(flat[0, 0], tp.tensor(2.0**-0.5, dtype=tp.complex64))
    tp.testing.assert_close(flat[0, 1], tp.tensor(0.0 + 0.0j))
    tp.testing.assert_close(flat[0, 2], tp.tensor(0.0 + 0.0j))
    tp.testing.assert_close(flat[0, 3], tp.tensor(2.0**-0.5, dtype=tp.complex64))


def test_product_state_is_the_tensor_product():
    plus = tp.tensor([2.0**-0.5, 2.0**-0.5], dtype=tp.complex64)
    one = tp.tensor([0.0, 1.0], dtype=tp.complex64)
    product = tq.product_state([plus, one]).reshape(1, -1)
    expected = tp.tensor([0.0, 2.0**-0.5, 0.0, 2.0**-0.5], dtype=tp.complex64)
    tp.testing.assert_close(product[0], expected)


def test_flatten_and_unflatten_round_trip():
    state = tq.plus_state(3)
    flat = tq.flatten(state)
    assert flat.shape == (1, 8)
    tp.testing.assert_close(tq.unflatten(flat, 3), state)


def test_unflatten_reads_the_qubit_count_when_unset():
    flat = tp.ones((1, 4), dtype=tp.complex64)
    assert tq.unflatten(flat).shape == (1, 2, 2)


def test_unflatten_rejects_a_non_power_of_two():
    with pytest.raises(ValueError):
        tq.unflatten(tp.ones((1, 3), dtype=tp.complex64))


def test_norm_and_normalize():
    state = tq.flatten(tq.zeros_state(2)) * 3.0
    tp.testing.assert_close(tq.norm(state), tp.tensor([3.0]), atol=1e-6, rtol=1e-6)
    tp.testing.assert_close(tq.norm(tq.normalize(state)), tp.tensor([1.0]), atol=1e-6, rtol=1e-6)


def test_inner_is_the_conjugate_overlap():
    bell = tq.apply_cnot(tq.apply_h(tq.zeros_state(2), 0), 0, 1)
    same = tq.inner(bell, bell)
    other = tq.inner(bell, tq.zeros_state(2))
    tp.testing.assert_close(same.real, tp.tensor([1.0]), atol=1e-6, rtol=1e-6)
    tp.testing.assert_close(same.imag, tp.tensor([0.0]), atol=1e-6, rtol=0)
    tp.testing.assert_close(other.abs(), tp.tensor([0.7071068]), atol=1e-6, rtol=1e-6)


def test_fidelity_of_a_state_with_itself_is_one():
    state = tq.plus_state(2)
    tp.testing.assert_close(tq.fidelity(state, state), tp.tensor([1.0]), atol=1e-6, rtol=1e-6)


def test_to_density_is_traceless_of_purity():
    state = tq.basis_state(1, 1)
    density = tq.to_density(state)
    assert density.shape == (1, 2, 2)
    tp.testing.assert_close(density[0, 0, 0], tp.tensor(0.0 + 0.0j))
    tp.testing.assert_close(density[0, 1, 1], tp.tensor(1.0 + 0.0j))


def test_density_round_trips_back_to_the_pure_state():
    state = tq.apply_h(tq.zeros_state(1), 0)
    recovered = tq.density_to_state(tq.to_density(state), 1)
    tp.testing.assert_close(recovered.reshape(1, -1), state.reshape(1, -1), atol=1e-6, rtol=1e-6)


def test_partial_trace_of_a_bell_pair_is_the_mixed_single_qubit_state():
    bell = tq.apply_cnot(tq.apply_h(tq.zeros_state(2), 0), 0, 1)
    reduced = tq.partial_trace(tq.to_density(bell), [0], 2)
    assert reduced.shape == (1, 2, 2)
    tp.testing.assert_close(reduced[0, 0, 0], tp.tensor(0.5 + 0.0j), atol=1e-6, rtol=0)
    tp.testing.assert_close(reduced[0, 1, 1], tp.tensor(0.5 + 0.0j), atol=1e-6, rtol=0)
    tp.testing.assert_close(reduced[0, 0, 1], tp.tensor(0.0 + 0.0j), atol=1e-6, rtol=0)


def test_partial_trace_of_the_whole_register_is_the_density_matrix():
    density = tq.to_density(tq.plus_state(2))
    # Keeping every qubit is the identity on density matrices.
    tp.testing.assert_close(
        tq.partial_trace(density, [0, 1], 2),
        density,
        atol=1e-6,
        rtol=0,
    )


def test_partial_trace_of_the_whole_register_keeps_a_unit_pure_state():
    density = tq.to_density(tq.basis_state(2, 2))  # |10>
    tp.testing.assert_close(
        tq.partial_trace(density, [0, 1], 2), density, atol=1e-6, rtol=0
    )


def test_partial_trace_rejects_a_repeated_wire():
    density = tq.to_density(tq.zeros_state(2))
    with pytest.raises(ValueError):
        tq.partial_trace(density, [0, 0], 2)


def test_product_of_plus_and_zero_is_the_superposition_of_one_qubit():
    plus = tp.tensor([2.0**-0.5, 2.0**-0.5], dtype=tp.complex64)
    zero = tp.tensor([1.0, 0.0], dtype=tp.complex64)
    product = tq.product_state([plus, zero]).reshape(1, -1)
    # The first factor owns wire 0, the leading bit of the amplitude index.
    expected = tp.tensor([2.0**-0.5, 0.0, 2.0**-0.5, 0.0], dtype=tp.complex64)
    tp.testing.assert_close(product[0], expected, atol=1e-6, rtol=0)


def test_state_keeps_its_norm_under_a_batch():
    states = tq.plus_state(2, batch=3)
    assert states.shape == (3, 2, 2)
    tp.testing.assert_close(tq.norm(states), tp.ones(3), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Global phase
# ---------------------------------------------------------------------------


def test_the_phase_between_two_equal_operators_is_one():
    state = tq.plus_state(3)
    assert abs(tq.global_phase(state, state) - 1) < 1e-9
    matrix = tq.pauli_string_matrix("XYZ")
    assert abs(tq.global_phase(matrix, matrix) - 1) < 1e-9


def test_the_phase_reads_the_factor_that_was_applied():
    for angle in (0.3, math.pi / 2, -2.1, math.pi):
        state = tq.apply_h(tq.zeros_state(3), 0)
        turned = tq.apply_global_phase(state, angle)
        tp.testing.assert_close(
            tp.tensor(tq.global_phase(state, turned)),
            tp.tensor(complex(math.cos(angle), math.sin(angle))),
            atol=1e-6,
            rtol=1e-6,
        )


def test_the_phase_sees_a_half_turn_as_minus_one():
    gate = tq.h_matrix()
    assert abs(tq.global_phase(gate, -gate) + 1) < 1e-9


def test_the_phase_is_read_off_the_whole_operator():
    # One tiny entry cannot decide it: the reference answer here is four, and
    # the first entry big enough to divide on is nowhere near the largest.
    reference = tp.tensor([[1e-6, 0.0], [0.0, 4.0]], dtype=tp.complex64)
    tp.testing.assert_close(
        tp.tensor(tq.global_phase(reference, 4 * reference)),
        tp.tensor(4.0 + 0.0j),
        atol=1e-6,
        rtol=1e-6,
    )


def test_the_phase_survives_a_real_operator():
    real = tp.tensor([[1.0, 0.0], [0.0, 2.0]])
    tp.testing.assert_close(
        tp.tensor(tq.global_phase(real, -real)), tp.tensor(-1.0 + 0.0j),
        atol=1e-6, rtol=1e-6,
    )


def test_a_state_that_differs_by_more_than_a_phase_is_rejected():
    state = tq.apply_h(tq.zeros_state(2), 0)
    with pytest.raises(ValueError):
        tq.global_phase(state, tq.zeros_state(2))
    with pytest.raises(ValueError):
        tq.global_phase(state, tq.apply_h(state, 0))


def test_the_zero_operator_carries_no_phase():
    zero = tp.zeros(2, 2, dtype=tp.complex64)
    with pytest.raises(ValueError):
        tq.global_phase(zero, zero)


def test_two_operators_of_different_shapes_are_rejected():
    state = tq.plus_state(2)
    with pytest.raises(ValueError):
        tq.global_phase(state, tq.h_matrix())


def test_a_gate_built_two_ways_agrees_up_to_a_phase():
    # The fused two-wire path and one dense product over the whole register are
    # two implementations of the same operator.  Comparing them through a phase
    # says they differ in that one factor and in nothing else, which an element
    # by element comparison would report as a list of mismatches instead.
    n_qubits = 3
    wires = [0, 2]
    matrix = tq.controlled_matrix(tq.rot_y_matrix(0.4)).to(tq.plus_state(n_qubits).dtype)
    state = tq.plus_state(n_qubits)
    fused = tq.flatten(tq.apply_gate(state, matrix, wires)).reshape(-1)
    wide = tq.expand_to_register(matrix, wires, n_qubits) @ tq.flatten(
        state
    ).reshape(-1, 1)
    tp.testing.assert_close(
        tp.tensor(tq.global_phase(fused, wide.reshape(-1))),
        tp.tensor(1.0 + 0.0j),
        atol=1e-6,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# One qubit as a point in the ball
# ---------------------------------------------------------------------------


def test_the_ball_reads_the_four_states_that_fix_the_axes():
    # |0> and |1> are the poles, |+> the equator along x, and a rotation about X
    # by a quarter turn is the equator along y -- which is the point where the
    # imaginary part of the coherence is the whole answer.
    tp.testing.assert_close(
        tq.sv2bloch(tq.zeros_state(1, dtype=tp.complex128)),
        tp.tensor([[0.0, 0.0, 1.0]], dtype=tp.float64),
        atol=1e-12,
        rtol=0,
    )
    tp.testing.assert_close(
        tq.sv2bloch(tq.basis_state(1, 1, dtype=tp.complex128)),
        tp.tensor([[0.0, 0.0, -1.0]], dtype=tp.float64),
        atol=1e-12,
        rtol=0,
    )
    tp.testing.assert_close(
        tq.sv2bloch(tq.plus_state(1, dtype=tp.complex128)),
        tp.tensor([[1.0, 0.0, 0.0]], dtype=tp.float64),
        atol=1e-12,
        rtol=0,
    )
    # A quarter turn about x is a pi/2 rotation about x, which walks |+> round to
    # -y; S is the same rotation about z by a quarter, and walks it to +y.
    quarter_x = tq.apply_rx(tq.zeros_state(1, dtype=tp.complex128), 0, math.pi / 2)
    tp.testing.assert_close(
        tq.sv2bloch(quarter_x),
        tp.tensor([[0.0, -1.0, 0.0]], dtype=tp.float64),
        atol=1e-12,
        rtol=0,
    )
    quarter_z = tq.apply_s(tq.plus_state(1, dtype=tp.complex128), 0)
    tp.testing.assert_close(
        tq.sv2bloch(quarter_z),
        tp.tensor([[0.0, 1.0, 0.0]], dtype=tp.float64),
        atol=1e-12,
        rtol=0,
    )


def test_the_ball_reads_a_state_by_what_each_pauli_operator_says_about_it():
    # The three components are the readings of the three Pauli operators: the
    # vector is the one the ball's axes are named after.  A reading that put y's
    # sign the other way would be the same for every state with real amplitudes
    # -- where y is zero either way -- and wrong for every state with a phase on
    # it, so the states below all carry one.
    paulis = {
        "X": tp.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=tp.complex128),
        "Y": tq.y_matrix(c_dtype=tp.complex128),
        "Z": tp.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=tp.complex128),
    }
    states = [
        tq.zeros_state(1, dtype=tp.complex128),
        tq.plus_state(1, dtype=tp.complex128),
        tq.apply_s(tq.plus_state(1, dtype=tp.complex128), 0),
        tq.apply_s(tq.plus_state(1, dtype=tp.complex128), 0, True),
        tq.apply_p(tq.plus_state(1, dtype=tp.complex128), 0, math.pi / 3),
        tq.apply_ry(tq.apply_h(tq.zeros_state(1, dtype=tp.complex128), 0), 0, 0.7),
        tq.apply_ry(tq.apply_s(tq.plus_state(1, dtype=tp.complex128), 0), 0, 0.7),
    ]
    for state in states:
        density = tq.to_density(state)[0]
        want = [float(tp.real(tp.trace(density @ matrix))) for matrix in paulis.values()]
        tp.testing.assert_close(
            tq.sv2bloch(state)[0],
            tp.tensor(want, dtype=tp.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        # The density matrix of a state reads the same way: the ball cannot tell
        # the two routes apart, which is the point of putting them in it.
        tp.testing.assert_close(
            tq.dens2bloch(tq.to_density(state)),
            tq.sv2bloch(state),
            atol=1e-12,
            rtol=1e-12,
        )


def test_a_mixed_state_sits_inside_the_ball_and_says_by_how_much():
    # The length of the vector is twice the larger eigenvalue less one, so the
    # mixture of three-quarters ground and a quarter excited is a quarter of the
    # way along z and reads back as exactly the mixture it was.
    density = 0.75 * tq.to_density(tq.zeros_state(1, dtype=tp.complex128)) + 0.25 * (
        tq.to_density(tq.basis_state(1, 1, dtype=tp.complex128))
    )
    vector = tq.dens2bloch(density)
    tp.testing.assert_close(
        vector, tp.tensor([[0.0, 0.0, 0.5]], dtype=tp.float64), atol=1e-12, rtol=0
    )
    assert abs(float(tp.linalg.vector_norm(vector)) - 0.5) < 1e-12
    rebuilt = tq.bloch2dens(vector)
    tp.testing.assert_close(rebuilt, density, atol=1e-12, rtol=1e-12)


def test_a_vector_round_trips_through_what_it_names():
    angles = tp.tensor([0.0, 0.3, 1.1, 2.4], dtype=tp.float64)
    states = tq.apply_ry(
        tq.apply_s(tq.zeros_state(1, batch=4, dtype=tp.complex128), 0), 0, angles
    )
    vectors = tq.sv2bloch(states)
    tp.testing.assert_close(
        tq.sv2bloch(tq.bloch2sv(vectors)), vectors, atol=1e-11, rtol=1e-11
    )
    # And a batch of vectors reads back as a batch of density matrices, which
    # agree with the density matrices of the states they came from.
    tp.testing.assert_close(
        tq.dens2bloch(tq.bloch2dens(vectors)),
        tq.sv2bloch(states),
        atol=1e-11,
        rtol=1e-11,
    )
    tp.testing.assert_close(
        tq.bloch2dens(vectors), tq.to_density(states), atol=1e-11, rtol=1e-11
    )
    for length in tp.linalg.vector_norm(vectors, dim=-1).tolist():
        assert abs(length - 1.0) < 1e-11


def test_the_south_pole_is_the_state_the_formula_cannot_reach():
    # The formula divides by the length of the vector's z component, which is
    # zero at -z.  There the state is |1>, and saying so is the whole job.
    south = tp.tensor([[0.0, 0.0, -1.0]], dtype=tp.float64)
    tp.testing.assert_close(
        tq.bloch2sv(south)[0],
        tp.tensor([0.0, 1.0 + 0j], dtype=tp.complex128),
        atol=1e-12,
        rtol=0,
    )
    tp.testing.assert_close(
        tq.sv2bloch(tq.bloch2sv(south)), south, atol=1e-12, rtol=0
    )


def test_the_ball_is_a_qubit_and_three_components():
    with pytest.raises(ValueError):
        tq.sv2bloch(tq.zeros_state(2))                    # two qubits
    with pytest.raises(ValueError):
        tq.sv2bloch(tq.to_density(tq.zeros_state(1)))      # a density, not a state
    with pytest.raises(ValueError):
        tq.dens2bloch(tq.to_density(tq.zeros_state(2)))    # two qubits
    with pytest.raises(ValueError):
        tq.dens2bloch(tq.zeros_state(1, batch=1))          # a state, not a density
    with pytest.raises(ValueError):
        tq.bloch2sv(tp.tensor([[1.0, 0.0]]))              # two components
    with pytest.raises(ValueError):
        tq.bloch2dens(tp.tensor([[1.0, 0.0, 0.0, 0.0]]))  # four


def test_reset_projects_the_wire_onto_zero():
    # (|00> + |11>)/sqrt2: either wire reset discards the |11> branch, leaving |00>
    bell = tq.apply_cnot(tq.apply_h(tq.zeros_state(2, dtype=tp.complex128), 0), 0, 1)
    ground = tq.basis_state(0, 2, dtype=tp.complex128)
    tp.testing.assert_close(tq.apply_reset(bell, 0), ground, atol=1e-12, rtol=1e-12)
    tp.testing.assert_close(tq.apply_reset(bell, 1), ground, atol=1e-12, rtol=1e-12)


def test_reset_keeps_the_state_normalized_and_is_idempotent():
    state = tq.random_state(2, dtype=tp.complex128)
    reset = tq.apply_reset(state, 1)
    tp.testing.assert_close(tq.norm(reset), tp.tensor([1.0], dtype=tp.float64), atol=1e-12, rtol=1e-12)
    tp.testing.assert_close(tq.apply_reset(reset, 1), reset, atol=1e-12, rtol=1e-12)


def test_reset_rejects_an_out_of_range_wire():
    with pytest.raises(IndexError):
        tq.apply_reset(tq.zeros_state(1), 5)


def test_density_mixture_of_a_pure_state_is_a_single_pair():
    pure = tq.to_density(tq.apply_h(tq.zeros_state(1, dtype=tp.complex128), 0))
    mixture = tq.density_mixture(pure)
    assert len(mixture) == 1 and len(mixture[0]) == 1
    weight, state = mixture[0][0]
    tp.testing.assert_close(tp.tensor([weight], dtype=tp.float64), tp.tensor([1.0], dtype=tp.float64),
                              atol=1e-12, rtol=1e-12)
    tp.testing.assert_close(state, tq.apply_h(tq.zeros_state(1, dtype=tp.complex128), 0),
                            atol=1e-12, rtol=1e-12)


def test_density_mixture_reconstructs_the_density_matrix():
    rho = 0.6 * tq.to_density(tq.zeros_state(1, dtype=tp.complex128)) + \
        0.4 * tq.to_density(tq.basis_state(1, 1, dtype=tp.complex128))
    mixture = tq.density_mixture(rho)[0]
    rebuilt = None
    for weight, state in mixture:
        term = weight * tq.to_density(state)
        rebuilt = term if rebuilt is None else rebuilt + term
    tp.testing.assert_close(rebuilt, rho, atol=1e-12, rtol=1e-12)


def test_density_mixture_orders_by_descending_weight():
    rho = 0.3 * tq.to_density(tq.zeros_state(1, dtype=tp.complex128)) + \
        0.7 * tq.to_density(tq.basis_state(1, 1, dtype=tp.complex128))
    mixture = tq.density_mixture(rho)[0]
    weights = [weight for weight, _ in mixture]
    assert weights == sorted(weights, reverse=True)
    assert len(weights) == 2


def test_density_mixture_handles_a_batch_and_filters_by_atol():
    pure = tq.to_density(tq.zeros_state(1, dtype=tp.complex128))
    mixed = 0.5 * tq.to_density(tq.zeros_state(1, dtype=tp.complex128)) + \
        0.5 * tq.to_density(tq.basis_state(1, 1, dtype=tp.complex128))
    mixtures = tq.density_mixture(tp.stack([pure[0], mixed[0]]))
    assert len(mixtures) == 2
    assert [len(entry) for entry in mixtures] == [1, 2]
    noisy = 0.999 * tq.to_density(tq.zeros_state(1, dtype=tp.complex128)) + \
        0.001 * tq.to_density(tq.basis_state(1, 1, dtype=tp.complex128))
    assert len(tq.density_mixture(noisy, atol=0.01)[0]) == 1
    assert len(tq.density_mixture(noisy, atol=1e-12)[0]) == 2


def test_density_mixture_rejects_a_malformed_density():
    with pytest.raises(ValueError):
        tq.density_mixture(tp.ones((2, 2)))
    with pytest.raises(ValueError):
        tq.density_mixture(tp.ones((1, 3, 3)))
