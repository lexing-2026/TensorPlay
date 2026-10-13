"""State layout, normalization, density matrices, and partial trace."""

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
