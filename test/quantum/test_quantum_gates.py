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


# ---------------------------------------------------------------------------
# Control readings, arbitrary axes, multi-wire rotations
# ---------------------------------------------------------------------------


def test_a_control_can_ask_for_a_zero_reading():
    # Both controls ask for 0, so a clear pair flips the target ...
    flipped = tq.apply_mcx(tq.basis_state(0, 3), [0, 1, 2], control_values=[0, 0])
    tp.testing.assert_close(
        tq.flatten(flipped), tq.flatten(tq.basis_state(1, 3)), atol=1e-6, rtol=1e-6
    )
    # ... and a pair that is already set does not.
    held = tq.apply_mcx(tq.basis_state(3, 3), [0, 1, 2], control_values=[0, 0])
    tp.testing.assert_close(tq.flatten(held), tq.flatten(tq.basis_state(3, 3)),
                            atol=1e-6, rtol=1e-6)


def test_the_first_control_leads_the_pattern():
    # control_values reads the controls in wire order, and wire 0 is the leading
    # bit of the index, so a lone 1 on wire 0 points at index 4 of 8.
    mat = tq.mcx_matrix(2, control_values=[1, 0])
    swapped = [
        (i, j)
        for i in range(8)
        for j in range(8)
        if abs(float(mat[i, j].real) - 1.0) < 1e-6 and i != j
    ]
    assert swapped == [(4, 5), (5, 4)]


def test_a_scalar_control_value_asks_for_it_on_every_control():
    one = tq.mcx_matrix(2, control_values=1)
    tp.testing.assert_close(one, tq.mcx_matrix(2), atol=1e-6, rtol=1e-6)


def test_control_values_need_one_entry_per_control():
    with pytest.raises(ValueError):
        tq.mcx_matrix(2, control_values=[1])
    with pytest.raises(ValueError):
        tq.mcx_matrix(2, control_values=[1, 0, 1])
    with pytest.raises(ValueError):
        tq.mcx_matrix(2, control_values=[2, 1])


def test_controlled_matrix_places_the_operator_on_the_matching_pattern():
    mat = tq.controlled_matrix(tq.x_matrix(), 2, control_values=[0, 1])
    # controls read 0 then 1: the guarded basis pair is |010> and |011>.
    flipped = [
        (i, j)
        for i in range(8)
        for j in range(8)
        if abs(float(mat[i, j].real) - 1.0) < 1e-6 and i != j
    ]
    assert flipped == [(2, 3), (3, 2)]


def test_the_generic_controlled_helper_matches_the_named_gates():
    state = tq.zeros_state(3)
    named = tq.apply_cx(state, 0, 2)
    generic = tq.apply_controlled(state, tq.x_matrix(device=state.device), [0, 2])
    tp.testing.assert_close(named, generic, atol=1e-6, rtol=1e-6)


def test_new_named_applications_match_their_own_matrices():
    state = tq.zeros_state(3)
    pairs = [
        (tq.apply_cp(state, 0, 2, 0.4), tq.controlled_matrix(tq.phase_matrix(0.4)), [0, 2]),
        (tq.apply_crx(state, 0, 2, 0.4), tq.controlled_matrix(tq.rot_x_matrix(0.4)), [0, 2]),
        (tq.apply_cry(state, 1, 2, 0.4), tq.controlled_matrix(tq.rot_y_matrix(0.4)), [1, 2]),
        (tq.apply_crz(state, 1, 2, 0.4), tq.controlled_matrix(tq.rot_z_matrix(0.4)), [1, 2]),
        (tq.apply_cy(state, 0, 1), tq.controlled_matrix(tq.y_matrix()), [0, 1]),
        (tq.apply_ccx(state, 0, 1, 2), tq.mcx_matrix(2), [0, 1, 2]),
        (tq.apply_u2(state, 2, 0.4, 0.5), tq.u2_matrix(0.4, 0.5), [2]),
        (tq.apply_i(state, 2), tq.i_matrix(), [2]),
        (tq.apply_sdg(state, 2), tq.sdg_matrix(), [2]),
        (tq.apply_tdg(state, 2), tq.tdg_matrix(), [2]),
        (tq.apply_rot(state, 2, [0, 1, 0], 0.4), tq.rot_matrix([0, 1, 0], 0.4), [2]),
        (tq.apply_multirz(state, 0.4), tq.multirz_matrix(0.4, 3), [0, 1, 2]),
    ]
    for out, mat, wires in pairs:
        expected = tq.apply_gate(state, mat.to(state.dtype), wires)
        tp.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_a_rotation_about_an_axis_reduces_to_the_coordinate_rotations():
    theta = 0.73
    for axis, coordinate in (
        ([1, 0, 0], tq.rot_x_matrix(theta)),
        ([0, 1, 0], tq.rot_y_matrix(theta)),
        ([0, 0, 1], tq.rot_z_matrix(theta)),
    ):
        tp.testing.assert_close(
            tq.rot_matrix(axis, theta), coordinate, atol=1e-6, rtol=1e-6
        )


def test_a_rotation_ignores_the_length_of_its_axis():
    theta = 0.73
    tp.testing.assert_close(
        tq.rot_matrix([0, 3, 4], theta),
        tq.rot_matrix([0, 3 / 5, 4 / 5], theta),
        atol=1e-6,
        rtol=1e-6,
    )


def test_a_rotation_about_no_direction_is_refused():
    with pytest.raises(ValueError):
        tq.rot_matrix([0, 0, 0], 0.5)


def test_an_axis_needs_three_components():
    with pytest.raises(ValueError):
        tq.rot_matrix([0, 1], 0.5)


def test_the_multi_wire_rotation_is_one_angle_on_every_wire():
    angle = 0.3
    per_wire = tp.eye(8, dtype=tp.complex64)
    for wire in range(3):
        per_wire = tq.expand_to_register(
            tq.rot_z_matrix(angle), [wire], 3
        ) @ per_wire
    tp.testing.assert_close(
        tq.multirz_matrix(angle, 3), per_wire, atol=1e-6, rtol=1e-6
    )


def test_multi_wire_eigenvalues_are_the_diagonal_of_the_matrix():
    # Each basis state carries e^{-i angle (n - 2k) / 2} with k ones in its index,
    # which for three wires walks the counts 3, 1, 1, -1, 1, -1, -1, -3.
    angle = 0.4
    counts = [3, 1, 1, -1, 1, -1, -1, -3]
    expected = tp.tensor(
        [complex(math.cos(-angle * k / 2), math.sin(-angle * k / 2)) for k in counts],
        dtype=tp.complex64,
    )
    tp.testing.assert_close(tq.multirz_eigvals(angle, 3), expected, atol=1e-6, rtol=1e-6)


def test_a_multi_wire_rotation_needs_at_least_one_wire():
    with pytest.raises(ValueError):
        tq.multirz_matrix(0.3, 0)


def test_global_phase_scales_the_state_without_touching_the_relations():
    state = tq.apply_h(tq.zeros_state(1), 0)
    out = tq.apply_global_phase(state, 0.7)
    expected = tq.flatten(state) * tp.tensor([0.76484219 + 0.64421769j])
    tp.testing.assert_close(tq.flatten(out), expected.to(state.dtype), atol=1e-6, rtol=1e-6)
    # The relative phase of the two amplitudes is untouched.
    ratio = out.reshape(2)[1] / out.reshape(2)[0]
    tp.testing.assert_close(ratio, tp.tensor(1.0 + 0j), atol=1e-6, rtol=1e-6)


def _basis_index_of(state):
    """The one basis entry a single-basis state sits on."""
    weights = tq.flatten(state).abs().reshape(-1)
    return int(weights.argmax())


def test_relative_phase_toffoli_flips_where_the_plain_one_flips():
    plain = tq.mcx_matrix(2)
    for index in range(8):
        start = tq.basis_state(index, 3)
        relative = _basis_index_of(tq.apply_rccx(start, 0, 1, 2))
        ordinary = _basis_index_of(tq.apply_gate(start, plain, [0, 1, 2]))
        if index >> 2:
            # control a at 1: the wire flip is the plain one.
            assert relative == ordinary
        else:
            # control a at 0: the target is left alone.
            assert relative == index


def test_relative_phase_toffoli_adds_a_sign_and_is_its_own_inverse():
    relative = tq.rccx_matrix()
    tp.testing.assert_close(relative @ relative, tp.eye(8, dtype=relative.dtype), atol=1e-6, rtol=1e-6)
    out = tq.apply_rccx(tq.basis_state(2, 3), 0, 1, 2)
    tp.testing.assert_close(
        tq.flatten(out),
        -tq.flatten(tq.basis_state(2, 3)),
        atol=1e-6,
        rtol=1e-6,
    )
    # The flip of the plain gate still happens on the two-control branch.
    flipped = tq.apply_rccx(tq.basis_state(6, 3), 0, 1, 2)
    tp.testing.assert_close(
        tq.flatten(flipped), tq.flatten(tq.basis_state(7, 3)), atol=1e-6, rtol=1e-6
    )


def test_global_phase_matrix_is_a_scalar_operator():
    tp.testing.assert_close(
        tq.globalphase_matrix(0.0).reshape(()),
        tp.tensor(1.0 + 0j),
        atol=1e-6,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Which wire plays which role
# ---------------------------------------------------------------------------


def _index_of(state):
    return int(tp.argmax(tq.probs(state).reshape(-1)).reshape(()))


def test_a_cnot_names_its_control_where_the_caller_put_it():
    # |01>: wire 1 holds the one, so a CNOT from wire 1 lands on wire 0.
    state = tq.apply_x(tq.zeros_state(2), 1)
    assert _index_of(tq.apply_cnot(state, 1, 0)) == 3
    assert _index_of(tq.apply_cnot(state, 0, 1)) == 1


def test_the_two_directions_of_a_cnot_differ():
    # |10> is the mirror of |01>, so each direction fires on a different one.
    state = tq.apply_x(tq.zeros_state(2), 0)
    assert _index_of(tq.apply_cnot(state, 1, 0)) == 2
    assert _index_of(tq.apply_cnot(state, 0, 1)) == 3


def test_the_control_of_a_controlled_phase_is_where_it_was_named():
    state = tq.apply_x(tq.zeros_state(2), 0)
    angle = 0.5
    # Control wire 1 is clear, so the phase never lands.
    assert _index_of(tq.apply_cp(state, 1, 0, angle)) == 2
    # Control wire 0 is set, so |10> picks up the phase and |11> does not.
    phased = tq.apply_cp(state, 0, 1, angle)
    probs = tq.probs(phased).reshape(-1)
    assert probs[2] > 0.99
    assert probs[3] < 1e-6


def test_a_controlled_rotation_turns_the_wire_it_was_named_on():
    state = tq.apply_x(tq.zeros_state(2), 1)
    turned = tq.probs(tq.apply_crx(state, 0, 1, math.pi / 2)).reshape(-1)
    # The control wire is clear, so the target stays put.
    assert turned[1] > 0.99


def test_a_multi_controlled_x_takes_its_target_last():
    state = tq.basis_state(5, 3)  # |101>: wires 0 and 2 are set, wire 1 is not
    assert _index_of(tq.apply_mcx(state, [0, 2, 1])) == 7
    assert _index_of(tq.apply_mcx(state, [0, 1, 2])) == 5


def test_a_toffoli_names_its_target_last():
    # |011>: wires 1 and 2 are set, wire 0 is clear.
    state = tq.apply_x(tq.apply_x(tq.zeros_state(3), 2), 1)
    assert _index_of(tq.apply_toffoli(state, 1, 2, 0)) == 7
    assert _index_of(tq.apply_toffoli(state, 0, 1, 2)) == 3
    # The two controls are interchangeable -- a gate over the same pair of wires
    # is the same gate whichever way round they were named.
    assert _index_of(tq.apply_toffoli(state, 2, 1, 0)) == 7


def test_a_fredkin_names_its_control():
    # |101>: wires 0 and 2 are set, so either naming of the control fires.
    state = tq.apply_x(tq.apply_x(tq.zeros_state(3), 2), 0)
    assert _index_of(tq.apply_cswap(state, 2, 0, 1)) == 3  # swaps wires 0 and 1
    assert _index_of(tq.apply_cswap(state, 0, 1, 2)) == 6  # swaps wires 1 and 2


def test_a_gate_matrix_reads_its_bits_in_the_named_order():
    # A plain controlled-X on (wire 1, wire 0) is the same operator as the
    # controlled-X on (wire 0, wire 1) with its bits swapped -- which the matrix
    # path must reproduce.
    control_high = tq.controlled_matrix(tq.x_matrix())
    state = tq.apply_x(tq.zeros_state(2), 0)  # |10>
    # Wire 1 named first means wire 1 is the control, and it is clear here.
    assert _index_of(tq.apply_gate(state, control_high, [1, 0])) == 2
    assert _index_of(tq.apply_gate(state, control_high, [0, 1])) == 3


def test_an_ascending_wire_list_behaves_as_before():
    state = tq.apply_x(tq.apply_x(tq.zeros_state(3), 2), 1)  # |011>
    tp.testing.assert_close(
        tq.flatten(tq.apply_mcx(state, [0, 1, 2])),
        tq.flatten(tq.basis_state(3, 3)),  # wire 0 is clear, so nothing happens
        atol=1e-6,
        rtol=1e-6,
    )
    tp.testing.assert_close(
        tq.flatten(tq.apply_mcx(state, [1, 2, 0])),
        tq.flatten(tq.basis_state(7, 3)),
        atol=1e-6,
        rtol=1e-6,
    )


def test_a_repeated_wire_is_still_refused():
    with pytest.raises(ValueError):
        tq.apply_cnot(tq.zeros_state(2), 1, 1)
