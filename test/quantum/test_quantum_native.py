"""The compiled state primitives: agreement with the decomposition, contracts.

The gate kernel is the fast path outside a recorded graph; the decomposition
carries gradients.  These check the two produce the same amplitudes, and pin
the argument rules both paths rely on.
"""
import math

import pytest

import tensorplay as tp
from tensorplay import quantum as tq
from tensorplay import _C
from tensorplay.quantum import _native

pytestmark = pytest.mark.skipif(
    not hasattr(_C, "_quantum_apply_gate"), reason="built before the state primitives"
)


def _flat(state):
    return state.reshape(state.shape[0], -1)


def _by_decomposition(state, mat, wires):
    """The gate as the recorded graph computes it, whatever the fast path does."""
    tracked = state.clone().requires_grad_(True)
    return tq.apply_gate(tracked, mat, wires).detach()


def _pairs():
    """A spread of widths, parameter values, and register sizes.

    Each entry names the case, so a failure points at one shape of gate.
    """
    for n_qubits in (1, 2, 3, 5):
        for wire in range(n_qubits):
            yield (
                f"one wire {wire} of {n_qubits}",
                tq.plus_state(n_qubits),
                tq.rot_y_matrix(0.4),
                [wire],
            )
    yield "far wires", tq.plus_state(3), tq.controlled_matrix(tq.rot_x_matrix(-1.1)), [0, 2]
    yield "controlled rotation", tq.plus_state(3), tq.controlled_matrix(tq.rot_z_matrix(0.9)), [1, 2]
    yield "adjacent wires", tq.plus_state(3), tq.swap_matrix(), [0, 1]
    yield "three wires", tq.zeros_state(3), tq.mcx_matrix(2), [0, 1, 2]
    yield "four wires", tq.zeros_state(4), tq.mcx_matrix(3), [0, 1, 2, 3]
    yield "basis state", tq.basis_state(5, 3), tq.swap_matrix(), [0, 2]
    yield "single wire on a basis state", tq.basis_state(5, 3), tq.h_matrix(), [2]


@pytest.mark.parametrize("name,state,mat,wires", list(_pairs()), ids=[case[0] for case in _pairs()])
def test_fast_path_matches_the_decomposition(name, state, mat, wires):
    tp.testing.assert_close(
        tq.apply_gate(state, mat, wires),
        _by_decomposition(state, mat, wires),
        atol=1e-6,
        rtol=1e-6,
    )


def test_fast_path_matches_the_decomposition_in_double_precision():
    state = tq.plus_state(3, dtype=tp.complex128)
    mat = tq.rot_y_matrix(0.4, c_dtype=tp.complex128)
    tp.testing.assert_close(
        tq.apply_gate(state, mat, [1]),
        _by_decomposition(state, mat, [1]),
        atol=1e-12,
        rtol=1e-12,
    )


def test_fast_path_carries_a_per_batch_matrix():
    theta = tp.tensor([0.0, math.pi, 2.0 * math.pi])
    states = tq.zeros_state(1, batch=3)
    out = _flat(tq.apply_gate(states, tq.rot_y_matrix(theta), [0]))
    tp.testing.assert_close(out[0], tp.tensor([1.0, 0.0], dtype=tp.complex64), atol=1e-6, rtol=0)
    tp.testing.assert_close(out[1], tp.tensor([0.0, 1.0], dtype=tp.complex64), atol=1e-6, rtol=0)
    tp.testing.assert_close(out[2], tp.tensor([-1.0, 0.0], dtype=tp.complex64), atol=1e-6, rtol=0)


def test_fast_path_keeps_the_state_shape():
    state = tq.plus_state(4)
    out = tq.apply_gate(state, tq.h_matrix(), [3])
    assert out.shape == state.shape


def test_fast_path_takes_the_wider_dtype():
    state = tq.plus_state(2, dtype=tp.complex64)
    out = tq.apply_gate(state, tq.h_matrix(c_dtype=tp.complex128), [0])
    assert out.dtype == tp.complex128
    tp.testing.assert_close(
        out.reshape(1, -1).to(tp.complex64),
        _flat(tq.apply_gate(state, tq.h_matrix(), [0])),
        atol=1e-6,
        rtol=1e-6,
    )


def test_state_probs_squares_the_amplitudes():
    state = tq.apply_h(tq.zeros_state(1), 0)
    tp.testing.assert_close(tq.probs(state), tp.tensor([[0.5, 0.5]]), atol=1e-6, rtol=1e-6)


def test_state_probs_of_a_bell_pair():
    bell = tq.apply_cnot(tq.apply_h(tq.zeros_state(2), 0), 0, 1)
    tp.testing.assert_close(tq.probs(bell), tp.tensor([[0.5, 0.0, 0.0, 0.5]]),
                            atol=1e-6, rtol=1e-6)


def test_state_probs_in_double_precision():
    state = tq.apply_h(tq.zeros_state(1, dtype=tp.complex128), 0)
    out = tq.probs(state)
    assert out.dtype == tp.float64
    tp.testing.assert_close(out, tp.tensor([[0.5, 0.5]], dtype=tp.float64), atol=1e-12, rtol=0)


def test_state_probs_over_a_batch():
    states = tq.plus_state(3, batch=4)
    out = tq.probs(states)
    assert out.shape == (4, 8)
    tp.testing.assert_close(out, tp.full((4, 8), 0.125), atol=1e-6, rtol=1e-6)


def test_gate_rejects_a_matrix_of_the_wrong_width():
    flat = tq.flatten(tq.zeros_state(2)).contiguous()
    with pytest.raises(ValueError):
        _C._quantum_apply_gate(flat, tp.eye(4, dtype=tp.complex64), [0])


def test_gate_rejects_a_repeated_wire():
    flat = tq.flatten(tq.zeros_state(2)).contiguous()
    with pytest.raises(ValueError):
        _C._quantum_apply_gate(flat, tp.eye(4, dtype=tp.complex64), [0, 0])


def test_gate_rejects_a_wire_outside_the_register():
    flat = tq.flatten(tq.zeros_state(2)).contiguous()
    with pytest.raises(IndexError):
        _C._quantum_apply_gate(flat, tq.i_matrix(), [2])


def test_gate_rejects_a_state_length_that_is_not_a_whole_register():
    flat = tp.zeros((1, 3), dtype=tp.complex64)
    with pytest.raises(ValueError):
        _C._quantum_apply_gate(flat, tq.i_matrix(), [0])


def test_gate_rejects_a_real_state():
    flat = tp.ones((1, 2), dtype=tp.float32)
    with pytest.raises(NotImplementedError):
        _C._quantum_apply_gate(flat, tq.i_matrix(), [0])


def test_state_probs_rejects_a_real_state():
    with pytest.raises(NotImplementedError):
        _C._quantum_state_probs(tp.ones((1, 2), dtype=tp.float32))


def test_decomposed_path_is_taken_when_a_gradient_is_recorded():
    state = tq.plus_state(2)
    tracked = state.clone().requires_grad_(True)
    tq.apply_gate(tracked, tq.h_matrix(), [0]).real.sum().backward()
    assert tracked.grad is not None
    assert not bool(tp.allclose(tracked.grad, tp.zeros_like(tracked)))


def test_no_grad_takes_the_fast_path():
    state = tq.plus_state(2)
    with tp.no_grad():
        out = tq.apply_gate(state, tq.h_matrix(), [0])
    assert not out.requires_grad
    tp.testing.assert_close(
        out, _by_decomposition(state, tq.h_matrix(), [0]), atol=1e-6, rtol=1e-6
    )


def test_native_helpers_report_absence_rather_than_failing():
    saved = _native._APPLY_GATE
    try:
        _native._APPLY_GATE = None
        assert tq.apply_gate(tq.zeros_state(1), tq.h_matrix(), [0]).shape == (1, 2)
    finally:
        _native._APPLY_GATE = saved


def test_gate_rejects_more_wires_than_either_backend_allows():
    # Eight wires is the widest gate either backend tabulates.
    state = tq.zeros_state(9)
    mat = tp.eye(1 << 9, dtype=tp.complex64)
    with pytest.raises(ValueError):
        _C._quantum_apply_gate(
            tq.flatten(state).contiguous(), mat, list(range(9))
        )


def test_gate_spans_every_width_it_allows():
    # Every control is zero on |0...0>, so the widest gate is inert there --
    # what it checks is that the width is handled at all.
    state = tq.zeros_state(8)
    mat = tq.mcx_matrix(7, c_dtype=tp.complex64)
    out = tq.apply_gate(state, mat, list(range(8)))
    tp.testing.assert_close(_flat(out), _flat(state))
    # With the controls set, the target flips.
    excited = tq.basis_state(254, 8)  # every control set, target clear
    flipped = tq.apply_gate(excited, mat, list(range(8)))
    tp.testing.assert_close(_flat(flipped), _flat(tq.basis_state(255, 8)))

# ---------------------------------------------------------------------------
# Marginal distribution and Pauli expectation
# ---------------------------------------------------------------------------


def _words(n_qubits):
    """A spread of words over a register, identities included."""
    letters = "IXYZ"
    yield "I" * n_qubits
    for letter in letters:
        yield letter * n_qubits
    yield "X" * n_qubits
    yield "Y" * n_qubits
    yield "Z" * n_qubits
    yield ("XZ" * n_qubits)[:n_qubits]
    yield ("YX" * n_qubits)[:n_qubits]
    yield ("ZY" * n_qubits)[:n_qubits]


def _by_pauli_gates(state, word):
    """The expectation value the one-factor-at-a-time route computes."""
    rotated = state
    for wire, letter in enumerate(word):
        if letter != "I":
            builder = {"X": tq.x_matrix, "Y": tq.y_matrix, "Z": tq.z_matrix}[letter]
            rotated = tq.apply_gate(rotated, builder(c_dtype=state.dtype), [wire])
    return (rotated.conj() * state).sum(dim=tuple(range(1, state.dim()))).real


def test_marginal_probs_matches_the_permuted_reduction():
    bell = tq.apply_cnot(tq.apply_h(tq.zeros_state(2), 0), 0, 1)
    tp.testing.assert_close(
        tq.marginal_probs(bell, [0]), tp.tensor([[0.5, 0.5]]), atol=1e-6, rtol=1e-6
    )
    tp.testing.assert_close(
        tq.marginal_probs(bell, [1]), tp.tensor([[0.5, 0.5]]), atol=1e-6, rtol=1e-6
    )
    tp.testing.assert_close(tq.marginal_probs(bell, [0, 1]), tq.probs(bell),
                            atol=1e-6, rtol=1e-6)


def test_marginal_probs_keeps_the_order_the_wires_were_listed_in():
    # |101>: wire 1 is clear and wire 2 is set, so the two listings name
    # different outcomes.
    state = tq.basis_state(5, 3)
    tp.testing.assert_close(tq.marginal_probs(state, [1, 2]),
                            tp.tensor([[0.0, 1.0, 0.0, 0.0]]), atol=1e-6, rtol=1e-6)
    tp.testing.assert_close(tq.marginal_probs(state, [2, 1]),
                            tp.tensor([[0.0, 0.0, 1.0, 0.0]]), atol=1e-6, rtol=1e-6)


def test_marginal_probs_of_every_subset_sums_to_one():
    state = tq.plus_state(4)
    for wires in ([0], [1], [3], [0, 1], [2, 3], [1, 2, 3], [0, 1, 2, 3]):
        tp.testing.assert_close(
            tq.marginal_probs(state, wires).sum(dim=-1),
            tp.ones(1),
            atol=1e-6,
            rtol=1e-6,
        )


def test_marginal_probs_reads_the_full_distribution_when_all_wires_are_listed():
    state = tq.apply_h(tq.zeros_state(3), 1)
    tp.testing.assert_close(tq.marginal_probs(state, [0, 1, 2]), tq.probs(state),
                            atol=1e-6, rtol=1e-6)


def test_marginal_probs_in_double_precision():
    state = tq.apply_h(tq.zeros_state(2, dtype=tp.complex128), 0)
    out = tq.marginal_probs(state, [0])
    assert out.dtype == tp.float64
    tp.testing.assert_close(out, tp.tensor([[0.5, 0.5]], dtype=tp.float64),
                            atol=1e-12, rtol=0)


def test_marginal_probs_over_a_batch():
    states = tq.plus_state(3, batch=4)
    out = tq.marginal_probs(states, [1])
    assert out.shape == (4, 2)
    tp.testing.assert_close(out, tp.full((4, 2), 0.5), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 5])
def test_compiled_expectation_matches_the_gate_route(n_qubits):
    state = tq.plus_state(n_qubits)
    state = tq.apply_ry(state, 0, 0.7)
    if n_qubits > 1:
        state = tq.apply_cnot(state, 0, n_qubits - 1)
    for word in _words(n_qubits):
        tp.testing.assert_close(
            tq.expectation(state, word),
            _by_pauli_gates(state, word),
            atol=1e-5,
            rtol=1e-5,
        )


def test_compiled_expectation_reads_the_eigenvalues_off_the_state():
    # An X eigenstate: |+> reads 1 on X and 0 on Z.
    plus = tq.apply_h(tq.zeros_state(1), 0)
    tp.testing.assert_close(tq.expectation(plus, "X"), tp.tensor([1.0]), atol=1e-6, rtol=1e-6)
    tp.testing.assert_close(tq.expectation(plus, "Z"), tp.tensor([0.0]), atol=1e-6, rtol=1e-6)
    # A Y eigenstate needs the quarter turn the Y factor contributes.
    y_state = tq.apply_s(tq.apply_h(tq.zeros_state(1), 0), 0)
    tp.testing.assert_close(tq.expectation(y_state, "Y"), tp.tensor([1.0]),
                            atol=1e-6, rtol=1e-6)
    tp.testing.assert_close(tq.expectation(y_state, "X"), tp.tensor([0.0]),
                            atol=1e-6, rtol=1e-6)


def _dense_expectation(state, word):
    """<psi|P|psi> from the dense word matrix, shared with nothing else."""
    n_qubits = state.dim() - 1
    word = word.ljust(n_qubits, "I")
    wires = [index for index, letter in enumerate(word) if letter != "I"]
    matrix = tq.expand_to_register(tq.pauli_string_matrix(word), wires, n_qubits)
    flat = tq.flatten(state)
    turned = matrix @ flat.reshape(-1, 1)
    return (flat.conj().reshape(-1) * turned.reshape(-1)).sum().real


def _twisted(n_qubits):
    """A state whose amplitudes carry phases, so a sign error cannot hide."""
    state = tq.plus_state(n_qubits)
    for wire in range(n_qubits):
        state = tq.apply_s(state, wire) if wire % 2 == 0 else tq.apply_t(state, wire)
    return state


_WORDS = ("X", "Y", "Z", "XX", "YY", "ZZ", "XY", "YZ", "XZ", "III")


def test_a_compiled_pauli_expectation_matches_the_dense_matrix():
    # A real overlap cannot tell a conjugated product from the right one: the
    # difference only shows up in the imaginary half, so these states carry
    # phases on every wire.
    for n_qubits in (1, 2, 3):
        for state in (_twisted(n_qubits), tq.plus_state(n_qubits)):
            for word in _WORDS:
                if len(word) > n_qubits:
                    continue
                padded = word.ljust(n_qubits, "I")
                got = _C._quantum_pauli_expectation(
                    tq.flatten(state).contiguous(),
                    _native.pauli_codes(padded, tp.device("cpu")),
                )
                want = _dense_expectation(state, word)
                tp.testing.assert_close(got, want.reshape(1), atol=1e-5, rtol=1e-5)


def test_a_compiled_y_factor_reads_the_state_and_not_its_conjugate():
    state = tq.apply_s(tq.apply_h(tq.zeros_state(1), 0), 0)
    tp.testing.assert_close(tq.expectation(state, "Y"), tp.tensor([1.0]),
                            atol=1e-6, rtol=1e-6)
    tp.testing.assert_close(tq.expectation(state, "X"), tp.tensor([0.0]),
                            atol=1e-6, rtol=1e-6)


def test_compiled_expectation_of_an_identity_word_is_the_norm():
    state = tq.apply_ry(tq.zeros_state(2), 1, 0.6)
    tp.testing.assert_close(
        tq.expectation(state, "II").reshape(()),
        (tp.norm(state) ** 2).reshape(()),
        atol=1e-6,
        rtol=1e-6,
    )


def test_compiled_expectation_over_a_batch():
    states = tq.zeros_state(2, batch=3)
    out = tq.expectation(states, "ZI")
    assert out.shape == (3,)
    tp.testing.assert_close(out, tp.ones(3), atol=1e-6, rtol=1e-6)


def test_compiled_expectation_in_double_precision():
    # Wire 1 in a plus state reads 1 on X; wire 0 stays at 0.
    state = tq.apply_h(tq.zeros_state(2, dtype=tp.complex128), 1)
    out = tq.expectation(state, "IX")
    assert out.dtype == tp.float64
    tp.testing.assert_close(out, tp.tensor([1.0], dtype=tp.float64), atol=1e-12, rtol=0)


def test_energy_of_a_hamiltonian_matches_term_by_term():
    terms = [(0.3, "XI"), (0.7, "IZ"), (-1.1, "ZZ"), (0.2, "II")]
    hamiltonian = tq.Hamiltonian(terms)
    state = tq.apply_cnot(tq.apply_h(tq.zeros_state(2), 0), 0, 1)
    total = tq.energy(state, hamiltonian)
    by_hand = sum(
        coeff * tq.expectation(state, word) for coeff, word in terms
    )
    tp.testing.assert_close(total, by_hand, atol=1e-5, rtol=1e-5)


def test_marginal_probs_rejects_a_repeated_wire():
    flat = tq.flatten(tq.zeros_state(2)).contiguous()
    with pytest.raises(ValueError):
        _C._quantum_marginal_probs(flat, [0, 0])


def test_marginal_probs_rejects_a_wire_outside_the_register():
    flat = tq.flatten(tq.zeros_state(2)).contiguous()
    with pytest.raises(IndexError):
        _C._quantum_marginal_probs(flat, [3])


def test_expectation_rejects_a_word_that_does_not_fit():
    flat = tq.flatten(tq.zeros_state(2)).contiguous()
    codes = _native.pauli_codes("XXX", flat.device)
    with pytest.raises(ValueError):
        _C._quantum_pauli_expectation(flat, codes)


def test_expectation_rejects_a_code_outside_the_alphabet():
    flat = tq.flatten(tq.zeros_state(2)).contiguous()
    codes = tp.tensor([0, 4], dtype=tp.int32, device=flat.device)
    with pytest.raises(ValueError):
        _C._quantum_pauli_expectation(flat, codes)


def test_expectation_rejects_a_real_state():
    with pytest.raises(NotImplementedError):
        _C._quantum_pauli_expectation(
            tp.ones((1, 2), dtype=tp.float32), tp.tensor([3], dtype=tp.int32)
        )


def test_the_new_paths_report_absence_rather_than_failing():
    saved = (_native._MARGINAL_PROBS, _native._PAULI_EXPECTATION)
    try:
        _native._MARGINAL_PROBS = None
        _native._PAULI_EXPECTATION = None
        state = tq.apply_h(tq.zeros_state(1), 0)
        tp.testing.assert_close(tq.marginal_probs(state, [0]), tp.tensor([[0.5, 0.5]]),
                                atol=1e-6, rtol=1e-6)
        tp.testing.assert_close(tq.expectation(state, "X"), tp.tensor([1.0]),
                                atol=1e-6, rtol=1e-6)
    finally:
        _native._MARGINAL_PROBS, _native._PAULI_EXPECTATION = saved


def test_the_new_paths_step_aside_for_a_recorded_gradient():
    tracked = tq.zeros_state(2).requires_grad_(True)
    rotated = tq.apply_ry(tracked, 0, 0.4)
    assert _native.native_pauli_expectation(
        rotated, _native.pauli_codes("ZI", rotated.device)
    ) is None
    assert _native.native_marginal_probs(rotated, [1]) is None
    # And the value still comes out right along the recorded path: a rotation
    # by theta about Y leaves Z reading cos(theta).
    tq.expectation(rotated, "ZI").backward()
    tp.testing.assert_close(
        tq.expectation(rotated.detach(), "ZI"),
        tp.tensor([math.cos(0.4)]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert tracked.grad is not None


def test_the_marginal_kernel_keeps_the_column_order_it_was_given():
    # |01> names wire 1, so the two orderings have to put the weight on
    # different columns -- the leading column is the wire listed first.
    flat = tq.flatten(tq.basis_state(1, 2)).contiguous()
    leading_one_first = _C._quantum_marginal_probs(flat, [1, 0])
    trailing_one_first = _C._quantum_marginal_probs(flat, [0, 1])
    assert abs(float(leading_one_first.reshape(-1)[2]) - 1.0) < 1e-6
    assert abs(float(trailing_one_first.reshape(-1)[1]) - 1.0) < 1e-6


def test_the_marginal_kernel_agrees_with_the_module_on_an_unusual_order():
    state = tq.apply_h(tq.zeros_state(3), 0)
    flat = tq.flatten(state).contiguous()
    tp.testing.assert_close(
        _C._quantum_marginal_probs(flat, [2, 0, 1]),
        tq.marginal_probs(state, [2, 0, 1]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_a_long_walk_is_cut_into_chunks_and_still_agrees():
    # Few columns over a wide register means each column walks the whole
    # register, so the walk is cut into pieces that are summed afterwards.  The
    # pieces have to add up to the same weights the decomposition gives.
    state = tq.plus_state(14)
    flat = tq.flatten(state).contiguous()
    tp.testing.assert_close(
        _C._quantum_marginal_probs(flat, [3, 11]),
        tq.marginal_probs(state, [3, 11]),
        atol=1e-6,
        rtol=1e-6,
    )
    tp.testing.assert_close(
        _C._quantum_marginal_probs(flat, [13, 0, 6]),
        tq.marginal_probs(state, [13, 0, 6]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_the_marginal_kernel_gives_the_same_numbers_twice():
    # The chunk partials are summed in a fixed order, so a repeat run is the
    # same number and not merely a close one.
    flat = tq.flatten(tq.plus_state(14)).contiguous()
    first = _C._quantum_marginal_probs(flat, [2, 9])
    second = _C._quantum_marginal_probs(flat, [2, 9])
    assert first.reshape(-1).tolist() == second.reshape(-1).tolist()


def test_the_marginal_kernel_reads_more_wires_than_a_gate_may_span():
    # A gate's matrix is 2**k by 2**k, which caps it well short of a wide
    # register; a histogram has no matrix and covers the whole thing.
    n_qubits = 10
    flat = tq.flatten(tq.basis_state(1, n_qubits)).contiguous()
    out = _C._quantum_marginal_probs(flat, list(range(n_qubits)))
    assert out.shape == (1, 1 << n_qubits)
    assert abs(float(out.reshape(-1)[1]) - 1.0) < 1e-6
