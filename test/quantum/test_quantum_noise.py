"""Channels on density matrices: semantics, invariants, and composition."""

import numpy as np
import pytest

import tensorplay as tp
from tensorplay import quantum as tq


def _trace(density):
    return tp.diagonal(density.reshape(density.shape[0], density.shape[1], -1), dim1=1, dim2=2).sum(-1)


def _hermitian(density):
    return tq.dagger(density)


def test_kraus_sets_have_the_stated_width():
    for kraus in (
        tq.bit_flip_kraus(0.1),
        tq.phase_flip_kraus(0.1),
        tq.amplitude_damping_kraus(0.2),
        tq.depolarizing_kraus(0.3),
    ):
        assert kraus.dim() == 3
        assert kraus.shape[-1] == kraus.shape[-2] == 2


def test_kraus_sets_are_complete():
    # sum_r K_r^dagger K_r == I for every trace-preserving channel
    identity = tp.eye(2, dtype=tp.complex64)
    for kraus in (
        tq.bit_flip_kraus(0.3),
        tq.phase_flip_kraus(0.7),
        tq.amplitude_damping_kraus(0.25),
        tq.depolarizing_kraus(0.45),
    ):
        total = None
        for operator in kraus:
            piece = tq.dagger(operator) @ operator
            total = piece if total is None else total + piece
        tp.testing.assert_close(total, identity, atol=1e-6, rtol=1e-6)


def test_kraus_operators_are_normalized():
    for kraus in (
        tq.bit_flip_kraus(0.3),
        tq.phase_flip_kraus(0.7),
        tq.amplitude_damping_kraus(0.25),
        tq.depolarizing_kraus(0.45),
    ):
        for operator in kraus:
            product = tq.dagger(operator) @ operator
            assert float(product.abs().max()) <= 1.0 + 1e-6


def test_channels_reject_a_probability_outside_the_unit_interval():
    with pytest.raises(ValueError):
        tq.bit_flip_kraus(1.5)


def test_bit_flip_at_zero_is_the_identity_channel():
    density = tq.to_density(tq.plus_state(2))
    out = tq.apply_bit_flip(density, 0.0, 1)
    tp.testing.assert_close(out, density, atol=1e-6, rtol=1e-6)


def test_bit_flip_at_one_always_flips():
    density = tq.to_density(tq.zeros_state(1))
    out = tq.apply_bit_flip(density, 1.0, 0)
    flipped = tq.to_density(tq.basis_state(1, 1))
    tp.testing.assert_close(out, flipped, atol=1e-6, rtol=1e-6)


def test_phase_flip_splits_the_coherences():
    # |+> turned about Z has an imaginary off-diagonal, which Z conjugates, so
    # a fair phase flip averages it with its own conjugate.
    state = tq.apply_rz(tq.apply_h(tq.zeros_state(1), 0), 0, 0.6)
    density = tq.to_density(state)
    out = tq.apply_phase_flip(density, 0.5, 0)
    tp.testing.assert_close(out[0, 0, 0], tp.tensor(0.5 + 0.0j), atol=1e-6, rtol=0)
    tp.testing.assert_close(out[0, 0, 1], tp.tensor(0.0 + 0.0j), atol=1e-6, rtol=0)
    tp.testing.assert_close(out[0, 1, 1], tp.tensor(0.5 + 0.0j), atol=1e-6, rtol=0)


def test_bit_flip_conjugates_the_coherences():
    # X is its own inverse, so a certain flip hands back the state with its
    # coherences conjugated and its populations exchanged.
    state = tq.apply_rz(tq.apply_h(tq.zeros_state(1), 0), 0, 0.6)
    density = tq.to_density(state)
    out = tq.apply_bit_flip(density, 1.0, 0)
    tp.testing.assert_close(out[0, 0, 1], density[0, 0, 1].conj(), atol=1e-6, rtol=0)
    tp.testing.assert_close(out[0, 0, 0], density[0, 1, 1], atol=1e-6, rtol=0)


def test_phase_flip_keeps_the_populations():
    density = tq.to_density(tq.basis_state(1, 1))
    out = tq.apply_phase_flip(density, 0.5, 0)
    tp.testing.assert_close(tq.density_probs(out), tp.tensor([[0.0, 1.0]]), atol=1e-6, rtol=1e-6)


def test_amplitude_damping_moves_population_to_the_ground_state():
    density = tq.to_density(tq.basis_state(1, 1))
    out = tq.apply_amplitude_damping(density, 0.25, 0)
    tp.testing.assert_close(
        tq.density_probs(out), tp.tensor([[0.25, 0.75]]), atol=1e-6, rtol=1e-6
    )


def test_amplitude_damping_leaves_the_ground_state_alone():
    density = tq.to_density(tq.zeros_state(1))
    out = tq.apply_amplitude_damping(density, 0.9, 0)
    tp.testing.assert_close(out, density, atol=1e-6, rtol=1e-6)


def test_depolarizing_spreads_a_pure_state():
    density = tq.to_density(tq.zeros_state(1))
    out = tq.apply_depolarizing(density, 0.9, 0)
    # (1-p)|0><0| + (p/3)(X + Y + Z) sends p/3 of the weight to |1> twice over.
    tp.testing.assert_close(
        tq.density_probs(out), tp.tensor([[0.4, 0.6]]), atol=1e-6, rtol=1e-6
    )


def test_channels_preserve_the_trace():
    density = tq.to_density(tq.plus_state(2))
    for apply, kwargs in (
        (tq.apply_bit_flip, {"p": 0.3, "wire": 0}),
        (tq.apply_phase_flip, {"p": 0.4, "wire": 1}),
        (tq.apply_depolarizing, {"p": 0.2, "wire": 1}),
    ):
        out = apply(density, **kwargs)
        tp.testing.assert_close(_trace(out).real, tp.tensor([1.0]), atol=1e-6, rtol=1e-6)


def test_amplitude_damping_moves_weight_to_the_ground_state():
    density = tq.to_density(tq.basis_state(3, 2))  # |11>: both wires excited
    out = tq.apply_amplitude_damping(density, 0.5, 0)
    # Half the population drops from wire 0, half stays on |11>.
    tp.testing.assert_close(
        tq.density_probs(out), tp.tensor([[0.0, 0.5, 0.0, 0.5]]), atol=1e-6, rtol=1e-6
    )


def test_channels_keep_the_density_matrix_hermitian():
    density = tq.to_density(tq.plus_state(2))
    out = tq.apply_bit_flip(density, 0.3, 0)
    tp.testing.assert_close(out, _hermitian(out), atol=1e-6, rtol=1e-6)


def test_channels_keep_eigenvalues_non_negative():
    density = tq.to_density(tq.plus_state(2))
    out = tq.apply_depolarizing(density, 0.6, 1)
    eigenvalues = tp.linalg.eigvalsh(out)
    assert float(eigenvalues.min().real) > -1e-6


def test_channels_compose():
    density = tq.to_density(tq.zeros_state(1))
    once = tq.apply_phase_flip(density, 0.5, 0)
    twice = tq.apply_phase_flip(once, 0.5, 0)
    # Z applied twice is the identity on the diagonals.
    tp.testing.assert_close(twice, density, atol=1e-6, rtol=1e-6)


def test_apply_channel_accepts_an_arbitrary_kraus_set():
    # A Hadamard "channel": one unitary operator is a valid Kraus set.
    density = tq.to_density(tq.zeros_state(1))
    out = tq.apply_channel(density, tq.h_matrix().unsqueeze(0), [0])
    tp.testing.assert_close(out, tq.to_density(tq.plus_state(1)), atol=1e-6, rtol=1e-6)


def test_apply_channel_rejects_a_malformed_density_matrix():
    with pytest.raises(ValueError):
        tq.apply_channel(tp.zeros((1, 4), dtype=tp.complex64), tq.h_matrix().unsqueeze(0), [0])


def test_apply_channel_rejects_a_non_stacked_kraus_set():
    density = tq.to_density(tq.zeros_state(1))
    with pytest.raises(ValueError):
        tq.apply_channel(density, tq.h_matrix(), [0])


def test_channel_on_a_non_ground_wire():
    density = tq.to_density(tq.basis_state(1, 2))  # |01>: wire 0 = 0
    out = tq.apply_bit_flip(density, 1.0, 0)
    # A certain flip on wire 0 carries |01> to |11>.
    tp.testing.assert_close(
        tq.density_probs(out), tp.tensor([[0.0, 0.0, 0.0, 1.0]]), atol=1e-6, rtol=0
    )


def test_two_qubit_channel_is_applied_to_the_listed_wires():
    density = tq.to_density(tq.zeros_state(2))
    # The operator that flips both wires at once: |00> <-> |11>.
    identity = tp.eye(4, dtype=tp.complex64)
    flip_both = tp.tensor(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=tp.complex64
    )
    share = 2.0**-0.5
    out = tq.apply_channel(
        density, tp.stack([share * identity, share * flip_both]), [0, 1]
    )
    # A coin flip on both wires keeps |00> half the time and moves it to |11>.
    tp.testing.assert_close(
        tq.density_probs(out), tp.tensor([[0.5, 0.0, 0.0, 0.5]]), atol=1e-6, rtol=1e-6
    )


# ---------------------------------------------------------------------------
# A Pauli channel written the way a noise model is written
# ---------------------------------------------------------------------------


def _pauli(word):
    """The dense matrix of a Pauli word, for the reference side of a check."""
    piece = {
        "I": tp.eye(2, dtype=tp.complex128),
        "X": tq.x_matrix(c_dtype=tp.complex128),
        "Y": tq.y_matrix(c_dtype=tp.complex128),
        "Z": tq.z_matrix(c_dtype=tp.complex128),
    }
    out = None
    for letter in word:
        out = piece[letter] if out is None else tp.kron(out, piece[letter])
    return out


def _bell_density():
    state = tq.apply_cnot(tq.apply_h(tq.zeros_state(2, dtype=tp.complex128), 0), 0, 1)
    return tq.to_density(state)


def test_named_errors_give_the_mixture_they_describe():
    # A noise model says which errors happen with what chance and leaves the rest
    # to the identity.  The state afterwards is the mixture those chances name,
    # each error applied to the state in front of the sum -- which is worked out
    # here with plain matrix algebra so the two sides are not the same code.
    density = _bell_density()
    out = tq.apply_channel(
        density, tq.pauli_kraus({"XX": 0.02, "ZI": 0.01}, c_dtype=tp.complex128), [0, 1]
    )
    longhand = (
        0.97 * density[0]
        + 0.02 * (_pauli("XX") @ density[0] @ _pauli("XX"))
        + 0.01 * (_pauli("ZI") @ density[0] @ _pauli("ZI"))
    )
    tp.testing.assert_close(out[0], longhand, atol=1e-12, rtol=1e-12)
    assert abs(complex(tp.trace(out[0])).real - 1.0) < 1e-12


def test_an_error_named_on_one_wire_arrives_on_the_whole_register():
    # The operator of "ZI" acts on the first wire only, but it has to reach
    # apply_channel as an operator on both -- otherwise it lands on the wrong
    # one, or on neither.
    density = _bell_density()
    out = tq.apply_channel(
        density, tq.pauli_kraus({"ZI": 0.3}, c_dtype=tp.complex128), [0, 1]
    )
    longhand = 0.7 * density[0] + 0.3 * (_pauli("ZI") @ density[0] @ _pauli("ZI"))
    tp.testing.assert_close(out[0], longhand, atol=1e-12, rtol=1e-12)


def test_an_error_set_can_be_given_as_pairs():
    # A list reads the same as a mapping, which is what a table of chances off a
    # measurement looks like.
    density = _bell_density()
    pairs = [("XX", 0.02), ("ZI", 0.01)]
    tp.testing.assert_close(
        tq.apply_channel(density, tq.pauli_kraus(pairs, c_dtype=tp.complex128), [0, 1]),
        tq.apply_channel(
            density, tq.pauli_kraus(dict(pairs), c_dtype=tp.complex128), [0, 1]
        ),
        atol=1e-12,
        rtol=1e-12,
    )


def test_a_channel_strength_is_read_in_double_precision():
    # An operator built from a written-down chance is only as good as the way that
    # chance was read: in single precision the channel's trace drifts off one by
    # a few parts in ten million, which is a few parts in ten million too many on
    # an expectation.
    density = _bell_density()
    kraus = tq.pauli_kraus({"XX": 0.02, "ZI": 0.01}, c_dtype=tp.complex128)
    out = tq.apply_channel(density, kraus, [0, 1])
    assert abs(complex(tp.trace(out[0])).real - 1.0) < 1e-14


def test_a_named_error_set_is_read_for_what_it_is():
    density = _bell_density()
    with pytest.raises(ValueError):
        tq.pauli_kraus({})                       # no error and no chance
    with pytest.raises(ValueError):
        tq.pauli_kraus({"XX": 0.7, "X": 0.4})    # words of different lengths
    with pytest.raises(ValueError):
        tq.pauli_kraus({"XX": 0.6, "ZI": 0.6})    # more chance than there is
    with pytest.raises(ValueError):
        tq.pauli_kraus({"QQ": 0.1})               # not a Pauli word
    with pytest.raises(ValueError):
        tq.pauli_kraus({"XX": "loud"})            # not a chance
    with pytest.raises(ValueError):
        tq.pauli_kraus([("XX", 0.1), ("YY",)])   # a pair without a chance


# ---------------------------------------------------------------------------
# The same channels over more than one wire
# ---------------------------------------------------------------------------


def test_two_qubit_kraus_sets_have_the_stated_width():
    assert tuple(tq.bit_flip_kraus(0.1, 2).shape) == (2, 4, 4)
    assert tuple(tq.phase_flip_kraus(0.1, 2).shape) == (2, 4, 4)
    assert tuple(tq.amplitude_damping_kraus(0.2, 2).shape) == (4, 4, 4)
    assert tuple(tq.depolarizing_kraus(0.3, 2).shape) == (16, 4, 4)
    assert tuple(tq.depolarizing_kraus(0.3, 3).shape) == (64, 8, 8)
    assert tuple(tq.depolarizing_kraus(0.3, 4).shape) == (256, 16, 16)


def test_two_qubit_kraus_sets_are_complete():
    # sum_r K_r^dagger K_r == I on four wires, for every trace-preserving channel
    identity = tp.eye(4, dtype=tp.complex128)
    for kraus in (
        tq.bit_flip_kraus(0.3, 2, c_dtype=tp.complex128),
        tq.phase_flip_kraus(0.7, 2, c_dtype=tp.complex128),
        tq.amplitude_damping_kraus(0.25, 2, c_dtype=tp.complex128),
        tq.depolarizing_kraus(0.45, 2, c_dtype=tp.complex128),
    ):
        total = None
        for operator in kraus:
            piece = tq.dagger(operator) @ operator
            total = piece if total is None else total + piece
        tp.testing.assert_close(total, identity, atol=1e-9, rtol=1e-9)


def test_depolarizing_on_two_qubits_spreads_evenly():
    # With chance p the channel applies a uniformly drawn non-identity Pauli
    # word.  The sum of every word's conjugation (identity included) takes a
    # trace-one two-qubit matrix to 4 I, so the non-identity share is 4 I - rho
    # and the map comes out closed form.
    p = 0.4
    stack = tq.depolarizing_kraus(p, 2, c_dtype=tp.complex128)
    # a diagonal two-qubit density, which is what a product of populations reads
    rho = tp.tensor(
        [[[0.42, 0.0, 0.0, 0.0],
          [0.0, 0.28, 0.0, 0.0],
          [0.0, 0.0, 0.18, 0.0],
          [0.0, 0.0, 0.0, 0.12]]],
        dtype=tp.complex128,
    )
    applied = tq.apply_channel(rho, stack, [0, 1])
    rho_np = np.array(rho[0].tolist(), dtype=complex)
    want = (1 - p) * rho_np + (p / 15) * (4 * np.eye(4) - rho_np)
    tp.testing.assert_close(applied, tp.tensor(want.reshape(1, 4, 4), dtype=tp.complex128),
                            atol=1e-12, rtol=1e-12)


def test_correlated_bit_flip_flips_both_wires_together():
    # One error on the whole register: either both wires flip or neither does,
    # so the channel is (1-p) rho + p (X X) rho (X X) and no single-wire flip
    # ever appears in the sum.
    p = 0.25
    stack = tq.bit_flip_kraus(p, 2, c_dtype=tp.complex128)
    rho = tp.tensor(
        [[[0.5, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.5]]],
        dtype=tp.complex128,
    )
    applied = tq.apply_channel(rho, stack, [0, 1])
    xx = np.kron(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[0.0, 1.0], [1.0, 0.0]]))
    rho_np = np.array(rho[0].tolist(), dtype=complex)
    want = (1 - p) * rho_np + p * (xx @ rho_np @ xx)
    tp.testing.assert_close(applied, tp.tensor(want.reshape(1, 4, 4), dtype=tp.complex128),
                            atol=1e-12, rtol=1e-12)


def test_independent_amplitude_damping_equals_sequential_wires():
    # The two-qubit damping channel is the tensor product of one single-qubit
    # damping per wire, and channels on different wires commute, so applying it
    # as one 4-operator stack has to agree with damping wire by wire.
    gamma = 0.3
    stack = tq.amplitude_damping_kraus(gamma, 2, c_dtype=tp.complex128)
    rho = tp.tensor(
        [[[0.7, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.3]]],
        dtype=tp.complex128,
    )
    applied = tq.apply_channel(rho, stack, [0, 1])
    sequential = tq.apply_amplitude_damping(tq.apply_amplitude_damping(rho, gamma, 0), gamma, 1)
    tp.testing.assert_close(applied, sequential, atol=1e-12, rtol=1e-12)


def test_multi_qubit_channels_reject_a_bad_width():
    with pytest.raises(ValueError):
        tq.bit_flip_kraus(0.3, 0)
    with pytest.raises(ValueError):
        tq.phase_flip_kraus(0.3, -1)
    with pytest.raises(ValueError):
        tq.depolarizing_kraus(0.3, 5)
    with pytest.raises(ValueError):
        tq.amplitude_damping_kraus(0.3, 2.5)
