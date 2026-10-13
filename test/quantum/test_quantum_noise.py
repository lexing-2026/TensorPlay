"""Channels on density matrices: semantics, invariants, and composition."""

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
