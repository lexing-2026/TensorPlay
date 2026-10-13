"""Open-system evolution: the master equation, its integrators, and the state path.

Everything here is checked against an operator written out in full -- the dense
reference ``L rho L^dagger - 1/2 {L^dagger L, rho}`` for a single application,
the explicit superoperator exponentiated for a whole evolution, and closed
forms where one exists (a decay rate, a Rabi frequency).  A matrix-free route is
only worth trusting if it agrees with the matrix it replaces.
"""
import math

import pytest

import tensorplay as tp
from tensorplay import quantum as tq
from tensorplay.quantum._dynamics import (
    Lindbladian,
    _series_degree,
    _series_span_limit,
)

WIDE = dict(c_dtype=tp.complex128)


def _wide(matrix, wires, n_qubits):
    """Lift a local operator onto a whole register."""
    return tq.expand_to_register(matrix, wires, n_qubits)


def _dense_action(generator: Lindbladian, density, n_qubits: int):
    """The generator's action, written out term by term over dense matrices."""
    rho = density[0] if density.dim() == 3 else density
    hamiltonian = generator.hamiltonian
    matrix = (
        hamiltonian.register_matrix(**WIDE)
        if isinstance(hamiltonian, tq.Hamiltonian)
        else hamiltonian
    )
    out = -1j * (matrix @ rho - rho @ matrix)
    for rate, matrix, wires in generator._words:
        operator = _wide(matrix, wires, n_qubits)
        out = out + rate * (operator @ rho @ tq.dagger(operator) - rho)
    for rate, operator, gram, wires in generator._operators:
        wide = _wide(operator, wires, n_qubits)
        wide_gram = _wide(gram, wires, n_qubits)
        out = out + rate * (
            wide @ rho @ tq.dagger(wide)
            - 0.5 * (wide_gram @ rho + rho @ wide_gram)
        )
    if generator._scaling is not None:
        axes = [2] * (2 * n_qubits)
        out = out + (rho.reshape(axes) * generator._scaling.reshape(axes)).reshape(
            1 << n_qubits, 1 << n_qubits
        )
    return out


def _superoperator(generator: Lindbladian, n_qubits: int, dtype):
    """The generator as an operator on matrices, one column per basis entry.

    Built by asking the generator what it does to each matrix unit in turn, which
    is the same thing the matrix-free route does -- so it checks the integrator
    and the arithmetic, not the definition of the action itself.
    """
    side = 1 << n_qubits
    basis = tp.eye(side * side, dtype=dtype).reshape(side, side, side, side)
    columns = tp.stack(
        [
            generator.apply(basis[:, :, row, column][None])[0].reshape(-1)
            for row in range(side)
            for column in range(side)
        ],
        dim=-1,
    )
    return columns


# ---------------------------------------------------------------------------
# One application
# ---------------------------------------------------------------------------


def test_the_action_matches_the_written_out_one():
    n_qubits = 2
    hamiltonian = tq.Hamiltonian([(0.7, "XI"), (-0.4, "ZZ"), (0.3, "IY")])
    damping = tq.amplitude_damping_kraus(0.2, **WIDE)[0]
    generator = Lindbladian(hamiltonian, ["XI", ([1], damping)], **WIDE)
    rho = tq.to_density(tq.plus_state(n_qubits, dtype=tp.complex128))
    tp.testing.assert_close(
        generator.apply(rho), _dense_action(generator, rho, n_qubits)[None],
        atol=1e-12, rtol=1e-12,
    )


def test_a_diagonal_jump_is_the_scaling_it_looks_like():
    # A word of identities and Zs has no off-diagonal part, so its dissipator is
    # a factor on each entry rather than a matrix through them.
    n_qubits = 3
    hamiltonian = tq.Hamiltonian([(0.4, "XIZ")])
    rho = tq.to_density(tq.random_state(n_qubits, dtype=tp.complex128))
    wide = _wide(tq.pauli_matrix("ZIZ", **WIDE), [0, 2], n_qubits)
    want = (
        -1j * (hamiltonian.register_matrix(**WIDE) @ rho[0] - rho[0] @ hamiltonian.register_matrix(**WIDE))
        + wide @ rho[0] @ tq.dagger(wide)
        - rho[0]
    )
    got = Lindbladian(hamiltonian, ["ZIZ"], **WIDE).apply(rho)
    tp.testing.assert_close(got, want[None], atol=1e-12, rtol=1e-12)


def test_an_identity_word_moves_nothing():
    # It commutes with everything, and on a density matrix a global phase is
    # not a state at all -- so a term made only of identities has no say here.
    hamiltonian = tq.Hamiltonian([(2.5, "II"), (0.3, "XI")])
    rho = tq.to_density(tq.plus_state(2, dtype=tp.complex128))
    with_phase = Lindbladian(hamiltonian, **WIDE).apply(rho)
    without = Lindbladian(tq.Hamiltonian([(0.3, "XI")]), **WIDE).apply(rho)
    tp.testing.assert_close(with_phase, without, atol=1e-12, rtol=1e-12)


def test_the_action_leaves_the_trace_alone():
    # The trace of any generator's action on a matrix is zero; that is what makes
    # a run trace-preserving rather than merely plausible.
    n_qubits = 3
    generator = Lindbladian(
        tq.Hamiltonian([(0.6, "XII"), (-0.2, "ZZZ")]), ["XII", "IZI"], **WIDE
    )
    rho = tq.to_density(tq.random_state(n_qubits, dtype=tp.complex128))
    action = generator.apply(rho)
    assert abs(complex(tp.trace(action[0]))) < 1e-12


def test_the_action_agrees_on_a_wider_register():
    n_qubits = 5
    hamiltonian = tq.Hamiltonian([(0.4, "XIIZI"), (-0.3, "ZZIII"), (0.2, "IIXYI")])
    generator = Lindbladian(hamiltonian, ["XIIZI", "IZIZI"], **WIDE)
    rho = tq.to_density(tq.random_state(n_qubits, dtype=tp.complex128))
    tp.testing.assert_close(
        generator.apply(rho), _dense_action(generator, rho, n_qubits)[None],
        atol=1e-11, rtol=1e-11,
    )


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------


def _exponential(generator, rho, n_qubits, time, dtype=tp.complex128):
    """``exp(time * G) rho`` from the explicit superoperator."""
    side = 1 << n_qubits
    operator = _superoperator(generator, n_qubits, dtype)
    flat = tp.reshape(rho, [1, -1]).to(dtype)
    return (tp.matrix_exp(time * operator) @ flat.transpose(-1, -2)).transpose(
        -1, -2
    ).reshape(rho.shape)


def test_both_routes_reach_the_exponential():
    n_qubits = 2
    hamiltonian = tq.Hamiltonian([(0.7, "XI"), (-0.4, "ZZ"), (0.3, "IY")])
    damping = tq.amplitude_damping_kraus(0.25, **WIDE)[0]
    generator = Lindbladian(hamiltonian, ["XI", ([1], damping)], **WIDE)
    rho = tq.to_density(tq.plus_state(n_qubits, dtype=tp.complex128))
    for time, method in (
        (0.05, "rk4"),
        (0.5, "rk4"),
        (2.0, "rk4"),
        (0.05, "taylor"),
        (0.5, "taylor"),
        (1.0, "taylor"),
    ):
        want = _exponential(generator, rho, n_qubits, time)
        got = generator.evolve(rho, time, method=method)
        tp.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)


def test_a_run_keeps_the_matrix_a_density_matrix():
    # A density matrix that loses Hermiticity or turns negative has stopped
    # being one, and no observable read off it can be trusted after that.
    n_qubits = 3
    generator = Lindbladian(
        tq.Hamiltonian([(1.1, "XXI"), (0.5, "ZZZ")]), ["XII", "IZI"], **WIDE
    )
    rho = tq.to_density(tq.random_state(n_qubits, dtype=tp.complex128))
    out = generator.evolve(rho, 3.0)
    side = 1 << n_qubits
    flat = out[0].reshape(1, side, side)
    assert float((flat - tq.dagger(flat)).abs().max()) < 1e-8
    assert float(tp.linalg.eigvalsh(flat[0]).min()) > -1e-8
    assert abs(complex(tp.trace(flat[0]).real) - 1.0) < 1e-8


def test_a_phase_error_decays_at_the_rate_it_should():
    # A single Z jump at rate gamma damps a coherence like exp(-2 gamma t): the
    # jump flips the sign of one side of every off-diagonal pair, and the
    # anticommutator halves that on top of the two populations it moves.
    state = tq.plus_state(1, dtype=tp.complex128)
    rho = tq.to_density(state)
    gamma = 0.4
    generator = Lindbladian(None, [("Z", gamma)], **WIDE)
    for time in (0.5, 1.0, 2.0):
        out = generator.evolve(rho, time, method="rk4", n_step=400)
        coherence = complex(out[0, 0, 1])
        assert abs(coherence - math.exp(-2 * gamma * time) * complex(rho[0, 0, 1])) < 1e-8


def test_a_bare_word_is_a_jump_at_unit_rate():
    # An operator carries its rate in its size, so a bare Pauli word means one --
    # which is a relaxation towards the diagonal, not a flip applied at every
    # moment: the coherence of |+> under a unit-rate Z decays as exp(-2t).
    rho = tq.to_density(tq.plus_state(1, dtype=tp.complex128))
    bare = Lindbladian(None, ["Z"], **WIDE).evolve(rho, 0.7, n_step=400)
    scaled = Lindbladian(
        None, [([0], tq.z_matrix(**WIDE))], n_qubits=1, **WIDE
    ).evolve(rho, 0.7, n_step=400)
    tp.testing.assert_close(bare, scaled, atol=1e-12, rtol=1e-12)
    assert abs(complex(bare[0, 0, 1]) - math.exp(-2 * 0.7) * 0.5) < 1e-8
    assert abs(complex(bare[0, 0, 0]) - 0.5) < 1e-8        # populations stay put


def test_a_rate_can_go_with_the_operator_it_belongs_to():
    rho = tq.to_density(tq.plus_state(1, dtype=tp.complex128))
    by_word = Lindbladian(None, [("X", 0.5)], **WIDE).evolve(rho, 0.3)
    half_operator = Lindbladian(
        None, [([0], 0.5 ** 0.5 * tq.x_matrix(**WIDE))], n_qubits=1, **WIDE
    ).evolve(rho, 0.3, n_step=400)
    tp.testing.assert_close(by_word, half_operator, atol=1e-12, rtol=1e-12)


def test_a_channel_can_be_handed_in_as_the_kraus_operators_it_is():
    # A channel is the exponent of a dissipator, and the two do not use the same
    # operators -- which is what the Kraus set is for.  Handed in as it stands,
    # the set must relax the excited population at the rate it damps at.
    rho = tq.to_density(tq.basis_state(1, 1, dtype=tp.complex128))
    gamma = 0.3
    generator = Lindbladian(
        None, [([0], tq.amplitude_damping_kraus(gamma, **WIDE))], n_qubits=1, **WIDE
    )
    for time in (0.5, 2.0, 4.0):
        out = generator.evolve(rho, time, method="rk4", n_step=2000)
        assert abs(complex(out[0, 1, 1]) - math.exp(-gamma * time)) < 1e-7
    # The ground state is a fixed point of the channel.
    settled = tq.to_density(tq.zeros_state(1, dtype=tp.complex128))
    tp.testing.assert_close(
        generator.evolve(settled, 3.0, n_step=200), settled, atol=1e-12, rtol=1e-12
    )


def test_a_channel_generator_matches_the_channel_it_was_given():
    # One step of the generator, taken small, is the channel's own action: the
    # two routes have to agree on the physics, not only on where they end up.
    rho = tq.to_density(tq.plus_state(1, dtype=tp.complex128))
    gamma = 0.3
    kraus = tq.amplitude_damping_kraus(gamma, **WIDE)
    channel_step = tq.apply_channel(rho, kraus, [0])
    generator = Lindbladian(None, [([0], kraus)], n_qubits=1, **WIDE)
    action = generator.apply(rho)
    tp.testing.assert_close(
        action / gamma, (channel_step - rho) / gamma, atol=1e-6, rtol=1e-6
    )


def test_a_single_dissipative_operator_gives_the_same_channel():
    # The same channel written the other way round, as one jump operator plus the
    # identity it leaves over.
    rho = tq.to_density(tq.basis_state(1, 1, dtype=tp.complex128))
    gamma = 0.3
    kraus = tq.amplitude_damping_kraus(gamma, **WIDE)
    residual = tp.eye(2, dtype=tp.complex128) - tq.dagger(kraus[1]) @ kraus[1]
    jumps = [kraus[1], tp.linalg.matrix_sqrth(residual)]
    from_stack = Lindbladian(None, [([0], kraus)], n_qubits=1, **WIDE).apply(rho)
    from_jumps = Lindbladian(None, [([0], j) for j in jumps], n_qubits=1, **WIDE).apply(rho)
    tp.testing.assert_close(from_jumps, from_stack, atol=1e-6, rtol=1e-6)



def test_a_hamiltonian_oscillates_at_the_rate_it_should():
    # Under -i [w X, rho] the expectation of Z of |0> swings as cos(2 w t).
    n_qubits = 1
    rho = tq.to_density(tq.zeros_state(1, dtype=tp.complex128))
    omega = 0.75
    generator = Lindbladian(tq.Hamiltonian([(omega, "X")]), **WIDE)
    for time in (0.3, 1.1, 2.7):
        out = generator.evolve(rho, time, method="rk4", n_step=400)
        # The population swings between 0 and 1 as cos(2 w t) does.
        assert abs(complex(out[0, 0, 0]) - 0.5 * (1.0 + math.cos(2 * omega * time))) < 1e-8
        assert abs(complex(out[0, 0, 1])) > 1e-6      # a superposition, not a fixed point


def test_each_entry_of_a_batch_evolves_on_its_own():
    n_qubits = 3
    hamiltonian = tq.Hamiltonian([(0.8, "XII"), (0.4, "IZZ")])
    generator = Lindbladian(hamiltonian, ["XII"], **WIDE)
    singles = [
        tq.to_density(tq.basis_state(index, n_qubits, dtype=tp.complex128))
        for index in range(4)
    ]
    batched = tp.stack([one[0] for one in singles], dim=0)
    together = generator.evolve(batched, 1.5)
    for index, one in enumerate(singles):
        alone = generator.evolve(one, 1.5)
        tp.testing.assert_close(together[index], alone[0], atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_a_trajectory_reports_the_times_it_was_asked_for():
    n_qubits = 2
    hamiltonian = tq.Hamiltonian([(0.9, "XI")])
    rho = tq.to_density(tq.plus_state(n_qubits, dtype=tp.complex128))
    generator = Lindbladian(hamiltonian, **WIDE)
    for n_sample in (2, 5, 11):
        states, values = generator.trajectory(
            rho, 1.0, n_sample=n_sample, observables=["XI"], n_step=40
        )
        assert states.shape[0] == n_sample
        assert values.shape == (n_sample, 1, 1)
        # The first report is the initial state and the last is the end point.
        tp.testing.assert_close(states[0], rho, atol=0, rtol=0)
        assert abs(complex(values[0, 0, 0]) - 1.0) < 1e-9
        tp.testing.assert_close(
            states[-1], generator.evolve(rho, 1.0, n_step=40), atol=1e-9, rtol=1e-9
        )


def test_observables_are_read_off_the_reported_times():
    n_qubits = 2
    hamiltonian = tq.Hamiltonian([(0.9, "XI")])
    rho = tq.to_density(tq.plus_state(n_qubits, dtype=tp.complex128))
    states, values = Lindbladian(hamiltonian, **WIDE).trajectory(
        rho, 1.0, n_sample=3, observables=[hamiltonian, "ZI"], n_step=40
    )
    assert values.shape == (3, 2, 1)
    for index, matrix in enumerate(states):
        for term, word in enumerate([tq.Hamiltonian([(0.9, "XI")]), "ZI"]):
            want = sum(
                coeff * complex((matrix[0] * _dense_word(letter, n_qubits)).sum())
                for coeff, letter in (word.terms if hasattr(word, "terms") else [(1.0, word)])
            )
            assert abs(float(values[index, term, 0]) - want.real) < 1e-8


def _dense_word(word, n_qubits):
    """A Pauli word as a dense matrix over the whole register."""
    from tensorplay.quantum._circuit import expand_to_register
    from tensorplay.quantum._gates import pauli_matrix
    from tensorplay.quantum._helpers import pauli_wires

    return expand_to_register(pauli_matrix(word, **WIDE), pauli_wires(word), n_qubits)


def test_mesolve_without_samples_answers_with_one_matrix():
    n_qubits = 2
    rho = tq.to_density(tq.plus_state(n_qubits, dtype=tp.complex128))
    out = tq.mesolve(rho, tq.Hamiltonian([(0.5, "XI")]), None, 1.0)
    assert out.shape == rho.shape


def test_mesolve_with_samples_answers_with_the_trajectory():
    n_qubits = 2
    rho = tq.to_density(tq.plus_state(n_qubits, dtype=tp.complex128))
    states, values = tq.mesolve(
        rho, tq.Hamiltonian([(0.5, "XI")]), ["ZI"], 1.0, n_sample=4,
        observables=["ZI"],
    )
    assert states.shape == (4, 1, 4, 4)
    assert values.shape == (4, 1, 1)


# ---------------------------------------------------------------------------
# The closed-system state path
# ---------------------------------------------------------------------------


def test_sesolve_reaches_the_exact_exponential():
    n_qubits = 3
    hamiltonian = tq.Hamiltonian([(0.7, "XII"), (-0.4, "IZZ"), (0.3, "IIY")])
    state = tq.apply_h(tq.zeros_state(n_qubits, dtype=tp.complex128), 0)
    propagator = tp.matrix_exp(-1j * 0.7 * hamiltonian.register_matrix(**WIDE))
    flat = tq.flatten(state)
    want = (propagator @ flat.transpose(-1, -2)).transpose(-1, -2).reshape(state.shape)
    tp.testing.assert_close(
        tq.sesolve(state, hamiltonian, 0.7, method="matrix_exp"),
        want,
        atol=1e-10,
        rtol=1e-10,
    )


def test_the_order_of_a_trotter_product_is_the_order_it_claims():
    # A product that is run forwards only stays first-order however often it is
    # repeated; the palindromic one is second, and the weighted composition of
    # three of those is fourth.  Each is checked by the rate its error falls at,
    # measured over the same span.
    n_qubits = 3
    hamiltonian = tq.Hamiltonian([(0.7, "XII"), (-0.4, "ZIZ"), (0.25, "IIY")])
    state = tq.apply_h(tq.zeros_state(n_qubits, dtype=tp.complex128), 0)
    state = tq.apply_ry(state, 1, 0.5)
    generator = hamiltonian.register_matrix(**WIDE)
    flat = tq.flatten(state)
    want = (
        tp.matrix_exp(-1j * 2.0 * generator) @ flat.transpose(-1, -2)
    ).transpose(-1, -2).reshape(state.shape)

    def error(order, steps):
        current = state
        for _ in range(steps):
            current = tq.trotter_evolve(current, hamiltonian, 2.0 / steps, order=order)
        return float((current - want).abs().max())

    for order, expected in ((1, 1.0), (2, 2.0), (4, 4.0)):
        coarse, fine = error(order, 2), error(order, 8)
        slope = math.log(coarse / fine) / math.log(4.0)
        assert abs(slope - expected) < 0.3, (order, slope)


def test_a_commuting_hamiltonian_needs_no_splitting_at_all():
    # Terms that commute come out exact whatever the order, so the product can be
    # checked against each term's own rotation without a reference exponential
    # to compare through -- whatever error that carries would hide the answer.
    from tensorplay.quantum._gates import pauli_rotation_matrix

    n_qubits = 3
    hamiltonian = tq.Hamiltonian([(0.7, "XII"), (-0.4, "XIZ")])
    state = tq.apply_h(tq.zeros_state(n_qubits, dtype=tp.complex128), 0)
    exact = state
    for coeff, word in hamiltonian.terms:
        rotation = pauli_rotation_matrix(word, 2.0 * 2.0 * coeff, **WIDE)
        exact = tq.apply_gate(exact, rotation, [w for w, c in enumerate(word) if c != "I"])
    # Order 1 applies each term's own rotation once, so it can be checked to the
    # last digit; the symmetric orders build the same rotation out of two halves
    # and come out a rounding different, which is the price of the palindrome.
    tp.testing.assert_close(
        tq.trotter_evolve(state, hamiltonian, 2.0, order=1),
        exact,
        atol=0,
        rtol=0,
    )
    for order in (2, 4):
        tp.testing.assert_close(
            tq.trotter_evolve(state, hamiltonian, 2.0, order=order),
            exact,
            atol=1e-6,
            rtol=1e-6,
        )


def test_an_order_that_is_not_offered_is_refused():
    state = tq.plus_state(2)
    with pytest.raises(ValueError):
        tq.trotter_evolve(state, tq.Hamiltonian([(0.5, "XI")]), 1.0, order=3)


def test_sesolve_takes_a_dense_generator_too():
    n_qubits = 2
    hamiltonian = tq.Hamiltonian([(0.7, "XI"), (-0.4, "ZZ")])
    state = tq.apply_h(tq.zeros_state(n_qubits, dtype=tp.complex128), 0)
    dense = hamiltonian.register_matrix(**WIDE)
    propagator = tp.matrix_exp(-1j * 0.7 * dense)
    flat = tq.flatten(state)
    want = (propagator @ flat.transpose(-1, -2)).transpose(-1, -2).reshape(state.shape)
    tp.testing.assert_close(
        tq.sesolve(state, dense, 0.7), want, atol=1e-10, rtol=1e-10
    )


def test_sesolve_reports_the_times_it_was_asked_for():
    n_qubits = 2
    hamiltonian = tq.Hamiltonian([(0.9, "XI")])
    state = tq.apply_h(tq.zeros_state(n_qubits, dtype=tp.complex128), 0)
    states, values = tq.sesolve(
        state, hamiltonian, 1.0, n_sample=3, observables=["XI"], n_step=4
    )
    assert states.shape == (3, 1, 2, 2)
    assert values.shape == (3, 1, 1)
    assert abs(float(values[0, 0, 0]) - 1.0) < 1e-9


def test_the_state_path_and_the_matrix_path_agree():
    # The same evolution, once on the amplitudes and once on their square: the
    # whole reason for having the state path is that it is the same answer for
    # 2**n times less work.
    n_qubits = 4
    hamiltonian = tq.Hamiltonian([(0.6, "XIIX"), (-0.3, "ZIZI"), (0.25, "IXIY")])
    state = tq.apply_h(tq.zeros_state(n_qubits, dtype=tp.complex128), 0)
    state = tq.apply_ry(state, 2, 0.4)
    through_state = tq.to_density(
        tq.sesolve(state, hamiltonian, 1.0, method="matrix_exp")
    )
    through_matrix = tq.mesolve(tq.to_density(state), hamiltonian, None, 1.0, n_step=64)
    tp.testing.assert_close(through_state, through_matrix, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_the_evolution_is_differentiable_in_the_state_it_starts_from():
    # Reading a coefficient out of a fitted decay curve is what the master
    # equation is for, and that needs the gradient of a reported value.
    real = tp.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True)
    imag = tp.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
    state = tq.unflatten(tp.complex(real, imag).reshape(1, 4), 2)
    rho = tq.to_density(state)
    evolved = tq.mesolve(rho, None, [("ZI", 0.4)], 1.0, n_step=40)
    population = tp.diagonal(evolved[0]).real.sum()
    population.backward()
    # A phase error moves coherences only, so the total population comes back
    # with the state it started from as its gradient.
    assert real.grad is not None
    assert abs(complex(real.grad[0])) > 1e-9


def test_a_reported_observable_is_differentiable():
    n_qubits = 1
    generator = Lindbladian(tq.Hamiltonian([(0.9, "X")]), **WIDE)
    angle = tp.tensor(0.4, requires_grad=True)
    state = tq.apply_ry(tq.zeros_state(1, dtype=tp.complex128), 0, angle)
    _, values = generator.trajectory(
        tq.to_density(state), 1.0, n_sample=2, observables=["X"], n_step=20
    )
    values[1, 0, 0].backward()
    assert angle.grad is not None
    assert abs(float(angle.grad)) > 1e-9


# ---------------------------------------------------------------------------
# What is refused, and why
# ---------------------------------------------------------------------------


def test_an_unknown_method_is_refused():
    rho = tq.to_density(tq.plus_state(2))
    with pytest.raises(ValueError):
        tq.mesolve(rho, tq.Hamiltonian([(0.5, "XI")]), None, 0.1, method="euler")


def test_the_default_method_takes_the_series_where_it_fits():
    # On a span the series can sum it costs less than stepping and lands closer
    # to the exact answer, so the default takes it; past that ceiling the steps
    # are the only route left and the default follows.
    n_qubits = 2
    hamiltonian = tq.Hamiltonian([(0.7, "XI"), (-0.4, "ZZ")])
    rho = tq.to_density(tq.plus_state(n_qubits, dtype=tp.complex128))
    generator = Lindbladian(hamiltonian, **WIDE)
    short = generator.evolve(rho, 0.5)
    tp.testing.assert_close(
        short, generator.evolve(rho, 0.5, method="taylor"), atol=1e-12, rtol=1e-12
    )
    long = generator.evolve(rho, 12.0)
    tp.testing.assert_close(
        long, generator.evolve(rho, 12.0, method="rk4"), atol=1e-6, rtol=1e-6
    )
    with pytest.raises(ValueError):
        tq.sesolve(tq.plus_state(2), tq.Hamiltonian([(0.5, "XI")]), 0.1, method="euler")


def test_asking_for_a_step_two_ways_is_refused():
    rho = tq.to_density(tq.plus_state(2))
    with pytest.raises(ValueError):
        tq.mesolve(rho, tq.Hamiltonian([(0.5, "XI")]), None, 0.1, dt=0.01, n_step=10)


def test_a_negative_number_of_reported_times_is_refused():
    rho = tq.to_density(tq.plus_state(2))
    with pytest.raises(ValueError):
        tq.mesolve(rho, tq.Hamiltonian([(0.5, "XI")]), None, 0.1, n_sample=-2)


def test_a_matrix_of_the_wrong_width_is_refused():
    with pytest.raises(ValueError):
        tq.mesolve(
            tq.to_density(tq.plus_state(2)),
            tq.Hamiltonian([(0.5, "XII")]),
            None,
            0.1,
        )
    with pytest.raises(ValueError):
        Lindbladian(tq.Hamiltonian([(0.5, "XI")])).apply(
            tq.to_density(tq.plus_state(3))
        )


def test_something_that_is_not_a_density_matrix_is_refused():
    with pytest.raises(ValueError):
        tq.mesolve(tq.plus_state(2), tq.Hamiltonian([(0.5, "XI")]), None, 0.1)
    odd = tp.zeros(1, 6, 6, dtype=tp.complex64)
    with pytest.raises(ValueError):
        tq.mesolve(odd, tq.Hamiltonian([(0.5, "XI")]), None, 0.1)


def test_a_jump_operator_wider_than_the_register_is_refused():
    with pytest.raises(ValueError):
        Lindbladian(tq.Hamiltonian([(0.5, "XI")]), ["XXXI"])


def test_a_jump_operator_that_is_not_an_operator_is_refused():
    hamiltonian = tq.Hamiltonian([(0.5, "XI")])
    with pytest.raises(ValueError):
        Lindbladian(hamiltonian, [1.0])
    with pytest.raises(ValueError):
        Lindbladian(hamiltonian, [([0], tp.rand(2, 3))])
    with pytest.raises(ValueError):
        Lindbladian(hamiltonian, [("XI", -1.0)])
    with pytest.raises(ValueError):
        Lindbladian(hamiltonian, [("XI", "fast")])
    with pytest.raises(ValueError):
        Lindbladian(hamiltonian, [([0, 1], tq.h_matrix())])


def test_a_hamiltonian_that_is_not_one_is_refused():
    with pytest.raises(ValueError):
        Lindbladian("ZZ")
    with pytest.raises(ValueError):
        Lindbladian(tp.zeros(4, 5, dtype=tp.complex64))
    with pytest.raises(ValueError):
        Lindbladian(tq.random_unitary(4).reshape(-1))


def test_an_observable_that_is_not_one_is_refused():
    rho = tq.to_density(tq.plus_state(2))
    with pytest.raises(ValueError):
        tq.mesolve(rho, tq.Hamiltonian([(0.5, "XI")]), None, 0.1, n_sample=2, observables=[3.0])


def test_sesolve_sends_a_density_matrix_to_mesolve():
    rho = tq.to_density(tq.plus_state(2))
    with pytest.raises(ValueError):
        tq.sesolve(rho, tq.Hamiltonian([(0.5, "XI")]), 0.1)


def test_a_series_is_refused_past_the_span_precision_carries():
    rho = tq.to_density(tq.plus_state(2))
    with pytest.raises(ValueError):
        tq.mesolve(
            rho, tq.Hamiltonian([(5.0, "XI")]), None, 40.0, method="taylor"
        )
    # The same run is fine on the step-by-step route, which never sums terms that
    # large in one go.
    assert tq.mesolve(rho, tq.Hamiltonian([(5.0, "XI")]), None, 40.0, n_step=400).shape == rho.shape


def test_a_time_dependent_generator_needs_the_step_route():
    rho = tq.to_density(tq.plus_state(2))
    drive = lambda t: tq.Hamiltonian([(math.cos(t), "XI")])
    with pytest.raises(ValueError):
        tq.mesolve(rho, drive, None, 0.5, method="taylor")
    out = tq.mesolve(rho, drive, None, 0.5, n_step=100)
    assert out.shape == rho.shape


def test_a_time_dependent_generator_reads_its_terms_as_it_goes():
    # A drive that turns on and off has to be re-read at every stage of a step.
    rho = tq.to_density(tq.plus_state(1, dtype=tp.complex128))
    drive = lambda t: tq.Hamiltonian([(math.cos(2.0 * t), "X")])
    generator = Lindbladian(drive)
    assert generator.n_qubits is None
    out = generator.evolve(rho, 1.0, n_step=200)
    assert abs(complex(tp.trace(out[0]).real) - 1.0) < 1e-6


def test_the_series_degree_grows_with_the_span():
    assert _series_degree(1.0) < _series_degree(5.0) < _series_degree(20.0)
    assert _series_span_limit(tp.complex128) > _series_span_limit(tp.complex64)


def test_the_generator_says_what_it_holds():
    generator = Lindbladian(tq.Hamiltonian([(0.5, "XI"), (0.2, "IZ")]), ["ZI"], **WIDE)
    assert "n_qubits=2" in repr(generator)
    assert "jumps=1" in repr(generator)
    assert "time-dependent" in repr(Lindbladian(lambda t: None))
