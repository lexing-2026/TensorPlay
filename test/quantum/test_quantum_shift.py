"""Gradients taken by displacing the angle, and values read off a finite number
of shots.

Both are checked against something independent: the shift rule against the
gradient the framework differentiates on its own, and the sampled value against
the exact one -- with the shot count raised until the two agree, which is the
only way to tell noise from a wrong answer.
"""
import math
import statistics

import pytest

import tensorplay as tp
from tensorplay import quantum as tq

H2 = tq.Hamiltonian(
    [(-1.0523732, "II"), (0.3979374, "IZ"), (-0.3979374, "ZI"), (-0.1809312, "XX")]
)


def _ansatz(angles):
    """A two-qubit circuit with one angle per qubit, ``[batch, 2]``."""
    state = tq.zeros_state(2, batch=angles.shape[0])
    state = tq.apply_ry(state, 0, angles[:, 0])
    state = tq.apply_ry(state, 1, angles[:, 1])
    return tq.apply_cnot(state, 0, 1)


def _energy(angles):
    return tq.energy(_ansatz(angles), H2)


def _single(angles):
    """The same circuit for one set of angles, shaped ``[2]``."""
    state = tq.zeros_state(2)
    state = tq.apply_ry(state, 0, angles[0])
    state = tq.apply_ry(state, 1, angles[1])
    return tq.apply_cnot(state, 0, 1)


def test_the_shifted_gradient_matches_the_differentiated_one():
    angles = tp.tensor([[0.2, -0.4], [0.35, 0.15]], requires_grad=True)
    value = _energy(angles)
    value.sum().backward()
    differentiated = angles.grad.reshape(-1).tolist()
    shifted = tq.param_shift(_energy, angles).reshape(-1).tolist()
    for got, want in zip(shifted, differentiated):
        assert abs(got - want) < 1e-5


def test_a_deeper_circuit_agrees_too():
    # The rule does not care how deep the circuit is, which is the point of it:
    # nothing about the value is differentiated, only displaced.  Each angle
    # enters through one rotation of its own, which is what the rule asks for --
    # an angle fed to several gates is not one exponential and has no single
    # displacement that reaches it.
    def energy(angles):
        state = tq.zeros_state(2, batch=angles.shape[0])
        for layer in range(6):
            state = tq.apply_ry(state, 0, angles[:, 2 * layer])
            state = tq.apply_rz(state, 1, angles[:, 2 * layer + 1])
            state = tq.apply_cnot(state, 0, 1)
        return tq.energy(state, H2)

    angles = tp.tensor(
        [[0.3, 1.1, -0.2, 0.5, 0.9, -1.3, 0.15, 0.75, -0.6, 0.4, 1.0, 0.05]],
        requires_grad=True,
    )
    energy(angles).sum().backward()
    differentiated = angles.grad.reshape(-1).tolist()
    shifted = tq.param_shift(energy, angles).reshape(-1).tolist()
    assert shifted, "a gradient of nothing proves nothing"
    for got, want in zip(shifted, differentiated):
        assert abs(got - want) < 1e-5


def test_the_rule_follows_the_generator_it_is_given():
    # A rotation by exp(-i theta X) puts the value at a frequency of four rather
    # than one, so the displacement has to be a quarter of the default and the
    # weight twice it: the difference between two displaced values then scales
    # the derivative exactly, which is what the general form says.
    def population(angles):
        state = tq.apply_pauli_rotation(tq.zeros_state(1, batch=angles.shape[0]), "X", 4.0 * angles)
        return tq.energy(state, tq.Hamiltonian([(1.0, "Z")]))

    angles = tp.tensor([[0.37], [1.1]], requires_grad=True)
    population(angles).sum().backward()
    differentiated = angles.grad.reshape(-1).tolist()
    shifted = tq.param_shift(
        population, angles, shift=math.pi / 8.0, weight=2.0
    ).reshape(-1).tolist()
    for got, want in zip(shifted, differentiated):
        assert abs(got - want) < 1e-4


def test_the_rule_does_not_reach_every_angle():
    # A controlled rotation is not one exponential: where the control is not
    # set its generator reads zero, so the value picks up half the frequency as
    # well as the whole one, and a single displacement cannot reach both.  The
    # rule says so rather than quietly returning half a gradient -- the
    # differentiated gradient covers these angles, and the two differ here by
    # far more than rounding.
    def energy(angles):
        state = tq.zeros_state(2, batch=angles.shape[0], dtype=tp.complex128)
        state = tq.apply_h(state, 0)
        state = tq.apply_cry(state, 0, 1, angles[:, 0])
        return tq.energy(state, tq.Hamiltonian([(1.0, "XI"), (0.5, "ZZ")]))

    base = tp.tensor([[0.7]], dtype=tp.float64)
    angles = base.clone().requires_grad_(True)
    energy(angles).sum().backward()
    differentiated = float(angles.grad[0, 0])
    shifted = float(tq.param_shift(energy, base)[0, 0])
    assert abs(differentiated) > 1e-3, differentiated
    assert abs(differentiated - shifted) > 1e-3, (differentiated, shifted)


def test_the_rule_reaches_a_value_that_cannot_be_differentiated():
    # The value here is a sample: no graph, nothing to backpropagate through.
    def sampled(angles):
        return tq.sample_energy(_ansatz(angles), H2, 20000)

    angles = tp.tensor([[0.2, -0.4]])
    gradients = tq.param_shift(sampled, angles)
    assert gradients.shape == angles.shape
    assert abs(float(gradients[0, 0]) - (-0.171084)) < 0.02
    assert abs(float(gradients[0, 1]) - 0.151875) < 0.02


def test_the_shot_noise_shrinks_as_the_shots_grow():
    # A wrong rule would not converge; noise does.
    state = _ansatz(tp.tensor([[0.2, -0.4]]))
    exact = float(tq.energy(state, H2)[0])
    errors = []
    for shots in (200, 4000):
        estimates = [float(tq.sample_energy(state, H2, shots)[0]) for _ in range(40)]
        errors.append(statistics.fmean(abs(value - exact) for value in estimates))
    assert errors[1] < errors[0] / 3.0


def test_the_sampled_value_falls_on_the_exact_one():
    state = _ansatz(tp.tensor([[0.2, -0.4], [0.9, 0.1]]))
    exact = tq.energy(state, H2)
    sampled = tq.sample_energy(state, H2, 40000)
    assert sampled.shape == exact.shape
    tp.testing.assert_close(sampled, exact, atol=2e-2, rtol=0)


def test_a_readout_that_lies_shows_up_in_the_estimate():
    # Half the shots reported upside down drags the estimate toward zero, which
    # is what a miscalibrated readout does and what the estimate has to carry.
    state = _ansatz(tp.tensor([[0.2, -0.4]]))
    honest = statistics.fmean(
        float(tq.sample_energy(state, H2, 8000)[0]) for _ in range(40)
    )
    lying = statistics.fmean(
        float(tq.sample_energy(state, H2, 8000, readout=[[0.1, 0.9], [0.1, 0.9]])[0])
        for _ in range(40)
    )
    exact = float(tq.energy(state, H2)[0])
    assert abs(honest - exact) < 0.05
    # Half the readings upside down lands a long way from the exact energy,
    # whichever way it falls: the estimate has to carry that.
    assert abs(lying - exact) > 5.0 * abs(honest - exact)


def _single_energy(angles):
    return tq.energy(_single(angles), H2)


def test_a_single_parameter_set_is_read_without_a_batch():
    angles = tp.tensor([0.2, -0.4], requires_grad=True)
    value = _single_energy(angles)
    value.backward()
    shifted = tq.param_shift(lambda a: _single_energy(a).reshape(-1)[0], angles)
    assert shifted.shape == angles.shape
    for got, want in zip(shifted.tolist(), angles.grad.tolist()):
        assert abs(got - want) < 1e-5


def test_a_parameter_axis_can_be_named():
    # The angles can sit anywhere in the tensor, which is what makes a circuit
    # with several parameter blocks usable without transposing them into place.
    def energy(block):
        angles = block[:, 2:4]
        return tq.energy(_ansatz(angles), H2)

    block = tp.tensor([[9.0, 9.0, 0.2, -0.4]], requires_grad=True)
    energy(block).sum().backward()
    differentiated = block.grad.tolist()[0]
    shifted = tq.param_shift(energy, block, axis=-1).tolist()[0]
    for index in (0, 1):
        assert abs(shifted[index] - differentiated[index]) < 1e-5
    for index in (0, 1):
        assert abs(shifted[index + 2] - differentiated[index + 2]) < 1e-5


def test_a_displacement_of_zero_is_refused():
    angles = tp.tensor([0.2, -0.4])
    with pytest.raises(ValueError):
        tq.param_shift(_energy, angles, shift=0.0)
    with pytest.raises(ValueError):
        tq.param_shift(_energy, angles, weight=0.0)


def test_an_axis_that_is_not_there_is_refused():
    angles = tp.tensor([0.2, -0.4])
    with pytest.raises(ValueError):
        tq.param_shift(_energy, angles, axis=3)
    with pytest.raises(ValueError):
        tq.param_shift(_energy, tp.tensor(0.2))


def test_a_value_that_keeps_an_extra_axis_is_refused():
    # The value is expected with the parameter axis taken out, so a value that
    # still carries a leading entry of one is a mistake worth naming rather than
    # a gradient quietly shaped wrong.
    with pytest.raises(ValueError):
        tq.param_shift(_single_energy, tp.tensor([0.2, -0.4]))


def _single_energy(angles):
    return tq.energy(_single(angles), H2)


def test_a_single_parameter_set_is_read_without_a_batch():
    angles = tp.tensor([0.2, -0.4], requires_grad=True)
    value = _single_energy(angles)
    value.backward()
    shifted = tq.param_shift(lambda a: _single_energy(a).reshape(-1)[0], angles)
    assert shifted.shape == angles.shape
    for got, want in zip(shifted.tolist(), angles.grad.tolist()):
        assert abs(got - want) < 1e-5


def test_a_parameter_axis_can_be_named():
    # The angles can sit anywhere in the tensor, which is what makes a circuit
    # with several parameter blocks usable without transposing them into place.
    def energy(block):
        angles = block[:, 2:4]
        return tq.energy(_ansatz(angles), H2)

    block = tp.tensor([[9.0, 9.0, 0.2, -0.4]], requires_grad=True)
    energy(block).sum().backward()
    differentiated = block.grad.tolist()[0]
    shifted = tq.param_shift(energy, block, axis=-1).tolist()[0]
    for index in (0, 1):
        assert abs(shifted[index] - differentiated[index]) < 1e-5
    for index in (0, 1):
        assert abs(shifted[index + 2] - differentiated[index + 2]) < 1e-5


def test_a_displacement_of_zero_is_refused():
    angles = tp.tensor([0.2, -0.4])
    with pytest.raises(ValueError):
        tq.param_shift(_energy, angles, shift=0.0)
    with pytest.raises(ValueError):
        tq.param_shift(_energy, angles, weight=0.0)


def test_an_axis_that_is_not_there_is_refused():
    angles = tp.tensor([0.2, -0.4])
    with pytest.raises(ValueError):
        tq.param_shift(_energy, angles, axis=3)
    with pytest.raises(ValueError):
        tq.param_shift(_energy, tp.tensor(0.2))


def test_a_value_that_keeps_an_extra_axis_is_refused():
    # The value is expected with the parameter axis taken out, so a value that
    # still carries a leading entry of one is a mistake worth naming rather than
    # a gradient quietly shaped wrong.
    with pytest.raises(ValueError):
        tq.param_shift(_single_energy, tp.tensor([0.2, -0.4]))


def test_a_parameter_block_of_one_column_reads_as_one_angle():
    # A circuit with one angle per qubit keeps its angles as [batch, n_qubits],
    # so a one-qubit circuit hands a [batch, 1] block to its gate.  That has to
    # come out as one angle per batch entry rather than a matrix with an extra
    # axis that quietly keeps the same angle for every entry.
    def population(block):
        state = tq.apply_pauli_rotation(tq.zeros_state(1, batch=block.shape[0]), "X", block)
        return tq.energy(state, tq.Hamiltonian([(1.0, "Z")]))

    block = tp.tensor([[0.37], [1.1]])
    together = population(block)
    apart = tp.stack([population(tp.tensor([[th]])) for th in (0.37, 1.1)])
    tp.testing.assert_close(together, apart.reshape(-1), atol=1e-6, rtol=1e-6)

    angles = tp.tensor([[0.37], [1.1]], requires_grad=True)
    population(angles).sum().backward()
    for got, want in zip(
        tq.param_shift(population, angles).reshape(-1).tolist(),
        angles.grad.reshape(-1).tolist(),
    ):
        assert abs(got - want) < 1e-5
