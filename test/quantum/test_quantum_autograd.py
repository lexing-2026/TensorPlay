"""Gradients through gates, circuits, expectation values, and evolution."""
import math


import tensorplay as tp
from tensorplay import quantum as tq

H2 = [(-1.0523732, "II"), (0.3979374, "IZ"), (-0.3979374, "ZI"), (-0.1809312, "XX")]


def _flat(state):
    return state.reshape(state.shape[0], -1)


def test_gate_parameters_carry_gradients():
    theta = tp.tensor(0.4, requires_grad=True)
    state = tq.zeros_state(1)
    out = tq.apply_ry(state, 0, theta)
    out.real.sum().backward()
    assert theta.grad is not None
    assert abs(float(theta.grad)) > 0


def test_probability_of_a_rotation_matches_the_analytic_derivative():
    theta = tp.tensor(0.7, requires_grad=True)
    state = tq.zeros_state(1)
    out = tq.apply_ry(state, 0, theta)
    tq.probs(out)[0, 1].backward()
    # d/dtheta sin^2(theta/2) = sin(theta)/2
    expected = math.sin(0.7) / 2.0
    tp.testing.assert_close(theta.grad, tp.tensor(expected), atol=1e-6, rtol=1e-5)


def test_finite_difference_agrees_with_the_rotation_gradient():
    theta = tp.tensor(0.9, requires_grad=True)
    state = tq.zeros_state(1)
    hamiltonian = tq.Hamiltonian([(1.0, "Z")])
    tq.energy(tq.apply_ry(state, 0, theta), hamiltonian).backward()
    analytic = float(theta.grad)

    # The difference is taken in double precision: a step of 1e-4 on a
    # single-precision value leaves too little signal to compare against.
    wide = tq.zeros_state(1, dtype=tp.complex128)
    step = 1e-4
    with tp.no_grad():
        up = tq.energy(
            tq.apply_ry(wide, 0, tp.tensor(0.9 + step, dtype=tp.float64)), hamiltonian
        )
        down = tq.energy(
            tq.apply_ry(wide, 0, tp.tensor(0.9 - step, dtype=tp.float64)), hamiltonian
        )
    numeric = float((up - down) / (2 * step))
    tp.testing.assert_close(
        tp.tensor(analytic), tp.tensor(numeric), atol=1e-5, rtol=1e-4
    )


def test_expectation_is_differentiable_in_the_rotation_angle():
    theta = tp.tensor(0.25, requires_grad=True)
    state = tq.zeros_state(1)
    tq.expectation(tq.apply_ry(state, 0, theta), "Z").backward()
    # A rotation about Y carries |0> toward |1>, so <Z> reads cos(theta).
    tp.testing.assert_close(theta.grad, tp.tensor(-math.sin(0.25)), atol=1e-6, rtol=1e-5)


def test_circuit_parameters_receive_gradients_through_its_own_list():
    circuit = tq.Circuit(1)
    theta = tp.tensor(0.3, requires_grad=True)
    circuit.ry(0, theta)
    tq.probs(circuit())[0, 1].backward()
    assert theta.grad is not None
    assert abs(float(theta.grad)) > 0


def test_vqe_style_gradient_flows_into_every_parameter():
    angles = tp.tensor([0.2, -0.4, 1.1], requires_grad=True)
    circuit = tq.Circuit(2)
    circuit.ry(0, angles[0])
    circuit.ry(1, angles[1])
    circuit.rx(0, angles[2])
    circuit.cx(0, 1)
    loss = tq.energy(circuit(), tq.Hamiltonian(H2)).sum()
    loss.backward()
    assert angles.grad is not None
    tp.testing.assert_close(angles.grad.shape, tp.Size([3]))
    assert float(angles.grad.abs().sum()) > 0


def test_a_registered_parameter_reaches_its_optimizer_slot():
    circuit = tq.Circuit(1)
    angle = tp.nn.Parameter(tp.tensor(0.4))
    circuit.rx(0, angle)
    optimizer = tp.optim.SGD(circuit.parameters(), lr=0.1)
    loss = tq.expectation(circuit(), "Z").sum()
    loss.backward()
    optimizer.step()
    assert not bool(tp.allclose(angle.detach(), tp.tensor(0.4)))


def test_gradient_of_the_energy_is_finite_and_smooth():
    theta = tp.tensor(0.0, requires_grad=True)
    state = tq.zeros_state(2)
    hamiltonian = tq.Hamiltonian(H2)
    energy = tq.energy(tq.apply_rx(tq.apply_ry(tq.apply_ry(state, 0, theta), 1, 0.4), 0, 0.3), hamiltonian)
    energy.backward()
    assert math.isfinite(float(theta.grad))


def test_trotter_evolution_is_differentiable_in_time():
    time = tp.tensor(0.3, requires_grad=True)
    state = tq.zeros_state(1)
    hamiltonian = tq.Hamiltonian([(0.7, "Z")])
    tq.trotter_evolve(state, hamiltonian, time).real.sum().backward()
    assert time.grad is not None
    assert abs(float(time.grad)) > 0


def test_trotter_gradient_matches_finite_differences():
    time = tp.tensor(0.45, requires_grad=True)
    state = tq.zeros_state(1)
    hamiltonian = tq.Hamiltonian([(1.3, "Z")])
    loss = tq.probs(tq.trotter_evolve(state, hamiltonian, time)).sum(dim=-1)
    (loss * tp.tensor([1.0, -1.0])).sum().backward()
    analytic = float(time.grad)

    step = 1e-4
    with tp.no_grad():
        up = tq.probs(tq.trotter_evolve(state, hamiltonian, time.detach() + step)).sum(dim=-1)
        down = tq.probs(tq.trotter_evolve(state, hamiltonian, time.detach() - step)).sum(dim=-1)
    weights = tp.tensor([1.0, -1.0])
    numeric = float(((up - down) * weights).sum() / (2 * step))
    tp.testing.assert_close(tp.tensor(analytic), tp.tensor(numeric), atol=1e-5, rtol=1e-4)


def test_density_matrix_stays_differentiable_through_a_channel():
    p = tp.tensor(0.2, requires_grad=True)
    density = tq.to_density(tq.basis_state(1, 1))
    out = tq.apply_bit_flip(density, p, 0)
    tq.density_probs(out).sum().backward()
    assert p.grad is not None


def test_no_grad_mode_freezes_the_parameters():
    theta = tp.tensor(0.5, requires_grad=True)
    state = tq.zeros_state(1)
    with tp.no_grad():
        out = tq.apply_ry(state, 0, theta)
    assert not out.requires_grad
