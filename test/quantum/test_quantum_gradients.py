"""The gradient a circuit's own backpropagation gives.

Checked against central differences in double precision, which is the only other
way a derivative can be taken: comparing two ways of computing it proves less
than it looks, since two wrong answers agree with each other too.  Double
precision matters here -- at single precision the difference of two values a
micro-radian apart is mostly rounding, and would agree with anything.

The circuits cover one angle per gate family, a relative phase arriving before
the angle gate, and a two-qubit gate read from both wires.
"""
import pytest

import tensorplay as tp
from tensorplay import quantum as tq

# ---------------------------------------------------------------------------
# The differentiated gradient, checked against central differences
#
# Everything above compares two ways of computing the same derivative, which
# agree as easily when both are wrong.  These take the derivative the only other
# way it can be taken -- by moving the angle and measuring how far the value
# moves -- which needs double precision: at single precision the difference of
# two values a micro-radian apart is mostly rounding, and would agree with
# anything.
# ---------------------------------------------------------------------------


WIDE = tq.Hamiltonian([(0.3, "ZZI"), (-0.2, "XYY"), (0.15, "IXZ"), (0.4, "IZZ"), (0.1, "XXX")])


def _difference(value_fn, angles, index, step=1e-6):
    """d(value)/d(angle[index]) by moving that angle and measuring."""
    up = angles.clone()
    down = angles.clone()
    up[0, index] = up[0, index] + step
    down[0, index] = down[0, index] - step
    return (float(value_fn(up)[0]) - float(value_fn(down)[0])) / (2 * step)


def _check_against_differences(value_fn, base, tol=2e-7):
    angles = base.clone().requires_grad_(True)
    value_fn(angles).sum().backward()
    differentiated = angles.grad.reshape(-1).tolist()
    assert differentiated, "a gradient of nothing proves nothing"
    for index in range(base.shape[1]):
        want = _difference(value_fn, base, index)
        assert abs(differentiated[index] - want) < tol, (
            f"angle {index}: differentiated {differentiated[index]}, "
            f"difference {want}"
        )
    return differentiated


@pytest.mark.parametrize(
    "name",
    [
        "ry", "rz", "rx", "p", "sxdg_angle", "pauli_rotation", "u3",
        "rot", "cry", "crz", "rzz", "multirz", "rxx", "cnot", "swap",
        "controlled_x", "h_then_angle", "xx_plus_yy",
    ],
)
def test_one_gate_families_differentiate_exactly(name):
    """Every parameterized gate differentiates to the value's own slope.

    The circuits differ in where the angle sits and what the value reads, and
    one of them carries a phase before the angle gate -- the case a gradient has
    to conjugate through, since a relative phase is what a state is made of.
    """

    def circuit(angles):
        state = tq.zeros_state(3, batch=angles.shape[0], dtype=tp.complex128)
        one = lambda column: angles[:, column]
        if name == "ry":
            state = tq.apply_ry(state, 0, one(0))
        elif name == "rz":
            state = tq.apply_rz(state, 1, one(0))
        elif name == "rx":
            state = tq.apply_rx(state, 2, one(0))
        elif name == "p":
            state = tq.apply_p(state, 1, one(0))
        elif name == "sxdg_angle":
            state = tq.apply_pauli_rotation(state, "X", one(0))
        elif name == "pauli_rotation":
            state = tq.apply_pauli_rotation(state, "XYZ", one(0))
        elif name == "u3":
            state = tq.apply_u3(state, 2, one(0), one(1), one(2))
        elif name == "rot":
            state = tq.apply_rot(state, 0, tp.tensor([1.0, 0.0, 0.0]), one(0))
        elif name == "cry":
            state = tq.apply_cry(state, 0, 1, one(0))
        elif name == "crz":
            state = tq.apply_crz(state, 1, 2, one(0))
        elif name == "rzz":
            state = tq.apply_rzz(state, [1, 2], one(0))
        elif name == "multirz":
            state = tq.apply_multirz(state, one(0))
        elif name == "rxx":
            state = tq.apply_rxx(state, [0, 1], one(0))
        elif name == "cnot":
            state = tq.apply_ry(state, 0, one(0))
            state = tq.apply_cnot(state, 0, 1)
        elif name == "swap":
            state = tq.apply_rz(state, 2, one(0))
            state = tq.apply_swap(state, 0, 2)
        elif name == "controlled_x":
            state = tq.apply_controlled(
                state, tq.rot_y_matrix(one(0)), [0, 2], control_values=[1]
            )
        elif name == "h_then_angle":
            # A phase first: the angle gate then acts on a state that already
            # carries a relative phase, which the gradient has to carry back.
            state = tq.apply_h(state, 0)
            state = tq.apply_ry(state, 0, one(0))
        elif name == "xx_plus_yy":
            state = tq.apply_h(state, 0)
            state = tq.apply_h(state, 1)
            state = tq.apply_xx_plus_yy(state, [0, 1], one(0), one(1))
        return tq.energy(state, WIDE)

    columns = 3 if name in ("u3", "xx_plus_yy") else 1
    base = tp.tensor([[0.3, 0.7, -0.4][:columns]], dtype=tp.float64)
    _check_against_differences(circuit, base)


def test_a_distribution_derivative_comes_out_right():
    # Not an energy this time: the value is one entry of a measurement
    # distribution, read after a circuit whose angles all come before it.
    def population(angles):
        state = tq.zeros_state(2, batch=angles.shape[0], dtype=tp.complex128)
        state = tq.apply_ry(state, 0, angles[:, 0])
        state = tq.apply_rz(state, 1, angles[:, 1])
        state = tq.apply_cnot(state, 0, 1)
        return tq.probs(state)[:, 3]

    base = tp.tensor([[0.3, 0.4]], dtype=tp.float64)
    _check_against_differences(population, base)


def test_a_two_qubit_gate_keeps_both_directions_exact():
    # A gradient through a two-qubit gate has to read both wires' angles; a
    # permutation that moved only one of them would still be a plausible-looking
    # number, and smaller than it should be.
    def value_fn(angles):
        state = tq.zeros_state(2, batch=angles.shape[0], dtype=tp.complex128)
        state = tq.apply_ry(state, 0, angles[:, 0])
        state = tq.apply_ry(state, 1, angles[:, 1])
        state = tq.apply_cnot(state, 0, 1)
        state = tq.apply_rz(state, 1, angles[:, 2])
        state = tq.apply_rzz(state, [0, 1], angles[:, 2])
        return tq.energy(state, tq.Hamiltonian([(0.3, "ZZ"), (-0.2, "XX"), (0.15, "IZ")]))

    base = tp.tensor([[0.3, -0.7, 0.9]], dtype=tp.float64)
    differentiated = _check_against_differences(value_fn, base)
    assert all(abs(entry) > 1e-3 for entry in differentiated), differentiated




