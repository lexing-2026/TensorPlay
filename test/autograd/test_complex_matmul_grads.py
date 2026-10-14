"""Gradients of the matrix family when the operands are complex.

A complex value is differentiated against the adjoint of the other operand.
Every backward written down for a real operand says the same thing and differs
by that one conjugation, so nothing about a complex gradient looks wrong: the
value is exact, the gradient is the right size and a plausible number, and it is
wrong by an amount that depends on the phase the operand carries.

Each check builds a one-parameter unitary -- so every gradient has a parameter
to read -- and compares the differentiated gradient with a central difference.
Double precision is what makes the difference meaningful: at single precision
the two values a micro-radian apart differ mostly in their rounding.
"""
import pytest

import tensorplay as tp

CDTYPE = tp.complex128
ANGLE = 0.7
STEP = 1e-6
COLUMN = tp.tensor([[0.9 + 0.1j], [0.2 - 0.5j]], dtype=CDTYPE)
SQUARE = tp.tensor([[0.9 + 0.1j, 0.3 - 0.2j], [0.2 - 0.5j, 0.7 + 0.4j]], dtype=CDTYPE)
ROW = tp.tensor([[0.6 + 0.3j, -0.4 + 0.7j]], dtype=CDTYPE)
BIAS = tp.tensor([0.2 + 0.1j, -0.3 + 0.5j], dtype=CDTYPE)
FOUR = tp.tensor([0.3 - 0.2j, 0.7 + 0.4j, -0.1 + 0.5j, 0.2 + 0.6j], dtype=CDTYPE)


def _rotation(angle):
    """``exp(-i angle X / 2)``, written out so the angle is the only parameter.

    One angle gives one matrix and several give one matrix each, which is the
    shape every product below wants.
    """
    flat = tp.reshape(angle, [-1])
    cosine = tp.cos(flat / 2) + 0j
    sine = -1j * tp.sin(flat / 2)
    stacked = tp.stack(
        [tp.stack([cosine, sine], dim=-1), tp.stack([-sine, cosine], dim=-1)], dim=-2
    )
    return stacked[0] if stacked.shape[0] == 1 else stacked


def _batched(matrix):
    return matrix.unsqueeze(0).expand(2, *matrix.shape)


def _squared(value):
    """A real value of a complex one: the squared modulus, which is what an
    energy or a loss actually is."""
    return value.abs().square().sum()


def _differentiates(value_fn):
    """The value's own slope, by differencing it and by differentiating it."""
    angle = tp.tensor([[ANGLE]], dtype=tp.float64, requires_grad=True)
    _squared(value_fn(angle)).backward()
    differentiated = angle.grad[0, 0]
    higher = _squared(value_fn(tp.tensor([[ANGLE + STEP]], dtype=tp.float64)))
    lower = _squared(value_fn(tp.tensor([[ANGLE - STEP]], dtype=tp.float64)))
    difference = (higher - lower) / (2 * STEP)
    # The gradient of a real parameter comes back carrying the convention's
    # conjugate side, so what has to match is the value it carries.
    return float(tp.real(differentiated)), float(difference)


@pytest.mark.parametrize(
    "name, value_fn",
    [
        ("matmul", lambda a: _rotation(a) @ SQUARE),
        ("mm", lambda a: _rotation(a) @ COLUMN),
        ("mv", lambda a: _rotation(a) @ tp.tensor([0.9 + 0.1j, 0.2 - 0.5j], dtype=CDTYPE)),
        ("addmm", lambda a: tp.addmm(BIAS, _rotation(a), SQUARE)),
        ("bmm", lambda a: tp.bmm(_batched(_rotation(a)), _batched(COLUMN))),
        ("baddbmm", lambda a: tp.baddbmm(
            tp.zeros(2, 2, 1, dtype=CDTYPE), _batched(_rotation(a)), _batched(COLUMN))),
        ("multi_dot", lambda a: tp.linalg.multi_dot([_rotation(a), SQUARE])),
        ("dot", lambda a: tp.dot(_rotation(a).reshape(-1), FOUR)),
        ("outer", lambda a: tp.outer(FOUR[:2], FOUR[:2]) * _rotation(a)),
    ],
)
def test_a_matrix_product_differentiates_a_complex_operand(name, value_fn):
    differentiated, difference = _differentiates(value_fn)
    assert abs(differentiated - difference) < 1e-6, (
        f"{name}: differentiated {differentiated}, difference {difference}"
    )


def test_a_gradient_taken_from_a_real_part_does_too():
    # A real part is real-linear, which is a weaker demand than the squared
    # modulus above and covers the branch of the backward the other does not.
    def value(angle):
        return (_rotation(angle) @ SQUARE).real

    angle = tp.tensor([[ANGLE]], dtype=tp.float64, requires_grad=True)
    value(angle).sum().backward()
    differentiated = float(tp.real(angle.grad[0, 0]))
    difference = float(
        (value(tp.tensor([[ANGLE + STEP]], dtype=tp.float64)).sum()
         - value(tp.tensor([[ANGLE - STEP]], dtype=tp.float64)).sum()) / (2 * STEP)
    )
    assert abs(differentiated - difference) < 1e-6


def test_each_entry_of_a_batch_gets_its_own_gradient():
    # A gradient summed over the batch would still be the right size and a
    # plausible number; a value taken one entry at a time tells the two apart.
    angles = tp.tensor([[0.4], [1.3]], dtype=tp.float64, requires_grad=True)
    _squared(tp.bmm(_rotation(angles), _batched(COLUMN))).backward()
    assert angles.grad.shape == angles.shape

    for index in range(2):
        one = tp.tensor([[float(angles[index, 0])]], dtype=tp.float64, requires_grad=True)
        _squared(tp.bmm(_rotation(one).unsqueeze(0), _batched(COLUMN)[:1])).backward()
        got = float(tp.real(one.grad[0, 0]))
        assert abs(got - float(tp.real(angles.grad[index, 0]))) < 1e-6, index
