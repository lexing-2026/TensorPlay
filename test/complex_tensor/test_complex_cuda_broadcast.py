"""Complex arithmetic on the device between operands of different extents.

An operand with a single element is read again for every element of the
other one.  The result has exactly as many elements as the larger operand, so
nothing beyond them may change: not the rest of the buffer the result is a
part of, and not another tensor that happens to sit next to it in memory.
Every expected value is worked out by hand from the small integers below.
"""
import cmath

import pytest

import tensorplay as tp

pytestmark = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="needs the device"
)

VALUES = [1 + 5j, 2 + 6j, 3 + 7j, 4 + 8j]
# Longer than one launch block, so a write past the result lands inside it.
LENGTH = 300


def _buffer():
    return tp.tensor(
        [complex(k + 1, -(k + 1)) for k in range(LENGTH)],
        dtype=tp.complex64, device="cuda",
    )


def _untouched_tail():
    return [complex(k + 1, -(k + 1)) for k in range(len(VALUES), LENGTH)]


def _values(device="cuda"):
    return tp.tensor(VALUES, dtype=tp.complex64, device=device)


def _single(value):
    return tp.tensor([value], dtype=tp.complex64, device="cuda")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("mul_", [complex(2, -2), complex(4, -4), complex(6, -6), complex(8, -8)]),
        ("add_", [complex(3, -1), complex(4, -2), complex(5, -3), complex(6, -4)]),
        ("sub_", [complex(-1, -1), complex(0, -2), complex(1, -3), complex(2, -4)]),
        ("div_", [complex(0.5, -0.5), complex(1, -1), complex(1.5, -1.5), complex(2, -2)]),
    ],
)
def test_writing_into_the_head_of_a_buffer_leaves_its_tail(name, expected):
    buffer = _buffer()
    head = buffer[: len(VALUES)]
    getattr(head, name)(_single(2 + 0j))
    got = buffer.cpu().tolist()
    assert got[: len(VALUES)] == expected
    assert got[len(VALUES):] == _untouched_tail()


def test_a_result_placed_in_the_head_of_a_buffer_leaves_its_tail():
    buffer = _buffer()
    tp.add(_single(1 + 0j), _values(), out=buffer[: len(VALUES)])
    got = buffer.cpu().tolist()
    assert got[: len(VALUES)] == [2 + 5j, 3 + 6j, 4 + 7j, 5 + 8j]
    assert got[len(VALUES):] == _untouched_tail()


@pytest.mark.parametrize(
    "op, expected",
    [
        (lambda one, z: one * z, VALUES),
        (lambda one, z: z * one, VALUES),
        (lambda one, z: one + z, [2 + 5j, 3 + 6j, 4 + 7j, 5 + 8j]),
        (lambda one, z: z - one, [0 + 5j, 1 + 6j, 2 + 7j, 3 + 8j]),
        (lambda one, z: z / one, VALUES),
    ],
)
def test_a_fresh_result_leaves_the_tensors_around_it(op, expected):
    # Small tensors are handed out side by side, so a result that ran past its
    # last element would show up in one of its neighbours.
    for _ in range(6):
        neighbours = [_values() for _ in range(8)]
        out = op(_single(1 + 0j).expand(len(VALUES)), neighbours[3])
        assert out.cpu().tolist() == expected
        for neighbour in neighbours:
            assert neighbour.cpu().tolist() == VALUES
        del neighbours, out


def test_gradient_of_a_square_is_twice_the_conjugate():
    # Both factors hand their gradient to the same leaf, each one the incoming
    # ones read again for every element times the conjugate of the other.
    for _ in range(6):
        z = _values()
        z.requires_grad_(True)
        (z * z).sum().backward()
        assert z.grad.cpu().tolist() == [2 - 10j, 4 - 12j, 6 - 14j, 8 - 16j]
        assert z.detach().cpu().tolist() == VALUES


def test_gradient_through_exp_times_itself():
    points = [0.25 + 0.5j, -0.5 + 0.125j, 0.75 - 0.25j, 0.125 + 1.0j]
    # d/dz exp(z) z = exp(z) (z + 1), conjugated for a sum that is not real.
    expected = [(cmath.exp(p) * (p + 1)).conjugate() for p in points]
    for device in ("cpu", "cuda"):
        for _ in range(4):
            z = tp.tensor(points, dtype=tp.complex64, device=device)
            z.requires_grad_(True)
            (tp.exp(z) * z).sum().backward()
            got = z.grad.cpu().tolist()
            for value, want in zip(got, expected):
                assert value.real == pytest.approx(want.real, rel=1e-5, abs=1e-6)
                assert value.imag == pytest.approx(want.imag, rel=1e-5, abs=1e-6)
            assert z.detach().cpu().tolist() == points
