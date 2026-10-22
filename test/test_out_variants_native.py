"""The destination variants the runtime derives.

A schema declares its destination in one of two ways: it names one
outright, or it declares the in-place form and lets the runtime derive the
destination form from it.  The derived spellings are destination variants
the runtime carries, and each of them answers the map it is derived from
and lays the answer on the destination the caller holds.

This walks a sample of the variants the project carries and checks that
each one answers the same as the map it is derived from: the destination
holds the answer, and the answer is the one the map gives.
"""

import numpy as np
import pytest

import tensorplay as tp

from tensorplay._ops import NATIVE_NAMESPACE

OPS = getattr(tp.ops, NATIVE_NAMESPACE)

# The maps that answer for the tensor alone.
SINGLE = [
    ("abs", "out"),
    ("neg", "out"),
    ("relu", "out"),
    ("sqrt", "out"),
    ("exp", "out"),
    ("log", "out"),
    ("sign", "out"),
    ("reciprocal", "out"),
    ("floor", "out"),
    ("ceil", "out"),
    ("square", "out"),
    ("sigmoid", "out"),
    ("tanh", "out"),
    ("silu", "out"),
]

# The maps that take more than the tensor they answer for: the seeds name
# the operands they take, and the formula says what they answer.
WITH_ARGUMENT = [
    ("add", "out", (5,), lambda x, o, v: x + o[0]),
    ("sub", "out", (5,), lambda x, o, v: x - o[0]),
    ("mul", "out", (5,), lambda x, o, v: x * o[0]),
    ("div", "out", (5,), lambda x, o, v: x / o[0]),
    ("pow", "out", (5,), lambda x, o, v: x ** o[0]),
    ("addcmul", "out", (5, 7), lambda x, o, v: x + v * o[0] * o[1]),
    ("addcdiv", "out", (5, 7), lambda x, o, v: x + v * o[0] / o[1]),
    ("lerp", "out", (5, 7), lambda x, o, v: x + o[1] * (o[0] - x)),
]

# The maps that answer with a list: the variant takes the list of
# destinations and lays each answer on one of them.
WITH_LIST = [
    ("unsafe_split", "Tensor_out"),
]


def _values(count=64, seed=3):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(count).astype(np.float32)


def _carried(operation, spelling):
    packet = getattr(OPS, operation, None)
    if packet is None:
        return False
    return spelling in packet.overloads()


def _call(operation, spelling, arguments, destination):
    overload = getattr(getattr(OPS, operation), spelling)
    return overload(*arguments, out=destination)


@pytest.mark.parametrize("operation,spelling", SINGLE)
def test_the_variant_answers_the_map_on_the_destination(operation, spelling):
    if not _carried(operation, spelling):
        pytest.skip(f"{operation}.{spelling} is not carried")
    values = _values()
    source = tp.from_numpy(values)
    destination = tp.empty_like(source)
    _call(operation, spelling, (source,), destination)
    # The destination holds the answer the map gives.
    wanted = getattr(tp, operation)(source)
    assert np.allclose(np.array(destination.tolist(), dtype=np.float32),
                       np.array(wanted.tolist(), dtype=np.float32),
                       rtol=1e-5, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("operation,spelling,seeds,formula", WITH_ARGUMENT)
def test_the_variant_answers_the_map_with_its_arguments(operation, spelling,
                                                        seeds, formula):
    if not _carried(operation, spelling):
        pytest.skip(f"{operation}.{spelling} is not carried")
    values = _values()
    operands = [_values(seed=seed) for seed in seeds]
    source = tp.from_numpy(values)
    arguments = [source] + [tp.from_numpy(item) for item in operands]
    destination = tp.empty_like(source)
    _call(operation, spelling, arguments, destination)
    wanted = formula(values, operands, np.float32(1.0))
    assert np.allclose(np.array(destination.tolist(), dtype=np.float32),
                       wanted, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_the_variant_forwards_the_arguments_it_is_handed():
    # The scale has a schema default; a caller that supplies one must have
    # it carried into the map the variant answers.
    if not _carried("add", "Scalar_out"):
        pytest.skip("add.Scalar_out is not carried")
    values = _values()
    source = tp.from_numpy(values)
    destination = tp.empty_like(source)
    _call("add", "Scalar_out", (source, 2.0, 3), destination)
    wanted = values + 2.0 * 3.0
    assert np.allclose(np.array(destination.tolist(), dtype=np.float32),
                       wanted, rtol=1e-6, atol=1e-6)


def test_the_variant_keeps_the_bounds_it_is_handed():
    # The bounds have schema defaults; a caller that supplies them must not
    # receive draws from the default interval.
    if not _carried("uniform", "out"):
        pytest.skip("uniform.out is not carried")
    source = tp.from_numpy(_values())
    destination = tp.empty_like(source)
    _call("uniform", "out", (source, 3.0, 4.0), destination)
    drawn = np.array(destination.tolist(), dtype=np.float32)
    assert (drawn >= 3.0).all() and (drawn < 4.0).all()


def test_the_variant_lays_a_list_answer_the_map_gives():
    # A map whose answer is a list of tensors: the destinations hold the
    # answer, and the tensors the caller handed in are left alone.
    if not _carried("_foreach_zero", "out"):
        pytest.skip("_foreach_zero.out is not carried")
    values = [_values(seed=seed) for seed in (11, 12, 13)]
    sources = [tp.from_numpy(item) for item in values]
    destinations = [tp.full_like(source, 9.0) for source in sources]
    _call("_foreach_zero", "out", (sources,), destinations)
    for item, source, destination in zip(values, sources, destinations):
        assert np.array_equal(np.array(destination.tolist(), dtype=np.float32),
                              np.zeros_like(item))
        assert np.array_equal(np.array(source.tolist(), dtype=np.float32), item)


@pytest.mark.parametrize("operation,spelling", WITH_LIST)
def test_the_variant_lays_each_answer_on_its_destination(operation, spelling):
    if not _carried(operation, spelling):
        pytest.skip(f"{operation}.{spelling} is not carried")
    values = _values(6)
    source = tp.from_numpy(values)
    destinations = [tp.empty(2, dtype=tp.float32) for _ in range(3)]
    _call(operation, spelling, (source, 2), destinations)
    laid = np.array([d.tolist() for d in destinations], dtype=np.float32)
    # Every answer landed on one of the destinations.
    assert laid.shape == (3, 2)
    wanted = np.array(values, dtype=np.float32).reshape(3, 2)
    assert np.allclose(laid, wanted, rtol=1e-5, atol=1e-5)
