"""Native factory, nested dispatch and indexed mutation regression coverage."""

import numpy as np
import pytest

import tensorplay as tp


@pytest.mark.parametrize("factory", ["new_zeros", "new_ones", "new_empty", "new_full"])
@pytest.mark.parametrize("size", [(), (0,), (4,)])
def test_nested_factory_shape(factory, size):
    def make(x):
        args = (size, 7) if factory == "new_full" else (size,)
        return getattr(x, factory)(*args, dtype=tp.int32)

    x = tp.ones((2, 3, 5))
    result = tp.func.vmap(tp.func.vmap(make))(x)
    assert tuple(result.shape) == (2, 3, *size)
    assert result.dtype == tp.int32
    if factory != "new_empty":
        expected = {"new_zeros": 0, "new_ones": 1, "new_full": 7}[factory]
        np.testing.assert_array_equal(result.numpy(), np.full((2, 3, *size), expected))


@pytest.mark.parametrize("factory", ["new_zeros", "new_ones", "new_empty", "new_full"])
def test_factory_result_keeps_batch_ownership_for_mutation(factory):
    def make_and_fill(x):
        args = ((4,), 0) if factory == "new_full" else ((4,),)
        result = getattr(x, factory)(*args)
        result[tp.arange(4, dtype=tp.int64)] = x
        return result

    x = tp.ones((2, 3, 4)) * 7
    result = tp.func.vmap(tp.func.vmap(make_and_fill))(x)
    np.testing.assert_array_equal(result.numpy(), x.numpy())


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_nested_indexed_scalar_assignment(depth):
    def fill(x):
        result = x.new_zeros((4, 5), dtype=tp.int32)
        row = tp.arange(4, dtype=tp.int64).unsqueeze(-1)
        col = tp.zeros((4, 4), dtype=tp.int64)
        result[row, col] = result.new_ones(())
        return result

    batch_shape = (2, 3, 2)[:depth]
    mapped = fill
    for _ in range(depth):
        mapped = tp.func.vmap(mapped)
    result = mapped(tp.ones((*batch_shape, 7)))
    expected = np.zeros((*batch_shape, 4, 5), dtype=np.int32)
    expected[..., 0] = 1
    np.testing.assert_array_equal(result.numpy(), expected)


def test_nested_advanced_indexing():
    e = tp.rand(7, 4)
    idx = tp.tensor([0, 1], dtype=tp.int64).view(2, 1)

    def fake_vmap(function, in_dims=0, out_dims=0):
        def wrapped(input):
            outputs = [
                function(input.select(in_dims, i))
                for i in range(input.size(in_dims))
            ]
            return tp.stack(outputs, out_dims)

        return wrapped

    def with_vmap(vectorize):
        def outer(index):
            def inner(value):
                return value[index]

            return vectorize(inner, in_dims=1)(e)

        return vectorize(outer)(idx)

    np.testing.assert_array_equal(with_vmap(tp.func.vmap).numpy(),
                                  with_vmap(fake_vmap).numpy())


def test_outer_only_operand_and_exception_restore_dispatch():
    def outer(x):
        def inner(y):
            return x.new_ones((4,)) + y
        return tp.func.vmap(inner)(tp.zeros((3, 4)))

    np.testing.assert_array_equal(
        tp.func.vmap(outer)(tp.zeros((2, 5))).numpy(), np.ones((2, 3, 4)))
    destination = tp.zeros((4,))
    with pytest.raises(RuntimeError, match="vmap"):
        tp.func.vmap(lambda x: destination.index_put_([tp.tensor([0])], x))(
            tp.ones((3, 1)))
    np.testing.assert_array_equal((tp.ones((4,)) + 2).numpy(), np.full(4, 3))
    np.testing.assert_array_equal(
        tp.func.vmap(lambda x: x + 2)(tp.ones((3, 4))).numpy(), np.full((3, 4), 3))


@pytest.mark.parametrize("accumulate", [False, True])
def test_native_index_put_broadcast_and_noncontiguous_destination(accumulate):
    base = tp.zeros((5, 4), dtype=tp.int32)
    destination = base.transpose(0, 1)
    row = tp.arange(4, dtype=tp.int64).unsqueeze(-1)
    col = tp.tensor([[0, -1]], dtype=tp.int64)
    destination.index_put_([row, col], tp.tensor(3, dtype=tp.int32), accumulate)
    expected = np.zeros((4, 5), dtype=np.int32)
    expected[:, [0, -1]] = 3
    np.testing.assert_array_equal(destination.numpy(), expected)
    np.testing.assert_array_equal(base.numpy(), expected.T)


def test_native_index_put_accumulates_repeated_indices():
    destination = tp.zeros((4, 5), dtype=tp.int32)
    row = tp.arange(4, dtype=tp.int64).unsqueeze(-1)
    col = tp.zeros((4, 3), dtype=tp.int64)
    destination.index_put_([row, col], tp.tensor(2, dtype=tp.int32), True)
    expected = np.zeros((4, 5), dtype=np.int32)
    expected[:, 0] = 6
    np.testing.assert_array_equal(destination.numpy(), expected)


def test_native_index_put_rejects_dtype_mismatch():
    with pytest.raises(RuntimeError, match="dtypes match"):
        tp.zeros((4,), dtype=tp.float32).index_put_(
            [tp.tensor([0], dtype=tp.int64)], tp.ones((1,), dtype=tp.float64))
