import numpy as np
import pytest

import tensorplay as tp


@pytest.mark.parametrize("dtype", [tp.float16, tp.bfloat16, tp.float32, tp.float64])
@pytest.mark.parametrize("expanded_operand", ["left", "right", "both"])
def test_expanded_matrices_use_dense_product_semantics(dtype, expanded_operand):
    left_base = tp.tensor([[1.0], [-2.0], [3.0]], dtype=dtype, requires_grad=True)
    right_base = tp.tensor([[0.5, -1.0]], dtype=dtype, requires_grad=True)
    left = left_base.expand(3, 4)
    right = right_base.expand(4, 2)
    if expanded_operand == "left":
        right = right.contiguous()
    elif expanded_operand == "right":
        left = left.contiguous()
    result = left @ right
    expected = np.array([[2, -4], [-4, 8], [6, -12]])
    np.testing.assert_allclose(result.detach().float().numpy(), expected)
    result.sum().backward()
    np.testing.assert_allclose(left_base.grad.float().numpy(), [[-2], [-2], [-2]])
    np.testing.assert_allclose(right_base.grad.float().numpy(), [[8, 8]])


@pytest.mark.parametrize("operation", ["amax", "amin"])
@pytest.mark.parametrize("dtype", [tp.float16, tp.bfloat16, tp.float32, tp.float64, tp.int64])
@pytest.mark.parametrize("layout", ["dense", "transposed", "expanded", "sliced"])
def test_extrema_reduce_strided_inputs(operation, dtype, layout):
    value = tp.tensor([[3, -2, 7, 1], [5, 0, -4, 9], [2, 6, -1, 8]], dtype=dtype)
    if layout == "transposed":
        value = value.t()
    elif layout == "expanded":
        value = value[:1].expand(3, 4)
    elif layout == "sliced":
        value = value[:, ::2]
    reference = value.float().numpy()
    reducer = getattr(np, "max" if operation == "amax" else "min")
    for dims in (0, 1, (0, 1)):
        for keepdim in (False, True):
            result = getattr(value, operation)(dim=dims, keepdim=keepdim)
            expected = reducer(reference, axis=dims, keepdims=keepdim)
            np.testing.assert_array_equal(result.float().numpy(), expected)


@pytest.mark.parametrize("dtype", [tp.float16, tp.bfloat16, tp.float32, tp.float64])
def test_extrema_preserve_nan_and_infinite_identities(dtype):
    value = tp.tensor([[float("-inf")] * 40, [float("inf")] * 40], dtype=dtype)
    assert value.amax(dim=1).tolist() == [float("-inf"), float("inf")]
    assert value.amin(dim=1).tolist() == [float("-inf"), float("inf")]
    value[0, 3] = float("nan")
    value[1, 37] = float("nan")
    assert np.isnan(value.amax(dim=1).float().numpy()).all()
    assert np.isnan(value.amin(dim=1).float().numpy()).all()


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_extrema_preserve_large_integer_values(dtype):
    maximum = np.iinfo(dtype).max
    values = np.array([[maximum, maximum - 1], [2**53 + 3, 2**53 + 1]], dtype=dtype)
    value = tp.tensor(values)
    np.testing.assert_array_equal(value.amax(dim=1).numpy(), values.max(axis=1))
    np.testing.assert_array_equal(value.amin(dim=1).numpy(), values.min(axis=1))


@pytest.mark.parametrize("operation", ["amax", "amin"])
def test_extrema_scalar_empty_and_invalid_dimensions(operation):
    scalar = tp.tensor(3.0)
    for dims in ((), 0, -1):
        assert getattr(scalar, operation)(dim=dims).item() == 3.0
    empty = tp.empty(0, 4)
    assert tuple(getattr(empty, operation)(dim=1).shape) == (0,)
    with pytest.raises((IndexError, RuntimeError)):
        getattr(empty, operation)(dim=0)
    with pytest.raises(RuntimeError, match="more than once"):
        getattr(tp.ones(2, 3), operation)(dim=(0, 0))
