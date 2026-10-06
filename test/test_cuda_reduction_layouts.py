import numpy as np
import pytest

import tensorplay as tp


pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("dtype", [tp.float16, tp.bfloat16, tp.float32, tp.float64, tp.int64])
def test_unaligned_reduction_head_body_tail(dtype):
    for length in (128, 129, 255, 4096, 16385, 1048577):
        for offset in range(1, 8):
            values = (np.arange(length + 8) % 9 - 4).astype(np.float64)
            storage = tp.tensor(values, dtype=dtype, device="cuda")
            view = tp.as_strided(storage, [length], [1], offset)
            expected = values[offset:offset + length]
            output_dtype = tp.float64 if dtype == tp.float64 else tp.float32
            assert view.storage_offset() == offset
            assert view.sum(dtype=output_dtype).item() == expected.sum()
            if dtype != tp.int64:
                np.testing.assert_allclose(
                    view.mean(dtype=output_dtype).item(), expected.mean(), atol=1e-6)
                assert tp.nansum(view).item() == expected.sum()


@pytest.mark.parametrize("stride", [264, 265])
@pytest.mark.parametrize("dtype", [tp.float16, tp.float32, tp.float64, tp.int64])
def test_unaligned_reduction_rows_and_indices(stride, dtype):
    length, rows = 257, 3
    for offset in range(1, 8):
        values = np.ones(rows * stride + 8)
        for row, winner in enumerate((0, 17, length - 1)):
            start = offset + row * stride
            values[start + winner] = 9
            if winner < length - 1:
                values[start + length - 1] = 9
        storage = tp.tensor(values, dtype=dtype, device="cuda")
        view = tp.as_strided(storage, [rows, length], [stride, 1], offset)
        expected = np.stack([values[offset + row * stride:offset + row * stride + length]
                             for row in range(rows)])
        np.testing.assert_array_equal(view.sum(dim=1).cpu().numpy(), expected.sum(axis=1))
        np.testing.assert_array_equal(view.argmax(dim=1).cpu().numpy(), expected.argmax(axis=1))
        np.testing.assert_array_equal(view.argmin(dim=1).cpu().numpy(), expected.argmin(axis=1))
        maximum, indices = view.max(dim=1)
        np.testing.assert_array_equal(maximum.cpu().numpy(), expected.max(axis=1))
        np.testing.assert_array_equal(indices.cpu().numpy(), expected.argmax(axis=1))
        minimum, indices = view.min(dim=1)
        np.testing.assert_array_equal(minimum.cpu().numpy(), expected.min(axis=1))
        np.testing.assert_array_equal(indices.cpu().numpy(), expected.argmin(axis=1))


@pytest.mark.parametrize("dtype", [tp.float16, tp.bfloat16, tp.float32, tp.float64])
def test_unaligned_reduction_nan_and_product(dtype):
    for offset in range(1, 8):
        values = np.ones(16385 + 8)
        values[offset] = 2
        values[offset + 17] = 3
        values[offset + 16384] = 4
        storage = tp.tensor(values, dtype=dtype, device="cuda")
        view = tp.as_strided(storage, [16385], [1], offset)
        assert view.prod(dtype=tp.float32).item() == 24
        values[offset] = np.nan
        values[offset + 17] = np.nan
        storage = tp.tensor(values, dtype=dtype, device="cuda")
        view = tp.as_strided(storage, [16385], [1], offset)
        assert view.argmax().item() == 0
        assert view.argmin().item() == 0
        assert np.isnan(view.max().item())
        assert np.isnan(view.min().item())
        expected = tp.tensor(np.nansum(values[offset:offset + 16385]), dtype=dtype).item()
        assert tp.nansum(view).item() == expected


@pytest.mark.parametrize("dtype", [tp.float16, tp.bfloat16, tp.float32, tp.float64, tp.int64])
@pytest.mark.parametrize("columns, stride, offset", [(128, 128, 0), (130, 136, 2), (129, 129, 1), (4, 8, 0)])
def test_reduction_across_contiguous_outputs(dtype, columns, stride, offset):
    for rows in (17, 257, 16385):
        values = np.arange(rows * stride + 8) % 7 - 3
        storage = tp.tensor(values, dtype=dtype, device="cuda")
        view = tp.as_strided(storage, [rows, columns], [stride, 1], offset)
        expected = np.stack([values[offset + row * stride:offset + row * stride + columns]
                             for row in range(rows)])
        np.testing.assert_array_equal(view.sum(dim=0, dtype=tp.float32).cpu().numpy(), expected.sum(axis=0))
        if dtype != tp.int64:
            np.testing.assert_allclose(view.mean(dim=0, dtype=tp.float32).cpu().numpy(), expected.mean(axis=0), atol=1e-6)
        np.testing.assert_array_equal(view.argmax(dim=0).cpu().numpy(), expected.argmax(axis=0))
        np.testing.assert_array_equal(view.argmin(dim=0).cpu().numpy(), expected.argmin(axis=0))
        maximum, indices = view.max(dim=0)
        np.testing.assert_array_equal(maximum.cpu().numpy(), expected.max(axis=0))
        np.testing.assert_array_equal(indices.cpu().numpy(), expected.argmax(axis=0))
        minimum, indices = view.min(dim=0)
        np.testing.assert_array_equal(minimum.cpu().numpy(), expected.min(axis=0))
        np.testing.assert_array_equal(indices.cpu().numpy(), expected.argmin(axis=0))
        minimum, maximum = tp.aminmax(view, dim=0)
        np.testing.assert_array_equal(minimum.cpu().numpy(), expected.min(axis=0))
        np.testing.assert_array_equal(maximum.cpu().numpy(), expected.max(axis=0))


@pytest.mark.parametrize("dtype", [tp.float32, tp.float64, tp.complex64, tp.complex128])
def test_output_groups_with_multiple_reduced_dimensions(dtype):
    values = (np.arange(5 * 9 * 7 * 16) % 11 - 5).reshape(5, 9, 7, 16)
    if dtype in (tp.complex64, tp.complex128):
        values = values + 1j * (values % 3)
    tensor = tp.tensor(values, dtype=dtype, device="cuda")
    np.testing.assert_allclose(tensor.sum(dim=[0, 2]).cpu().numpy(), values.sum(axis=(0, 2)), atol=1e-5)
    np.testing.assert_allclose(tensor.mean(dim=[0, 2]).cpu().numpy(), values.mean(axis=(0, 2)), atol=1e-5)


def test_output_groups_nan_and_moments():
    values = (np.arange(1025 * 8) % 7 - 3).astype(np.float32).reshape(1025, 8)
    values[0, 0] = np.nan
    values[1, 0] = np.nan
    values[-1, 7] = np.nan
    tensor = tp.tensor(values, device="cuda")
    maximum, indices = tensor.max(dim=0)
    np.testing.assert_array_equal(indices.cpu().numpy()[[0, 7]], [0, 1024])
    assert np.isnan(maximum.cpu().numpy()[[0, 7]]).all()
    minimum, indices = tensor.min(dim=0)
    np.testing.assert_array_equal(indices.cpu().numpy()[[0, 7]], [0, 1024])
    assert np.isnan(minimum.cpu().numpy()[[0, 7]]).all()
    np.testing.assert_allclose(tp.nansum(tensor, dim=[0]).cpu().numpy(), np.nansum(values, axis=0))
    variance, mean = tp.var_mean(tensor, dim=[0], correction=0)
    np.testing.assert_allclose(mean.cpu().numpy(), values.astype(np.float64).mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(variance.cpu().numpy(), values.astype(np.float64).var(axis=0), atol=1e-5)


@pytest.mark.parametrize("rank", [65, 96, 128])
def test_reduction_with_many_input_dimensions(rank):
    shape = [1] * rank
    shape[0], shape[32], shape[-1] = 3, 5, 8
    values = np.arange(120).reshape(3, 5, 8)
    tensor = tp.tensor(values, dtype=tp.float32, device="cuda").reshape(shape)
    assert tensor.sum().item() == values.sum()
    result = tensor.sum(dim=[0, 32])
    assert len(result.shape) == rank - 2
    np.testing.assert_array_equal(result.reshape([8]).cpu().numpy(), values.sum(axis=(0, 1)))
    maximum, indices = tensor.max(dim=-1, keepdim=True)
    assert len(maximum.shape) == rank
    np.testing.assert_array_equal(maximum.reshape([3, 5]).cpu().numpy(), values.max(axis=-1))
    np.testing.assert_array_equal(indices.reshape([3, 5]).cpu().numpy(), values.argmax(axis=-1))
