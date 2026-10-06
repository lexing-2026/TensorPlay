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
