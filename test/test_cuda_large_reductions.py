import os

import numpy as np
import pytest

import tensorplay as tp


pytestmark = pytest.mark.skipif(
    not tp.cuda.is_available() or os.environ.get("TP_TEST_LARGE_REDUCTIONS") != "1",
    reason="requires CUDA and TP_TEST_LARGE_REDUCTIONS=1",
)


def test_large_reduced_axis_indices():
    length = (1 << 31) + 17
    tensor = tp.ones([length], dtype=tp.int8, device="cuda")
    assert tensor.sum().item() == length
    assert tensor.argmax().item() == 0
    assert tensor.argmin().item() == 0
    tensor[-1] = 7
    assert tensor.argmax().item() == length - 1
    maximum, indices = tensor.max(dim=0)
    assert maximum.item() == 7
    assert indices.item() == length - 1
    tensor[-1] = -7
    assert tensor.argmin().item() == length - 1
    minimum, indices = tensor.min(dim=0)
    assert minimum.item() == -7
    assert indices.item() == length - 1


def test_split_output_rows_rebase_second_output():
    length = (1 << 28) + 1
    tensor = tp.ones([2, length], dtype=tp.int64, device="cuda")
    tensor[0, -1] = 7
    tensor[1, -2] = 9
    maximum, indices = tensor.max(dim=1)
    np.testing.assert_array_equal(maximum.cpu().numpy(), [7, 9])
    np.testing.assert_array_equal(indices.cpu().numpy(), [length - 1, length - 2])


def test_large_output_groups_rebase_reduced_indices():
    rows = (1 << 29) + 3
    tensor = tp.ones([rows, 4], dtype=tp.float32, device="cuda")
    tensor[-1, 0] = 7
    tensor[-2, 1] = 8
    tensor[-3, 2] = 9
    tensor[-4, 3] = 10
    np.testing.assert_allclose(tensor.sum(dim=0).cpu().numpy(), rows, rtol=1e-6)
    maximum, indices = tensor.max(dim=0)
    np.testing.assert_array_equal(maximum.cpu().numpy(), [7, 8, 9, 10])
    np.testing.assert_array_equal(indices.cpu().numpy(), [rows - 1, rows - 2, rows - 3, rows - 4])
    np.testing.assert_array_equal(tensor.argmax(dim=0).cpu().numpy(), [rows - 1, rows - 2, rows - 3, rows - 4])
    tensor[0, 0] = 7
    assert tensor.argmax(dim=0)[0].item() == 0
    tensor[0, 0] = 1
    tensor[-1, 0] = -7
    tensor[-2, 1] = -8
    tensor[-3, 2] = -9
    tensor[-4, 3] = -10
    minimum, indices = tensor.min(dim=0)
    np.testing.assert_array_equal(minimum.cpu().numpy(), [-7, -8, -9, -10])
    np.testing.assert_array_equal(indices.cpu().numpy(), [rows - 1, rows - 2, rows - 3, rows - 4])
    np.testing.assert_array_equal(tensor.argmin(dim=0).cpu().numpy(), [rows - 1, rows - 2, rows - 3, rows - 4])
    tensor[0, 0] = -7
    assert tensor.argmin(dim=0)[0].item() == 0
    tensor[0, 0] = 1
    tensor[-2, 0] = np.nan
    tensor[-1, 0] = np.nan
    assert tensor.argmax(dim=0)[0].item() == rows - 2
    assert tensor.argmin(dim=0)[0].item() == rows - 2
    maximum, indices = tensor.max(dim=0)
    assert np.isnan(maximum[0].item())
    assert indices[0].item() == rows - 2
    minimum, indices = tensor.min(dim=0)
    assert np.isnan(minimum[0].item())
    assert indices[0].item() == rows - 2
