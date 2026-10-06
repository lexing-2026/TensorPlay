import numpy as np
import pytest

import tensorplay as tp


pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("dtype", [
    tp.bool, tp.uint8, tp.int8, tp.int16, tp.int32, tp.int64,
    tp.float16, tp.bfloat16, tp.float32, tp.float64,
])
@pytest.mark.parametrize("dim", [0, 1])
def test_hash_tensor_cuda_value_bits(dtype, dim):
    values = (np.arange(257 * 4) % 17 - 8).reshape(257, 4)
    if dtype in (tp.float16, tp.bfloat16, tp.float32, tp.float64):
        values = values * 0.25
        tensor = tp.tensor(values, dtype=dtype, device="cuda")
        bits = tensor.cpu().to(tp.float64).numpy().view(np.uint64)
    else:
        tensor = tp.tensor(values, dtype=dtype, device="cuda")
        bits = tensor.cpu().numpy().astype(np.uint64)
    expected = np.bitwise_xor.reduce(bits, axis=dim)
    actual = tp.hash_tensor(tensor, [dim])
    assert actual.dtype == tp.uint64
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


def test_hash_tensor_cuda_unaligned_out_and_empty():
    values = np.linspace(-4, 4, 1031, dtype=np.float32)
    values[1:5] = [-0.0, np.inf, -np.inf, np.nan]
    storage = tp.tensor(values, device="cuda")
    tensor = tp.as_strided(storage, [1029], [1], 1)
    expected = np.bitwise_xor.reduce(values[1:1030].astype(np.float64).view(np.uint64))
    output = tp.empty([1], dtype=tp.uint64, device="cuda")
    actual = tp.hash_tensor(tensor, [0], keepdim=True, out=output)
    assert actual.data_ptr() == output.data_ptr()
    assert output.item() == expected
    assert tp.hash_tensor(tp.empty([0, 2], device="cuda"), [1]).numel() == 0
    with pytest.raises(RuntimeError):
        tp.hash_tensor(tp.empty([0], device="cuda"), [0])
    with pytest.raises(RuntimeError):
        tp.hash_tensor(tensor, [0], mode=7)
