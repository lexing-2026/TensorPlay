import math

import pytest

import tensorplay as tp


pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")


def _cuda(values):
    return tp.tensor(values, device="cuda")


def test_foreach_minmax_propagate_nan_and_canonicalize_zero():
    values = [_cuda([float("nan"), -0.0, -2.0, 0.0, 2.0]) for _ in range(32)]
    maximum = tp._C._foreach_maximum(values, 1.0)[0].cpu().tolist()
    minimum = tp._C._foreach_minimum(values, 1.0)[0].cpu().tolist()
    absolute = tp._C._foreach_abs(values)[0].cpu().tolist()

    assert math.isnan(maximum[0])
    assert math.isnan(minimum[0])
    assert maximum[1:] == [1.0, 1.0, 1.0, 2.0]
    assert minimum[1:] == [-0.0, -2.0, 0.0, 1.0]
    assert absolute[0] != absolute[0]
    assert math.copysign(1.0, absolute[1]) == 1.0
    assert absolute[2:] == [2.0, 0.0, 2.0]


def test_foreach_minmax_list_propagate_nan():
    lhs = [_cuda([float("nan"), -2.0, 4.0]) for _ in range(32)]
    rhs = [_cuda([1.0, float("nan"), 3.0]) for _ in range(32)]

    maximum = tp._C._foreach_maximum(lhs, rhs)[0].cpu().tolist()
    minimum = tp._C._foreach_minimum(lhs, rhs)[0].cpu().tolist()

    assert all(math.isnan(maximum[i]) and math.isnan(minimum[i]) for i in (0, 1))
    assert maximum[2] == 4.0
    assert minimum[2] == 3.0


def test_foreach_norm_empty_tensor_uses_zero_identity():
    values = [_cuda([]), _cuda([3.0, 4.0]), _cuda([])]

    norms = tp._foreach_norm(values, 2)

    assert [value.item() for value in norms] == [0.0, 5.0, 0.0]


def test_foreach_norm_batches_more_than_one_metadata_group():
    values = [_cuda([float(index % 7 + 1)]) for index in range(129)]

    norms = tp._foreach_norm(values, 2)

    assert len(norms) == len(values)
    for index, norm in enumerate(norms):
        assert norm.item() == float(index % 7 + 1)


@pytest.mark.parametrize(
    "dtype", [tp.float16, tp.bfloat16, tp.float32, tp.float64]
)
def test_foreach_norm_chunk_boundaries_and_offset_slice(dtype):
    base = tp.ones(65538, dtype=dtype, device="cuda")
    values = [base[:65535], base[:65536], base[:65537], base[1:]]

    norms = tp._foreach_norm(values, 2)

    expected = [math.sqrt(value.numel()) for value in values]
    for norm, want in zip(norms, expected):
        assert abs(norm.item() - want) < 0.01


def test_foreach_norm_preserves_autograd_fallback():
    value = tp.tensor([3.0, 4.0], device="cuda", requires_grad=True)
    norm = tp._foreach_norm([value], 2)[0]

    norm.backward()

    assert all(
        abs(actual - expected) < 1e-6
        for actual, expected in zip(value.grad.tolist(), [0.6, 0.8])
    )


@pytest.mark.skipif(tp.cuda.device_count() < 2, reason="Multiple GPUs are required")
def test_foreach_norm_guards_the_input_device():
    with tp.cuda.device(0):
        value = tp.tensor([3.0, 4.0], device="cuda:1")
        norm = tp._foreach_norm([value], 2)[0]
        tp.cuda.synchronize(1)

    assert norm.item() == 5.0
