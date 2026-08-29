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
