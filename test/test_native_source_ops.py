
import numpy as np
import torch
import torch.nn.functional as torch_F

import tensorplay as tp
import tensorplay.functional as tp_F


def _np(value):
    return value.detach().cpu().numpy()


def _tp(value):
    return tp.tensor(value.detach().cpu().numpy(), device="cpu")


def _assert_close(got, expected, *, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(_np(got), expected.detach().cpu().numpy(),
                               rtol=rtol, atol=atol)


def test_unary_aliases_are_native_dispatch_entries():
    cases = (
        ("absolute", torch.tensor([-2.0, 0.0, 3.0])),
        ("arccos", torch.tensor([-0.8, 0.0, 0.8])),
        ("arccosh", torch.tensor([1.1, 2.0, 5.0])),
        ("arcsin", torch.tensor([-0.8, 0.0, 0.8])),
        ("arcsinh", torch.tensor([-2.0, 0.0, 3.0])),
        ("arctan", torch.tensor([-2.0, 0.0, 3.0])),
        ("arctanh", torch.tensor([-0.8, 0.0, 0.8])),
    )
    for name, value in cases:
        _assert_close(getattr(tp_F, name)(_tp(value)), getattr(torch, name)(value))


def test_unary_inplace_aliases_write_through():
    value = torch.tensor([-0.8, 0.0, 0.8])
    for name in ("arccos_", "arcsin_", "arctan_", "arctanh_"):
        expected = getattr(torch, name[:-1])(value)
        got = _tp(value)
        result = getattr(tp_F, name)(got)
        _assert_close(result, expected)
        _assert_close(got, expected)


def test_pooling_1d_adapters_match_torch():
    value = torch.randn(2, 3, 13)
    x = _tp(value)
    _assert_close(
        tp_F.avg_pool1d(x, 3, stride=2, padding=1, ceil_mode=True,
                        count_include_pad=False),
        torch_F.avg_pool1d(value, 3, stride=2, padding=1, ceil_mode=True,
                            count_include_pad=False),
    )
    _assert_close(
        tp_F.max_pool1d(x, 3, stride=2, padding=1, dilation=2,
                        ceil_mode=True),
        torch_F.max_pool1d(value, 3, stride=2, padding=1, dilation=2,
                           ceil_mode=True),
    )
    got_values, got_indices = tp_F.max_pool1d_with_indices(x, 3, stride=2)
    expected_values, expected_indices = torch_F.max_pool1d_with_indices(
        value, 3, stride=2)
    _assert_close(got_values, expected_values)
    np.testing.assert_array_equal(_np(got_indices), _np(expected_indices))

    got_values, got_indices = tp_F.adaptive_max_pool1d(x, 5)
    expected_values, expected_indices = torch_F.adaptive_max_pool1d(
        value, 5, return_indices=True)
    _assert_close(got_values, expected_values)
    np.testing.assert_array_equal(_np(got_indices), _np(expected_indices))
    _assert_close(tp_F.adaptive_avg_pool1d(x, 5),
                  torch_F.adaptive_avg_pool1d(value, 5))


def test_native_aliases_and_source_adapters_match_torch():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0])
    _assert_close(tp_F.concat([_tp(a), _tp(b)]), torch.cat([a, b]))
    _assert_close(tp_F.concatenate([_tp(a), _tp(b)]), torch.cat([a, b]))
    _assert_close(tp_F.diagflat(_tp(a), offset=1), torch.diagflat(a, offset=1))
    _assert_close(tp_F.ger(_tp(a), _tp(b)), torch.ger(a, b))

    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[0.0, 5.0], [6.0, 7.0]])
    _assert_close(tp_F.kron(_tp(left), _tp(right)), torch.kron(left, right))
    _assert_close(tp_F.matrix_power(_tp(left), 5), torch.linalg.matrix_power(left, 5))

    integral = torch.tensor([1, 2, 3], dtype=torch.int32)
    _assert_close(tp_F.vander(_tp(integral), N=4, increasing=True),
                  torch.vander(integral, N=4, increasing=True))

    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([10.0, 20.0, 30.0])
    _assert_close(tp_F.cartesian_prod(_tp(x), _tp(y)),
                  torch.cartesian_prod(x, y))
