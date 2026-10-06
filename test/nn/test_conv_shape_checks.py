"""Convolutions reject inputs, weights and biases that disagree with groups.

A weight whose output channels the groups do not divide, an input with the
wrong channel count or a bias of the wrong length would otherwise be read
past its end; every convolution entry point names the mismatch instead.
"""
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])

INPUT = {1: (2, 4, 9), 2: (2, 4, 7, 7), 3: (1, 4, 5, 5, 5)}
CONV = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}
CONV_T = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}


def randn(*shape, device):
    return tp.randn(*shape, device=device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_out_channels_not_divisible_by_groups(device, rank):
    x = randn(*INPUT[rank], device=device)
    w = randn(5, 2, *([3] * rank), device=device)
    with pytest.raises(RuntimeError, match="expected weight to be divisible by 2"):
        CONV[rank](x, w, groups=2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_input_channels_must_match_weight(device, rank):
    x = randn(*INPUT[rank], device=device)
    w = randn(4, 3, *([3] * rank), device=device)
    with pytest.raises(RuntimeError, match="to have 3 channels, but got 4 channels"):
        CONV[rank](x, w)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_bias_length_must_match_output_channels(device, rank):
    x = randn(*INPUT[rank], device=device)
    w = randn(6, 4, *([3] * rank), device=device)
    with pytest.raises(RuntimeError, match="expected bias to be 1-dimensional with 6 elements"):
        CONV[rank](x, w, bias=randn(5, device=device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_transposed_input_channels_must_match_weight(device, rank):
    x = randn(*INPUT[rank], device=device)
    w = randn(3, 2, *([3] * rank), device=device)
    with pytest.raises(RuntimeError, match="to have 3 channels, but got 4 channels"):
        CONV_T[rank](x, w)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_transposed_in_channels_not_divisible_by_groups(device, rank):
    x = randn(*INPUT[rank], device=device)
    w = randn(4, 3, *([3] * rank), device=device)
    with pytest.raises(RuntimeError, match="expected weight to be divisible by 3"):
        CONV_T[rank](x, w, groups=3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_valid_grouped_shapes_still_run(device, rank):
    x = randn(*INPUT[rank], device=device)
    out = CONV[rank](x, randn(6, 2, *([3] * rank), device=device),
                     bias=randn(6, device=device), groups=2)
    assert out.shape[1] == 6
    out = CONV_T[rank](x, randn(4, 3, *([3] * rank), device=device),
                       bias=randn(6, device=device), groups=2)
    assert out.shape[1] == 6
