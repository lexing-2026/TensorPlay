"""Average pooling and nearest upsampling, compiled, with their gradients.

Both are written into the loop of whatever consumes them: an output element
reads its window, and an input element's gradient reads the few windows that
covered it.  Each case is checked against the sums worked out by hand below,
on the value and on the gradient, and on a region that was actually built.
"""
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])

SIZE = 7

#: kernel, stride, padding, ceil_mode, count_include_pad
POOLINGS = [
    (2, 2, 0, False, True),
    (3, 2, 1, False, True),
    (3, 2, 1, False, False),
    (3, 1, 0, False, True),
    (2, 2, 0, True, True),
    (3, 2, 1, True, True),
    (3, 2, 0, True, False),
]


def _source():
    return [
        [((3 * row + 5 * col) % 11) / 4.0 - 1.0 for col in range(SIZE)]
        for row in range(SIZE)
    ]


def _weights(extent):
    return [[(row * extent + col + 1) / 8.0 for col in range(extent)] for row in range(extent)]


def _pooled_extent(kernel, stride, padding, ceil_mode):
    span = SIZE + 2 * padding - kernel
    if not ceil_mode:
        return span // stride + 1
    count = -(-span // stride) + 1
    # A window has to start inside the source or its leading padding.
    if (count - 1) * stride >= SIZE + padding:
        count -= 1
    return count


def _pool_by_hand(kernel, stride, padding, ceil_mode, count_include_pad):
    """The pooled values and the gradient of their weighted sum."""

    x = _source()
    extent = _pooled_extent(kernel, stride, padding, ceil_mode)
    weights = _weights(extent)
    pooled = [[0.0] * extent for _ in range(extent)]
    grad = [[0.0] * SIZE for _ in range(SIZE)]
    for i in range(extent):
        for j in range(extent):
            top, left = i * stride - padding, j * stride - padding
            bottom = min(top + kernel, SIZE + padding)
            right = min(left + kernel, SIZE + padding)
            padded = (bottom - top) * (right - left)
            top, left = max(top, 0), max(left, 0)
            bottom, right = min(bottom, SIZE), min(right, SIZE)
            divisor = padded if count_include_pad else (bottom - top) * (right - left)
            total = 0.0
            for row in range(top, bottom):
                for col in range(left, right):
                    total += x[row][col]
                    grad[row][col] += weights[i][j] / divisor
            pooled[i][j] = total / divisor
    return pooled, grad, weights


def _run(fn, device, weights):
    x = tp.tensor([[_source()]], device=device, requires_grad=True)
    compiled = tp.compile(lambda v: fn(v) * 1.0, backend="stax")
    out = compiled(x)
    (out * tp.tensor([[weights]], device=device)).sum().backward()
    # The answers only say something about a region that was built.
    lowering = next(iter(compiled._tensorplay_cache.values()))
    assert lowering._tensorplay_codegen is not None
    assert lowering._tensorplay_backward_codegen is not None
    return out.detach().cpu().tolist()[0][0], x.grad.cpu().tolist()[0][0]


def _assert_close(got, expected):
    assert len(got) == len(expected)
    for got_row, expected_row in zip(got, expected):
        assert got_row == pytest.approx(expected_row, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pooling", POOLINGS)
def test_average_pooling_value_and_gradient(device, pooling):
    kernel, stride, padding, ceil_mode, count_include_pad = pooling
    pooled, grad, weights = _pool_by_hand(*pooling)

    def pool(v):
        return F.avg_pool2d(
            v, kernel, stride, padding, ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    got, got_grad = _run(pool, device, weights)
    _assert_close(got, pooled)
    _assert_close(got_grad, grad)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("extent", [14, 21, 10])
def test_nearest_upsampling_value_and_gradient(device, extent):
    x = _source()
    weights = _weights(extent)
    upsampled = [[0.0] * extent for _ in range(extent)]
    grad = [[0.0] * SIZE for _ in range(SIZE)]
    for i in range(extent):
        for j in range(extent):
            row, col = i * SIZE // extent, j * SIZE // extent
            upsampled[i][j] = x[row][col]
            grad[row][col] += weights[i][j]

    def upsample(v):
        return F.interpolate(v, size=(extent, extent), mode="nearest")

    got, got_grad = _run(upsample, device, weights)
    _assert_close(got, upsampled)
    _assert_close(got_grad, grad)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pooling", POOLINGS)
def test_average_pooling_run_directly(device, pooling):
    kernel, stride, padding, ceil_mode, count_include_pad = pooling
    pooled, grad, weights = _pool_by_hand(*pooling)
    x = tp.tensor([[_source()]], device=device, requires_grad=True)
    out = F.avg_pool2d(
        x, kernel, stride, padding, ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )
    (out * tp.tensor([[weights]], device=device)).sum().backward()
    _assert_close(out.detach().cpu().tolist()[0][0], pooled)
    _assert_close(x.grad.cpu().tolist()[0][0], grad)
