import pytest

import tensorplay as tp


@pytest.mark.parametrize(
    "view",
    [
        lambda x: x.view(2, 3),
        lambda x: x.view((2, 3)),
        lambda x: x.view(size=[2, -1]),
        lambda x: tp.view(x, (2, 3)),
        lambda x: tp.ops.tp.view.default(x, [2, 3]),
    ],
)
def test_cpu_view_shape_and_storage(view):
    base = tp.arange(6, device="cpu")
    result = view(base)

    assert tuple(result.shape) == (2, 3)
    assert result.stride() == (3, 1)
    assert result.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert result.data_ptr() == base.data_ptr()
    result[1, 2] = 42
    assert base.tolist() == [0, 1, 2, 3, 4, 42]


def test_cpu_view_preserves_storage_offset():
    base = tp.arange(10, device="cpu")
    sliced = base[2:8]
    result = sliced.view(2, 3)

    assert result.storage_offset() == 2
    assert result.data_ptr() == sliced.data_ptr()
    assert result.tolist() == [[2, 3, 4], [5, 6, 7]]


def test_cpu_view_accepts_compatible_noncontiguous_strides():
    base = tp.arange(12, device="cpu")
    sliced = base[::2]
    result = sliced.view(2, 3)

    assert result.stride() == (6, 2)
    assert result.data_ptr() == base.data_ptr()
    assert result.tolist() == [[0, 2, 4], [6, 8, 10]]


def test_cpu_view_rejects_incompatible_strides():
    transposed = tp.arange(6, device="cpu").reshape(2, 3).t()

    with pytest.raises(RuntimeError, match="size and stride"):
        transposed.view(6)


def test_cpu_view_empty_and_scalar():
    base = tp.empty((0, 3), device="cpu")
    empty = base.view(3, 0)
    assert tuple(empty.shape) == (3, 0)
    assert empty.numel() == 0
    assert tuple(base.view(3, -1).shape) == (3, 0)
    with pytest.raises(RuntimeError, match="ambiguous|unspecified dimension"):
        base.view(0, -1)

    scalar = tp.tensor(7, device="cpu")
    assert scalar.view(1).tolist() == [7]
    assert scalar.view(()).item() == 7


@pytest.mark.parametrize("shape", [(4, 2), (-1, -1), (0, -1)])
def test_cpu_view_rejects_invalid_shapes(shape):
    with pytest.raises((RuntimeError, ValueError), match="shape|dimension|size"):
        tp.arange(6, device="cpu").view(shape)


def test_cpu_view_backward():
    base = tp.arange(6, dtype=tp.float32, device="cpu", requires_grad=True)
    result = base.view(2, 3)

    (result * result).sum().backward()

    assert base.grad.tolist() == [0, 2, 4, 6, 8, 10]
