"""Tensor.to / Module.to with a target memory format."""

import tensorplay as tp


def _channels_last_strides(shape):
    n, c, h, w = shape
    return (c * h * w, 1, w * c, c)


def test_to_accepts_memory_format_in_every_spelling():
    x = tp.randn(2, 3, 4, 5)
    expected = _channels_last_strides(x.shape)

    assert x.to(memory_format=tp.channels_last).stride() == expected
    # Device-and-dtype form with both left unset (the module conversion path).
    assert x.to(None, None, False, memory_format=tp.channels_last).stride() == expected

    narrowed = x.to(tp.float16, memory_format=tp.channels_last)
    assert narrowed.dtype == tp.float16
    assert narrowed.stride() == expected

    like = x.to(tp.zeros(1, dtype=tp.float64), memory_format=tp.channels_last)
    assert like.dtype == tp.float64
    assert like.stride() == expected

    moved = x.to("cpu", tp.float32, memory_format=tp.channels_last)
    assert moved.stride() == expected
    assert tp.equal(moved, x)


def test_to_without_changes_returns_self():
    x = tp.randn(2, 3)
    assert x.to() is x
    assert x.to(None, None) is x
    assert x.to(x.dtype) is x
    assert x.to("cpu") is x
    assert x.to(x) is x
    assert x.to(x.dtype, copy=True) is not x


def test_to_keeps_positional_non_blocking_after_device():
    x = tp.randn(2, 3)
    assert tp.equal(x.to("cpu", True), x)


def test_module_to_memory_format_converts_four_dimensional_parameters():
    conv = tp.nn.Conv2d(3, 4, 3)
    conv.to(memory_format=tp.channels_last)
    assert conv.weight.is_contiguous(memory_format=tp.channels_last)
    assert conv.bias.dim() == 1
