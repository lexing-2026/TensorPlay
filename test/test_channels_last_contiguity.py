"""A channels-last tensor is dense in its own order and is not row-major."""
import numpy as np

import tensorplay as tp


def _source():
    return tp.tensor(np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5))


def test_channels_last_copy_is_not_row_major():
    a = _source()
    b = a.contiguous(memory_format=tp.channels_last)
    assert tuple(b.stride()) == (60, 1, 15, 3)
    assert not b.is_contiguous()
    assert b.is_contiguous(memory_format=tp.channels_last)
    np.testing.assert_array_equal(b.numpy(), a.numpy())


def test_every_way_of_asking_for_channels_last_agrees():
    a = _source()
    made = {
        "contiguous": a.contiguous(memory_format=tp.channels_last),
        "to": a.to(memory_format=tp.channels_last),
        "clone": a.contiguous(memory_format=tp.channels_last).clone(),
    }
    for name, t in made.items():
        assert not t.is_contiguous(), name
        assert t.is_contiguous(memory_format=tp.channels_last), name


def test_row_major_copy_of_channels_last_moves_the_data():
    a = _source()
    b = a.contiguous(memory_format=tp.channels_last)
    c = b.contiguous()
    assert tuple(c.stride()) == (60, 20, 5, 1)
    assert c.is_contiguous()
    np.testing.assert_array_equal(c.numpy(), a.numpy())
