import pytest

import tensorplay as tp


def test_untyped_storage_is_native_and_resizable():
    storage = tp.UntypedStorage(4)
    assert tp.is_storage(storage)
    assert not tp.is_storage(tp.ones((1,)))
    assert storage.nbytes() == 4
    assert storage.size() == 4
    assert len(storage) == 4
    assert storage.resizable()
    storage.resize_(8)
    assert storage.nbytes() == 8


def test_set_tensor_overloads_share_native_storage():
    source = tp.ones((2, 2), dtype=tp.float32)
    target = tp.empty((0,), dtype=tp.float32)

    assert target.set_(source) is target
    assert tuple(target.shape) == (2, 2)
    assert target.untyped_storage()._cdata == source.untyped_storage()._cdata

    target.set_(source, 0, [1, 4])
    assert tuple(target.shape) == (1, 4)
    assert tuple(target.stride()) == (4, 1)

    target.set_(source, 0, [2, 2], [1, 2])
    assert tuple(target.stride()) == (1, 2)


def test_set_storage_overloads_and_reset():
    source = tp.ones((2, 2), dtype=tp.float32)
    storage = source.untyped_storage()
    target = tp.empty((0,), dtype=tp.float32)

    target.set_(storage)
    assert tuple(target.shape) == (4,)
    assert target.untyped_storage()._cdata == storage._cdata

    target.set_(storage, 0, [2, 2], [2, 1])
    assert tuple(target.shape) == (2, 2)
    assert tuple(target.stride()) == (2, 1)

    target.set_()
    assert tuple(target.shape) == (0,)
    assert target.untyped_storage().nbytes() == 0


def test_set_storage_rejects_unchanged_geometry_out_of_bounds():
    target = tp.empty((2,), dtype=tp.float32)
    with pytest.raises(RuntimeError, match="out of bounds"):
        target.set_(tp.UntypedStorage(1), 0, [2])
