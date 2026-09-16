import numpy as np
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


_TP_DTYPE = {
    np.float16: tp.float16,
    np.float32: tp.float32,
    np.float64: tp.float64,
}


def test_addbmm_dtype_promotion_and_broadcast():
    rng = np.random.default_rng(7)
    cases = (
        (np.float32, np.float32),
        (np.float64, np.float64),
        # Mixed factors promote to the wider element type.
        (np.float64, np.float32),
        (np.float32, np.float64),
        # The half pair runs the accumulate-fallback path on every build.
        (np.float16, np.float16),
    )
    for seed_dt, factor_dt in cases:
        batch1 = tp.tensor(rng.standard_normal((4, 8, 16)).astype(factor_dt) * 0.5)
        batch2 = tp.tensor(rng.standard_normal((4, 16, 8)).astype(factor_dt) * 0.5)
        seed = tp.tensor(rng.standard_normal((8, 8)).astype(seed_dt) * 0.5)
        got = tp.addbmm(seed, batch1, batch2, beta=1.5, alpha=0.5)

        assert tuple(got.shape) == (8, 8)
        assert got.dtype == _TP_DTYPE[factor_dt]

        want = 1.5 * seed.numpy().astype(np.float64)
        for i in range(4):
            want = want + 0.5 * (
                batch1.numpy()[i].astype(np.float64)
                @ batch2.numpy()[i].astype(np.float64)
            )
        actual = got.numpy().astype(np.float64)
        if factor_dt is np.float16:
            assert np.allclose(actual, want, rtol=1e-2, atol=5e-2)
        elif factor_dt is np.float32:
            assert np.allclose(actual, want, rtol=1e-4, atol=1e-4)
        else:
            assert np.allclose(actual, want, rtol=1e-12, atol=1e-12)
