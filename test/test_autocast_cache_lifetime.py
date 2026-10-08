import pytest

import tensorplay as tp


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_transposed_cast_releases_source(cache_enabled):
    with tp.autocast("cpu", cache_enabled=cache_enabled):
        weight = tp.randn(32, 32, requires_grad=True)
        source = tp._C._WeakTensorRef(weight)
        result = tp.randn(2, 32) @ weight.t()
        del result, weight
        if not cache_enabled:
            assert source.expired()
    assert source.expired()


@pytest.mark.skipif(not tp.cuda.is_available(), reason="GPU required")
def test_transposed_cast_memory_does_not_accumulate():
    tp.clear_autocast_cache()
    tp.cuda.synchronize()
    initial = tp.cuda.memory_allocated()
    for _ in range(4):
        with tp.autocast("cuda"):
            weight = tp.randn(1024, 1024, device="cuda", requires_grad=True)
            result = tp.randn(2, 1024, device="cuda") @ weight.t()
        del result, weight
        tp.cuda.synchronize()
        assert tp.cuda.memory_allocated() == initial
