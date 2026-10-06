"""Backward workers preserve nested vectorization and restore their local state."""
import pytest

import tensorplay as tp


DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nested", [False, True])
def test_batched_vjp_preserves_worker_transform_state(device, nested):
    x = (tp.arange(12, dtype=tp.float64, device=device) / 10).reshape(3, 4)
    x.requires_grad_(True)
    y = x.sin()
    shape = (2, 3, 3, 4) if nested else (3, 3, 4)
    count = 72 if nested else 36
    vectors = tp.arange(count, dtype=tp.float64, device=device).reshape(shape) / count

    def vjp(v):
        return tp.autograd.grad(y, x, v, retain_graph=True)[0]

    mapped = tp.vmap(tp.vmap(vjp)) if nested else tp.vmap(vjp)
    for _ in range(3):
        assert tp.allclose(mapped(vectors), vectors * x.cos(), atol=1e-12, rtol=1e-12)
    # A later graph runs without the previous vectorization state.
    y.sum().backward()
    assert tp.allclose(x.grad, x.cos(), atol=1e-12, rtol=1e-12)
