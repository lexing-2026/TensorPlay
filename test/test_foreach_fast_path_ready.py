"""Native fast-path restriction check used by the fused optimizer dispatch."""

import tensorplay as tp

check = tp._C._foreach_fast_path_ready


def _lists(n=3, dtype=tp.float32):
    params = [tp.randn(4, 5, dtype=dtype) for _ in range(n)]
    grads = [tp.randn(4, 5, dtype=dtype) for _ in range(n)]
    return params, grads


def test_homogeneous_contiguous_lists_are_ready():
    params, grads = _lists()
    steps = [tp.tensor(0.0) for _ in params]
    assert check([params, grads], steps, True)
    assert check([params, grads, []], [], False)


def test_mismatched_dtype_shape_or_layout_is_rejected():
    params, grads = _lists()
    assert not check([params, [g.double() for g in grads]], [], False)
    assert not check([params, [tp.randn(5, 4) for _ in grads]], [], False)
    assert not check([params, [tp.randn(5, 4).t() for _ in grads]], [], False)
    assert not check([params, grads[:2]], [], False)


def test_step_placement_and_size_are_checked():
    params, grads = _lists()
    assert not check([params, grads], [tp.zeros(2) for _ in params], True)
    assert not check([params, grads], [tp.tensor(0.0) for _ in params[:2]], True)
    if tp.cuda.is_available():
        cuda_params = [p.cuda() for p in params]
        cuda_grads = [g.cuda() for g in grads]
        host_steps = [tp.tensor(0.0) for _ in params]
        device_steps = [tp.tensor(0.0, device="cuda") for _ in params]
        assert check([cuda_params, cuda_grads], host_steps, True)
        assert not check([cuda_params, cuda_grads], host_steps, False)
        assert check([cuda_params, cuda_grads], device_steps, False)
