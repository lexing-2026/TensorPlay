"""Torch parity tests for matmul's shape, dtype, layout, and autograd contract."""

import numpy as np
import pytest
import torch

import tensorplay as tp


TP_DTYPES = {
    torch.uint8: tp.uint8,
    torch.float16: tp.float16,
    torch.bfloat16: tp.bfloat16,
    torch.float32: tp.float32,
    torch.float64: tp.float64,
    torch.int8: tp.int8,
    torch.int16: tp.int16,
    torch.int32: tp.int32,
    torch.int64: tp.int64,
    torch.complex64: tp.complex64,
    torch.complex128: tp.complex128,
}


def _torch_host_numpy(value):
    value = value.detach().cpu()
    # NumPy has no bfloat16 dtype; TensorPlay exposes bfloat16 as float32 at
    # the Python/NumPy boundary, just as its existing dtype tests do.
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy()


def _tp_numpy(value):
    return value.cpu().numpy() if str(value.device).startswith("cuda") else value.numpy()


def _make_tp(value, device="cpu", requires_grad=False):
    dtype = TP_DTYPES[value.dtype]
    host = tp.tensor(_torch_host_numpy(value), dtype=dtype)
    if device == "cpu":
        return tp.tensor(_torch_host_numpy(value), dtype=dtype, requires_grad=requires_grad)

    # Keep this path explicit: constructing a CUDA TensorPlay tensor directly
    # from a NumPy object is a separate binding issue, while CPU -> CUDA copy
    # exercises the matmul implementation itself.
    result = tp.empty(
        tuple(value.shape),
        dtype=dtype,
        device=tp.device("cuda"),
        requires_grad=requires_grad,
    )
    result.copy_(host)
    return result


def _torch_inputs(shape_a, shape_b, dtype=torch.float32):
    if dtype.is_complex:
        real_a = torch.randn(shape_a, dtype=torch.float32)
        imag_a = torch.randn(shape_a, dtype=torch.float32)
        real_b = torch.randn(shape_b, dtype=torch.float32)
        imag_b = torch.randn(shape_b, dtype=torch.float32)
        return (real_a + 1j * imag_a).to(dtype), (real_b + 1j * imag_b).to(dtype)
    if dtype.is_floating_point:
        if dtype == torch.bfloat16:
            return (
                torch.randn(shape_a, dtype=torch.float32).to(dtype),
                torch.randn(shape_b, dtype=torch.float32).to(dtype),
            )
        return torch.randn(shape_a, dtype=dtype), torch.randn(shape_b, dtype=dtype)
    if dtype == torch.uint8:
        return (
            torch.randint(0, 256, shape_a, dtype=dtype),
            torch.randint(0, 256, shape_b, dtype=dtype),
        )
    return (
        torch.randint(-5, 6, shape_a, dtype=dtype),
        torch.randint(-5, 6, shape_b, dtype=dtype),
    )


MATMUL_SHAPES = (
    ((3,), (3,)),
    ((3,), (3, 2)),
    ((2, 3), (3,)),
    ((2, 3), (3, 4)),
    ((5, 2, 3), (3, 4)),
    ((2, 1, 3, 4), (5, 4, 6)),
    ((2, 3, 4), (1, 4, 5)),
    ((2, 3, 4), (2, 4, 5)),
    ((0, 3), (3, 2)),
    ((2, 0), (0, 4)),
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matmul_forward_matches_torch_across_shapes(device):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("requires CUDA build of TensorPlay")

    torch.manual_seed(20260821)
    for shape_a, shape_b in MATMUL_SHAPES:
        a, b = _torch_inputs(shape_a, shape_b)
        torch_a = a.to(device)
        torch_b = b.to(device)
        torch_out = torch.matmul(torch_a, torch_b)

        tp_out = tp.matmul(_make_tp(a, device), _make_tp(b, device))
        assert tuple(tp_out.shape) == tuple(torch_out.shape)
        assert tp_out.dtype == tp.float32
        np.testing.assert_allclose(
            _tp_numpy(tp_out),
            _torch_host_numpy(torch_out),
            rtol=2e-4,
            atol=2e-4,
        )

    # A transposed matrix and a broadcasted batch must retain Torch's
    # non-contiguous/zero-stride semantics.
    a_base, b = _torch_inputs((2, 3, 4), (1, 3, 5))
    torch_a = a_base.transpose(-1, -2).to(device)
    torch_b = b.to(device)
    torch_out = torch.matmul(torch_a, torch_b)
    tp_a = _make_tp(a_base, device).transpose(-1, -2)
    tp_b = _make_tp(b, device)
    tp_out = tp.matmul(tp_a, tp_b)
    np.testing.assert_allclose(
        _tp_numpy(tp_out),
        _torch_host_numpy(torch_out),
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.complex64,
        torch.complex128,
    ],
)
def test_matmul_same_dtype_matches_torch(dtype):
    a, b = _torch_inputs((2, 3), (3, 4), dtype)
    torch_out = torch.matmul(a, b)
    tp_out = tp.matmul(_make_tp(a), _make_tp(b))
    assert tp_out.dtype == TP_DTYPES[dtype]
    rtol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    atol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-4
    np.testing.assert_allclose(
        _tp_numpy(tp_out), _torch_host_numpy(torch_out), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matmul_out_matches_torch(device):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("requires CUDA build of TensorPlay")

    a, b = _torch_inputs((2, 1, 3, 4), (5, 4, 6))
    torch_a = a.to(device)
    torch_b = b.to(device)
    expected = torch.matmul(torch_a, torch_b)

    # Exercise both the existing-storage path and the write-only resize path.
    cases = (
        (torch.empty((2, 5, 3, 6), dtype=a.dtype, device=device),
         tp.empty((2, 5, 3, 6), dtype=tp.float32,
                  device=tp.device("cuda")) if device == "cuda" else
         tp.empty((2, 5, 3, 6), dtype=tp.float32)),
        (torch.empty((1, 1), dtype=a.dtype, device=device),
         tp.empty((1, 1), dtype=tp.float32,
                  device=tp.device("cuda")) if device == "cuda" else
         tp.empty((1, 1), dtype=tp.float32)),
    )
    # A non-contiguous output must still be written in its existing layout.
    torch_out_base = torch.empty((2, 3, 5, 6), dtype=a.dtype, device=device)
    tp_out_base = (tp.empty((2, 3, 5, 6), dtype=tp.float32,
                            device=tp.device("cuda")) if device == "cuda" else
                  tp.empty((2, 3, 5, 6), dtype=tp.float32))
    cases += ((torch_out_base.transpose(1, 2), tp_out_base.transpose(1, 2)),)

    for torch_out, tp_out in cases:
        torch_result = torch.matmul(torch_a, torch_b, out=torch_out)
        tp_result = tp.matmul(_make_tp(a, device), _make_tp(b, device), out=tp_out)
        assert torch_result is torch_out
        assert tp_result._impl_id == tp_out._impl_id
        assert tuple(tp_out.shape) == tuple(expected.shape)
        np.testing.assert_allclose(
            _tp_numpy(tp_out), _torch_host_numpy(expected), rtol=2e-4, atol=2e-4
        )


def test_matmul_out_rejects_dtype_and_autograd_like_torch():
    a = torch.randn((2, 3), dtype=torch.float32)
    b = torch.randn((3, 4), dtype=torch.float32)
    tp_a, tp_b = _make_tp(a), _make_tp(b)
    with pytest.raises(RuntimeError, match="dtype"):
        tp.matmul(tp_a, tp_b, out=tp.empty((2, 4), dtype=tp.float64))

    tp_a_req = _make_tp(a, requires_grad=True)
    with pytest.raises(RuntimeError, match="automatic differentiation"):
        tp.matmul(tp_a_req, tp_b, out=tp.empty((2, 4), dtype=tp.float32))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_matmul_matches_torch_on_both_devices(device, dtype):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("requires CUDA build of TensorPlay")

    a, b = _torch_inputs((2, 1, 3, 4), (5, 4, 6), dtype)
    torch_out = torch.matmul(a.to(device), b.to(device))
    tp_out = tp.matmul(_make_tp(a, device), _make_tp(b, device))
    assert tp_out.dtype == TP_DTYPES[dtype]
    np.testing.assert_allclose(
        _tp_numpy(tp_out), _torch_host_numpy(torch_out), rtol=2e-4, atol=2e-4
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_matmul_high_rank_broadcast_matches_torch(device, dtype):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("requires CUDA build of TensorPlay")

    # Eleven-dimensional operands exercise the CUDA strided clone path that
    # is needed when both sides contribute non-singleton broadcast batches.
    a_shape = (2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3)
    b_shape = (1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 4)
    a, b = _torch_inputs(a_shape, b_shape, dtype)
    torch_out = torch.matmul(a.to(device), b.to(device))
    tp_out = tp.matmul(_make_tp(a, device), _make_tp(b, device))
    np.testing.assert_allclose(
        _tp_numpy(tp_out), _torch_host_numpy(torch_out), rtol=2e-4, atol=2e-4
    )


def test_cpu_matmul_rejects_dtypes_torch_does_not_implement():
    unsupported = (tp.bool, tp.uint16, tp.uint32, tp.uint64, tp.complex32)
    for dtype in unsupported:
        a = tp.ones((2, 3), dtype=dtype)
        b = tp.ones((3, 4), dtype=dtype)
        with pytest.raises(NotImplementedError):
            tp.matmul(a, b)


@pytest.mark.parametrize(
    "dtype",
    [tp.bool, tp.int8, tp.uint8, tp.int16, tp.int32, tp.int64],
)
def test_cuda_matmul_rejects_integer_and_bool_even_for_empty_output(dtype):
    if not tp.cuda.is_available():
        pytest.skip("requires CUDA build of TensorPlay")

    a = tp.ones((0, 3), dtype=dtype, device=tp.device("cuda"))
    b = tp.ones((3, 4), dtype=dtype, device=tp.device("cuda"))
    with pytest.raises(NotImplementedError):
        tp.matmul(a, b)


def test_matmul_rejects_mixed_dtypes_like_torch():
    a = torch.randn((2, 3), dtype=torch.float32)
    b = torch.randn((3, 4), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="same dtype"):
        torch.matmul(a, b)
    with pytest.raises(RuntimeError, match="same dtype"):
        tp.matmul(_make_tp(a), _make_tp(b))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matmul_rejects_invalid_rank_and_shapes_like_torch(device):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("requires CUDA build of TensorPlay")

    invalid_cases = (
        (torch.tensor(1.0), torch.randn(2, 2)),
        (torch.randn(2, 3), torch.randn(4, 2)),
        (torch.randn(2, 3, 4), torch.randn(5, 4, 6)),
        (torch.randn(3), torch.randn(4)),
    )
    for a, b in invalid_cases:
        with pytest.raises(RuntimeError):
            torch.matmul(a.to(device), b.to(device))
        with pytest.raises(RuntimeError):
            tp.matmul(_make_tp(a, device), _make_tp(b, device))


AUTOGRAD_SHAPES = (
    ((3,), (3,)),
    ((3,), (3, 2)),
    ((2, 3), (3,)),
    ((2, 3), (3, 4)),
    ((5, 2, 3), (3, 4)),
    ((2, 1, 3, 4), (5, 4, 6)),
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matmul_backward_matches_torch(device):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("requires CUDA build of TensorPlay")

    torch.manual_seed(20260822)
    for shape_a, shape_b in AUTOGRAD_SHAPES:
        a_host, b_host = _torch_inputs(shape_a, shape_b)
        torch_a = a_host.to(device).requires_grad_()
        torch_b = b_host.to(device).requires_grad_()
        torch_out = torch.matmul(torch_a, torch_b)
        torch_loss = torch_out.square()
        if torch_loss.numel() != 1:
            torch_loss = torch_loss.sum()
        torch_loss.backward()

        tp_a = _make_tp(a_host, device, requires_grad=True)
        tp_b = _make_tp(b_host, device, requires_grad=True)
        tp_out = tp.matmul(tp_a, tp_b)
        tp_loss = tp_out * tp_out
        if tp_loss.numel() != 1:
            tp_loss = tp_loss.sum()
        tp_loss.backward()

        np.testing.assert_allclose(
            _tp_numpy(tp_a.grad),
            _torch_host_numpy(torch_a.grad),
            rtol=2e-4,
            atol=2e-4,
        )
        np.testing.assert_allclose(
            _tp_numpy(tp_b.grad),
            _torch_host_numpy(torch_b.grad),
            rtol=2e-4,
            atol=2e-4,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "shapes",
    [((2, 3), (3, 4)), ((2, 1, 3, 4), (5, 4, 6)), ((2, 3, 4), (1, 4, 5))],
)
def test_complex_matmul_backward_helpers_match_torch(device, dtype, shapes):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("requires CUDA build of TensorPlay")

    a_shape, b_shape = shapes
    a_host, b_host = _torch_inputs(a_shape, b_shape, dtype)
    torch_a = a_host.to(device).requires_grad_()
    torch_b = b_host.to(device).requires_grad_()
    torch_output = torch.matmul(torch_a, torch_b)
    grad_host = _torch_inputs(tuple(torch_output.shape), tuple(torch_output.shape), dtype)[0]
    torch_grad_a, torch_grad_b = torch.autograd.grad(
        torch_output,
        (torch_a, torch_b),
        grad_outputs=grad_host.to(device),
    )

    tp_a = _make_tp(a_host, device)
    tp_b = _make_tp(b_host, device)
    tp_grad = _make_tp(grad_host, device)
    tp_grad_a = tp.matmul_backward_self(tp_grad, tp_a, tp_b)
    tp_grad_b = tp.matmul_backward_other(tp_grad, tp_a, tp_b)
    np.testing.assert_allclose(
        _tp_numpy(tp_grad_a), _torch_host_numpy(torch_grad_a), rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(
        _tp_numpy(tp_grad_b), _torch_host_numpy(torch_grad_b), rtol=2e-4, atol=2e-4
    )

    # Exercise the generated autograd formula as well as the public backward
    # helpers used by that formula.
    tp_a_req = _make_tp(a_host, device, requires_grad=True)
    tp_b_req = _make_tp(b_host, device, requires_grad=True)
    tp.matmul(tp_a_req, tp_b_req).backward(tp_grad)
    np.testing.assert_allclose(
        _tp_numpy(tp_a_req.grad), _torch_host_numpy(torch_grad_a), rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(
        _tp_numpy(tp_b_req.grad), _torch_host_numpy(torch_grad_b), rtol=2e-4, atol=2e-4
    )
