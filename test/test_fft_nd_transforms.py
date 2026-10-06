"""n-dimensional and Hermitian transforms, the primitive transforms under
them, and the frequency-grid helpers.

The n-D and Hermitian ops resize their input and run one primitive transform
(_fft_c2c / _fft_r2c / _fft_c2r) over every listed dimension; the primitives
scale the whole signal once by their `normalization` (0 none, 1 by
1/sqrt(n), 2 by 1/n) whatever the direction.  Everything is checked against
numpy, and gradients through gradcheck/gradgradcheck.
"""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import tensorplay as tp
from tensorplay import functional as F
from tensorplay.autograd import gradcheck, gradgradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
NORMS = ["backward", "forward", "ortho"]


def as_tensor(array, device="cpu"):
    return tp.from_numpy(np.ascontiguousarray(array)).to(device)


def as_numpy(value):
    return (value.cpu() if value.is_cuda else value).numpy()


def signal(shape, complex_input, seed=0):
    rng = np.random.default_rng(seed)
    array = rng.standard_normal(shape)
    if complex_input:
        array = array + 1j * rng.standard_normal(shape)
    return array


def real_view(value):
    return tp.view_as_real(value) if value.is_complex() else value


def assert_close(got, expected, tol=1e-10):
    got = as_numpy(got)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    np.testing.assert_allclose(got, expected, rtol=tol, atol=tol)


def hfftn_reference(array, s, axes, norm):
    out = array
    for axis, n in zip(axes[:-1], s[:-1]):
        out = np.fft.fft(out, n=n, axis=axis, norm=norm)
    return np.fft.hfft(out, n=s[-1], axis=axes[-1], norm=norm)


def ihfftn_reference(array, s, axes, norm):
    out = np.fft.ihfft(array, n=s[-1], axis=axes[-1], norm=norm)
    for axis, n in zip(axes[:-1], s[:-1]):
        out = np.fft.ifft(out, n=n, axis=axis, norm=norm)
    return out


# (name, tensorplay call, numpy call, complex input)
ND_CASES = [
    ("fftn", lambda x, n: tp.fft.fftn(x, norm=n), lambda a, n: np.fft.fftn(a, norm=n), True),
    ("fftn_real", lambda x, n: tp.fft.fftn(x, norm=n), lambda a, n: np.fft.fftn(a, norm=n), False),
    ("fftn_s_dim", lambda x, n: tp.fft.fftn(x, s=(5, 3), dim=(2, 0), norm=n),
     lambda a, n: np.fft.fftn(a, s=(5, 3), axes=(2, 0), norm=n), True),
    ("fftn_s_tail", lambda x, n: tp.fft.fftn(x, s=(-1, 2), norm=n),
     lambda a, n: np.fft.fftn(a, s=(a.shape[1], 2), axes=(1, 2), norm=n), True),
    ("ifftn", lambda x, n: tp.fft.ifftn(x, norm=n), lambda a, n: np.fft.ifftn(a, norm=n), True),
    ("ifftn_dim", lambda x, n: tp.fft.ifftn(x, dim=1, norm=n),
     lambda a, n: np.fft.ifftn(a, axes=(1,), norm=n), True),
    ("rfftn", lambda x, n: tp.fft.rfftn(x, norm=n), lambda a, n: np.fft.rfftn(a, norm=n), False),
    ("rfftn_s_dim", lambda x, n: tp.fft.rfftn(x, s=(6, 2), dim=(0, 1), norm=n),
     lambda a, n: np.fft.rfftn(a, s=(6, 2), axes=(0, 1), norm=n), False),
    ("irfftn", lambda x, n: tp.fft.irfftn(x, norm=n), lambda a, n: np.fft.irfftn(a, norm=n), True),
    ("irfftn_s", lambda x, n: tp.fft.irfftn(x, s=(4, 7), norm=n),
     lambda a, n: np.fft.irfftn(a, s=(4, 7), axes=(1, 2), norm=n), True),
    ("irfftn_dim", lambda x, n: tp.fft.irfftn(x, dim=(2, 0), norm=n),
     lambda a, n: np.fft.irfftn(a, axes=(2, 0), norm=n), True),
    ("hfftn", lambda x, n: tp.fft.hfftn(x, norm=n),
     lambda a, n: hfftn_reference(a, [None, None, 2 * (a.shape[2] - 1)], (0, 1, 2), n), True),
    ("hfftn_s", lambda x, n: tp.fft.hfftn(x, s=(5, 9), norm=n),
     lambda a, n: hfftn_reference(a, (5, 9), (1, 2), n), True),
    ("hfft2_dim", lambda x, n: tp.fft.hfft2(x, dim=(2, 0), norm=n),
     lambda a, n: hfftn_reference(a, (None, 2 * (a.shape[0] - 1)), (2, 0), n), True),
    ("ihfftn", lambda x, n: tp.fft.ihfftn(x, norm=n),
     lambda a, n: ihfftn_reference(a, [None] * 3, (0, 1, 2), n), False),
    ("ihfftn_s", lambda x, n: tp.fft.ihfftn(x, s=(2, 8), dim=(0, 2), norm=n),
     lambda a, n: ihfftn_reference(a, (2, 8), (0, 2), n), False),
    ("ihfft2", lambda x, n: tp.fft.ihfft2(x, norm=n),
     lambda a, n: ihfftn_reference(a, (None, None), (1, 2), n), False),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("case", ND_CASES, ids=[c[0] for c in ND_CASES])
def test_nd_transforms_match_numpy(device, norm, case):
    _, fn, reference, complex_input = case
    array = signal((3, 4, 5), complex_input)
    assert_close(fn(as_tensor(array, device), norm), reference(array, norm))


@pytest.mark.parametrize("device", DEVICES)
def test_hermitian_one_dim_signal_length(device):
    spectrum = signal((3, 6), True, seed=1)
    real = signal((3, 6), False, seed=2)
    x = as_tensor(spectrum, device)
    assert_close(tp.fft.hfft(x, n=7, dim=0, norm="ortho"),
                 np.fft.hfft(spectrum, n=7, axis=0, norm="ortho"))
    assert_close(tp.fft.hfft(x, n=4), np.fft.hfft(spectrum, n=4))
    assert_close(tp.fft.ihfft(as_tensor(real, device), n=9, norm="forward"),
                 np.fft.ihfft(real, n=9, norm="forward"))


@pytest.mark.parametrize("device", DEVICES)
def test_inputs_are_promoted(device):
    ints = np.arange(24).reshape(2, 3, 4)
    x = as_tensor(ints, device)
    assert tp.fft.fftn(x).dtype == tp.complex64
    assert_close(tp.fft.fftn(x), np.fft.fftn(ints).astype(np.complex64), tol=1e-4)
    assert tp.fft.rfftn(x).dtype == tp.complex64
    assert tp.fft.ihfftn(x).dtype == tp.complex64
    # Complex-to-real transforms accept a real input as a complex one.
    real = signal((3, 5), False)
    assert_close(tp.fft.irfftn(as_tensor(real, device)), np.fft.irfftn(real))
    assert_close(tp.fft.hfft(as_tensor(real, device)), np.fft.hfft(real))
    assert tp.fft.fftn(as_tensor(real.astype(np.float32), device)).dtype == tp.complex64


@pytest.mark.parametrize("device", DEVICES)
def test_empty_transform_dims_copy_the_input(device):
    x = as_tensor(signal((2, 3), True), device)
    y = tp.fft.fftn(x, dim=[])
    assert tp.equal(y, x)
    y.zero_()
    assert not tp.equal(y, x)


@pytest.mark.parametrize("device", DEVICES)
def test_out_variants(device):
    array = signal((3, 4, 5), True)
    x = as_tensor(array, device)
    out = tp.empty(0, dtype=tp.complex128, device=device)
    result = tp.fft.fftn(x, dim=(0, 2), out=out)
    assert result.data_ptr() == out.data_ptr()
    assert_close(out, np.fft.fftn(array, axes=(0, 2)))
    real_out = tp.empty(0, dtype=tp.float64, device=device)
    tp.fft.hfftn(x, out=real_out)
    assert_close(real_out, hfftn_reference(array, [None, None, 8], (0, 1, 2), None))
    with pytest.raises(RuntimeError, match="complex output"):
        tp.fft.fftn(x, out=tp.empty(0, dtype=tp.float64, device=device))
    with pytest.raises(RuntimeError, match="floating point output"):
        tp.fft.hfftn(x, out=tp.empty(0, dtype=tp.complex128, device=device))


def test_invalid_arguments():
    x = tp.randn(3, 4, dtype=tp.complex64)
    with pytest.raises(RuntimeError, match="Invalid number of data points"):
        tp.fft.fftn(x, s=(0, 4))
    with pytest.raises(RuntimeError, match="Invalid number of data points"):
        tp.fft.fft(x, n=0)
    with pytest.raises(RuntimeError, match="Invalid number of data points"):
        tp.fft.hfft(x, n=0)
    with pytest.raises(RuntimeError, match="dims must be unique"):
        tp.fft.fftn(x, dim=(0, -2))
    with pytest.raises(RuntimeError, match="same length"):
        tp.fft.fftn(x, s=(2, 2), dim=(0,))
    with pytest.raises(RuntimeError, match="Invalid normalization mode"):
        tp.fft.ifftn(x, norm="sideways")
    with pytest.raises(TypeError, match="real input"):
        tp.fft.rfftn(x)
    with pytest.raises(TypeError, match="real input"):
        tp.fft.ihfft(x)
    with pytest.raises(RuntimeError, match="at least one axis"):
        tp.fft.irfftn(x, dim=[])


# --- Primitive transforms ----------------------------------------------------

SCALE = {0: lambda n: 1.0, 1: lambda n: 1.0 / np.sqrt(n), 2: lambda n: 1.0 / n}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("normalization", [0, 1, 2])
@pytest.mark.parametrize("dims", [[1], [0, 2], [2, 0, 1]])
def test_primitives_scale_the_whole_signal_once(device, normalization, dims):
    shape = (4, 3, 6)
    n = int(np.prod([shape[d] for d in dims]))
    scale = SCALE[normalization](n)
    z = signal(shape, True, seed=3)
    r = signal(shape, False, seed=4)
    zt, rt = as_tensor(z, device), as_tensor(r, device)

    assert_close(F._fft_c2c(zt, dims, normalization, True),
                 np.fft.fftn(z, axes=dims) * scale)
    assert_close(F._fft_c2c(zt, dims, normalization, False),
                 np.fft.ifftn(z, axes=dims) * n * scale)
    assert_close(F._fft_r2c(rt, dims, normalization, True),
                 np.fft.rfftn(r, axes=dims) * scale)
    # Two-sided: the half spectrum completed by conjugate symmetry.
    assert_close(F._fft_r2c(rt, dims, normalization, False),
                 np.fft.fftn(r, axes=dims) * scale)
    last = 7
    half = as_tensor(np.fft.rfftn(signal(shape[:dims[-1]] + (last,) + shape[dims[-1] + 1:],
                                         False, seed=5), axes=dims), device)
    out_n = n // shape[dims[-1]] * last
    assert_close(F._fft_c2r(half, dims, normalization, last),
                 np.fft.irfftn(as_numpy(half), s=[shape[d] for d in dims[:-1]] + [last],
                               axes=dims) * out_n * SCALE[normalization](out_n))


@pytest.mark.parametrize("device", DEVICES)
def test_primitives_read_strided_inputs(device):
    z = as_tensor(signal((5, 4, 6), True, seed=6), device)
    view = z.transpose(0, 2)[1:, :, ::2]
    expected = np.fft.fftn(as_numpy(view.contiguous()), axes=(0, 2))
    assert_close(F._fft_c2c(view, [0, 2], 0, True), expected)
    r = as_tensor(signal((6, 5), False, seed=7), device).t()
    assert_close(F._fft_r2c(r, [0, 1], 0, True), np.fft.rfftn(as_numpy(r.contiguous())))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["irfft2", "irfftn"])
def test_inverse_real_transforms_preserve_input(device, name):
    x = as_tensor(signal((3, 4), True, seed=14), device)
    original = x.clone()
    fn = getattr(tp.fft, name)
    expected = fn(x)
    assert tp.equal(x, original)
    out = tp.empty(0, dtype=expected.dtype, device=device)
    assert fn(x, out=out).data_ptr() == out.data_ptr()
    assert_close(out, as_numpy(expected))
    assert tp.equal(x, original)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fft_backward_initializes_a_fresh_worker():
    code = """
import tensorplay as tp
from tensorplay import functional as F
x = tp.randn(64, 128, 128, dtype=tp.complex64, device='cuda', requires_grad=True)
y = F._fft_c2c(x, [0, 1, 2], 0, True)
g = tp.autograd.grad(y, x, tp.ones_like(y))[0]
expected = tp.zeros_like(x)
expected[0, 0, 0] = x.numel()
assert tp.allclose(g, expected, atol=1e-4, rtol=1e-5)
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=60)


PRIMITIVES = [
    ("c2c", lambda t: F._fft_c2c(t, [0, 2], 1, True), True, (3, 2, 4)),
    ("c2c_inverse", lambda t: F._fft_c2c(t, [1], 2, False), True, (3, 2, 4)),
    ("r2c", lambda t: F._fft_r2c(t, [0, 2], 0, True), False, (3, 2, 5)),
    ("r2c_twosided", lambda t: F._fft_r2c(t, [2, 1], 2, False), False, (3, 2, 4)),
    ("c2r", lambda t: F._fft_c2r(t, [0, 2], 1, 6), True, (3, 2, 4)),
    ("c2r_odd", lambda t: F._fft_c2r(t, [2], 0, 5), True, (3, 2, 3)),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", PRIMITIVES, ids=[c[0] for c in PRIMITIVES])
def test_primitives_differentiate(device, case):
    _, fn, complex_input, shape = case
    x = as_tensor(signal(shape, complex_input, seed=8), device).requires_grad_(True)
    assert gradcheck(lambda t: real_view(fn(t)), (x,), atol=1e-6, rtol=1e-4,
                     check_forward_ad=True, check_batched_grad=True,
                     check_batched_forward_grad=True)
    assert gradgradcheck(lambda t: real_view(fn(t)), (x,), atol=1e-6, rtol=1e-4)


ND_GRAD_CASES = [
    ("fftn", lambda t: tp.fft.fftn(t, s=(4, 3), dim=(0, 2), norm="ortho"), True),
    ("ifftn_real", lambda t: tp.fft.ifftn(t, dim=(1, 2)), False),
    ("rfftn", lambda t: tp.fft.rfftn(t, s=(2, 6)), False),
    ("irfftn", lambda t: tp.fft.irfftn(t, s=(3, 2, 5), norm="forward"), True),
    ("hfftn", lambda t: tp.fft.hfftn(t, dim=(0, 2)), True),
    ("hfft2", lambda t: tp.fft.hfft2(t, s=(2, 6)), True),
    ("ihfftn", lambda t: tp.fft.ihfftn(t, norm="ortho"), False),
    ("ihfft2", lambda t: tp.fft.ihfft2(t, s=(3, 5)), False),
    ("hfft", lambda t: tp.fft.hfft(t, n=5, dim=0), True),
    ("ihfft", lambda t: tp.fft.ihfft(t, n=6), False),
    ("fftshift", lambda t: tp.fft.fftshift(t, dim=(0, 2)), False),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ND_GRAD_CASES, ids=[c[0] for c in ND_GRAD_CASES])
def test_nd_transforms_differentiate(device, case):
    _, fn, complex_input = case
    x = as_tensor(signal((3, 2, 4), complex_input, seed=9), device).requires_grad_(True)
    assert gradcheck(lambda t: real_view(fn(t)), (x,), atol=1e-6, rtol=1e-4,
                     check_forward_ad=True, check_batched_grad=True,
                     check_batched_forward_grad=True)
    assert gradgradcheck(lambda t: real_view(fn(t)), (x,), atol=1e-6, rtol=1e-4)


VMAP_CASES = [
    ("fft", lambda t: tp.fft.fft(t, n=5, dim=0), True),
    ("ifft", lambda t: tp.fft.ifft(t, dim=-1, norm="ortho"), True),
    ("rfft", lambda t: tp.fft.rfft(t, n=5, dim=0), False),
    ("irfft", lambda t: tp.fft.irfft(t, n=5), True),
    ("fft2", lambda t: tp.fft.fft2(t, s=(5, 3)), True),
    ("ifft2", lambda t: tp.fft.ifft2(t, dim=(1, 0)), True),
    ("rfft2", lambda t: tp.fft.rfft2(t), False),
    ("irfft2", lambda t: tp.fft.irfft2(t, s=(3, 5)), True),
    ("hfft", lambda t: tp.fft.hfft(t, n=5), True),
    ("ihfft", lambda t: tp.fft.ihfft(t, n=5), False),
    ("hfft2", lambda t: tp.fft.hfft2(t), True),
    ("ihfft2", lambda t: tp.fft.ihfft2(t), False),
    ("fftn", lambda t: tp.fft.fftn(t), True),
    ("ifftn", lambda t: tp.fft.ifftn(t, s=(5,)), True),
    ("rfftn", lambda t: tp.fft.rfftn(t, dim=(1, 0)), False),
    ("irfftn", lambda t: tp.fft.irfftn(t, s=(3, 5)), True),
    ("hfftn", lambda t: tp.fft.hfftn(t), True),
    ("ihfftn", lambda t: tp.fft.ihfftn(t), False),
    ("fftshift", lambda t: tp.fft.fftshift(t), False),
    ("ifftshift", lambda t: tp.fft.ifftshift(t), False),
    ("c2c", lambda t: F._fft_c2c(t, [1, 0], 1, False), True),
    ("r2c", lambda t: F._fft_r2c(t, [0, 1], 2, True), False),
    ("c2r", lambda t: F._fft_c2r(t, [0, 1], 0, 5), True),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("in_dim", [0, 1, 2])
@pytest.mark.parametrize("case", VMAP_CASES, ids=[c[0] for c in VMAP_CASES])
def test_vmap_transforms_each_sample(device, in_dim, case):
    _, fn, complex_input = case
    x = as_tensor(signal((2, 3, 4), complex_input, seed=11), device)
    got = tp.vmap(fn, in_dims=in_dim)(x)
    expected = tp.stack([fn(x.select(in_dim, i)) for i in range(x.size(in_dim))])
    assert_close(got, as_numpy(expected))


@pytest.mark.parametrize("device", DEVICES)
def test_nested_vmap_and_gradients(device):
    x = as_tensor(signal((2, 3, 4, 5), False, seed=12), device)
    fn = lambda t: tp.fft.rfftn(t, norm="ortho")
    got = tp.vmap(tp.vmap(fn))(x)
    expected = tp.stack([tp.stack([fn(sample) for sample in group]) for group in x])
    assert_close(got, as_numpy(expected))
    loss = lambda t: tp.view_as_real(fn(t)).square().sum()
    got_grad = tp.vmap(tp.func.grad(loss))(x[0])
    expected_grad = tp.stack([tp.func.grad(loss)(sample) for sample in x[0]])
    assert_close(got_grad, as_numpy(expected_grad))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("in_dim", [0, 1, 2])
@pytest.mark.parametrize("dims", [[], [0, -1]])
def test_vmap_roll_preserves_sample_boundaries(device, in_dim, dims):
    x = as_tensor(signal((2, 3, 4), False, seed=15), device)
    shifts = [3] if not dims else [1, -2]
    fn = lambda t: tp.roll(t, shifts, dims)
    got = tp.vmap(fn, in_dims=in_dim)(x)
    expected = tp.stack([fn(x.select(in_dim, i)) for i in range(x.size(in_dim))])
    assert_close(got, as_numpy(expected))


# --- Spectrum re-ordering and frequency grids ---------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_shifts_match_numpy(device):
    array = np.arange(60.0).reshape(3, 4, 5)
    x = as_tensor(array, device)
    assert_close(tp.fft.fftshift(x), np.fft.fftshift(array))
    assert_close(tp.fft.ifftshift(x), np.fft.ifftshift(array))
    assert_close(tp.fft.fftshift(x, dim=1), np.fft.fftshift(array, axes=1))
    assert_close(tp.fft.ifftshift(x, dim=(0, -1)), np.fft.ifftshift(array, axes=(0, -1)))
    assert_close(tp.fft.ifftshift(tp.fft.fftshift(x)), array)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [1, 2, 7, 8])
def test_frequency_grids_match_numpy(device, n):
    got = tp.fft.fftfreq(n, d=0.25, dtype=tp.float64, device=device)
    assert got.device.type == device
    assert_close(got, np.fft.fftfreq(n, d=0.25))
    assert_close(tp.fft.rfftfreq(n, 2.0, dtype=tp.float64, device=device),
                 np.fft.rfftfreq(n, d=2.0))
    assert tp.fft.fftfreq(n).dtype == tp.get_default_dtype()
    complex_grid = tp.fft.rfftfreq(n, dtype=tp.complex128, device=device)
    assert_close(complex_grid, np.fft.rfftfreq(n).astype(np.complex128))


def test_frequency_grid_edges():
    assert tp.fft.fftfreq(0).shape == (0,)
    with pytest.raises(RuntimeError, match="negative dimension"):
        tp.fft.fftfreq(-1)
    with pytest.raises(RuntimeError, match="floating point or complex"):
        tp.fft.fftfreq(4, dtype=tp.int64)
    out = tp.empty(1, dtype=tp.float64)
    tp.fft.fftfreq(5, out=out)
    assert_close(out, np.fft.fftfreq(5))
    assert tp.fft.rfftfreq(4, requires_grad=True).requires_grad


# --- Compiled graphs -------------------------------------------------------------

COMPILED = [
    ("fftn", lambda a: tp.fft.fftn(a, dim=(0, 2), norm="ortho"), True),
    ("fftn_real", lambda a: tp.fft.fftn(a, norm="forward"), False),
    ("ifftn", lambda a: tp.fft.ifftn(a, s=(5, 3)), True),
    ("rfftn", lambda a: tp.fft.rfftn(a), False),
    ("irfftn", lambda a: tp.fft.irfftn(a, s=(4, 6)), True),
    ("hfftn", lambda a: tp.fft.hfftn(a, norm="forward"), True),
    ("ihfftn", lambda a: tp.fft.ihfftn(a, dim=(1, 2)), False),
    ("hfft", lambda a: tp.fft.hfft(a, n=7), True),
    ("ihfft", lambda a: tp.fft.ihfft(a, norm="ortho"), False),
    ("ifft", lambda a: tp.fft.ifft(a, dim=1), True),
    ("irfft", lambda a: tp.fft.irfft(a, n=9), True),
    ("fftshift", lambda a: tp.fft.ifftshift(tp.fft.fftshift(a, dim=1)), False),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", COMPILED, ids=[c[0] for c in COMPILED])
def test_compiled_transforms_match_eager(device, case):
    _, fn, complex_input = case
    x = as_tensor(signal((3, 4, 5), complex_input, seed=10), device)
    expected = fn(x)
    compiled = tp.compile(fn, strict_native=True)
    got = compiled(x)
    assert got.dtype == expected.dtype
    assert tuple(got.shape) == tuple(expected.shape)
    assert tp.allclose(got, expected, rtol=1e-10, atol=1e-10)
    eager_input = x.clone().requires_grad_(True)
    compiled_input = x.clone().requires_grad_(True)
    eager_result = real_view(fn(eager_input))
    weights = tp.arange(eager_result.numel(), dtype=tp.float64, device=device)
    weights = weights.reshape(eager_result.shape) / eager_result.numel()
    (eager_result * weights).square().sum().backward()
    (real_view(compiled(compiled_input)) * weights).square().sum().backward()
    assert_close(compiled_input.grad, as_numpy(eager_input.grad), tol=1e-8)
