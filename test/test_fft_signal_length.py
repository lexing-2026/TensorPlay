"""One-dimensional and Hermitian transforms against numpy.

Covers what the 2-D tests do not: a signal length ``n`` that zero-pads or
truncates the transformed dimension, any transformed dimension, inputs of
one and three dimensions, the Hermitian family's scaling and conjugation,
and gradients of those cases (each backward kernel undoes the padding or
truncation its forward applied).
"""
import numpy as np
import pytest

import tensorplay as tp
from tensorplay.autograd import gradcheck, gradgradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def as_tensor(array, device):
    return tp.from_numpy(np.ascontiguousarray(array)).to(device)


def as_numpy(value):
    value = value.cpu() if value.is_cuda else value
    return value.numpy()


def real_view(value):
    return tp.view_as_real(value) if value.is_complex() else value


def signal(shape, complex_input, seed=0):
    rng = np.random.default_rng(seed)
    array = rng.standard_normal(shape)
    if complex_input:
        array = array + 1j * rng.standard_normal(shape)
    return array


# (name, call taking the module that provides `.fft`, complex input)
ONE_DIM = [
    ("fft", lambda m, x: m.fft.fft(x), True),
    ("fft_pad", lambda m, x: m.fft.fft(x, n=9), True),
    ("fft_trunc", lambda m, x: m.fft.fft(x, n=4), True),
    ("fft_real_trunc", lambda m, x: m.fft.fft(x, n=5, dim=-1), False),
    ("fft_dim0_pad", lambda m, x: m.fft.fft(x, n=6, dim=0), True),
    ("ifft_trunc", lambda m, x: m.fft.ifft(x, n=3, dim=0, norm="ortho"), True),
    ("rfft", lambda m, x: m.fft.rfft(x), False),
    ("rfft_pad", lambda m, x: m.fft.rfft(x, n=11), False),
    ("rfft_trunc", lambda m, x: m.fft.rfft(x, n=5, norm="forward"), False),
    ("rfft_dim1", lambda m, x: m.fft.rfft(x, dim=1), False),
    ("irfft", lambda m, x: m.fft.irfft(x), True),
    ("irfft_fewer_bins", lambda m, x: m.fft.irfft(x, n=9), True),
    ("irfft_more_bins", lambda m, x: m.fft.irfft(x, n=16, norm="ortho"), True),
    ("irfft_dim0", lambda m, x: m.fft.irfft(x, dim=0), True),
    ("hfft", lambda m, x: m.fft.hfft(x), True),
    ("hfft_n", lambda m, x: m.fft.hfft(x, n=11, norm="forward"), True),
    ("ihfft", lambda m, x: m.fft.ihfft(x), False),
    ("ihfft_ortho", lambda m, x: m.fft.ihfft(x, n=5, norm="ortho"), False),
]


class _NumpyFFT:
    """np.fft under the frontend's keyword: ``dim`` names numpy's ``axis``."""

    def __getattr__(self, name):
        transform = getattr(np.fft, name)

        def call(x, *args, dim=None, **kwargs):
            if dim is not None:
                kwargs["axis"] = dim
            return transform(x, *args, **kwargs)

        return call


def numpy_call(fn, array):
    class _np:
        fft = _NumpyFFT()
    return fn(_np, array)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(7,), (3, 7), (2, 3, 7)])
@pytest.mark.parametrize("case", ONE_DIM, ids=[c[0] for c in ONE_DIM])
def test_one_dim_transforms_match_numpy(device, shape, case):
    name, fn, complex_input = case
    if "dim0" in name or "dim1" in name:
        if len(shape) < 2:
            pytest.skip("needs a second dimension")
    array = signal(shape, complex_input)
    got = as_numpy(fn(tp, as_tensor(array, device)))
    expected = numpy_call(fn, array)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def hfftn_reference(array, axes, norm):
    out = array
    for axis in axes[:-1]:
        out = np.fft.fft(out, axis=axis, norm=norm)
    return np.fft.hfft(out, axis=axes[-1], norm=norm)


def ihfftn_reference(array, axes, norm):
    out = np.fft.ihfft(array, axis=axes[-1], norm=norm)
    for axis in axes[:-1]:
        out = np.fft.ifft(out, axis=axis, norm=norm)
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
def test_hermitian_nd_transforms_match_their_definition(device, norm):
    spectrum = signal((3, 4, 5), True, seed=1)
    real = signal((3, 4, 5), False, seed=2)
    cases = [
        (tp.fft.hfft2(as_tensor(spectrum, device), norm=norm),
         hfftn_reference(spectrum, (1, 2), norm)),
        (tp.fft.hfftn(as_tensor(spectrum, device), norm=norm),
         hfftn_reference(spectrum, (0, 1, 2), norm)),
        (tp.fft.ihfft2(as_tensor(real, device), norm=norm),
         ihfftn_reference(real, (1, 2), norm)),
        (tp.fft.ihfftn(as_tensor(real, device), norm=norm),
         ihfftn_reference(real, (0, 1, 2), norm)),
    ]
    for got, expected in cases:
        got = as_numpy(got)
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", ONE_DIM, ids=[c[0] for c in ONE_DIM])
def test_one_dim_transforms_differentiate(device, case):
    _, fn, complex_input = case
    x = as_tensor(signal((3, 7), complex_input, seed=3), device).requires_grad_(True)
    assert gradcheck(lambda t: real_view(fn(tp, t)), (x,), atol=1e-6, rtol=1e-4)


# The backward kernels apply the adjoint of each transform; differentiating
# them again applies the transform, so a Hessian-vector product through a
# spectral loss is exact.
SECOND_ORDER = ONE_DIM + [
    ("fft2", lambda m, x: m.fft.fft2(x, s=(2, 9)), True),
    ("ifft2", lambda m, x: m.fft.ifft2(x, norm="forward"), True),
    ("rfft2", lambda m, x: m.fft.rfft2(x, s=(4, 5)), False),
    ("irfft2", lambda m, x: m.fft.irfft2(x, s=(3, 9)), True),
    ("rfftn", lambda m, x: m.fft.rfftn(x), False),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", SECOND_ORDER, ids=[c[0] for c in SECOND_ORDER])
def test_spectral_backward_kernels_differentiate_twice(device, case):
    _, fn, complex_input = case
    x = as_tensor(signal((3, 7), complex_input, seed=4), device).requires_grad_(True)
    assert gradgradcheck(lambda t: real_view(fn(tp, t)), (x,), atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_spectral_hessian_vector_product(device):
    # loss = sum(w * |rfft(x, n)|^2) is quadratic in x, so its Hessian is
    # constant and H @ v equals the change of the gradient along v.
    x = as_tensor(signal((2, 6), False, seed=5), device).requires_grad_(True)
    v = as_tensor(signal((2, 6), False, seed=6), device)
    w = as_tensor(np.linspace(0.5, 1.5, 2 * 5).reshape(2, 5), device)

    def loss(t):
        return (w * tp.fft.rfft(t, n=8).abs() ** 2).sum()

    (g,) = tp.autograd.grad(loss(x), x, create_graph=True)
    (hv,) = tp.autograd.grad((g * v).sum(), x)
    xs = (x.detach() + v).requires_grad_(True)
    (g_shifted,) = tp.autograd.grad(loss(xs), xs)
    np.testing.assert_allclose(
        as_numpy(hv), as_numpy(g_shifted - g.detach()), rtol=1e-10, atol=1e-10)
