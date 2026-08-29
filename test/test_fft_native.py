import numpy as np
import pytest

import tensorplay as tp


def as_tensor(array, device=None):
    value = tp.from_numpy(np.ascontiguousarray(array))
    return value if device is None else value.to(device)


def as_numpy(value):
    return value.cpu().numpy() if value.is_cuda else value.numpy()


@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
@pytest.mark.parametrize("complex_input", [False, True])
def test_fft2_native_cpu(norm, complex_input):
    rng = np.random.default_rng(11)
    real = rng.standard_normal((2, 5, 7), dtype=np.float64)
    array = real + 1j * rng.standard_normal(real.shape) if complex_input else real
    got = as_numpy(tp.fft.fft2(as_tensor(array), norm=norm))
    expected = np.fft.fft2(array, axes=(-2, -1), norm=norm)
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
def test_rfft2_irfft2_native_cpu(norm):
    rng = np.random.default_rng(12)
    array = rng.standard_normal((3, 4, 6), dtype=np.float64)
    input_tensor = as_tensor(array)
    spectrum = tp.fft.rfft2(input_tensor, norm=norm)
    expected_spectrum = np.fft.rfft2(array, axes=(-2, -1), norm=norm)
    np.testing.assert_allclose(
        as_numpy(spectrum), expected_spectrum, rtol=1e-11, atol=1e-11)

    restored = tp.fft.irfft2(spectrum, s=[4, 6], norm=norm)
    expected_restored = np.fft.irfft2(expected_spectrum, s=[4, 6], axes=(-2, -1), norm=norm)
    np.testing.assert_allclose(
        as_numpy(restored), expected_restored, rtol=1e-11, atol=1e-11)


def test_fft2_native_cpu_resize_and_interior_dims():
    rng = np.random.default_rng(13)
    array = rng.standard_normal((4, 3, 5), dtype=np.float64)
    target = [3, 7]
    cropped = array[:3, :, :]
    padded = np.zeros((3, 3, 7), dtype=np.float64)
    padded[:, :, :5] = cropped
    expected = np.fft.fft2(padded, axes=(0, 2))
    got = tp.fft.fft2(as_tensor(array), s=target, dim=[0, 2])
    np.testing.assert_allclose(as_numpy(got), expected, rtol=1e-11, atol=1e-11)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("operation", ["fft2", "ifft2", "rfft2", "irfft2"])
def test_fft2_native_cuda(operation):
    rng = np.random.default_rng(14)
    real = rng.standard_normal((2, 5, 6), dtype=np.float32)
    if operation in {"fft2", "ifft2"}:
        array = real + 1j * rng.standard_normal(real.shape, dtype=np.float32)
        value = as_tensor(array, "cuda")
        cpu_value = as_tensor(array)
    elif operation == "irfft2":
        array = real
        value = tp.fft.rfft2(as_tensor(array, "cuda"), s=[5, 6])
        cpu_value = tp.fft.rfft2(as_tensor(array), s=[5, 6])
    else:
        array = real
        value = as_tensor(array, "cuda")
        cpu_value = as_tensor(array)
    result = getattr(tp.fft, operation)(value, s=[5, 6], dim=[-2, -1])
    cpu = getattr(tp.fft, operation)(cpu_value, s=[5, 6], dim=[-2, -1])
    np.testing.assert_allclose(
        as_numpy(result), as_numpy(cpu), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("operation", ["fft2", "ifft2", "rfft2", "irfft2"])
def test_fft2_native_autograd(operation):
    devices = ["cpu"]
    if tp.cuda.is_available():
        devices.append("cuda")
    rng = np.random.default_rng(15)
    backward_name = {
        "fft2": "fft_fft2_backward",
        "ifft2": "fft_ifft2_backward",
        "rfft2": "fft_rfft2_backward",
        "irfft2": "fft_irfft2_backward",
    }[operation]
    for device in devices:
        if operation == "irfft2":
            array = rng.standard_normal((2, 3, 3)) + 1j * rng.standard_normal((2, 3, 3))
        elif operation in {"fft2", "ifft2"}:
            array = rng.standard_normal((2, 3, 4)) + 1j * rng.standard_normal((2, 3, 4))
        else:
            array = rng.standard_normal((2, 3, 4))
        value = as_tensor(array, device).requires_grad_(True)
        result = getattr(tp.fft, operation)(value, s=[3, 4], dim=[1, 2], norm="ortho")
        if operation == "irfft2":
            gradient_array = rng.standard_normal(tuple(result.shape))
        else:
            gradient_array = rng.standard_normal(tuple(result.shape)) + 1j * rng.standard_normal(tuple(result.shape))
        gradient = as_tensor(gradient_array, device)
        result.backward(gradient)
        expected = getattr(tp, backward_name)(gradient, value, [3, 4], [1, 2], "ortho")
        np.testing.assert_allclose(
            as_numpy(value.grad), as_numpy(expected), rtol=2e-5, atol=2e-5)
