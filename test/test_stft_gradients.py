"""Short-time Fourier transform: values, and gradients in signal and window.

The reference frames a (reflect- or zero-) padded signal, multiplies each
frame by the window centered in n_fft and transforms it, the definition the
kernels implement.  Gradients reach the signal and the window, and the
backward kernel differentiates again.
"""
import numpy as np
import pytest

import tensorplay as tp
from tensorplay.autograd import gradcheck, gradgradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def reference_stft(x, n_fft, hop, win_length, window, center, pad_mode, normalized, onesided):
    x = np.atleast_2d(x)
    if center:
        mode = "reflect" if pad_mode == "reflect" else "constant"
        x = np.pad(x, ((0, 0), (n_fft // 2, n_fft // 2)), mode=mode)
    w = np.ones(win_length) if window is None else window
    left = (n_fft - win_length) // 2
    full = np.zeros(n_fft)
    full[left:left + win_length] = w
    n_frames = 1 + (x.shape[1] - n_fft) // hop
    frames = np.stack([x[:, t * hop:t * hop + n_fft] * full for t in range(n_frames)], axis=1)
    spec = np.fft.rfft(frames, axis=-1) if onesided else np.fft.fft(frames, axis=-1)
    if normalized:
        spec = spec / np.sqrt(n_fft)
    return np.swapaxes(spec, 1, 2)


CASES = {
    "default": dict(n_fft=8, hop_length=3),
    "window": dict(n_fft=8, hop_length=3, window=True),
    "not_centered": dict(n_fft=6, hop_length=2, center=False),
    "zero_padded": dict(n_fft=8, hop_length=2, pad_mode="constant", window=True),
    "normalized": dict(n_fft=8, hop_length=4, normalized=True),
    "twosided": dict(n_fft=8, hop_length=3, onesided=False, window=True),
    "short_window": dict(n_fft=8, hop_length=2, win_length=6, window=True),
    "odd_margin_window": dict(n_fft=8, hop_length=2, win_length=5, window=True),
    "short_rectangle": dict(n_fft=8, hop_length=2, win_length=5),
    "short_rectangle_not_centered": dict(n_fft=8, hop_length=3, win_length=6, center=False),
}


def arguments(spec, device, window=None):
    kw = dict(spec)
    n_fft = kw["n_fft"]
    win_length = kw.get("win_length", n_fft)
    if kw.pop("window", False):
        kw["window"] = window if window is not None else tp.from_numpy(
            np.linspace(0.5, 1.5, win_length)).to(device)
    return kw


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("name", list(CASES))
def test_stft_matches_reference(device, batched, name):
    spec = CASES[name]
    rng = np.random.default_rng(3)
    signal = rng.standard_normal((2, 29) if batched else (29,))
    kw = arguments(spec, device)
    got = tp.stft(tp.from_numpy(signal).to(device), return_complex=True, **kw)
    window = kw.get("window")
    expected = reference_stft(
        signal, spec["n_fft"], spec["hop_length"], spec.get("win_length", spec["n_fft"]),
        None if window is None else window.cpu().numpy(), spec.get("center", True),
        spec.get("pad_mode", "reflect"), spec.get("normalized", False),
        spec.get("onesided", True))
    if not batched:
        expected = expected[0]
    got = got.cpu().numpy()
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(CASES))
def test_stft_gradients_reach_signal_and_window(device, name):
    spec = CASES[name]
    tp.manual_seed(5)
    x = tp.randn(2, 23, dtype=tp.float64).to(device).requires_grad_(True)
    if spec.get("window"):
        w = tp.from_numpy(np.linspace(0.5, 1.5, spec.get("win_length", spec["n_fft"])))
        w = w.to(device).requires_grad_(True)

        def fn(signal, window):
            return tp.view_as_real(
                tp.stft(signal, return_complex=True, **arguments(spec, device, window)))

        inputs = (x, w)
    else:
        def fn(signal):
            return tp.view_as_real(tp.stft(signal, return_complex=True, **arguments(spec, device)))

        inputs = (x,)
    assert gradcheck(fn, inputs, atol=1e-6, rtol=1e-4)
    assert gradgradcheck(fn, inputs, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_windows_of_changing_length_do_not_leak_between_calls(device):
    # The window is centered in an n_fft buffer; the margins must read zero
    # whatever earlier, longer windows left in reused memory.
    signal = np.random.default_rng(0).standard_normal((2, 40))
    x = tp.from_numpy(signal).to(device)
    for step, win_length in enumerate([5, 7, 6, 5, 3, 8, 5, 4, 3, 6, 5]):
        window = np.random.default_rng(step).random(win_length) + 0.5
        got = tp.stft(x, n_fft=8, hop_length=2, win_length=win_length,
                      window=tp.from_numpy(window).to(device), return_complex=True)
        expected = reference_stft(signal, 8, 2, win_length, window, True, "reflect", False, True)
        np.testing.assert_allclose(got.cpu().numpy(), expected, rtol=1e-10, atol=1e-10,
                                   err_msg=f"call {step}, win_length {win_length}")


@pytest.mark.parametrize("device", DEVICES)
def test_istft_default_window_is_win_length_wide(device):
    # Without a window, istft frames with win_length ones centered in n_fft,
    # the same rectangle stft is given explicitly here.
    signal = np.random.default_rng(1).standard_normal(48)
    x = tp.from_numpy(signal).to(device)
    rectangle = tp.ones(5, dtype=tp.float64).to(device)
    spec = tp.stft(x, n_fft=8, hop_length=2, win_length=5, window=rectangle, return_complex=True)
    back = tp.istft(spec, n_fft=8, hop_length=2, win_length=5, length=48)
    np.testing.assert_allclose(back.cpu().numpy(), signal, rtol=1e-9, atol=1e-9)
