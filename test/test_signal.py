"""Closed-form value tests for tensorplay.signal.windows, plus an
stft/istft round-trip.

Reference values are computed here directly from the defining formulas of
each window using numpy/scalar math; tensorplay itself is never used to
derive the expected results.
"""

import math

import numpy as np
import pytest

import tensorplay as tp
from tensorplay.signal import windows
from tensorplay.signal.windows import (
    bartlett,
    blackman,
    cosine,
    exponential,
    gaussian,
    general_cosine,
    general_hamming,
    hamming,
    hann,
    kaiser,
    nuttall,
)

# Small lengths, covering even/odd M and the M == 1 special case.
SMALL_M = [1, 2, 3, 5, 8, 9]

WINDOW_BUILDERS = [
    bartlett,
    blackman,
    cosine,
    exponential,
    gaussian,
    general_cosine,
    general_hamming,
    hamming,
    hann,
    kaiser,
    nuttall,
]


# ---------------------------------------------------------------------------
# Reference formulas (float64 numpy, derived from the window definitions)
# ---------------------------------------------------------------------------

def _period_len(M, sym):
    """Number of grid steps used in the denominator: M for a periodic
    window (with M > 1), M - 1 for a symmetric window or M == 1."""
    return M if (not sym and M > 1) else M - 1


def ref_bartlett(M, sym):
    if M == 0:
        return np.zeros(0)
    if M == 1:
        return np.ones(1)
    n = np.arange(M, dtype=np.float64)
    return 1.0 - np.abs(2.0 * n / _period_len(M, sym) - 1.0)


def ref_cosine(M, sym):
    if M == 0:
        return np.zeros(0)
    n = np.arange(M, dtype=np.float64)
    denom = M if (sym or M == 1) else M + 1
    return np.sin(math.pi * (n + 0.5) / denom)


def ref_exponential(M, sym, tau=1.0):
    if M == 0:
        return np.zeros(0)
    center = (M - 1) / 2.0 if (sym or M == 1) else M / 2.0
    n = np.arange(M, dtype=np.float64)
    return np.exp(-np.abs((n - center) / tau))


def ref_gaussian(M, sym, std=1.0):
    if M == 0:
        return np.zeros(0)
    start = -_period_len(M, sym) / 2.0
    n = np.arange(M, dtype=np.float64)
    k = (start + n) / (std * math.sqrt(2.0))
    return np.exp(-(k**2))


def ref_kaiser(M, sym, beta=12.0):
    if M == 0:
        return np.zeros(0)
    if M == 1:
        return np.ones(1)
    n = np.arange(M, dtype=np.float64)
    half = _period_len(M, sym) / 2.0
    arg = np.maximum(0.0, 1.0 - ((n - half) / half) ** 2)
    return np.i0(beta * np.sqrt(arg)) / np.i0(beta)


def ref_general_cosine(M, sym, a):
    if M == 0:
        return np.zeros(0)
    if M == 1:
        return np.ones(1)
    n = np.arange(M, dtype=np.float64)
    out = np.zeros(M)
    for i, coeff in enumerate(a):
        out += (-1.0) ** i * coeff * np.cos(2.0 * math.pi * i * n / _period_len(M, sym))
    return out


def ref_hann(M, sym):
    return ref_general_cosine(M, sym, [0.5, 0.5])


def ref_hamming(M, sym):
    return ref_general_cosine(M, sym, [0.54, 0.46])


def ref_blackman(M, sym):
    return ref_general_cosine(M, sym, [0.42, 0.5, 0.08])


def ref_nuttall(M, sym):
    return ref_general_cosine(M, sym, [0.3635819, 0.4891775, 0.1365995, 0.0106411])


# ---------------------------------------------------------------------------
# Closed-form value comparisons
# ---------------------------------------------------------------------------

def _assert_window_close(actual_np, expected_np, dtype):
    if dtype == tp.float64:
        np.testing.assert_allclose(actual_np, expected_np, rtol=1e-12, atol=1e-12)
    else:
        np.testing.assert_allclose(actual_np, expected_np, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_bartlett_matches_formula(M, sym, dtype):
    got = bartlett(M, sym=sym, dtype=dtype).numpy()
    _assert_window_close(got, ref_bartlett(M, sym), dtype)


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_cosine_matches_formula(M, sym, dtype):
    got = cosine(M, sym=sym, dtype=dtype).numpy()
    _assert_window_close(got, ref_cosine(M, sym), dtype)


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_exponential_matches_formula(M, sym, dtype):
    got = exponential(M, sym=sym, dtype=dtype).numpy()
    _assert_window_close(got, ref_exponential(M, sym), dtype)
    # A custom tau rescales the exponent.
    got_tau = exponential(M, sym=sym, tau=4.0, dtype=dtype).numpy()
    _assert_window_close(got_tau, ref_exponential(M, sym, tau=4.0), dtype)


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_gaussian_matches_formula(M, sym, dtype):
    got = gaussian(M, sym=sym, dtype=dtype).numpy()
    _assert_window_close(got, ref_gaussian(M, sym), dtype)
    got_std = gaussian(M, sym=sym, std=0.9, dtype=dtype).numpy()
    _assert_window_close(got_std, ref_gaussian(M, sym, std=0.9), dtype)


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_kaiser_matches_formula(M, sym, dtype):
    got = kaiser(M, sym=sym, dtype=dtype).numpy()
    _assert_window_close(got, ref_kaiser(M, sym), dtype)
    got_beta = kaiser(M, sym=sym, beta=0.9, dtype=dtype).numpy()
    _assert_window_close(got_beta, ref_kaiser(M, sym, beta=0.9), dtype)


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_general_cosine_matches_formula(M, sym, dtype):
    coeffs = [0.46, 0.23, 0.31]
    got = general_cosine(M, a=coeffs, sym=sym, dtype=dtype).numpy()
    _assert_window_close(got, ref_general_cosine(M, sym, coeffs), dtype)


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_general_hamming_matches_formula(M, sym, dtype):
    got = general_hamming(M, sym=sym, dtype=dtype).numpy()
    _assert_window_close(got, ref_hamming(M, sym), dtype)
    got_alpha = general_hamming(M, alpha=0.5, sym=sym, dtype=dtype).numpy()
    _assert_window_close(got_alpha, ref_hann(M, sym), dtype)


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_hamming_matches_formula(M, sym, dtype):
    got = hamming(M, sym=sym, dtype=dtype).numpy()
    _assert_window_close(got, ref_hamming(M, sym), dtype)


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_hann_matches_formula(M, sym, dtype):
    got = hann(M, sym=sym, dtype=dtype).numpy()
    _assert_window_close(got, ref_hann(M, sym), dtype)


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_blackman_matches_formula(M, sym, dtype):
    got = blackman(M, sym=sym, dtype=dtype).numpy()
    _assert_window_close(got, ref_blackman(M, sym), dtype)


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("M", SMALL_M)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_nuttall_matches_formula(M, sym, dtype):
    got = nuttall(M, sym=sym, dtype=dtype).numpy()
    _assert_window_close(got, ref_nuttall(M, sym), dtype)


def test_kaiser_zero_beta_is_rectangular():
    got = kaiser(7, beta=0.0, sym=False, dtype=tp.float64).numpy()
    np.testing.assert_allclose(got, np.ones(7), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------

def _call(builder, M, **kwargs):
    """Invoke a window builder, supplying the coefficients that
    general_cosine requires."""
    if builder is general_cosine:
        kwargs.setdefault("a", [0.46, 0.23, 0.31])
    return builder(M, **kwargs)


@pytest.mark.parametrize("builder", WINDOW_BUILDERS)
def test_zero_length_is_empty(builder):
    for dtype in (tp.float32, tp.float64):
        w = _call(builder, 0, dtype=dtype)
        assert w.shape == (0,)
        assert w.dtype == dtype


@pytest.mark.parametrize("builder", WINDOW_BUILDERS)
def test_unit_length_is_one(builder):
    for dtype in (tp.float32, tp.float64):
        w = _call(builder, 1, dtype=dtype).numpy()
        np.testing.assert_allclose(w, np.ones(1), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("builder", WINDOW_BUILDERS)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_dtype_passthrough(builder, dtype):
    assert _call(builder, 5, dtype=dtype).dtype == dtype


@pytest.mark.parametrize("builder", WINDOW_BUILDERS, ids=lambda f: f.__name__)
def test_default_dtype_is_float32(builder):
    assert _call(builder, 5).dtype == tp.float32


def test_window_peak_normalization():
    # A periodic window of even length and a symmetric window of odd length
    # both contain the sample where the builder reaches its maximum of 1.
    for builder in WINDOW_BUILDERS:
        for M, sym in ((8, False), (9, True)):
            w = _call(builder, M, sym=sym, dtype=tp.float64).numpy()
            assert w.max() == pytest.approx(1.0, abs=1e-12), (
                builder.__name__,
                M,
                sym,
            )


def test_symmetric_even_window_peak_below_one():
    w = hann(8, sym=True, dtype=tp.float64).numpy()
    assert w.max() < 1.0


def test_periodic_prefix_of_symmetric():
    """A periodic window of length M repeats the first M points of the
    symmetric window of length M + 1 (both use the same grid spacing)."""
    for M in (4, 7, 10):
        for ref, kwargs in [
            (hann, {}),
            (hamming, {}),
            (bartlett, {}),
            (blackman, {}),
            (kaiser, {}),
            (cosine, {}),
            (exponential, {}),
            (gaussian, {}),
            (general_cosine, {"a": [0.46, 0.23, 0.31]}),
        ]:
            periodic = ref(M, sym=False, dtype=tp.float64, **kwargs).numpy()
            symmetric = ref(M + 1, sym=True, dtype=tp.float64, **kwargs).numpy()[:M]
            np.testing.assert_allclose(
                periodic, symmetric, rtol=1e-12, atol=1e-12, err_msg=ref.__name__
            )


def test_hann_matches_native_window_op():
    """Cross-check against the natively implemented window op."""
    for M in (1, 5, 8):
        for periodic in (True, False):
            got = hann(M, sym=not periodic, dtype=tp.float64).numpy()
            expected = tp.hann_window(M, periodic=periodic, dtype=tp.float64).numpy()
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_kaiser_matches_native_window_op():
    for M in (1, 6):
        for periodic in (True, False):
            got = kaiser(M, sym=not periodic, dtype=tp.float64).numpy()
            expected = tp.kaiser_window(M, periodic=periodic, dtype=tp.float64).numpy()
            np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_requires_grad_forwarding():
    for w in (
        hann(4, requires_grad=True),
        kaiser(4, requires_grad=True),
        exponential(4, requires_grad=True),
        bartlett(4, requires_grad=True),
        cosine(4, requires_grad=True),
        general_cosine(4, a=[0.4, 0.3, 0.3], requires_grad=True),
    ):
        assert w.requires_grad


def test_package_exports():
    assert windows.__all__ == [
        "bartlett",
        "blackman",
        "cosine",
        "exponential",
        "gaussian",
        "general_cosine",
        "general_hamming",
        "hamming",
        "hann",
        "kaiser",
        "nuttall",
    ]
    for name in windows.__all__:
        assert callable(getattr(windows, name))


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder", WINDOW_BUILDERS)
@pytest.mark.parametrize("bad_M", [-1, -7])
def test_negative_length_raises(builder, bad_M):
    with pytest.raises(ValueError):
        _call(builder, bad_M)


@pytest.mark.parametrize("builder", WINDOW_BUILDERS)
def test_unsupported_dtype_raises(builder):
    with pytest.raises(ValueError):
        _call(builder, 5, dtype=tp.int64)


@pytest.mark.parametrize("builder", WINDOW_BUILDERS)
def test_unsupported_layout_raises(builder):
    with pytest.raises(ValueError):
        _call(builder, 5, layout=tp.sparse_coo)


def test_exponential_argument_checks():
    with pytest.raises(ValueError):
        exponential(5, tau=0.0)
    with pytest.raises(ValueError):
        exponential(5, tau=-1.0)
    with pytest.raises(ValueError):
        exponential(5, sym=True, center=0.0)


def test_gaussian_argument_checks():
    with pytest.raises(ValueError):
        gaussian(5, std=0.0)
    with pytest.raises(ValueError):
        gaussian(5, std=-0.5)


def test_kaiser_argument_checks():
    with pytest.raises(ValueError):
        kaiser(5, beta=-0.1)


def test_general_cosine_coefficient_checks():
    with pytest.raises(TypeError):
        general_cosine(5, a=3.0)
    with pytest.raises(ValueError):
        general_cosine(5, a=[])


# ---------------------------------------------------------------------------
# stft / istft
# ---------------------------------------------------------------------------

def test_stft_matches_manual_frame_fft():
    """Compare against the definition: frame the (unpadded) signal, apply
    the window, and take a real FFT of each frame."""
    n, n_fft, hop = 64, 16, 4
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n)

    # Periodic Hann window from its closed form.
    n_win = np.arange(n_fft, dtype=np.float64)
    win = 0.5 - 0.5 * np.cos(2.0 * math.pi * n_win / n_fft)

    spec = tp.stft(
        tp.tensor(x),
        n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=tp.tensor(win),
        center=False,
        return_complex=True,
    )

    n_frames = 1 + (n - n_fft) // hop
    assert spec.shape == (n_fft // 2 + 1, n_frames)
    ref = np.stack(
        [np.fft.rfft(x[i * hop : i * hop + n_fft] * win) for i in range(n_frames)],
        axis=-1,
    )
    np.testing.assert_allclose(spec.numpy(), ref, rtol=1e-10, atol=1e-10)


def test_stft_istft_roundtrip():
    n, n_fft, hop = 512, 128, 32
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n)
    win = tp.hann_window(n_fft, periodic=True, dtype=tp.float64)

    spec = tp.stft(
        tp.tensor(x),
        n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        return_complex=True,
    )
    y = tp.istft(
        spec,
        n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        length=n,
    )
    assert y.shape == (n,)
    np.testing.assert_allclose(y.numpy(), x, rtol=1e-9, atol=1e-9)
