"""

functional/transforms layers. Native kernels live in
p10/src/backend/cpu/SpectralKernels.cpp (pocketfft) and
p10/src/backend/cuda/SpectralKernels.cu (cuFFT).
"""
import math

import numpy as np
import pytest
import torch

import tensorplay as tp
import tensorplay.audio as ta


def to_tp(t, device="cpu"):
    return tp.from_dlpack(t.detach().to(device).contiguous().__dlpack__()) \
        if str(device).startswith("cuda") else \
        tp.from_dlpack(t.contiguous().__dlpack__())


def to_torch(t):
    return torch.from_dlpack(t.__dlpack__())


SIZES = [1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 30, 64, 100, 1024]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 4, 16, 512])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("fn,tp_fn", [
    ("hann_window", "hann_window"),
    ("hamming_window", "hamming_window"),
    ("bartlett_window", "bartlett_window"),
    ("blackman_window", "blackman_window"),
])
def test_windows_match_torch(fn, tp_fn, n, periodic):
    ref = getattr(torch, fn)(n, periodic=periodic)
    got = getattr(tp, tp_fn)(n, periodic=periodic)
    np.testing.assert_allclose(to_torch(got).numpy(), ref.numpy(), rtol=1e-6, atol=1e-7)


def test_hamming_window_coeffs():
    a, b = 0.42, 0.7
    ref = torch.hamming_window(32, periodic=True, alpha=a, beta=b)
    got = tp.hamming_window(32, periodic=True, alpha=a, beta=b)
    np.testing.assert_allclose(to_torch(got).numpy(), ref.numpy(), rtol=1e-6, atol=2e-7)


def test_window_dtype():
    assert to_torch(tp.hann_window(8)).dtype == torch.float32
    f64 = tp.hann_window(8, dtype=tp.float64)
    assert to_torch(f64).dtype == torch.float64


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
@pytest.mark.parametrize("batch", [None, 3])
def test_fft_ifft_match_numpy(n, norm, batch):
    shape = (batch,) if batch else ()
    # numpy.fft always computes in double precision; construct the input in
    # complex128 so the 1e-9 bound measures transform accuracy, not f32
    x_np = np.random.randn(*shape, n).astype(np.float64) + \
        1j * np.random.randn(*shape, n).astype(np.float64)
    ref_f = np.fft.fft(x_np, axis=-1, norm=norm)
    ref_b = np.fft.ifft(x_np, axis=-1, norm=norm)

    x = tp.from_dlpack(torch.from_numpy(
        np.ascontiguousarray(x_np)).__dlpack__())
    got_f = to_torch(tp.fft_fft(x, -1, -1, norm)).numpy()
    got_b = to_torch(tp.fft_ifft(x, -1, -1, norm)).numpy()
    np.testing.assert_allclose(got_f.real, ref_f.real, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(got_f.imag, ref_f.imag, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(got_b.real, ref_b.real, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(got_b.imag, ref_b.imag, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
def test_rfft_irfft_match_numpy(n, norm):
    x_np = np.random.randn(2, n).astype(np.float64)
    ref = np.fft.rfft(x_np, n=n, axis=-1, norm=norm)
    x = tp.from_dlpack(torch.from_numpy(
        np.ascontiguousarray(x_np)).__dlpack__())
    got = to_torch(tp.fft_rfft(x, n, -1, norm)).numpy()
    np.testing.assert_allclose(got.real, ref.real, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(got.imag, ref.imag, rtol=1e-9, atol=1e-9)

    ref_i = np.fft.irfft(ref, n=n, axis=-1, norm=norm)
    got_i = to_torch(tp.fft_irfft(tp.from_dlpack(torch.from_numpy(
        np.ascontiguousarray(ref)).__dlpack__()), n, -1, norm)).numpy()
    np.testing.assert_allclose(got_i, ref_i, rtol=1e-9, atol=1e-9)


def test_fft_interior_dim():
    x_np = np.random.randn(2, 5, 16).astype(np.float64)
    ref = np.fft.fft(x_np, axis=1)
    x = tp.from_dlpack(torch.from_numpy(
        np.ascontiguousarray(x_np)).__dlpack__())
    got = to_torch(tp.fft_fft(x, -1, 1, "backward")).numpy()
    np.testing.assert_allclose(got.real, ref.real, rtol=1e-9, atol=1e-9)


def test_fft_resize_semantics():
    # n < size truncates from the front; n > size zero-pads at the end
    x_np = np.arange(10, dtype=np.float64) + 1j
    ref_trunc = np.fft.fft(x_np[:6])
    ref_pad = np.fft.fft(np.concatenate([x_np, np.zeros(2)]))
    x = tp.from_dlpack(torch.from_numpy(np.ascontiguousarray(x_np)).__dlpack__())
    np.testing.assert_allclose(
        to_torch(tp.fft_fft(x, 6, -1, "backward")).numpy(), ref_trunc, rtol=1e-12)
    np.testing.assert_allclose(
        to_torch(tp.fft_fft(x, 12, -1, "backward")).numpy(), ref_pad, rtol=1e-12)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

STFT_CASES = [
    dict(n_fft=16, hop=None, win=None),
    dict(n_fft=64, hop=16, win=48),
    dict(n_fft=128, hop=32, win=None),
    dict(n_fft=256, hop=64, win=None),
]


@pytest.mark.parametrize("kw", STFT_CASES)
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant"])
@pytest.mark.parametrize("normalized", [False, True])
def test_stft_matches_torch(kw, center, pad_mode, normalized):
    torch.manual_seed(0)
    wav = torch.randn(2, 8000, dtype=torch.float64)
    n_fft = kw["n_fft"]
    hop = kw["hop"] or n_fft // 4
    win_len = kw["win"] or n_fft
    window = torch.hann_window(win_len, dtype=torch.float64, periodic=True)

    ref = torch.stft(wav, n_fft, hop_length=hop, win_length=win_len,
                     window=window, center=center, pad_mode=pad_mode,
                     normalized=normalized, onesided=True, return_complex=True)
    got = to_torch(tp.stft(to_tp(wav), n_fft, hop, win_len, to_tp(window),
                           center=center, pad_mode=pad_mode,
                           normalized=normalized, onesided=True,
                           return_complex=True))
    np.testing.assert_allclose(
        got.numpy(), ref.numpy(), rtol=1e-8, atol=1e-8)


def test_stft_onesided_twosided():
    wav = torch.randn(1, 2000, dtype=torch.float64)
    w = torch.ones(64, dtype=torch.float64)
    ref_1 = torch.stft(wav, 64, window=w, onesided=True, return_complex=True)
    ref_2 = torch.stft(wav, 64, window=w, onesided=False, return_complex=True)
    g1 = to_torch(tp.stft(to_tp(wav), 64, None, None, to_tp(w),
                          onesided=True, return_complex=True))
    g2 = to_torch(tp.stft(to_tp(wav), 64, None, None, to_tp(w),
                          onesided=False, return_complex=True))
    assert g1.shape[-2] == 33 and g2.shape[-2] == 64
    np.testing.assert_allclose(g1.numpy(), ref_1.numpy(), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(g2.numpy(), ref_2.numpy(), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("kw", STFT_CASES[:3])
def test_istft_roundtrip_and_match_torch(kw):
    torch.manual_seed(1)
    n_fft = kw["n_fft"]
    hop = kw["hop"] or n_fft // 4
    win_len = kw["win"] or n_fft
    window = torch.hann_window(win_len, dtype=torch.float64, periodic=True)
    spec = torch.randn(2, n_fft // 2 + 1, 40, dtype=torch.float64) * (1 + 1j)

    ref = torch.istft(spec, n_fft, hop_length=hop, win_length=win_len,
                      window=window, center=True)
    got = to_torch(tp.istft(to_tp(spec.contiguous()), n_fft, hop, win_len,
                            to_tp(window), center=True, normalized=False,
                            onesided=True))
    np.testing.assert_allclose(got.numpy(), ref.numpy(), rtol=1e-7, atol=1e-7)


def test_stft_istft_roundtrip_signal():
    torch.manual_seed(2)
    wav = torch.randn(4000, dtype=torch.float64) * 0.1
    w = torch.hann_window(256, dtype=torch.float64)
    spec = torch.stft(wav, 256, hop_length=64, window=w,
                      return_complex=True)
    back = torch.istft(spec, 256, hop_length=64, window=w,
                       length=wav.numel())
    # istft(length=N) returns N samples; the center padding regions at both
    # edges carry boundary transients, so compare on the interior.
    rel = (back[64:-64] - wav[64:-64]).norm() / wav.norm()
    assert rel.item() < 1e-6


def test_stft_backward_matches_torch_grad():
    torch.manual_seed(3)
    wav_t = torch.randn(1, 1500, dtype=torch.float64, requires_grad=True)
    w = torch.hann_window(64)
    spec = torch.stft(wav_t, 64, hop_length=16, window=w, return_complex=True)
    loss = spec.abs().pow(2).sum()
    loss.backward()
    ref_grad = wav_t.grad.clone()

    wav_p = to_tp(wav_t.detach()).requires_grad_(True)
    spec_p = tp.stft(wav_p, 64, 16, None, to_tp(w), center=True,
                     pad_mode="reflect", normalized=False,
                     onesided=True, return_complex=True)
    # magnitude-square loss computed on the tp graph (complex abs -> real)
    proxy = spec_p.abs().pow(2).sum()
    proxy.backward()
    assert wav_p.grad is not None
    got = to_torch(wav_p.grad).numpy()
    np.testing.assert_allclose(got, ref_grad.numpy(), rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_melscale_fbanks_shape_and_rows():
    F = ta.functional.melscale_fbanks(201, 0.0, 8000.0, 40, 16000)
    t = to_torch(F)
    assert t.shape == (201, 40)
    # (freq bins) below/above all triangles legitimately sum to zero.
    col_sums = t.sum(-2)
    assert bool((col_sums > 0).all())


def test_amplitude_to_db_db_to_amplitude_roundtrip():
    x = torch.rand(8, 100, dtype=torch.float64) + 1e-6
    ref_a2db = 20.0 * torch.log10(torch.clamp(x, min=1e-5))
    # top_db); db_multiplier=log10(max(ref, amin))=0 for ref=1.
    got = to_torch(ta.functional.amplitude_to_DB(
        to_tp(x), 20.0, 1e-5, 0.0)).numpy()
    np.testing.assert_allclose(got, ref_a2db.numpy(), rtol=1e-9, atol=1e-9)


def test_mu_law_roundtrip():
    x = torch.linspace(-1, 1, 257, dtype=torch.float32)[:-1]
    q = 256
    enc = to_torch(ta.functional.mu_law_encoding(to_tp(x), q))
    assert enc.min().item() >= 0 and enc.max().item() < q
    dec = to_torch(ta.functional.mu_law_decoding(to_tp(enc), q))
    # mu-law companding is not linear: the decode grid spacing near |x|=1 is
    # for q=256 on this exact input; our port matches bit-for-bit.
    assert float((dec - x).abs().max()) < 0.02


def test_create_dct():
    d = to_torch(ta.functional.create_dct(13, 40, "ortho"))
    # row-wise (n_mels, n_mfcc) mel data.
    assert d.shape == (40, 13)
    # orthonormal columns
    eye = d.T @ d
    np.testing.assert_allclose(eye.numpy(), np.eye(13), atol=1e-5)


def test_resample_identity_when_same_rate():
    x = torch.randn(1, 1000, dtype=torch.float64)
    out = to_torch(ta.functional.resample(to_tp(x), 16000, 16000))
    assert abs(out.shape[-1] - x.shape[-1]) <= 2


def test_compute_deltas_shape():
    spec = torch.rand(2, 20, 50, dtype=torch.float64)
    out = to_torch(ta.functional.compute_deltas(to_tp(spec)))
    assert out.shape == spec.shape


def test_spectrogram_functional_matches_manual():
    torch.manual_seed(4)
    wav = torch.randn(1, 4000, dtype=torch.float64)
    w = torch.hann_window(256, dtype=torch.float64)
    ref_spec = torch.stft(wav, 256, hop_length=128, window=w,
                          center=True, pad_mode="reflect",
                          onesided=True, return_complex=True)
    ref_mag = ref_spec.abs()
    got = to_torch(ta.functional.spectrogram(
        to_tp(wav), pad=0, window=to_tp(w), n_fft=256, hop_length=128,
        win_length=256, power=1.0, normalized=False, center=True,
        pad_mode="reflect", onesided=True))
    np.testing.assert_allclose(got.numpy(), ref_mag.numpy(), rtol=1e-7, atol=1e-8)


# ---------------------------------------------------------------------------
# transforms modules
# ---------------------------------------------------------------------------

def test_transforms_forward_shapes():
    T = ta.transforms
    wav = torch.randn(1, 8000, dtype=torch.float64)
    # mel fb buffer defaults to float32; cast the module for float64 audio
    mel = T.MelSpectrogram(sample_rate=16000, n_fft=512,
                           win_length=512, hop_length=256,
                           n_mels=64).double()(to_tp(wav))
    assert to_torch(mel).shape[-2] == 64

    spec = T.Spectrogram(n_fft=512, hop_length=256, power=2.0)(to_tp(wav))
    assert to_torch(spec).shape[-2] == 257


def test_melspectrogram_values_close_reference():
    torch.manual_seed(5)
    wav = torch.randn(1, 4000, dtype=torch.float64).abs()
    T = ta.transforms
    m = T.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=200,
                         n_mels=32).double()(to_tp(wav.double()))
    fb = to_torch(ta.functional.melscale_fbanks(
        201, 0.0, 8000.0, 32, 16000)).double()
    spec = torch.stft(torch.abs(wav), 400, hop_length=200,
                      window=torch.hann_window(400, dtype=torch.float64),
                      center=True, onesided=True, return_complex=True)
    mag = spec.abs() ** 2
    # spec/mag carry a leading channel dim of 1.
    ref = (mag.transpose(-1, -2) @ fb).transpose(-1, -2)  # MelScale contract
    got = to_torch(m)
    np.testing.assert_allclose(
        got.numpy(), ref.numpy(), rtol=1e-4, atol=1e-6)


def test_amplitude_to_db_module():
    T = ta.transforms
    x = torch.rand(4, 10, dtype=torch.float64)
    y = T.AmplitudeToDB(top_db=60)(to_tp(x))
    ref = 20.0 * torch.log10(torch.clamp(x, min=1e-12))
    diff = to_torch(y).numpy() - ref.numpy()
    assert diff.max() <= 60.0 + 1e-6


# ---------------------------------------------------------------------------
# io roundtrips
# ---------------------------------------------------------------------------

def test_io_save_load_roundtrip(tmp_path):
    pytest.importorskip("soundfile")
    sr = 16000
    x = np.sin(2 * math.pi * 440 * np.arange(sr) / sr).astype(np.float32)[None, :]
    path = tmp_path / "t.wav"
    ta.save(str(path), to_tp(torch.from_numpy(x)), sr, channels_first=True)
    meta = ta.info(str(path))
    assert meta.sample_rate == sr
    assert meta.num_frames == sr
    assert meta.num_channels == 1
    y, sr2 = ta.load(str(path))
    assert sr2 == sr
    got = to_torch(y).numpy()
    # save writes the backend default WAV subtype (PCM_16), so the roundtrip
    # carries 1/32768 quantization
    np.testing.assert_allclose(got, x, atol=4e-5)


def test_load_channels_first_flag(tmp_path):
    sf = pytest.importorskip("soundfile")
    data = np.random.randn(1000, 2).astype(np.float32)
    path = tmp_path / "st.wav"
    sf.write(str(path), data, 22050, subtype="FLOAT")
    y_cf, _ = ta.load(str(path), channels_first=True)
    y_tc, _ = ta.load(str(path), channels_first=False)
    assert tuple(to_torch(y_cf).shape) == (2, 1000)
    assert tuple(to_torch(y_tc).shape) == (1000, 2)


def test_audio_meta_type():
    assert ta.AudioMetaData._fields == (
        "sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding")


def test_scipy_load_int16_bitexact(tmp_path):
    wavfile = pytest.importorskip("scipy.io.wavfile")
    ta_module = pytest.importorskip("tensorplay.audio")
    prev = ta.get_audio_backend()
    ta.set_audio_backend("scipy")
    try:
        rng = np.random.RandomState(0)
        raw = (rng.randn(8000, 2) * 8000).astype(np.int16)
        path = tmp_path / "i16.wav"
        wavfile.write(str(path), 8000, raw)
        y, sr = ta.load(str(path))
        assert sr == 8000
        got = to_torch(y).numpy()
        assert got.dtype == np.float32 and got.shape == (2, 8000)
        ref = (raw.astype(np.float32) / 32768.0).T
        np.testing.assert_array_equal(got, ref)
        # partial read honors the frame window
        y2, _ = ta.load(str(path), frame_offset=10, num_frames=100)
        assert to_torch(y2).shape == (2, 100)
        np.testing.assert_array_equal(to_torch(y2).numpy(), ref[:, 10:110])
    finally:
        ta.set_audio_backend(prev)


# ---------------------------------------------------------------------------
# backend registry
# ---------------------------------------------------------------------------

def test_backend_registry():
    backs = ta.list_audio_backends()
    assert isinstance(backs, list)
    cur = ta.get_audio_backend()
    assert cur in backs or cur is None
    with pytest.raises(ValueError):
        ta.set_audio_backend("nope")
    ta.set_audio_backend(None)  # reset to auto


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_deepspeech_forward_shape():
    from tensorplay.audio.models import DeepSpeech
    # dropout); forward(x) -> (batch, time, n_class).
    m = DeepSpeech(n_feature=40, n_hidden=32, n_class=11)
    m.eval()
    with torch.no_grad():
        out = m(to_tp(torch.randn(4, 1, 50, 40, dtype=torch.float32)))
    assert out.shape[0] == 4 and out.shape[-1] == 11


def test_wav2letter_forward_shape():
    from tensorplay.audio.models import Wav2Letter
    m = Wav2Letter(num_classes=11, num_features=1)
    m.eval()
    with torch.no_grad():
        out = m(to_tp(torch.randn(2, 1, 320, dtype=torch.float32)))
    assert out.shape[0] == 2 and out.shape[1] == 11


# ---------------------------------------------------------------------------
# datasets import surface (network datasets are exercised in CI only)
# ---------------------------------------------------------------------------

def test_datasets_importable():
    D = ta.datasets
    for name in ["YESNO", "LIBRISPEECH", "LJSPEECH", "COMMONVOICE",
                 "GTZAN", "SPEECHCOMMANDS", "VCTK_092"]:
        assert hasattr(D, name), name


# ---------------------------------------------------------------------------
# CUDA behavior checks (skipped when no GPU)
# ---------------------------------------------------------------------------

CUDA = pytest.mark.skipif(not (hasattr(tp, "cuda") and tp.cuda.is_available()),
                          reason="CUDA not available")


@CUDA
def test_cuda_fft_matches_cpu():
    x = torch.randn(3, 128, dtype=torch.complex128).cpu()
    cpu = to_torch(tp.fft_fft(to_tp(x), -1, -1, "backward"))
    gpu = to_torch(tp.fft_fft(to_tp(x.to("cuda")), -1, -1, "backward").cpu())
    np.testing.assert_allclose(cpu.numpy(), gpu.numpy(), rtol=1e-8, atol=1e-8)


@CUDA
def test_cuda_stft_matches_cpu():
    wav = torch.randn(1, 4000, dtype=torch.float64)
    w = torch.hann_window(256, dtype=torch.float64)
    cpu = to_torch(tp.stft(to_tp(wav), 256, 64, None, to_tp(w),
                           True, "reflect", False, True, True))
    gpu = to_torch(tp.stft(to_tp(wav.cuda()), 256, 64, None,
                           to_tp(w.cuda()), True, "reflect", False, True, True).cpu())
    np.testing.assert_allclose(cpu.numpy(), gpu.numpy(), rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# regressions: float32 mel pipeline, complex phase stretching, VAD, window
# device placement, weak-scalar pow, complex index_select
# ---------------------------------------------------------------------------

def test_melscale_fbanks_default_dtype():
    fb = to_torch(ta.functional.melscale_fbanks(201, 0.0, 8000.0, 32, 16000))
    assert fb.dtype == torch.float32


def test_melspectrogram_default_float32():
    torch.manual_seed(7)
    wav = torch.randn(1, 4000).abs()
    m = ta.transforms.MelSpectrogram(sample_rate=16000, n_fft=400,
                                     hop_length=200, n_mels=32)
    got = to_torch(m(to_tp(wav)))
    spec = torch.stft(wav, 400, hop_length=200,
                      window=torch.hann_window(400), center=True,
                      onesided=True, return_complex=True)
    fb = to_torch(ta.functional.melscale_fbanks(201, 0.0, 8000.0, 32, 16000))
    ref = (spec.abs() ** 2).transpose(-1, -2) @ fb
    np.testing.assert_allclose(got.numpy(), ref.transpose(-1, -2).numpy(),
                               rtol=1e-4, atol=1e-6)


def test_time_stretch_complex_specgram():
    torch.manual_seed(11)
    spec = torch.randn(2, 201, 50, dtype=torch.cfloat)
    out = to_torch(ta.transforms.TimeStretch(fixed_rate=1.2)(to_tp(spec)))
    assert out.dtype == torch.cfloat
    assert out.shape == (2, 201, math.ceil(50 / 1.2))


def test_pitch_shift_runs():
    torch.manual_seed(13)
    wav = torch.randn(1, 8000)
    out = to_torch(ta.transforms.PitchShift(sample_rate=8000, n_steps=2)(to_tp(wav)))
    assert out.dtype == torch.float32 and out.shape[-1] > 0


def test_vad_transform_runs():
    torch.manual_seed(17)
    sr = 16000
    # near-silence, a loud burst, then near-silence again: VAD trims some of
    # the leading quiet while keeping the burst
    quiet = torch.randn(1, sr // 2) * 0.005
    burst = torch.sin(torch.linspace(0, 2 * math.pi * 440, sr)) * 0.9
    wav = torch.cat([quiet, burst[None], quiet], dim=-1)
    out = to_torch(ta.transforms.Vad(sample_rate=sr)(to_tp(wav)))
    assert out.ndim == 2 and out.shape[0] == 1
    assert 0 < out.shape[1] < wav.shape[-1]
    assert out.shape[1] >= sr


@pytest.mark.parametrize("fn,kw", [
    ("hann_window", {}),
    ("hamming_window", {"alpha": 0.6, "beta": 0.4}),
    ("bartlett_window", {}),
    ("blackman_window", {}),
])
def test_window_factory_device_kwarg(fn, kw):
    plain = to_torch(getattr(tp, fn)(16, **kw))
    placed = to_torch(getattr(tp, fn)(16, device=tp.device("cpu"),
                                      layout=tp.strided, **kw))
    np.testing.assert_allclose(plain.numpy(), placed.numpy(), rtol=1e-6, atol=1e-6)
    if tp.cuda.is_available():
        gpu = getattr(tp, fn)(16, device=tp.device("cuda"), **kw)
        assert "cuda" in str(gpu.device)
        np.testing.assert_allclose(to_torch(gpu.cpu()).numpy(), plain.numpy(),
                                   rtol=1e-6, atol=1e-6)


def test_rpow_scalar_base_matches_torch():
    x = torch.linspace(1.0, 4.0, 8)
    got = to_torch(2.0 ** to_tp(x))
    ref = 2.0 ** x
    assert got.dtype == ref.dtype == torch.float32
    np.testing.assert_allclose(got.numpy(), ref.numpy(), rtol=1e-6, atol=1e-7)
    assert to_torch(2.0 ** to_tp(x.double())).dtype == torch.float64
    assert to_torch(2.0 ** to_tp(torch.arange(1, 5))).dtype == torch.float32


def test_index_select_complex_cpu():
    torch.manual_seed(19)
    z = torch.randn(3, 8, dtype=torch.cfloat)
    idx = torch.tensor([0, 2, 2, 5])
    got = to_torch(tp.index_select(to_tp(z), -1, to_tp(idx)))
    ref = z.index_select(-1, idx)
    np.testing.assert_allclose(got.numpy().real, ref.numpy().real, rtol=1e-6)
    np.testing.assert_allclose(got.numpy().imag, ref.numpy().imag, rtol=1e-6)


# ---------------------------------------------------------------------------
# Multiformat loading: native RIFF/WAVE codec and the FFmpeg (PyAV) fallback
# ---------------------------------------------------------------------------

def test_native_wav_subtypes_match_libsndfile(tmp_path):
    sf = pytest.importorskip("soundfile")
    sr = 16000
    sig = (np.random.default_rng(7).standard_normal((sr // 2, 2)) * 4000).astype(np.int16)
    for sub in ("PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE",
                "PCM_U8", "ULAW", "ALAW"):
        path = str(tmp_path / f"t_{sub}.wav")
        sf.write(path, sig, sr, subtype=sub)
        w, sr2 = ta.load(path)
        assert sr2 == sr
        ref, _ = sf.read(path, dtype="float32", always_2d=True)
        got = to_torch(w).numpy().T
        assert got.shape == ref.shape
        # Every subtype the native codec covers must decode bit-identically
        # to the reference library, including the companded G.711 forms.
        np.testing.assert_array_equal(got, ref)
        meta = ta.info(path)
        assert meta.sample_rate == sr
        assert meta.num_frames == sig.shape[0]
        assert meta.num_channels == 2


def test_native_wav_partial_reads_match_soundfile(tmp_path):
    sf = pytest.importorskip("soundfile")
    sr = 16000
    sig = (np.random.default_rng(3).standard_normal((sr, 2)) * 4000).astype(np.int16)
    path = str(tmp_path / "t.wav")
    sf.write(path, sig, sr, subtype="PCM_16")
    n = sig.shape[0]
    for off, nf in ((0, -1), (7, 33), (1000, 500), (n - 10, -1), (n + 100, -1), (5, 0)):
        ref, _ = sf.read(path, dtype="float32", start=off,
                         frames=nf if nf >= 0 else -1, always_2d=True)
        w, _ = ta.load(path, frame_offset=off, num_frames=nf)
        got = to_torch(w).numpy().T
        assert got.shape == ref.shape
        np.testing.assert_array_equal(got, ref)


def test_native_wav_save_roundtrip(tmp_path):
    x = (np.random.default_rng(5).standard_normal((2, 8000)) * 0.2).astype(np.float32)
    path = str(tmp_path / "rt.wav")
    ta.save(path, to_tp(torch.from_numpy(x)), 16000, channels_first=True)
    w, sr = ta.load(path)
    assert sr == 16000
    got = to_torch(w).numpy()
    assert got.shape == x.shape
    # PCM_16 quantization: one LSB of the 16-bit domain is 1/32768.
    assert np.abs(got - x).max() <= 3.1e-5


def _encode_av(path, fmt, codec, rate=44100, seconds=0.3, ch=2):
    import av
    n = int(rate * seconds)
    t = np.arange(n)
    s = (0.3 * np.sin(2 * math.pi * 440 * t / rate)).astype(np.float32)
    inter = np.stack([s] * ch, axis=1)
    layout = "stereo" if ch == 2 else "mono"
    c = av.open(str(path), "w", format=fmt)
    st = c.add_stream(codec, rate=rate, layout=layout)
    pts = 0
    for i in range(0, n, 1024):
        chunk = inter[i:i + 1024]
        f = av.AudioFrame.from_ndarray(np.ascontiguousarray(chunk.T),
                                       format="fltp", layout=layout)
        f.sample_rate = rate
        f.pts = pts
        pts += chunk.shape[0]
        for pk in st.encode(f):
            c.mux(pk)
    for pk in st.encode(None):
        c.mux(pk)
    c.close()


@pytest.mark.parametrize("ext,fmt,codec", [("m4a", "ipod", "aac"),
                                           ("mp3", "mp3", "libmp3lame")])
def test_av_fallback_containers(tmp_path, ext, fmt, codec):
    pytest.importorskip("av")
    import av.codec as ac
    try:
        ac.Codec(codec, "w")
    except Exception:
        pytest.skip(f"{codec} encoder unavailable in this PyAV build")
    path = str(tmp_path / f"t.{ext}")
    _encode_av(path, fmt, codec)
    w, sr = ta.load(path)
    assert sr == 44100
    full = to_torch(w).numpy()
    assert full.shape[0] == 2
    rms = float(np.sqrt((full ** 2).mean()))
    assert abs(rms - 0.21) < 0.05  # sine amplitude 0.3
    meta = ta.info(path)
    assert meta.sample_rate == 44100
    assert meta.num_channels == 2
    # Windowed load must equal a slice of the full decode: the container has
    # no frame-addressable seeks, so both paths share one deterministic decode.
    w2, _ = ta.load(path, frame_offset=100, num_frames=555)
    np.testing.assert_array_equal(to_torch(w2).numpy(), full[:, 100:655])
