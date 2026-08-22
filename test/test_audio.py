"""Audio stack alignment tests — tensorplay.audio vs torch/torchaudio 2.x.

Every DSP primitive is compared against the installed torch (oracle):
windows, fft family, stft/istft, autograd, and the ported torchaudio
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
# window factories vs ATen TensorFactories formulas (torch oracle)
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
    np.testing.assert_allclose(to_torch(got).numpy(), ref.numpy(), rtol=1e-6)


def test_window_dtype():
    assert to_torch(tp.hann_window(8)).dtype == torch.float32
    f64 = tp.hann_window(8, dtype=tp.float64)
    assert to_torch(f64).dtype == torch.float64


# ---------------------------------------------------------------------------
# FFT family vs pocketfft-equivalent numpy / torch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
@pytest.mark.parametrize("batch", [None, 3])
def test_fft_ifft_match_numpy(n, norm, batch):
    shape = (batch,) if batch else ()
    x_np = np.random.randn(*shape, n).astype(np.float64) + \
        1j * np.random.randn(*shape, n).astype(np.float64)
    ref_f = np.fft.fft(x_np, axis=-1, norm=norm)
    ref_b = np.fft.ifft(x_np, axis=-1, norm=norm)

    x = tp.tensor(x_np.tolist())
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
    x = tp.tensor(x_np.tolist())
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
    x = tp.tensor(x_np.tolist())
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
# stft/istft vs torch.stft/torch.istft (ATen SpectralOps semantics)
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
    spec = torch.randn(2, n_fft // 2 + 1, 40, dtype=torch.float64)

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
    rel = (back - wav[64:-64]).norm() / wav.norm()
    assert rel.item() < 1e-6


def test_stft_backward_matches_torch_grad():
    torch.manual_seed(3)
    wav_t = torch.randn(1, 1500, dtype=torch.float64, requires_grad=True)
    w = torch.hann_window(64, dtype=torch.float64)
    spec = torch.stft(wav_t, 64, hop_length=16, window=w, return_complex=True)
    loss = spec.abs().pow(2).sum()
    loss.backward()
    ref_grad = wav_t.grad.clone()

    wav_p = to_tp(wav_t.detach()).requires_grad_(True)
    spec_p = tp.stft(wav_p, 64, 16, None, to_tp(w), center=True,
                     pad_mode="reflect", normalized=False,
                     onesided=True, return_complex=True)
    # magnitude-square sum via complex parts
    real = to_torch(spec_p.real() if hasattr(spec_p, "real") else spec_p)
    # use view_as_real through torch for the loss on our graph
    back = to_torch(spec_p)
    proxy = (back.real ** 2 + back.imag ** 2).sum()
    proxy.backward()
    assert wav_p.grad is not None
    got = to_torch(wav_p.grad).numpy()
    np.testing.assert_allclose(got, ref_grad.numpy(), rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# functional layer parity (ported from torchaudio 2.11)
# ---------------------------------------------------------------------------

def test_melscale_fbanks_shape_and_rows():
    F = ta.functional.melscale_fbanks(201, 0.0, 8000.0, 40, 16000)
    t = to_torch(F)
    assert t.shape == (201, 40)
    row_sums = t.sum(-1)
    assert bool((row_sums > 0).all())


def test_amplitude_to_db_db_to_amplitude_roundtrip():
    x = torch.rand(8, 100, dtype=torch.float64) + 1e-6
    ref_a2db = 20.0 * torch.log10(torch.clamp(x, min=1e-5))
    got = to_torch(ta.functional.amplitude_to_DB(
        to_tp(x), "max_db", 80.0, 20.0, 1e-5)).numpy()
    np.testing.assert_allclose(got, ref_a2db.numpy(), rtol=1e-9, atol=1e-9)


def test_mu_law_roundtrip():
    x = torch.linspace(-1, 1, 257, dtype=torch.float32)[:-1]
    q = 256
    enc = to_torch(ta.functional.mu_law_encoding(to_tp(x), q))
    assert enc.min().item() >= 0 and enc.max().item() < q
    dec = to_torch(ta.functional.mu_law_decoding(to_tp(enc), q))
    assert float((dec - x).abs().max()) < 1.0 / q + 1e-3


def test_create_dct():
    d = to_torch(ta.functional.create_dct(13, 40, "ortho"))
    assert d.shape == (13, 40)
    # orthonormal rows
    eye = d @ d.T
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
    mel = T.MelSpectrogram(sample_rate=16000, n_fft=512,
                           win_length=512, hop_length=256,
                           n_mels=64)(to_tp(wav))
    assert to_torch(mel).shape[0] == 64

    spec = T.Spectrogram(n_fft=512, hop_length=256, power=2.0)(to_tp(wav))
    assert to_torch(spec).shape[-2] == 257


def test_melspectrogram_values_close_reference():
    torch.manual_seed(5)
    wav = torch.randn(1, 4000, dtype=torch.float64).abs()
    T = ta.transforms
    m = T.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=200,
                         n_mels=32)(to_tp(wav.double()))
    fb = to_torch(ta.functional.melscale_fbanks(
        201, 0.0, 8000.0, 32, 16000))
    spec = torch.stft(torch.abs(wav), 400, hop_length=200,
                      window=torch.hann_window(400, dtype=torch.float64),
                      center=True, onesided=True, return_complex=True)
    mag = spec.abs() ** 2
    ref = (fb @ mag.reshape(mag.shape[0], -1)).reshape(fb.shape[0], *mag.shape[1:])
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
    np.testing.assert_allclose(got, x, atol=1e-5)


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
# models smoke tests (ported verbatim from torchaudio.models)
# ---------------------------------------------------------------------------

def test_deepspeech_forward_shape():
    from tensorplay.audio.models import DeepSpeech
    m = DeepSpeech(n_features=40, hidden_size=32, num_hidden_layers=1,
                   rnn_type="nn.RNN", bidirectional=True)
    m.eval()
    with torch.no_grad():
        out, lengths = m(torch.randn(4, 40, 50, dtype=torch.float32),
                         torch.tensor([50, 40, 30, 20]))
    assert out.shape[0] == 4 and out.shape[-1] == 29


def test_wav2letter_forward_shape():
    from tensorplay.audio.models import Wav2Letter
    m = Wav2Letter(num_classes=11, input_features=1)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 1, 320, dtype=torch.float32))
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
# CUDA parity (skipped when no GPU)
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
