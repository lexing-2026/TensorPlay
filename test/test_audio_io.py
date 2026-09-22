"""Native WAV codec and batch tests.

The native RIFF/WAVE codec in ``tensorplay._C.io`` is checked against the
reference implementations installed in the environment (scipy's ``wavfile``
for PCM containers and soundfile for everything else).  PCM16 decode of a
file written by an external tool is bit-exact (raw samples / 32768), and
round-trips through our own encoder stay within one quantization step.
"""
import io as _io
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

import tensorplay as tp
from tensorplay.audio import decode_wav, decode_wav_batch, encode_wav, wav_info
from tensorplay.audio import io as tp_audio_io

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:  # pragma: no cover
    _HAS_SOUNDFILE = False


def _bytes_tensor(data: bytes):
    return tp.tensor(np.frombuffer(data, dtype=np.uint8).copy())


def _tone(C=2, T=8000, sr=16000, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-1.0, 1.0, size=(C, T))).astype(np.float32), sr


class WavRoundtripTest(unittest.TestCase):
    """Our encoder -> our decoder round-trips within one quantization step."""

    def test_pcm16_mono_stereo(self):
        for C in (1, 2):
            x, sr = _tone(C=C, seed=1 + C)
            data = encode_wav(tp.tensor(np.ascontiguousarray(x)), sr, 16)
            wav, got_sr = decode_wav(data)
            self.assertEqual(got_sr, sr)
            self.assertEqual(tuple(wav.shape), (C, x.shape[1]))
            err = np.abs(wav.numpy() - x).max()
            self.assertLessEqual(err, 1.0 / 32768.0 + 1e-7)

    def test_pcm8_24_32_roundtrip(self):
        x, sr = _tone(C=1, T=4000)
        xt = tp.tensor(np.ascontiguousarray(x))
        for bits in (8, 24, 32):
            data = encode_wav(xt, sr, bits)
            wav, _ = decode_wav(data)
            step = 2.0 / float(1 << (bits - 1))
            err = np.abs(wav.numpy() - x).max()
            self.assertLessEqual(err, step + 1e-7, f"bits={bits}")

    def test_float32_input_keeps_precision(self):
        x, sr = _tone(C=2, T=5000, seed=7)
        data = encode_wav(tp.tensor(np.ascontiguousarray(x)), sr, 16)
        wav, _ = decode_wav(data)
        # The int16 grid is the only loss: reconstruct and compare exactly.
        quant = np.round(x * 32768.0)
        quant = np.clip(quant, -32768, 32767) / 32768.0
        np.testing.assert_array_equal(wav.numpy(), quant)


class WavExternalTest(unittest.TestCase):
    """Decodes files written by external tools bit-exactly."""

    def test_scipy_pcm16_exact(self):
        raw = (np.random.default_rng(3).integers(-32768, 32768, size=(6000,)).astype(np.int16))
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/a.wav"
            wavfile.write(p, 16000, raw)
            with open(p, "rb") as f:
                data = _bytes_tensor(f.read())
        wav, sr = decode_wav(data)
        self.assertEqual(sr, 16000)
        np.testing.assert_array_equal(
            wav.numpy()[0], raw.astype(np.float32) / 32768.0)

    def test_scipy_pcm16_stereo_exact(self):
        raw = (np.random.default_rng(4).integers(-32768, 32768, size=(4000, 2)).astype(np.int16))
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/b.wav"
            wavfile.write(p, 22050, raw)
            with open(p, "rb") as f:
                data = _bytes_tensor(f.read())
        wav, sr = decode_wav(data)
        self.assertEqual(sr, 22050)
        np.testing.assert_array_equal(
            wav.numpy(), (raw.T / 32768.0).astype(np.float32))

    @unittest.skipUnless(_HAS_SOUNDFILE, "soundfile not installed")
    def test_soundfile_float32_exact(self):
        x, sr = _tone(C=2, T=4000, seed=9)
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/c.wav"
            sf.write(p, x.T, sr, subtype="FLOAT")
            with open(p, "rb") as f:
                data = _bytes_tensor(f.read())
        wav, got_sr = decode_wav(data)
        self.assertEqual(got_sr, sr)
        np.testing.assert_array_equal(wav.numpy(), x)

    @unittest.skipUnless(_HAS_SOUNDFILE, "soundfile not installed")
    def test_soundfile_float64_downcasts(self):
        x, sr = _tone(C=1, T=3000, seed=11)
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/d.wav"
            # mono clips are written 1-D; a (1, T) 2-D array would be
            # interpreted as T channels by the reference writer.
            sf.write(p, x[0], sr, subtype="DOUBLE")
            with open(p, "rb") as f:
                data = _bytes_tensor(f.read())
        wav, _ = decode_wav(data)
        np.testing.assert_array_equal(wav.numpy()[0], x[0].astype(np.float32))

    @unittest.skipUnless(_HAS_SOUNDFILE, "soundfile not installed")
    def test_ulaw_matches_reference(self):
        raw = (np.random.default_rng(5).integers(-32768, 32768, size=(5000,)).astype(np.int16))
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/e.wav"
            sf.write(p, raw / 32768.0, 8000, subtype="ULAW")
            with open(p, "rb") as f:
                data = _bytes_tensor(f.read())
            ref, _ = sf.read(p, dtype="float32", always_2d=True)
        wav, _ = decode_wav(data)
        np.testing.assert_allclose(wav.numpy(), ref.T, rtol=0, atol=1.1 / 32768.0)

    @unittest.skipUnless(_HAS_SOUNDFILE, "soundfile not installed")
    def test_alaw_matches_reference(self):
        raw = (np.random.default_rng(6).integers(-32768, 32768, size=(5000,)).astype(np.int16))
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/f.wav"
            sf.write(p, raw / 32768.0, 8000, subtype="ALAW")
            with open(p, "rb") as f:
                data = _bytes_tensor(f.read())
            ref, _ = sf.read(p, dtype="float32", always_2d=True)
        wav, _ = decode_wav(data)
        np.testing.assert_allclose(wav.numpy(), ref.T, rtol=0, atol=1.1 / 32768.0)

    def test_info_metadata(self):
        x, sr = _tone(C=2, T=1000)
        data = encode_wav(tp.tensor(np.ascontiguousarray(x)), sr, 16)
        info = wav_info(data)
        self.assertEqual(info, (16000, 1000, 2, 16, 1))

    def test_corrupt_wav_raises(self):
        with self.assertRaises(Exception):
            decode_wav(_bytes_tensor(b"RIFF\x00\x00\x00\x00WAVEgarbage"))


class WavSliceTest(unittest.TestCase):
    """frame_offset/num_frames decode exactly the requested window."""

    def test_slice_matches_full(self):
        x, sr = _tone(C=2, T=8000)
        data = encode_wav(tp.tensor(np.ascontiguousarray(x)), sr, 16)
        full, _ = decode_wav(data)
        part, _ = decode_wav(data, frame_offset=1000, num_frames=500)
        np.testing.assert_array_equal(part.numpy(), full.numpy()[:, 1000:1500])

    def test_offset_beyond_end_empty(self):
        x, sr = _tone(C=1, T=100)
        data = encode_wav(tp.tensor(np.ascontiguousarray(x)), sr, 16)
        part, _ = decode_wav(data, frame_offset=1000, num_frames=10)
        self.assertEqual(tuple(part.shape), (1, 0))

    def test_negative_offset_rejected(self):
        x, sr = _tone(C=1, T=100)
        data = encode_wav(tp.tensor(np.ascontiguousarray(x)), sr, 16)
        with self.assertRaises(ValueError):
            decode_wav(data, frame_offset=-5)


class WavBatchTest(unittest.TestCase):
    """Batch decode equals sequential decode and preserves order."""

    def test_batch_matches_single(self):
        xs = [_tone(C=2, T=2000, seed=20 + i) for i in range(5)]
        datas = [encode_wav(tp.tensor(np.ascontiguousarray(x)), sr, 16) for x, sr in xs]
        singles = [decode_wav(d) for d in datas]
        batch = decode_wav_batch(datas)
        self.assertEqual(len(batch), len(datas))
        for (w1, s1), (w2, s2) in zip(batch, singles):
            self.assertEqual(s1, s2)
            np.testing.assert_array_equal(w1.numpy(), w2.numpy())


class AudioIoTest(unittest.TestCase):
    """load()/save() route .wav files through the native codec."""

    def test_load_uses_native_wav(self):
        x, sr = _tone(C=2, T=3000)
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/x.wav"
            with open(p, "wb") as f:
                f.write(bytes(encode_wav(tp.tensor(np.ascontiguousarray(x)), sr, 16).numpy().tobytes()))
            wav, got_sr = tp_audio_io.load(p)
        self.assertEqual(got_sr, sr)
        self.assertEqual(tuple(wav.shape), (2, 3000))

    def test_load_partial_uses_native_wav(self):
        x, sr = _tone(C=2, T=3000)
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/x.wav"
            with open(p, "wb") as f:
                f.write(bytes(encode_wav(tp.tensor(np.ascontiguousarray(x)), sr, 16).numpy().tobytes()))
            wav, _ = tp_audio_io.load(p, frame_offset=500, num_frames=200)
        full, _ = decode_wav(encode_wav(tp.tensor(np.ascontiguousarray(x)), sr, 16))
        np.testing.assert_array_equal(wav.numpy(), full.numpy()[:, 500:700])

    def test_save_writes_decodable_wav(self):
        x, sr = _tone(C=1, T=2000)
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/y.wav"
            tp_audio_io.save(p, tp.tensor(np.ascontiguousarray(x)), sr)
            sr_back, raw = wavfile.read(p)
        self.assertEqual(sr_back, sr)
        ref = np.round(x[0] * 32768.0)
        ref = np.clip(ref, -32768, 32767).astype(np.int16)
        np.testing.assert_array_equal(raw, ref)

    def test_audio_to_tensor_submodule(self):
        native = getattr(getattr(tp, "_C", None), "io", None)
        self.assertIsNotNone(native, "native io submodule missing")
        self.assertTrue(hasattr(native, "audio_to_tensor"))
        audio = (np.arange(200, dtype=np.int32) - 100).reshape(100, 2)
        out = native.audio_to_tensor(np.ascontiguousarray(audio))
        self.assertEqual(tuple(out.shape), (2, 100))
        expected = audio.T.astype(np.float32) / 2147483648.0
        np.testing.assert_allclose(out.numpy(), expected, rtol=0, atol=1e-12)


class WavPerfProbeTest(unittest.TestCase):
    """Reports the batch speedup; informational, never asserted."""

    def test_batch_speedup_report(self):
        import time
        x, sr = _tone(C=2, T=16000)
        data = encode_wav(tp.tensor(np.ascontiguousarray(x)), sr, 16)
        datas = [data] * 16
        t0 = time.perf_counter()
        for d in datas:
            decode_wav(d)
        single = time.perf_counter() - t0
        t0 = time.perf_counter()
        decode_wav_batch(datas)
        batch = time.perf_counter() - t0
        print(f"\n[perf] 16x(2ch,16k) WAV: sequential {single * 1e3:.1f} ms, "
              f"batch {batch * 1e3:.1f} ms, speedup {single / batch:.2f}x")


if __name__ == "__main__":
    unittest.main()