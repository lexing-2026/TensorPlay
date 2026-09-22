"""Native image IO codec and batch tests.

The native codecs in ``tensorplay._C.io`` (libjpeg-turbo / libpng on CPU,
the hardware decoder on CUDA) are compared against independently decoded
references: torchvision's own codecs and PIL.  JPEG expectations allow a
1 LSB tolerance because SIMD IDCT rounding differs between builds; PNG
decoding is deterministic per implementation and is compared exactly.

The batch entry points are the speedup surface: CPU batches run on the
shared thread pool and CUDA batches on the hardware batch decoder.
"""
import io as _io
import unittest

import numpy as np
from PIL import Image

import tensorplay as tp
from tensorplay.vision import io as tp_io

try:
    import torch
    import torchvision
    import torchvision.io as tv_io
    _HAS_TORCHVISION = True
except ImportError:  # pragma: no cover
    _HAS_TORCHVISION = False


def _rgb(seed=0, h=37, w=53):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _gray(seed=1, h=29, w=41):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def _jpeg_bytes(arr_hwc, quality=90):
    buf = _io.BytesIO()
    Image.fromarray(arr_hwc).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _png_bytes(arr, mode=None):
    buf = _io.BytesIO()
    if isinstance(arr, Image.Image):
        img = arr
    else:
        img = Image.fromarray(arr, mode=mode) if mode else Image.fromarray(arr)
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bytes_tensor(data: bytes):
    return tp.tensor(np.frombuffer(data, dtype=np.uint8).copy())


@unittest.skipUnless(_HAS_TORCHVISION, "torchvision not installed")
class JpegDecodeReferenceTest(unittest.TestCase):
    """Native JPEG decode vs the reference codec path."""

    def test_rgb_jpeg_matches_reference(self):
        rgb = _rgb()
        data = _bytes_tensor(_jpeg_bytes(rgb))
        got = tp_io.decode_jpeg(data)
        ref = tv_io.decode_jpeg(torch.from_numpy(np.frombuffer(_jpeg_bytes(rgb), dtype=np.uint8).copy()))
        self.assertEqual(tuple(got.shape), tuple(ref.shape))
        self.assertEqual(got.dtype, tp.uint8)
        d = np.abs(got.numpy() - ref.numpy())
        self.assertLessEqual(d.max(), 1, f"max diff {d.max()}")

    def test_gray_jpeg_mode_gray(self):
        rgb = _rgb(seed=2)
        data = _bytes_tensor(_jpeg_bytes(rgb))
        got = tp_io.decode_jpeg(data, mode=tp_io.ImageReadMode.GRAY.value)
        ref = tv_io.decode_jpeg(
            torch.from_numpy(np.frombuffer(_jpeg_bytes(rgb), dtype=np.uint8).copy()),
            mode=torchvision.io.ImageReadMode.GRAY)
        self.assertEqual(tuple(got.shape), tuple(ref.shape))
        d = np.abs(got.numpy() - ref.numpy())
        self.assertLessEqual(d.max(), 1)

    def test_gray_jpeg_stays_gray(self):
        g = _gray()
        data = _bytes_tensor(_jpeg_bytes(g, quality=85))
        got = tp_io.decode_jpeg(data)  # unchanged: output keeps 1 channel
        self.assertEqual(tuple(got.shape), (1,) + g.shape)

    def test_jpeg_decode_shares_bytes_with_pil(self):
        # same stream produces byte-identical output vs torchvision when the
        # codec build matches; both funnel through the same IDCT tables.
        rgb = _rgb(seed=3)
        raw = _jpeg_bytes(rgb)
        got = tp_io.decode_jpeg(_bytes_tensor(raw))
        ref = tv_io.decode_jpeg(torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()))
        np.testing.assert_array_equal(got.numpy(), ref.numpy())

    def test_corrupt_jpeg_raises(self):
        with self.assertRaises(Exception):
            tp_io.decode_jpeg(_bytes_tensor(b"\xff\xd8\xff garbage"))


@unittest.skipUnless(_HAS_TORCHVISION, "torchvision not installed")
class PngDecodeReferenceTest(unittest.TestCase):
    """Native PNG decode vs the reference codec path."""

    def test_rgb_png_exact(self):
        rgb = _rgb(seed=4)
        raw = _png_bytes(rgb)
        got = tp_io.decode_png(_bytes_tensor(raw))
        ref = tv_io.decode_png(torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()))
        self.assertEqual(tuple(got.shape), tuple(ref.shape))
        np.testing.assert_array_equal(got.numpy(), ref.numpy())

    def test_rgba_png_mode_unchanged(self):
        rng = np.random.default_rng(5)
        rgba = rng.integers(0, 256, size=(31, 47, 4), dtype=np.uint8)
        raw = _png_bytes(rgba)
        got = tp_io.decode_png(_bytes_tensor(raw))
        self.assertEqual(tuple(got.shape), (4, 31, 47))
        ref = np.array(Image.open(_io.BytesIO(raw)).convert("RGBA"))
        np.testing.assert_array_equal(got.numpy(), np.ascontiguousarray(ref.transpose(2, 0, 1)))

    def test_palette_png_expands_to_rgb(self):
        img = Image.new("P", (23, 19))
        img.putpalette(list(range(256)) * 3)
        for x in range(23):
            for y in range(19):
                img.putpixel((x, y), (x * 7 + y * 3) % 256)
        raw = _png_bytes(img)
        got = tp_io.decode_png(_bytes_tensor(raw))
        self.assertEqual(tuple(got.shape), (3, 19, 23))
        ref = np.array(img.convert("RGB")).transpose(2, 0, 1)
        np.testing.assert_array_equal(got.numpy(), np.ascontiguousarray(ref))

    def test_png_mode_conversions_match_pil(self):
        rgb = _rgb(seed=6)
        raw = _png_bytes(rgb)
        for mode, pil_mode in [(1, "L"), (2, "LA"), (3, "RGB"), (4, "RGBA")]:
            got = tp_io.decode_png(_bytes_tensor(raw), mode=mode)
            ref = np.array(Image.open(_io.BytesIO(raw)).convert(pil_mode))
            expected_shape = ((1,) + ref.shape) if ref.ndim == 2 else (ref.shape[2],) + ref.shape[:2]
            self.assertEqual(tuple(got.shape), expected_shape, f"mode {mode} shape")
            expected = np.ascontiguousarray(ref.transpose(2, 0, 1) if ref.ndim == 3 else ref[None])
            d = np.abs(got.numpy().astype(int) - expected.astype(int))
            if mode in (2, 4):
                # opaque alpha is appended verbatim on the last plane
                np.testing.assert_array_equal(got.numpy()[-1], 255, err_msg=f"mode {mode} alpha")
                self.assertLessEqual(d.max(), 2, f"mode {mode}")
            else:
                self.assertLessEqual(d.max(), 2, f"mode {mode}")


class EncodeRoundtripTest(unittest.TestCase):
    """Encode paths: PNG is lossless, JPEG is lossy within LSB bounds."""

    def test_png_roundtrip_exact(self):
        rgb = _rgb(seed=7)
        encoded = tp_io.encode_png(tp.tensor(np.ascontiguousarray(rgb.transpose(2, 0, 1))))
        decoded = tp_io.decode_png(encoded)
        np.testing.assert_array_equal(
            decoded.numpy(), np.ascontiguousarray(rgb.transpose(2, 0, 1)))

    def test_png_roundtrip_float_input(self):
        rgb = _rgb(seed=8)
        t = tp.tensor(np.ascontiguousarray(rgb.transpose(2, 0, 1) / 255.0, dtype=np.float32))
        decoded = tp_io.decode_png(tp_io.encode_png(t))
        # float -> uint8 quantization costs at most one LSB per channel
        err = np.abs(decoded.numpy() / 255.0 - t.numpy())
        self.assertLessEqual(err.max(), 1.0 / 255.0, "float [0,1] roundtrip")

    def test_jpeg_roundtrip_lossy_bounds(self):
        # A smooth image keeps the DCT error measurable but tight; random
        # noise is not a meaningful JPEG fidelity probe.
        yy, xx = np.mgrid[0:64, 0:64]
        rgb = np.stack([(yy * 4) % 256, (xx * 4) % 256, ((yy + xx) * 2) % 256],
                       axis=-1).astype(np.uint8)
        t = tp.tensor(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        encoded = tp_io.encode_jpeg(t, quality=90)
        decoded = tp_io.decode_jpeg(encoded)
        self.assertEqual(tuple(decoded.shape), tuple(t.shape))
        d = np.abs(decoded.numpy().astype(np.int16) - t.numpy().astype(np.int16))
        self.assertLess(d.mean(), 8.0)
        self.assertLess(d.max(), 96)

    def test_jpeg_gray_roundtrip_keeps_gray(self):
        g = _gray()
        t = tp.tensor(np.ascontiguousarray(g[None]))
        encoded = tp_io.encode_jpeg(t, quality=85)
        decoded = tp_io.decode_jpeg(encoded, mode=1)
        self.assertEqual(tuple(decoded.shape), (1,) + g.shape)

    def test_write_jpeg_write_png_files(self):
        import tempfile, os
        rgb = _rgb(seed=10)
        t = tp.tensor(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        with tempfile.TemporaryDirectory() as d:
            jp = os.path.join(d, "a.jpg")
            tp_io.write_jpeg(t, jp, quality=80)
            self.assertTrue(os.path.getsize(jp) > 0)
            back = tp_io.read_image(jp)
            self.assertEqual(tuple(back.shape), tuple(t.shape))
            pn = os.path.join(d, "a.png")
            tp_io.write_png(t, pn)
            back = tp_io.read_image(pn)
            self.assertEqual(tuple(back.shape), tuple(t.shape))


class ImageApiTest(unittest.TestCase):
    """read_image / decode_image keep the float32 [0,1] contract."""

    def test_decode_image_jpeg_float_range(self):
        rgb = _rgb(seed=11)
        raw = _jpeg_bytes(rgb)
        img = tp_io.decode_image(_bytes_tensor(raw))
        self.assertEqual(img.dtype, tp.float32)
        self.assertEqual(tuple(img.shape), (3,) + rgb.shape[:2])
        self.assertGreaterEqual(img.min().item(), 0.0)
        self.assertLessEqual(img.max().item(), 1.0)

    def test_decode_image_png_matches_read_image(self):
        rgb = _rgb(seed=12)
        raw = _png_bytes(rgb)
        a = tp_io.decode_image(_bytes_tensor(raw))
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.png")
            with open(p, "wb") as f:
                f.write(raw)
            b = tp_io.read_image(p)
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_read_file_write_file(self):
        import tempfile, os
        data = np.arange(256, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bytes.bin")
            tp_io.write_file(p, tp.tensor(data))
            back = tp_io.read_file(p)
            np.testing.assert_array_equal(back.numpy(), data)
            part = tp_io.read_file(p, start=10, size=20)
            np.testing.assert_array_equal(part.numpy(), data[10:30])


class JpegBatchTest(unittest.TestCase):
    """Batch decode equals sequential decode and preserves order."""

    def test_cpu_batch_matches_single(self):
        streams = [_jpeg_bytes(_rgb(seed=i), quality=80) for i in range(6)]
        singles = [tp_io.decode_jpeg(_bytes_tensor(s)).numpy() for s in streams]
        batch = tp_io.decode_jpeg_batch([_bytes_tensor(s) for s in streams])
        self.assertEqual(len(batch), len(streams))
        for got, ref in zip(batch, singles):
            np.testing.assert_array_equal(got.numpy(), ref)

    def test_cpu_batch_mixed_gray_rgb(self):
        streams = [_jpeg_bytes(_gray(seed=i), quality=85) for i in range(3)]
        streams += [_jpeg_bytes(_rgb(seed=20 + i)) for i in range(3)]
        singles = [tp_io.decode_jpeg(_bytes_tensor(s)).numpy() for s in streams]
        batch = tp_io.decode_jpeg_batch([_bytes_tensor(s) for s in streams])
        for got, ref in zip(batch, singles):
            np.testing.assert_array_equal(got.numpy(), ref)

    def test_batch_invalid_device(self):
        assert tp._C is not None
        with self.assertRaises(ValueError):
            tp_io.decode_jpeg_batch([_bytes_tensor(_jpeg_bytes(_rgb()))], device="tpu")


@unittest.skipUnless(hasattr(tp, "cuda") and tp.cuda.is_available(), "CUDA not available")
class JpegGpuTest(unittest.TestCase):
    """Hardware decode agrees with the CPU decode within IDCT rounding."""

    def _gpu_ok(self):
        native = getattr(getattr(tp, "_C", None), "io", None)
        return native is not None and hasattr(native, "decode_jpeg_cuda")

    def test_single_cuda_vs_cpu(self):
        if not self._gpu_ok():
            self.skipTest("native CUDA JPEG decoder not compiled in")
        # Smooth images keep the two IDCT implementations within a few LSBs;
        # white noise amplifies the rounding difference without being a
        # meaningful fidelity probe.
        yy, xx = np.mgrid[0:64, 0:64]
        rgb = np.stack([(yy * 4) % 256, (xx * 4) % 256, ((yy + xx) * 2) % 256],
                       axis=-1).astype(np.uint8)
        raw = _bytes_tensor(_jpeg_bytes(rgb, quality=95))
        cpu = tp_io.decode_jpeg(raw).numpy()
        gpu = tp_io.decode_jpeg(raw, device="cuda")
        self.assertEqual(gpu.device.type, "cuda")
        got = gpu.cpu().numpy()
        self.assertEqual(got.shape, cpu.shape)
        d = np.abs(got.astype(np.int16) - cpu.astype(np.int16))
        self.assertLessEqual(d.max(), 8, f"max diff {d.max()}")

    def test_batch_cuda_vs_cpu(self):
        if not self._gpu_ok():
            self.skipTest("native CUDA JPEG decoder not compiled in")
        yy, xx = np.mgrid[0:64, 0:64]
        base = np.stack([(yy * 4) % 256, (xx * 4) % 256, ((yy + xx) * 2) % 256],
                        axis=-1).astype(np.uint8)
        streams = [_jpeg_bytes(base, quality=92) for _ in range(5)]
        cpu = [tp_io.decode_jpeg(_bytes_tensor(s)).numpy() for s in streams]
        gpu = tp_io.decode_jpeg_batch([_bytes_tensor(s) for s in streams], device="cuda")
        self.assertEqual(len(gpu), len(streams))
        for got, ref in zip(gpu, cpu):
            d = np.abs(got.cpu().numpy().astype(np.int16) - ref.astype(np.int16))
            self.assertLessEqual(d.max(), 8, f"max diff {d.max()}")


class BatchIoTest(unittest.TestCase):
    """Batch entry points return byte-identical results to single calls."""

    def test_png_batch_decode_matches_single(self):
        raw = [_bytes_tensor(_png_bytes(_rgb(seed=50 + i))) for i in range(4)]
        got = tp_io.decode_png_batch(raw)
        ref = [tp_io.decode_png(t) for t in raw]
        self.assertEqual(len(got), len(raw))
        for g, r in zip(got, ref):
            np.testing.assert_array_equal(g.numpy(), r.numpy())

    def test_encode_jpeg_batch_matches_single(self):
        imgs = [tp.tensor(np.ascontiguousarray(_rgb(seed=60 + i).transpose(2, 0, 1)))
                for i in range(4)]
        got = tp_io.encode_jpeg_batch(imgs, quality=85)
        ref = [tp_io.encode_jpeg(i, quality=85) for i in imgs]
        self.assertEqual(len(got), len(imgs))
        for g, r in zip(got, ref):
            np.testing.assert_array_equal(g.numpy(), r.numpy())

    def test_encode_png_batch_matches_single(self):
        imgs = [tp.tensor(np.ascontiguousarray(_rgb(seed=70 + i).transpose(2, 0, 1)))
                for i in range(4)]
        got = tp_io.encode_png_batch(imgs)
        ref = [tp_io.encode_png(i) for i in imgs]
        self.assertEqual(len(got), len(imgs))
        for g, r in zip(got, ref):
            np.testing.assert_array_equal(g.numpy(), r.numpy())


class PerfProbeTest(unittest.TestCase):
    """Reports the batch speedup; informational, never asserted."""

    def test_batch_speedup_report(self):
        streams = [_jpeg_bytes(_rgb(seed=100 + i, h=256, w=384)) for i in range(12)]
        ts = [_bytes_tensor(s) for s in streams]

        import time
        t0 = time.perf_counter()
        for t in ts:
            tp_io.decode_jpeg(t)
        single = time.perf_counter() - t0

        t0 = time.perf_counter()
        tp_io.decode_jpeg_batch(ts)
        batch = time.perf_counter() - t0

        print(f"\n[perf] 12x(384x256) JPEG: sequential {single * 1e3:.1f} ms, "
              f"batch {batch * 1e3:.1f} ms, speedup {single / batch:.2f}x")


if __name__ == "__main__":
    unittest.main()