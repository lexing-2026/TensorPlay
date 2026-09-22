"""Throughput benchmark for the native image/audio IO codecs.

Each native path (CPU single, CPU batch, CUDA batch) is timed on identical
inputs alongside the equivalent calls from the libraries installed in this
environment, so the comparison is apples-to-apples.  Timings are best-of-N
after warm-up.  This script only prints; it asserts nothing.

Usage: python test/bench_io_native.py
"""

import io as _io
import time

import numpy as np
from PIL import Image

import tensorplay as tp
from tensorplay.vision import io as tp_io
from tensorplay.audio import decode_wav, decode_wav_batch, encode_wav, encode_wav_batch

try:
    import torch
    import torchvision.io as tv_io
except ImportError:  # pragma: no cover
    torch = None
    tv_io = None

from scipy.io import wavfile
import soundfile as sf

_NATIVE = getattr(getattr(tp, "_C", None), "io", None)

_N_IMAGES = 16
_IMG_H, _IMG_W = 1080, 1920
_REPEATS = 3
_WARMUP = 1


def _bytes_tensor(data: bytes):
    return tp.tensor(np.frombuffer(data, dtype=np.uint8).copy())


def _best_of(fn, repeats=_REPEATS, warmup=_WARMUP):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _row(name, seconds, n, per="item", size=None):
    rate = n / seconds
    extra = ""
    if size is not None:
        mbs = size / (seconds * 1e6)
        extra = f"  {mbs:8.1f} MB/s"
    print(f"{name:<42} {seconds * 1e3:9.2f} ms   {rate:9.1f} {per}/s{extra}")


def _imgs(seed=0, n=_N_IMAGES, h=_IMG_H, w=_IMG_W):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for _ in range(n)]


def _jpeg_blobs(imgs, quality=90):
    blobs = []
    for img in imgs:
        buf = _io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        blobs.append(buf.getvalue())
    return blobs


def bench_jpeg():
    imgs = _imgs()
    blobs = _jpeg_blobs(imgs)
    n = len(blobs)
    mpix = n * _IMG_H * _IMG_W / 1e6

    def chunks():
        for img in imgs:
            yield np.ascontiguousarray(img.transpose(2, 0, 1))

    print(f"\nJPEG decode  {_IMG_W}x{_IMG_H} RGB, {n} images ({mpix:.1f} MP), quality 90")
    sec = _best_of(lambda: [tp_io.decode_jpeg(_bytes_tensor(b)) for b in blobs])
    _row("native single (loop)", sec, n, "img")
    sec = _best_of(lambda: tp_io.decode_jpeg_batch([_bytes_tensor(b) for b in blobs]))
    _row("native CPU batch", sec, n, "img")
    if tv_io is not None:
        sec = _best_of(lambda: [
            tv_io.decode_jpeg(torch.from_numpy(np.frombuffer(b, dtype=np.uint8).copy()))
            for b in blobs])
        _row("torchvision single (loop)", sec, n, "img")
    sec = _best_of(lambda: [np.asarray(Image.open(_io.BytesIO(b)).convert("RGB")) for b in blobs])
    _row("PIL single (loop)", sec, n, "img")

    print("CUDA JPEG decode (same inputs)")
    if torch is not None and _NATIVE is not None and hasattr(_NATIVE, "decode_jpeg_cuda"):
        dev = torch.cuda.get_device_properties(0)
        print(f"  device: {dev.name} (compute capability {dev.major}.{dev.minor})")
        print("  note: the dedicated hardware JPEG engine ships with data-center")
        print("  Ampere and newer; nvjpeg elsewhere runs its software path.")
        sec = _best_of(lambda: [tp_io.decode_jpeg(_bytes_tensor(b), device="cuda") for b in blobs])
        _row("native CUDA single (loop)", sec, n, "img")
        sec = _best_of(lambda: tp_io.decode_jpeg_batch(
            [_bytes_tensor(b) for b in blobs], device="cuda"))
        _row("native CUDA batch", sec, n, "img")
        sec = _best_of(lambda: [
            tp_io.decode_jpeg(_bytes_tensor(b), device="cuda").cpu() for b in blobs])
        _row("native CUDA single + transfer", sec, n, "img")

    print("JPEG encode (same images)")
    chws = list(chunks())
    tensors = [tp.tensor(c) for c in chws]
    sec = _best_of(lambda: [tp_io.encode_jpeg(tp.tensor(c), quality=90) for c in chws])
    _row("native encode (loop)", sec, n, "img")
    sec = _best_of(lambda: tp_io.encode_jpeg_batch(tensors, quality=90))
    _row("native encode batch", sec, n, "img")
    sec = _best_of(lambda: [Image.fromarray(img).save(
        _io.BytesIO(), format="JPEG", quality=90) for img in imgs])
    _row("PIL encode (loop)", sec, n, "img")


def bench_png():
    imgs = _imgs(seed=7, n=8)
    chws = [np.ascontiguousarray(img.transpose(2, 0, 1)) for img in imgs]
    blobs = []
    for img in imgs:
        buf = _io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        blobs.append(buf.getvalue())
    n = len(imgs)
    size = len(blobs[0])
    print(f"\nPNG decode  {_IMG_W}x{_IMG_H} RGB x{n} ({size / 1e6:.1f} MB each)")
    sec = _best_of(lambda: tp_io.decode_png(_bytes_tensor(blobs[0])))
    _row("native single", sec, 1, "img", size)
    if tv_io is not None:
        sec = _best_of(lambda: tv_io.decode_png(torch.from_numpy(
            np.frombuffer(blobs[0], dtype=np.uint8).copy())))
        _row("torchvision single", sec, 1, "img", size)
    sec = _best_of(lambda: np.asarray(Image.open(_io.BytesIO(blobs[0])).convert("RGB")))
    _row("PIL single", sec, 1, "img", size)
    sec = _best_of(lambda: tp_io.decode_png_batch([_bytes_tensor(b) for b in blobs]))
    _row("native CPU batch", sec, n, "img", size * n)

    print("PNG encode")
    tensors = [tp.tensor(c) for c in chws]
    mb = chws[0].nbytes
    sec = _best_of(lambda: tp_io.encode_png(tensors[0]))
    _row("native single", sec, 1, "img", mb)
    sec = _best_of(lambda: tp_io.encode_png_batch(tensors))
    _row("native encode batch", sec, n, "img", mb * n)
    sec = _best_of(lambda: [Image.fromarray(img).save(_io.BytesIO(), format="PNG")
                            for img in imgs])
    _row("PIL encode (loop)", sec, n, "img", mb * n)


def bench_wav():
    sr = 44100
    t = 30.0
    frames = int(sr * t)
    rng = np.random.default_rng(11)
    wave = (0.3 * np.sin(2 * np.pi * 440 * np.arange(frames) / sr)
            + 0.2 * rng.standard_normal(frames)).astype(np.float32)
    stereo = np.stack([wave, wave], axis=0)
    t_wav = tp.tensor(stereo)
    blob_t = encode_wav(t_wav, sr, bits=16)
    blob = np.frombuffer(blob_t.numpy(), dtype=np.uint8).tobytes()
    size = len(blob)
    n = 8
    blobs = [blob_t] * n
    raws = [blob] * n
    print(f"\nWAV decode  {t:.0f}s stereo PCM16 {sr} Hz ({size / 1e6:.1f} MB x{n})")
    sec = _best_of(lambda: [decode_wav(b) for b in blobs])
    _row("native single (loop)", sec, n, "file", size * n)
    sec = _best_of(lambda: decode_wav_batch(blobs))
    _row("native CPU batch", sec, n, "file", size * n)
    sec = _best_of(lambda: [wavfile.read(_io.BytesIO(r)) for r in raws])
    _row("scipy single (loop)", sec, n, "file", size * n)
    sec = _best_of(lambda: [sf.read(_io.BytesIO(r)) for r in raws])
    _row("soundfile single (loop)", sec, n, "file", size * n)

    print("WAV encode (PCM16)")
    i16 = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)
    sec = _best_of(lambda: encode_wav(t_wav, sr, bits=16))
    _row("native encode (single call)", sec, 1, "file", stereo.nbytes)
    sec = _best_of(lambda: encode_wav_batch([t_wav] * n, sr, bits=16))
    _row("native encode batch", sec, n, "file", stereo.nbytes * n)
    sec = _best_of(lambda: [wavfile.write(_io.BytesIO(), sr, i16.T) for _ in range(n)])
    _row("scipy encode (loop)", sec, n, "file", stereo.nbytes * n)
    sec = _best_of(lambda: [sf.write(
        _io.BytesIO(), stereo.T, sr, format="WAV", subtype="PCM_16") for _ in range(n)])
    _row("soundfile encode (loop)", sec, n, "file", stereo.nbytes * n)


def bench_thread_scaling():
    # The pool size is fixed at process start (OMP_NUM_THREADS), so each
    # thread count is measured in a fresh subprocess on the same inputs.
    import glob
    import os
    import subprocess
    import sys
    import tempfile

    imgs = _imgs(seed=3, n=_N_IMAGES)
    blobs = _jpeg_blobs(imgs, quality=90)
    with tempfile.TemporaryDirectory() as d:
        for i, b in enumerate(blobs):
            with open(os.path.join(d, f"{i}.jpg"), "wb") as f:
                f.write(b)
        code = (
            "import glob, os, time\n"
            "import numpy as np\n"
            "import tensorplay as tp\n"
            "from tensorplay.vision import io as io_\n"
            "files = [open(p, 'rb').read()\n"
            "         for p in sorted(glob.glob(os.path.join(os.environ['IMGDIR'], '*.jpg')))]\n"
            "ts = [tp.tensor(np.frombuffer(b, dtype=np.uint8).copy()) for b in files]\n"
            "io_.decode_jpeg_batch(ts)\n"
            "best = float('inf')\n"
            "for _ in range(3):\n"
            "    t0 = time.perf_counter()\n"
            "    io_.decode_jpeg_batch(ts)\n"
            "    best = min(best, time.perf_counter() - t0)\n"
            "print(best)\n"
        )
        print(f"\nCPU batch thread scaling ({_N_IMAGES} images, 20 cores)")
        for k in (1, 4, 8, 12, 16, 20):
            env = dict(os.environ, OMP_NUM_THREADS=str(k), IMGDIR=d)
            out = subprocess.run([sys.executable, "-c", code], env=env,
                                 capture_output=True, text=True)
            if out.returncode != 0:
                print(f"threads={k:2d}  subprocess failed: {out.stderr.strip()[-200:]}")
                continue
            sec = float(out.stdout.strip().splitlines()[-1])
            _row(f"threads={k:2d} (fresh proc)", sec, _N_IMAGES, "img")


def main():
    bench_jpeg()
    bench_png()
    bench_wav()
    bench_thread_scaling()


if __name__ == "__main__":
    main()
