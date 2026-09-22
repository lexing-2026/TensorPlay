"""tensorplay.vision.io — image reading/writing.

The public surface covers ``tensorplay.vision.io`` (ImageReadMode, read_image,
decode_image, decode_jpeg, decode_png, encode_jpeg, encode_png, write_jpeg,
write_png, read_file, write_file, decode_jpeg_batch).

Decoding is routed through the native codec bindings (``tensorplay._C.io``)
when they are compiled into the extension: JPEG goes through the system
codec with SIMD-accelerated IDCT (and the hardware decoder on CUDA), PNG
through the system codec, and both land directly in CHW channel planes.  The
PIL path remains as a fallback for builds without the native codecs and for
container variants the native path rejects.

``decode_jpeg`` / ``decode_png`` return uint8 CHW tensors; ``read_image`` /
``decode_image`` return float32 tensors in [0, 1] keeping the classic
semantics of this module.
"""

import os
from enum import Enum

import numpy as np

import tensorplay as tensorplay
from PIL import Image

__all__ = [
    "ImageReadMode",
    "read_file",
    "write_file",
    "read_image",
    "decode_image",
    "decode_jpeg",
    "decode_png",
    "decode_jpeg_batch",
    "encode_jpeg",
    "encode_png",
    "write_jpeg",
    "write_png",
]


class ImageReadMode(Enum):
    """Support for various modes while reading images (tensorplay.vision.io)."""

    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4


_PIL_MODE_MAP = {
    ImageReadMode.UNCHANGED: None,
    ImageReadMode.GRAY: "L",
    ImageReadMode.GRAY_ALPHA: "LA",
    ImageReadMode.RGB: "RGB",
    ImageReadMode.RGB_ALPHA: "RGBA",
}


def _native_io():
    """The ``tensorplay._C.io`` submodule when the extension provides it."""
    return getattr(getattr(tensorplay, "_C", None), "io", None)


def _mode_to_n(mode: int):
    try:
        return ImageReadMode(mode)
    except ValueError:
        raise ValueError(f"mode should be a value between 0 and {len(ImageReadMode)-1}, got {mode}") from None


def read_file(path: str, start=None, size=None) -> tensorplay.Tensor:
    """Returns the bytes of ``path`` as a uint8 1-D tensor (tensorplay.vision.io.read_file)."""
    data = np.fromfile(path, dtype=np.uint8)
    if start is not None or size is not None:
        s = int(start or 0)
        e = s + int(size) if size is not None else len(data)
        data = data[s:e]
    return tensorplay.tensor(data)


def write_file(filename: str, data: tensorplay.Tensor) -> None:
    """Writes the contents of a uint8 tensor into a file (tensorplay.vision.io.write_file)."""
    with open(filename, "wb") as f:
        f.write(bytes(data.cpu().numpy().tobytes()))


def _apply_mode(img: Image.Image, mode: ImageReadMode) -> Image.Image:
    target = _PIL_MODE_MAP[mode]
    if target is None:
        return img
    if img.mode != target:
        if target == "L" and img.mode in ("P", "1"):
            img = img.convert("L")
        elif target in ("LA", "RGBA"):
            img = img.convert(target)
        else:
            img = img.convert(target)
    return img


def _pil_to_uint8_chw(img: Image.Image) -> tensorplay.Tensor:
    """PIL image -> uint8 CHW tensor (fallback for the native codecs)."""
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return tensorplay.tensor(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def _uint8_to_float(t: tensorplay.Tensor) -> tensorplay.Tensor:
    """uint8 CHW -> float32 CHW in [0, 1]."""
    return t.to(tensorplay.float32) / 255.0


def _as_uint8_chw(img: tensorplay.Tensor) -> tensorplay.Tensor:
    """Accepts float32 [0,1] or uint8 [0,255] CHW and returns contiguous uint8."""
    if img.dtype == tensorplay.float32:
        img = img.clamp(0.0, 1.0).mul(255.0).to(tensorplay.uint8)
    if img.dtype != tensorplay.uint8:
        raise TypeError(f"expected a float32 [0, 1] or uint8 [0, 255] tensor, got {img.dtype}")
    if img.dim() != 3:
        raise ValueError(f"expected a CHW (channels, height, width) tensor, got {img.shape}")
    if not img.is_contiguous():
        img = img.contiguous()
    return img


def decode_jpeg(data: tensorplay.Tensor, mode: int = ImageReadMode.UNCHANGED.value,
                device="cpu"):
    """Decodes JPEG bytes (uint8 tensor) into a uint8 CHW image tensor.

    Args:
        data: uint8 1-D tensor holding the image bytes.
        mode: ImageReadMode value (0 unchanged, 1 gray, 3 rgb).
        device: ``"cpu"`` or ``"cuda"``; CUDA uses the hardware decoder.
    """
    native = _native_io()
    if device == "cuda" and native is not None and hasattr(native, "decode_jpeg_cuda"):
        return native.decode_jpeg_cuda(data, int(mode))
    if device == "cpu" and native is not None and hasattr(native, "decode_jpeg"):
        return native.decode_jpeg(data, int(mode))
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    from io import BytesIO

    img = Image.open(BytesIO(bytes(data.cpu().numpy().tobytes())))
    img.load()
    img = _apply_mode(img, _mode_to_n(mode))
    return _pil_to_uint8_chw(img)


def decode_png(data: tensorplay.Tensor, mode: int = ImageReadMode.UNCHANGED.value):
    """Decodes PNG bytes (uint8 tensor) into a uint8 CHW image tensor."""
    native = _native_io()
    if native is not None and hasattr(native, "decode_png"):
        return native.decode_png(data, int(mode))
    from io import BytesIO

    img = Image.open(BytesIO(bytes(data.cpu().numpy().tobytes())))
    img.load()
    img = _apply_mode(img, _mode_to_n(mode))
    return _pil_to_uint8_chw(img)


def decode_jpeg_batch(data: list, mode: int = ImageReadMode.UNCHANGED.value, device="cpu"):
    """Decodes a sequence of JPEG byte tensors.

    ``device="cpu"`` parallelizes single-image decode across the shared
    thread pool; ``device="cuda"`` uses the hardware batch decoder when the
    extension provides it (mixed grayscale/color batches fall back to
    per-image hardware decode internally).  Returns a list of uint8 CHW
    tensors in input order.
    """
    native = _native_io()
    if native is not None and hasattr(native, "decode_jpeg_batch"):
        return native.decode_jpeg_batch(list(data), int(mode), device)
    return [decode_jpeg(d, mode, device) for d in data]


def encode_jpeg(img: tensorplay.Tensor, quality: int = 75) -> tensorplay.Tensor:
    """Encodes a CHW image tensor into JPEG bytes (uint8 1-D tensor).

    Accepts float32 in [0, 1] or uint8 in [0, 255]; quality is 0-100.
    """
    native = _native_io()
    img = _as_uint8_chw(img)
    if native is not None and hasattr(native, "encode_jpeg"):
        return native.encode_jpeg(img.cpu(), int(quality))
    import io as _io

    arr = img.cpu().numpy().transpose(1, 2, 0)
    arr = arr[:, :, 0] if arr.shape[2] == 1 else arr
    buf = _io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
    return tensorplay.tensor(np.frombuffer(buf.getvalue(), dtype=np.uint8).copy())


def encode_png(img: tensorplay.Tensor, compression_level: int = 6) -> tensorplay.Tensor:
    """Encodes a CHW image tensor into PNG bytes (uint8 1-D tensor).

    Accepts float32 in [0, 1] or uint8 in [0, 255]; compression level is 0-9.
    """
    native = _native_io()
    img = _as_uint8_chw(img)
    if native is not None and hasattr(native, "encode_png"):
        return native.encode_png(img.cpu(), int(compression_level))
    import io as _io

    arr = img.cpu().numpy().transpose(1, 2, 0)
    arr = arr[:, :, 0] if arr.shape[2] == 1 else arr
    buf = _io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=int(compression_level))
    return tensorplay.tensor(np.frombuffer(buf.getvalue(), dtype=np.uint8).copy())


def write_jpeg(img: tensorplay.Tensor, filename: str, quality: int = 75) -> None:
    """Encodes and writes a tensor as JPEG (tensorplay.vision.io.write_jpeg)."""
    with open(filename, "wb") as f:
        f.write(bytes(encode_jpeg(img, quality).cpu().numpy().tobytes()))


def write_png(img: tensorplay.Tensor, filename: str, compression_level: int = 6) -> None:
    """Encodes and writes a tensor as PNG (tensorplay.vision.io.write_png)."""
    with open(filename, "wb") as f:
        f.write(bytes(encode_png(img, compression_level).cpu().numpy().tobytes()))


_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _sniff(data: bytes) -> str:
    if data.startswith(_JPEG_MAGIC):
        return "jpeg"
    if data.startswith(_PNG_MAGIC):
        return "png"
    return "unknown"


def read_image(path: str, mode: int = ImageReadMode.UNCHANGED.value,
               device="cpu") -> tensorplay.Tensor:
    """Reads an image from ``path`` as a float32 CHW tensor in [0, 1].

    The container is detected from the file header; native codecs take the
    fast path and PIL handles the rest.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        data = f.read()
    return decode_image(tensorplay.tensor(np.frombuffer(data, dtype=np.uint8).copy()), mode, device)


def decode_image(data: tensorplay.Tensor, mode: int = ImageReadMode.UNCHANGED.value,
                 device="cpu"):
    """Decodes image bytes (auto-detected container) as float32 CHW in [0, 1]."""
    raw = data.cpu().numpy().tobytes()
    kind = _sniff(raw)
    if kind == "jpeg":
        return _uint8_to_float(decode_jpeg(data, mode, device))
    if kind == "png":
        return _uint8_to_float(decode_png(data, mode))
    from io import BytesIO

    img = Image.open(BytesIO(raw))
    img.load()
    img = _apply_mode(img, _mode_to_n(mode))
    return _uint8_to_float(_pil_to_uint8_chw(img))
