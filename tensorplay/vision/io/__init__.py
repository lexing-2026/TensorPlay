"""tensorplay.vision.io — image reading/writing.

The public surface mirrors ``torchvision.io`` (ImageReadMode, read_image,
decode_image, decode_jpeg, decode_png, encode_jpeg, encode_png, write_jpeg,
write_png, read_file, write_file).  Decoding is delegated to PIL so no native
codec dependency is required; tensors are produced through the optimized
``vision_to_tensor`` path when available.
"""

import os
from enum import Enum

import numpy as np

import tensorplay as torch
from PIL import Image

__all__ = [
    "ImageReadMode",
    "read_file",
    "write_file",
    "read_image",
    "decode_image",
    "decode_jpeg",
    "decode_png",
    "encode_jpeg",
    "encode_png",
    "write_jpeg",
    "write_png",
]


class ImageReadMode(Enum):
    """Support for various modes while reading images (torchvision.io)."""

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


def _pil_to_tensor(pic: Image.Image) -> torch.Tensor:
    """HWC uint8 numpy -> CHW float32 [0,1] via the C++ fast path."""
    arr = np.array(pic)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if hasattr(torch, "vision_to_tensor") and arr.dtype == np.uint8:
        return torch.vision_to_tensor(np.ascontiguousarray(arr))
    t = torch.tensor(np.ascontiguousarray(arr.transpose(2, 0, 1)))
    if t.dtype == torch.uint8:
        t = t.to(torch.float32) / 255.0
    return t


def _mode_to_n(mode: int):
    try:
        return ImageReadMode(mode)
    except ValueError:
        raise ValueError(f"mode should be a value between 0 and {len(ImageReadMode)-1}, got {mode}") from None


def read_file(path: str, start=None, size=None) -> torch.Tensor:
    """Returns the bytes of ``path`` as a uint8 1-D tensor (torchvision.io.read_file)."""
    data = np.fromfile(path, dtype=np.uint8)
    if start is not None or size is not None:
        s = int(start or 0)
        e = s + int(size) if size is not None else len(data)
        data = data[s:e]
    return torch.tensor(data)


def write_file(filename: str, data: torch.Tensor) -> None:
    """Writes the contents of a uint8 tensor into a file (torchvision.io.write_file)."""
    with open(filename, "wb") as f:
        f.write(bytes(data.cpu().numpy().tobytes()))


def _apply_mode(img: Image.Image, mode: ImageReadMode) -> Image.Image:
    target = _PIL_MODE_MAP[mode]
    if target is None:
        # UNCHANGED: promote P / 1-mode images the way torchvision documents
        return img
    if img.mode != target:
        if target == "L" and img.mode in ("P", "1"):
            img = img.convert("L")
        elif target in ("LA", "RGBA"):
            img = img.convert(target)
        else:
            img = img.convert(target)
    return img


def decode_jpeg(data: torch.Tensor, mode: int = ImageReadMode.UNCHANGED.value, device="cpu"):
    """Decodes JPEG bytes (uint8 tensor) into an image tensor."""
    from io import BytesIO

    img = Image.open(BytesIO(bytes(data.cpu().numpy().tobytes())))
    img.load()
    img = _apply_mode(img, _mode_to_n(mode))
    return _pil_to_tensor(img)


def decode_png(data: torch.Tensor, mode: int = ImageReadMode.UNCHANGED.value):
    """Decodes PNG bytes (uint8 tensor) into an image tensor."""
    return decode_jpeg(data, mode)


def encode_jpeg(img: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Encodes a CHW uint8 [0,255] tensor into JPEG bytes as a uint8 tensor."""
    import io as _io

    arr = img.cpu().numpy()
    if arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    buf = _io.BytesIO()
    Image.fromarray(arr.squeeze() if arr.ndim == 3 and arr.shape[2] == 1 else arr).save(
        buf, format="JPEG", quality=int(quality)
    )
    return torch.tensor(np.frombuffer(buf.getvalue(), dtype=np.uint8))


def encode_png(img: torch.Tensor, compression_level: int = 6) -> torch.Tensor:
    """Encodes a CHW uint8 [0,255] tensor into PNG bytes as a uint8 tensor."""
    import io as _io

    arr = img.cpu().numpy()
    if arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    buf = _io.BytesIO()
    Image.fromarray(arr.squeeze() if arr.ndim == 3 and arr.shape[2] == 1 else arr).save(
        buf, format="PNG", compress_level=int(compression_level)
    )
    return torch.tensor(np.frombuffer(buf.getvalue(), dtype=np.uint8))


def write_jpeg(img: torch.Tensor, filename: str, quality: int = 75) -> None:
    """Encodes and writes a tensor as JPEG (torchvision.io.write_jpeg)."""
    with open(filename, "wb") as f:
        f.write(bytes(encode_jpeg(img, quality).cpu().numpy().tobytes()))


def write_png(img: torch.Tensor, filename: str, compression_level: int = 6) -> None:
    """Encodes and writes a tensor as PNG (torchvision.io.write_png)."""
    with open(filename, "wb") as f:
        f.write(bytes(encode_png(img, compression_level).cpu().numpy().tobytes()))


def read_image(path: str, mode: int = ImageReadMode.UNCHANGED.value) -> torch.Tensor:
    """Reads an image from ``path`` and returns it as a tensor
    (torchvision.io.read_image semantics: CHW, float32 in [0,1])."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    img = Image.open(path)
    img.load()
    img = _apply_mode(img, _mode_to_n(mode))
    return _pil_to_tensor(img)


def decode_image(data: torch.Tensor, mode: int = ImageReadMode.UNCHANGED.value):
    """Decodes image bytes, auto-detecting the format (torchvision.io.decode_image)."""
    from io import BytesIO

    img = Image.open(BytesIO(bytes(data.cpu().numpy().tobytes())))
    img.load()
    img = _apply_mode(img, _mode_to_n(mode))
    return _pil_to_tensor(img)
