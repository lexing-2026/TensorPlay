
from __future__ import annotations

import tensorplay._C as _C

__all__ = ["finfo", "iinfo"]


class finfo:
    __slots__ = ("dtype", "bits", "eps", "max", "min", "tiny",
                 "smallest_normal", "resolution")

    def __init__(self, dtype):
        info = {
            _C.float16: (16, 0.0009765625, 65504.0, 6.103515625e-05, 0.001),
            _C.bfloat16: (16, 0.0078125, 3.3895313892515355e+38,
                          1.1754943508222875e-38, 0.01),
            _C.float32: (32, 1.1920928955078125e-07, 3.4028234663852886e+38,
                         1.1754943508222875e-38, 1e-06),
            _C.float64: (64, 2.220446049250313e-16, 1.7976931348623157e+308,
                         2.2250738585072014e-308, 1e-15),
        }
        complex_map = {
            _C.complex64: _C.float32,
            _C.complex128: _C.float64,
        }
        if hasattr(_C, "complex32"):
            complex_map[_C.complex32] = _C.float16
        resolved = complex_map.get(dtype)
        if resolved is None:
            if dtype not in info:
                raise TypeError(
                    f"TensorPlay doesn't support {dtype!r} for finfo: "
                    "expected a floating point or complex dtype"
                )
        else:
            dtype = resolved
        bits, eps, max_, tiny, resolution = info[dtype]
        self.dtype = dtype
        self.bits = bits
        self.eps = eps
        self.max = max_
        self.min = -max_
        self.tiny = tiny
        self.smallest_normal = tiny
        self.resolution = resolution

    def __repr__(self):
        return (
            f"finfo(resolution={self.resolution}, min={self.min}, max={self.max}, "
            f"eps={self.eps}, smallest_normal={self.smallest_normal}, tiny={self.tiny})"
        )


class iinfo:
    __slots__ = ("dtype", "bits", "min", "max")

    def __init__(self, dtype):
        info = {
            _C.int8: (8, -128, 127),
            _C.uint8: (8, 0, 255),
            _C.int16: (16, -32768, 32767),
            getattr(_C, "uint16", None): (16, 0, 65535),
            _C.int32: (32, -2147483648, 2147483647),
            getattr(_C, "uint32", None): (32, 0, 4294967295),
            _C.int64: (64, -(2 ** 63), 2 ** 63 - 1),
            getattr(_C, "uint64", None): (64, 0, 2 ** 64 - 1),
            getattr(_C, "bool", None): (8, 0, 1),
        }
        entry = info.get(dtype)
        if entry is None:
            raise TypeError(
                f"TensorPlay doesn't support {dtype!r} for iinfo: "
                "expected an integer dtype"
            )
        bits, min_, max_ = entry
        self.dtype = dtype
        self.bits = bits
        self.min = min_
        self.max = max_

    def __repr__(self):
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})"
