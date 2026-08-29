"""Dtype collection helpers used to sweep the test suite over dtypes."""

import tensorplay as tp

__all__ = [
    "get_all_dtypes",
    "get_all_math_dtypes",
    "get_all_complex_dtypes",
    "get_all_int_dtypes",
    "get_all_fp_dtypes",
    "highest_precision_float",
]


def get_all_dtypes(
    include_half=True,
    include_bfloat16=True,
    include_bool=True,
    include_complex=True,
    include_complex32=False,
) -> list[tp.dtype]:
    dtypes = get_all_int_dtypes() + get_all_fp_dtypes(
        include_half=include_half, include_bfloat16=include_bfloat16
    )
    if include_bool:
        dtypes.append(tp.bool)
    if include_complex:
        dtypes += get_all_complex_dtypes(include_complex32)
    return dtypes


def get_all_math_dtypes(device) -> list[tp.dtype]:
    return (
        get_all_int_dtypes()
        + get_all_fp_dtypes(
            include_half=str(device).startswith("cuda"), include_bfloat16=False
        )
        + get_all_complex_dtypes()
    )


def get_all_complex_dtypes(include_complex32=False) -> list[tp.dtype]:
    return (
        [tp.complex32, tp.complex64, tp.complex128]
        if include_complex32
        else [tp.complex64, tp.complex128]
    )


def get_all_int_dtypes() -> list[tp.dtype]:
    return [tp.uint8, tp.int8, tp.int16, tp.int32, tp.int64]


def get_all_fp_dtypes(include_half=True, include_bfloat16=True) -> list[tp.dtype]:
    dtypes = [tp.float32, tp.float64]
    if include_half:
        dtypes.append(tp.float16)
    if include_bfloat16:
        dtypes.append(tp.bfloat16)
    return dtypes


def highest_precision_float(device) -> tp.dtype:
    return tp.float64
