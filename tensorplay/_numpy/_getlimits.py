import tensorplay

from . import _dtypes


def finfo(dtyp: object) -> tensorplay.finfo:
    torch_dtype = _dtypes.dtype(dtyp).torch_dtype
    return tensorplay.finfo(torch_dtype)


def iinfo(dtyp: object) -> tensorplay.iinfo:
    torch_dtype = _dtypes.dtype(dtyp).torch_dtype
    return tensorplay.iinfo(torch_dtype)
