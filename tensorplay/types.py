# In some cases, these basic types are shadowed by corresponding
# top-level values.  The underscore variants let us refer to these
# types.  See https://github.com/python/mypy/issues/4146 for why these
# workarounds is necessary
import os
from builtins import (  # noqa: F401
    bool as _bool,
    bytes as _bytes,
    complex as _complex,
    float as _float,
    int as _int,
    str as _str,
)
from collections.abc import Sequence
from typing import Any, IO, TYPE_CHECKING, Union
from typing_extensions import Self, TypeAlias

# `as` imports have better static analysis support than assignment `ExposedType: TypeAlias = HiddenType`
from tensorplay._C import (  # noqa: F401
    Device as _device,
    DType as _dtype,
    Size as Size,
)
from tensorplay._tensor import Tensor as Tensor

# Placeholder for DispatchKey, layout, qscheme, Sym* if they are added later
# from tensorplay import (
#     DispatchKey as DispatchKey,
#     layout as _layout,
#     qscheme as _qscheme,
#     SymBool as SymBool,
#     SymFloat as SymFloat,
#     SymInt as SymInt,
# )


if TYPE_CHECKING:
    # from tensorplay.autograd.graph import GradientEdge
    pass


__all__ = ["Number", "Device", "FileLike", "Storage"]

# Convenience aliases for common composite types that we need
# to talk about in TensorPlay
_TensorOrTensors: TypeAlias = Union[Tensor, Sequence[Tensor]]
_TensorOrTensorsOrGradEdge: TypeAlias = Union[  # noqa: PYI047
    Tensor,
    Sequence[Tensor],
    # "GradientEdge",
    # Sequence["GradientEdge"],
]

_size: TypeAlias = Union[Size, list[int], tuple[int, ...]]  # noqa: PYI042,PYI047
# _symsize: TypeAlias = Union[Size, Sequence[Union[int, SymInt]]]  # noqa: PYI042,PYI047
# _dispatchkey: TypeAlias = Union[str, DispatchKey]  # noqa: PYI042,PYI047

# int or SymInt
IntLikeType: TypeAlias = int # Union[int, SymInt]
# float or SymFloat
FloatLikeType: TypeAlias = float # Union[float, SymFloat]
# bool or SymBool
BoolLikeType: TypeAlias = bool # Union[bool, SymBool]

# py_sym_types = (SymInt, SymFloat, SymBool)  # left un-annotated intentionally
# PySymType: TypeAlias = Union[SymInt, SymFloat, SymBool]

# Meta-type for "numeric" things; matches our docs
Number: TypeAlias = Union[int, float, bool]
# tuple for isinstance(x, Number) checks.
# FIXME: refactor once python 3.9 support is dropped.
_Number = (int, float, bool)

FileLike: TypeAlias = Union[str, os.PathLike[str], IO[bytes]]

# Meta-type for "device-like" things.  Not to be confused with 'device' (a
# literal device object).  This nomenclature is consistent with PythonArgParser.
# None means use the default device (typically CPU)
Device: TypeAlias = Union[_device, str, int, None]


# Storage protocol implemented by ${Type}StorageBase classes
class Storage:
    _cdata: int
    device: _device
    dtype: _dtype
    _load_uninitialized: bool

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        raise NotImplementedError

    def _new_shared(self, size: int) -> Self:
        raise NotImplementedError

    def _write_file(
        self,
        f: Any,
        is_real_file: bool,
        save_size: bool,
        element_size: int,
    ) -> None:
        raise NotImplementedError

    def element_size(self) -> int:
        raise NotImplementedError

    def is_shared(self) -> bool:
        raise NotImplementedError

    def share_memory_(self) -> Self:
        raise NotImplementedError

    def nbytes(self) -> int:
        raise NotImplementedError

    def cpu(self) -> Self:
        raise NotImplementedError

    def data_ptr(self) -> int:
        raise NotImplementedError

    def from_file(
        self,
        filename: str,
        shared: bool = False,
        byte: int = 0,
    ) -> Self:
        raise NotImplementedError

    def _new_with_file(
        self,
        f: Any,
        element_size: int,
    ) -> Self:
        raise NotImplementedError
