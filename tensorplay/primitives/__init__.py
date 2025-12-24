from __future__ import annotations

import typing
from typing import (
    Any,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import TypeAlias

import tensorplay


if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.

    class _WorksWithInt(typing.Protocol):
        def __add__(self, other: Any) -> typing.Self: ...

        def __radd__(self, other: Any) -> typing.Self: ...

        def __mul__(self, other: Any) -> typing.Self: ...

        def __rmul__(self, other: Any) -> typing.Self: ...

    _IntLikeT = TypeVar("_IntLikeT", bound=_WorksWithInt)


ShapeType: TypeAlias = Union[tensorplay.Size, list[int], tuple[int, ...]]
StrideType: TypeAlias = Union[list[int], tuple[int, ...]]
DimsType: TypeAlias = Union[int, list[int], tuple[int, ...]]
DimsSequenceType: TypeAlias = Union[list[int], tuple[int, ...]]
# TODO: Type[tensorplay.SymInt], Type[tensorplay.SymFloat]
NumberTypeType: TypeAlias = Union[type[bool], type[int], type[float], type[complex]]
# TODO: This needs a lot more type annotations
# NumberType = Union[bool, int, float, complex, tensorplay.SymInt, tensorplay.SymFloat]
NumberType: TypeAlias = Union[bool, int, float, complex]
RealNumberType: TypeAlias = Union[bool, int, float]

Number = (bool, int, float, complex)
# I don't call it Integral because numbers.Integral includes bool, but IntLike
# does not
Dim = int
IntWithoutSymInt = int
FloatWithoutSymFloat = float
DeviceLikeType: TypeAlias = Union[str, tensorplay.device, int]
Tensor = tensorplay.Tensor