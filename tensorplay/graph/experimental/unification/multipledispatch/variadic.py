from __future__ import annotations

from .utils import typename

__all__ = ["Variadic", "VariadicSignatureMeta", "VariadicSignatureType", "isvariadic"]


class VariadicSignatureType(type):
    def __subclasscheck__(cls, subclass: type) -> bool:
        values = subclass.variadic_type if isvariadic(subclass) else (subclass,)
        return all(issubclass(value, cls.variadic_type) for value in values)

    def __eq__(cls, other: object) -> bool:
        return isvariadic(other) and set(cls.variadic_type) == set(other.variadic_type)

    def __hash__(cls) -> int:
        return hash((type(cls), frozenset(cls.variadic_type)))


def isvariadic(value: object) -> bool:
    return isinstance(value, VariadicSignatureType)


class VariadicSignatureMeta(type):
    def __getitem__(cls, value: type | tuple[type, ...]) -> VariadicSignatureType:
        values = value if isinstance(value, tuple) else (value,)
        if not values or not all(isinstance(item, type) for item in values):
            raise ValueError("variadic signatures require one or more types")
        return VariadicSignatureType(
            f"Variadic[{typename(values)}]",
            (),
            {"variadic_type": values, "__slots__": ()},
        )


class Variadic(metaclass=VariadicSignatureMeta):
    pass
