"""Runtime model of operator schemas.

Parses the schema text carried by every operator overload
(``add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor``)
into :class:`FunctionSchema`: arguments and returns with their types,
defaults, keyword-only markers and alias annotations.  Alias annotations
drive mutation and view analysis: ``Tensor(a!)`` is written in place,
``Tensor(a)`` on a return aliases the argument annotated with the same set.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AliasInfo",
    "Argument",
    "BoolType",
    "DeviceObjType",
    "FloatType",
    "FunctionSchema",
    "GeneratorType",
    "IntType",
    "LayoutType",
    "ListType",
    "MemoryFormatType",
    "NumberType",
    "OptionalType",
    "ScalarTypeType",
    "SchemaParseError",
    "SchemaType",
    "StorageType",
    "StringType",
    "SymBoolType",
    "SymFloatType",
    "SymIntType",
    "TensorType",
    "parse_schema",
]


class SchemaParseError(ValueError):
    """Schema text that does not follow the operator schema grammar."""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class SchemaType:
    """Base of every schema type; types compare and hash by spelling."""

    name = "?"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SchemaType) and str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    def is_tensor_like(self) -> bool:
        return False


class TensorType(SchemaType):
    name = "Tensor"

    def is_tensor_like(self) -> bool:
        return True


class IntType(SchemaType):
    name = "int"


class SymIntType(SchemaType):
    name = "SymInt"


class SymBoolType(SchemaType):
    name = "SymBool"


class SymFloatType(SchemaType):
    name = "SymFloat"


class FloatType(SchemaType):
    name = "float"


class BoolType(SchemaType):
    name = "bool"


class NumberType(SchemaType):
    """``Scalar``: an int, float, bool or complex number."""

    name = "Scalar"


class StringType(SchemaType):
    name = "str"


class DeviceObjType(SchemaType):
    name = "Device"


class ScalarTypeType(SchemaType):
    """A dtype (``ScalarType``)."""

    name = "ScalarType"


class LayoutType(SchemaType):
    name = "Layout"


class MemoryFormatType(SchemaType):
    name = "MemoryFormat"


class GeneratorType(SchemaType):
    name = "Generator"


class StorageType(SchemaType):
    name = "Storage"


class OptionalType(SchemaType):
    def __init__(self, element: SchemaType) -> None:
        self.element = element

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self.element}?"

    def getElementType(self) -> SchemaType:  # noqa: N802 - schema object API
        return self.element

    def is_tensor_like(self) -> bool:
        return self.element.is_tensor_like()


class ListType(SchemaType):
    def __init__(self, element: SchemaType, size: int | None = None) -> None:
        self.element = element
        self.size = size

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self.element}[{'' if self.size is None else self.size}]"

    def getElementType(self) -> SchemaType:  # noqa: N802 - schema object API
        return self.element

    def is_tensor_like(self) -> bool:
        return self.element.is_tensor_like()


_BASE_TYPES: dict[str, type[SchemaType]] = {
    "Tensor": TensorType,
    "int": IntType,
    "int64_t": IntType,
    "SymInt": SymIntType,
    "SymBool": SymBoolType,
    "SymFloat": SymFloatType,
    "float": FloatType,
    "bool": BoolType,
    "Scalar": NumberType,
    "str": StringType,
    "Device": DeviceObjType,
    "ScalarType": ScalarTypeType,
    "Layout": LayoutType,
    "MemoryFormat": MemoryFormatType,
    "Generator": GeneratorType,
    "Storage": StorageType,
}


# ---------------------------------------------------------------------------
# Arguments and schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasInfo:
    """Alias sets of one value: ``Tensor(a!)`` is ``before_set={"a"}``,
    ``is_write=True``; ``Tensor(a -> *)`` adds ``after_set={"*"}``."""

    before_set: frozenset[str]
    after_set: frozenset[str]
    is_write: bool


@dataclass(frozen=True)
class Argument:
    name: str
    type: SchemaType
    default_value: Any = None
    has_default: bool = False
    kwarg_only: bool = False
    alias_info: AliasInfo | None = None
    is_out: bool = False
    #: Declared length of a fixed-size list (``int[2]``).
    N: int | None = None

    @property
    def real_type(self) -> SchemaType:
        return self.type

    def has_default_value(self) -> bool:
        return self.has_default

    def __str__(self) -> str:
        text = _type_text(self.type, self.alias_info)
        if self.name:
            text += f" {self.name}"
        if self.has_default:
            text += f"={_default_text(self.default_value)}"
        return text


@dataclass(frozen=True)
class FunctionSchema:
    #: Operator name without the overload (``"add"``).
    name: str
    #: Overload name (``"Tensor"``), empty for the default overload.
    overload_name: str
    arguments: tuple[Argument, ...]
    returns: tuple[Argument, ...]
    namespace: str
    text: str = field(default="", compare=False)

    @property
    def is_mutable(self) -> bool:
        return any(
            a.alias_info is not None and a.alias_info.is_write for a in self.arguments
        )

    @property
    def is_vararg(self) -> bool:
        return False

    @property
    def is_varret(self) -> bool:
        return False

    def qualified_name(self) -> str:
        return f"{self.namespace}::{self.name}"

    def _is_view_op(self) -> bool:
        """A non-mutating op whose return aliases an argument."""

        if self.is_mutable:
            return False
        arg_sets = {
            name
            for a in self.arguments
            if a.alias_info is not None
            for name in a.alias_info.before_set
        }
        return any(
            r.alias_info is not None and r.alias_info.before_set & arg_sets
            for r in self.returns
        )

    def mutated_arguments(self) -> tuple[Argument, ...]:
        return tuple(
            a for a in self.arguments if a.alias_info is not None and a.alias_info.is_write
        )

    def __str__(self) -> str:
        head = self.name + (f".{self.overload_name}" if self.overload_name else "")
        parts: list[str] = []
        star = False
        for a in self.arguments:
            if a.kwarg_only and not star:
                parts.append("*")
                star = True
            parts.append(str(a))
        if len(self.returns) == 1 and not self.returns[0].name:
            rets = str(self.returns[0])
        else:
            rets = "(" + ", ".join(str(r) for r in self.returns) + ")"
        return f"{head}({', '.join(parts)}) -> {rets}"


def _type_text(t: SchemaType, alias: AliasInfo | None) -> str:
    if alias is None:
        return str(t)
    sets = "|".join(sorted(alias.before_set))
    if alias.after_set and alias.after_set != alias.before_set:
        sets += " -> " + "|".join(sorted(alias.after_set))
    ann = f"({sets}{'!' if alias.is_write else ''})"
    base = t
    suffix = ""
    while isinstance(base, (ListType, OptionalType)):
        suffix = ("?" if isinstance(base, OptionalType) else
                  f"[{'' if base.size is None else base.size}]") + suffix
        base = base.element
    return f"{base}{ann}{suffix}"


def _default_text(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_default_text(v) for v in value) + "]"
    return str(value)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def error(self, message: str) -> SchemaParseError:
        return SchemaParseError(f"{message} at offset {self.pos} in {self.text!r}")

    def skip(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def peek(self, token: str) -> bool:
        self.skip()
        return self.text.startswith(token, self.pos)

    def accept(self, token: str) -> bool:
        if self.peek(token):
            self.pos += len(token)
            return True
        return False

    def expect(self, token: str) -> None:
        if not self.accept(token):
            raise self.error(f"expected {token!r}")

    def ident(self) -> str:
        self.skip()
        start = self.pos
        while self.pos < len(self.text) and (
            self.text[self.pos].isalnum() or self.text[self.pos] in "_:"
        ):
            self.pos += 1
        if start == self.pos:
            raise self.error("expected an identifier")
        return self.text[start:self.pos]

    def at_end(self) -> bool:
        self.skip()
        return self.pos >= len(self.text)


def _parse_alias(lex: _Lexer) -> AliasInfo:
    before: list[str] = []
    after: list[str] = []
    target = before
    while True:
        lex.skip()
        if lex.accept("*"):
            target.append("*")
        else:
            target.append(lex.ident())
        if lex.accept("|"):
            continue
        if lex.accept("->"):
            target = after
            continue
        break
    is_write = lex.accept("!")
    lex.expect(")")
    before_set = frozenset(before)
    return AliasInfo(before_set, frozenset(after) or before_set, is_write)


def _parse_type(lex: _Lexer) -> tuple[SchemaType, AliasInfo | None, int | None]:
    base_name = lex.ident()
    base = _BASE_TYPES.get(base_name)
    if base is None:
        raise lex.error(f"unknown schema type {base_name!r}")
    t: SchemaType = base()
    alias: AliasInfo | None = None
    size: int | None = None
    if lex.peek("("):
        lex.expect("(")
        alias = _parse_alias(lex)
    while True:
        if lex.accept("?"):
            t = OptionalType(t)
        elif lex.peek("["):
            lex.expect("[")
            lex.skip()
            start = lex.pos
            while lex.pos < len(lex.text) and lex.text[lex.pos].isdigit():
                lex.pos += 1
            digits = lex.text[start:lex.pos]
            lex.expect("]")
            size = int(digits) if digits else None
            t = ListType(t, size)
        else:
            break
    return t, alias, size


_NAMED_DEFAULTS = {
    "None": None,
    "True": True,
    "False": False,
    "true": True,
    "false": False,
}


def _parse_default(lex: _Lexer) -> Any:
    lex.skip()
    text = lex.text
    start = lex.pos
    if text[start] in "\"'":
        quote = text[start]
        pos = start + 1
        while pos < len(text):
            if text[pos] == "\\":
                pos += 2
                continue
            if text[pos] == quote:
                break
            pos += 1
        lex.pos = pos + 1
        return ast.literal_eval(text[start:lex.pos])
    if text[start] == "[":
        depth = 0
        pos = start
        while pos < len(text):
            if text[pos] == "[":
                depth += 1
            elif text[pos] == "]":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        lex.pos = pos + 1
        inner = text[start + 1:pos].strip()
        if not inner:
            return []
        return [_literal(item.strip()) for item in inner.split(",")]
    pos = start
    while pos < len(text) and text[pos] not in ",)":
        pos += 1
    lex.pos = pos
    return _literal(text[start:pos].strip())


def _literal(token: str) -> Any:
    if token in _NAMED_DEFAULTS:
        return _NAMED_DEFAULTS[token]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    # Enum spellings (contiguous_format, Mean, Float32, long, ...) stay
    # symbolic: their meaning depends on the argument type.
    return token


def _parse_arguments(lex: _Lexer) -> tuple[Argument, ...]:
    args: list[Argument] = []
    kwarg_only = False
    lex.expect("(")
    if lex.accept(")"):
        return ()
    while True:
        if lex.accept("*"):
            kwarg_only = True
        else:
            t, alias, size = _parse_type(lex)
            name = lex.ident()
            has_default = lex.accept("=")
            default = _parse_default(lex) if has_default else None
            args.append(
                Argument(
                    name=name,
                    type=t,
                    default_value=default,
                    has_default=has_default,
                    kwarg_only=kwarg_only,
                    alias_info=alias,
                    is_out=kwarg_only and alias is not None and alias.is_write,
                    N=size,
                )
            )
        if lex.accept(")"):
            return tuple(args)
        lex.expect(",")


def _parse_returns(lex: _Lexer) -> tuple[Argument, ...]:
    def one() -> Argument:
        t, alias, size = _parse_type(lex)
        lex.skip()
        name = ""
        if not lex.at_end() and not lex.peek(",") and not lex.peek(")"):
            name = lex.ident()
        return Argument(name=name, type=t, alias_info=alias, N=size)

    if lex.accept("("):
        if lex.accept(")"):
            return ()
        rets = [one()]
        while lex.accept(","):
            rets.append(one())
        lex.expect(")")
        return tuple(rets)
    return (one(),)


def parse_schema(text: str, *, namespace: str) -> FunctionSchema:
    """Parse ``[ns::]name[.overload](args) -> returns`` into a FunctionSchema.

    ``namespace`` applies when the text carries no ``ns::`` qualifier.
    """

    lex = _Lexer(text)
    qualified = lex.ident()
    if "::" in qualified:
        namespace, qualified = qualified.split("::", 1)
    overload = ""
    if lex.accept("."):
        overload = lex.ident()
    arguments = _parse_arguments(lex)
    lex.expect("->")
    returns = _parse_returns(lex)
    if not lex.at_end():
        raise lex.error("trailing text after the return declaration")
    return FunctionSchema(
        name=qualified,
        overload_name=overload,
        arguments=arguments,
        returns=returns,
        namespace=namespace,
        text=text,
    )
