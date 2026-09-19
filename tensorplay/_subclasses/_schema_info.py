"""Alias and mutation facts of one operator schema, refined by call values.

A schema states which arguments an operator may write and which outputs may
alias which inputs.  Actual argument values sharpen that picture: two input
tensors sharing storage alias each other, and a tensor contained in a list
argument can reach anything that list can reach.  :class:`SchemaInfo`
combines both, and additionally knows the operators whose running statistics
(or noise buffer) are written only in training mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorplay as tp

__all__ = ["SchemaArgType", "SchemaArgument", "SchemaInfo"]


class SchemaArgType:
    input = "input"
    output = "output"


@dataclass(frozen=True)
class SchemaArgument:
    type: str
    index: int


# Operators whose listed arguments are written only when the boolean
# ``training`` / ``train`` / ``use_input_stats`` argument is true.
_TRAINING_OPS: dict[str, frozenset[str]] = {
    "batch_norm": frozenset({"running_mean", "running_var"}),
    "instance_norm": frozenset({"running_mean", "running_var"}),
    "_batch_norm_impl_index": frozenset({"running_mean", "running_var"}),
    "cudnn_batch_norm": frozenset({"running_mean", "running_var"}),
    "miopen_batch_norm": frozenset({"running_mean", "running_var"}),
    "native_batch_norm": frozenset({"running_mean", "running_var"}),
    "native_batch_norm.out": frozenset({"running_mean", "running_var"}),
    "rrelu_with_noise": frozenset({"noise"}),
    "rrelu_with_noise.out": frozenset({"noise"}),
    "rrelu_with_noise_": frozenset({"noise"}),
}


def _alias_type_set(schema_type: Any) -> frozenset[str] | None:
    """Mutable types a value of ``schema_type`` can alias (None: none)."""

    from tensorplay._function_schema import ListType, OptionalType

    if isinstance(schema_type, OptionalType):
        return _alias_type_set(schema_type.element)
    if isinstance(schema_type, ListType):
        return frozenset({f"{schema_type.element}[]"})
    if schema_type.is_tensor_like():
        return frozenset({"Tensor"})
    return None


def _contained_type_set(schema_type: Any) -> frozenset[str] | None:
    """Mutable types reachable through the elements of a container type."""

    from tensorplay._function_schema import ListType, OptionalType

    if isinstance(schema_type, OptionalType):
        return _contained_type_set(schema_type.element)
    if isinstance(schema_type, ListType):
        return _alias_type_set(schema_type.element) or frozenset()
    return None


def _can_alias(lhs: frozenset[str] | None, rhs: frozenset[str] | None) -> bool:
    return bool(lhs) and bool(rhs) and bool(lhs & rhs)


def _after_sets(alias_info: Any) -> frozenset[str]:
    return alias_info.after_set or alias_info.before_set


def _is_wildcard_after(alias_info: Any) -> bool:
    return alias_info is not None and "*" in _after_sets(alias_info)


def _storage_id(value: Any) -> Any:
    return value.untyped_storage()._cdata


def _is_alias_of(lhs: Any, rhs: Any) -> bool:
    """Values that share mutable state: tensors sharing storage, or the
    same list object."""

    if isinstance(lhs, tp.Tensor) and isinstance(rhs, tp.Tensor):
        return _storage_id(lhs) == _storage_id(rhs)
    if isinstance(lhs, list) and isinstance(rhs, list):
        return lhs is rhs
    return False


def _sub_values(value: Any) -> list[Any]:
    """The value itself plus everything reachable through its elements."""

    found = [value]
    if isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_sub_values(item))
    return found


def _contains(container: Any, value: Any) -> bool:
    return any(
        item is not container and _is_alias_of(item, value) for item in _sub_values(container)
    )


class SchemaInfo:
    """Alias queries on ``schema`` refined by the argument values of one call."""

    def __init__(self, schema: Any) -> None:
        self.schema = schema
        self.values: dict[str, Any] = {}
        self._maps_current = False
        self._wildcards: set[SchemaArgument] = set()
        self._containers: set[SchemaArgument] = set()
        self._init_sets()

    # -- construction ---------------------------------------------------------

    def _init_sets(self) -> None:
        duplicates: set[str] = set()
        for kind, arguments in ((SchemaArgType.input, self.schema.arguments),
                                (SchemaArgType.output, self.schema.returns)):
            seen: set[str] = set()
            for index, argument in enumerate(arguments):
                alias = argument.alias_info
                if alias is not None:
                    if _is_wildcard_after(alias):
                        self._wildcards.add(SchemaArgument(kind, index))
                    else:
                        for name in _after_sets(alias):
                            if name in seen:
                                duplicates.add(name)
                            seen.add(name)
                if _contained_type_set(argument.type):
                    self._containers.add(SchemaArgument(kind, index))
        # A set shared by two arguments of one list makes the answers for
        # that set conservative: treat its members as wildcards.
        for kind, arguments in ((SchemaArgType.input, self.schema.arguments),
                                (SchemaArgType.output, self.schema.returns)):
            for index, argument in enumerate(arguments):
                alias = argument.alias_info
                if alias is not None and _after_sets(alias) & duplicates:
                    self._wildcards.add(SchemaArgument(kind, index))
        self._base_wildcards = set(self._wildcards)

    def add_argument_values(self, values: dict[str, Any]) -> None:
        names = {a.name for a in self.schema.arguments}
        for name, value in values.items():
            if name not in names:
                raise RuntimeError(f"Schema has no argument named {name}")
            self.values[name] = value
        self._maps_current = False

    def _generate_alias_maps(self) -> None:
        arguments = self.schema.arguments
        self._wildcards = set(self._base_wildcards)
        self.input_alias_map = [{i} for i in range(len(arguments))]
        for i in range(len(arguments)):
            for j in range(i + 1, len(arguments)):
                a, b = arguments[i].name, arguments[j].name
                if a in self.values and b in self.values and _is_alias_of(self.values[a], self.values[b]):
                    self.input_alias_map[i].add(j)
                    self.input_alias_map[j].add(i)
                    if SchemaArgument(SchemaArgType.input, i) in self._wildcards:
                        self._wildcards.add(SchemaArgument(SchemaArgType.input, j))
                    elif SchemaArgument(SchemaArgType.input, j) in self._wildcards:
                        self._wildcards.add(SchemaArgument(SchemaArgType.input, i))
        # A value held inside a container argument can alias whatever that
        # container's elements may alias.
        for i in range(len(arguments)):
            for j in range(len(arguments)):
                if j in self.input_alias_map[i]:
                    continue
                a, b = arguments[i].name, arguments[j].name
                if a in self.values and b in self.values and _contains(self.values[a], self.values[b]):
                    self._wildcards.add(SchemaArgument(SchemaArgType.input, j))
        self.output_alias_map = [set() for _ in self.schema.returns]
        for i in range(len(arguments)):
            for j in range(len(self.schema.returns)):
                if self._schema_may_alias(SchemaArgument(SchemaArgType.input, i),
                                          SchemaArgument(SchemaArgType.output, j)):
                    if SchemaArgument(SchemaArgType.input, i) in self._wildcards:
                        self._wildcards.add(SchemaArgument(SchemaArgType.output, j))
                    self.output_alias_map[j] |= self.input_alias_map[i]
        self._maps_current = True

    def _ensure_maps(self) -> None:
        if not self._maps_current:
            self._generate_alias_maps()

    # -- schema-only relations ------------------------------------------------

    def _argument(self, argument: SchemaArgument) -> Any:
        items = self.schema.arguments if argument.type == SchemaArgType.input else self.schema.returns
        return items[argument.index]

    def _schema_may_alias(self, lhs: SchemaArgument, rhs: SchemaArgument) -> bool:
        a, b = self._argument(lhs), self._argument(rhs)
        if not _can_alias(_alias_type_set(a.type), _alias_type_set(b.type)):
            return False
        if a.alias_info is not None and b.alias_info is not None:
            return bool(_after_sets(a.alias_info) & _after_sets(b.alias_info))
        return False

    def _schema_may_contain_alias(self, lhs: SchemaArgument, rhs: SchemaArgument,
                                  bidirectional: bool = True) -> bool:
        if self._schema_may_alias(lhs, rhs):
            return True
        a, b = self._argument(lhs), self._argument(rhs)
        a_types, b_types = _alias_type_set(a.type), _alias_type_set(b.type)
        a_contained, b_contained = _contained_type_set(a.type), _contained_type_set(b.type)
        lhs_wildcard = _is_wildcard_after(a.alias_info) and _can_alias(a_types, b_contained)
        rhs_wildcard = _is_wildcard_after(b.alias_info) and _can_alias(b_types, a_contained)
        if bidirectional:
            return lhs_wildcard or rhs_wildcard or _can_alias(a_contained, b_contained)
        return rhs_wildcard or _can_alias(a_contained, b_contained)

    # -- queries --------------------------------------------------------------

    def _flag(self, name: str) -> bool:
        has = any(a.name == name for a in self.schema.arguments)
        if not has:
            return False
        return bool(self.values[name]) if name in self.values else True

    def is_mutable(self, argument: SchemaArgument | None = None) -> bool:
        if argument is None:
            return any(self.is_mutable(SchemaArgument(SchemaArgType.input, i))
                       for i in range(len(self.schema.arguments)))
        self._ensure_maps()
        key = self.schema.name + (f".{self.schema.overload_name}" if self.schema.overload_name else "")
        special = _TRAINING_OPS.get(key, frozenset())
        aliases = (self.input_alias_map if argument.type == SchemaArgType.input
                   else self.output_alias_map)[argument.index]
        for index in aliases:
            arg = self.schema.arguments[index]
            if arg.name in special:
                if self._flag("training") or self._flag("train") or self._flag("use_input_stats"):
                    return True
            elif arg.alias_info is not None and arg.alias_info.is_write:
                return True
        return False

    def may_alias(self, lhs: SchemaArgument, rhs: SchemaArgument) -> bool:
        if self._schema_may_alias(lhs, rhs):
            return True
        a, b = self._argument(lhs), self._argument(rhs)
        if not _can_alias(_alias_type_set(a.type), _alias_type_set(b.type)):
            return False
        self._ensure_maps()
        if lhs in self._wildcards and rhs in self._wildcards:
            return True
        if lhs.type == SchemaArgType.input and rhs.type == SchemaArgType.input:
            return rhs.index in self.input_alias_map[lhs.index]
        if lhs.type == SchemaArgType.output and rhs.type == SchemaArgType.output:
            return bool(self.output_alias_map[lhs.index] & self.output_alias_map[rhs.index])
        if lhs.type == SchemaArgType.output:
            return rhs.index in self.output_alias_map[lhs.index]
        return lhs.index in self.output_alias_map[rhs.index]

    def _may_contain_alias_impl(self, lhs: SchemaArgument, rhs: SchemaArgument) -> bool:
        a, b = self._argument(lhs), self._argument(rhs)
        return (_can_alias(_contained_type_set(a.type), _alias_type_set(b.type))
                and lhs in self._containers and rhs in self._wildcards)

    def may_contain_alias(self, lhs: SchemaArgument, rhs: SchemaArgument,
                          bidirectional: bool = True) -> bool:
        if self._schema_may_contain_alias(lhs, rhs) or self.may_alias(lhs, rhs):
            return True
        self._ensure_maps()
        if bidirectional:
            return self._may_contain_alias_impl(lhs, rhs) or self._may_contain_alias_impl(rhs, lhs)
        return self._may_contain_alias_impl(lhs, rhs)
