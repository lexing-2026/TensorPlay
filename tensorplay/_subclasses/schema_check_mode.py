"""Dispatch mode that checks operator calls against their schemas.

Every dispatched operator is run on the real arguments while a copy of the
inputs is kept.  Afterwards the mode verifies that

* an argument whose value changed is declared as written (``Tensor(a!)``);
* an output sharing storage with an input is declared to alias it;
* an operator that is not mutable does not hand back an input object as its
  output;
* two outputs sharing storage are declared to alias each other.

A violation raises ``RuntimeError``; permitted mutations and aliasing are
recorded in :attr:`SchemaCheckMode.mutated` and
:attr:`SchemaCheckMode.aliasing`.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, NamedTuple

import tensorplay as tp
from tensorplay.utils._dispatch import TensorPlayDispatchMode
from tensorplay.utils._pytree import tree_leaves, tree_map

from ._schema_info import SchemaArgType, SchemaArgument, SchemaInfo

__all__ = ["Aliasing", "Mutation", "SchemaCheckMode"]


class Mutation(NamedTuple):
    op_name: str
    arg_name: str


class Aliasing(NamedTuple):
    op_name: str
    arg_name: str
    output_number: str


# Operators whose alias annotation is intentionally incomplete.
_UNSAFE_OPS = ("_unsafe_view", "unsafe_split")


def is_iterable_of_tensors(iterable: Any) -> bool:
    if isinstance(iterable, tp.Tensor):
        return False
    try:
        if len(iterable) == 0:
            return False
        return all(isinstance(t, tp.Tensor) for t in iter(iterable))
    except TypeError:
        return False


def clone_inputs(args: Any) -> list[Any]:
    inputs: list[Any] = []
    for arg in args:
        if isinstance(arg, tp.Tensor):
            inputs.append(arg.detach().clone())
        elif is_iterable_of_tensors(arg):
            inputs.append([t.detach().clone() for t in arg])
        else:
            inputs.append(arg)
    return inputs


def _normalize(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    from tensorplay._higher_order_ops.auto_functionalize import normalize_arguments

    return normalize_arguments(func, tuple(args), dict(kwargs))


def _unwrap(e: Any) -> Any:
    if isinstance(e, tp.Tensor) and type(e) is not tp.Tensor:
        return getattr(e, "elem", e)
    return e


def _storage_id(e: Any) -> Any:
    return e.untyped_storage()._cdata


def _parse_metadata(e: Any) -> tuple[tuple[int, ...], Any] | None:
    if isinstance(e, tp.Tensor):
        if type(e) is not tp.Tensor:
            inner = getattr(e, "elem", None)
            if inner is None:
                return None
            return (tuple(inner.stride()), _storage_id(inner))
        if getattr(e, "layout", None) != tp.sparse_csr:
            return (tuple(e.stride()), _storage_id(e))
    return None


def _bitwise_equal(lhs: Any, rhs: Any) -> bool:
    if getattr(lhs, "is_quantized", False):
        return tp.equal(lhs, rhs)
    return tp.allclose(lhs, rhs, equal_nan=True)


def _has_mutated(before: Any, after: Any, md: Any) -> bool:
    are_tensors = type(before) is tp.Tensor and type(after) is tp.Tensor
    if (are_tensors and getattr(before, "layout", None) != tp.sparse_csr
            and getattr(after, "layout", None) != tp.sparse_csr):
        return md is not None and not (
            tuple(before.shape) == tuple(after.shape)
            and _bitwise_equal(before, after)
            and md[0] == tuple(after.stride())
            and md[1] == _storage_id(after)
        )
    return False


def _has_aliased(lhs: Any, rhs: Any) -> bool:
    """Whether the two values share mutable state (storage or list)."""

    def leaves(value: Any) -> list[Any]:
        if isinstance(value, tp.Tensor):
            return [value]
        if isinstance(value, (list, tuple)):
            return [leaf for item in value for leaf in leaves(item)]
        return []

    right = {_storage_id(t) for t in leaves(rhs)}
    return any(_storage_id(t) in right for t in leaves(lhs))


class SchemaCheckMode(TensorPlayDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []
        self.mutated: list[Mutation] = []
        self.aliasing: list[Aliasing] = []

    def reset_cache(self) -> None:
        self.ops.clear()
        self.mutated.clear()
        self.aliasing.clear()

    def display_ops(self) -> None:
        print(*self.ops, sep=",")

    def __tensorplay_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        schema = func._schema
        self.ops.append(schema.name)

        pre_arguments = _normalize(func, args, kwargs)
        c_p_args = dict(zip(pre_arguments.keys(), clone_inputs(pre_arguments.values())))
        cloned_arguments = {name: tree_map(_unwrap, c_p_args.get(name)) for name in c_p_args}
        cloned_metadata = {
            name: [_parse_metadata(a) for a in tree_leaves(pre_arguments.get(name))]
            for name in pre_arguments
        }

        out = func(*args, **kwargs)
        arguments = {name: tree_map(_unwrap, pre_arguments.get(name)) for name in pre_arguments}
        tuple_out = out if isinstance(out, tuple) else (out,)
        tuple_out = tree_map(_unwrap, tuple_out)

        schema_info = SchemaInfo(schema)
        schema_info.add_argument_values(pre_arguments)

        for i, arg in enumerate(schema.arguments):
            name = arg.name
            if arguments.get(name) is None:
                continue
            before = cloned_arguments.get(name)
            md = cloned_metadata.get(name)
            after = arguments.get(name)
            for j in range(len(tuple_out)):
                if _has_aliased(tuple_out[j], after) and schema.name not in _UNSAFE_OPS:
                    if not schema_info.may_contain_alias(
                        SchemaArgument(SchemaArgType.output, j),
                        SchemaArgument(SchemaArgType.input, i),
                    ):
                        raise RuntimeError(
                            f"Argument {name} is not defined to alias output but was aliasing"
                        )
                    self.aliasing.append(Aliasing(schema.name, name, f"output_{j}"))
                if after is tuple_out[j] and isinstance(after, tp.Tensor):
                    # Only mutable operators (add_, add.out) may return an input.
                    if not schema_info.is_mutable(SchemaArgument(SchemaArgType.input, i)) \
                            and schema.name not in ("lift", "lift_fresh"):
                        raise RuntimeError(
                            "Dispatcher operators below autograd are not allowed to directly "
                            f"return inputs.\nHowever, we found that `outputs[{j}] is {name}"
                        )
            if md is not None and any(
                _has_mutated(a, b, c)
                for a, b, c in zip(tree_leaves(before), tree_leaves(after), md)
            ):
                if not schema_info.is_mutable(SchemaArgument(SchemaArgType.input, i)):
                    raise RuntimeError(f"Argument {name} is not defined as mutable but was mutated")
                self.mutated.append(Mutation(schema.name, name))

        for i, j in combinations(range(len(schema.returns)), 2):
            if _has_aliased(tuple_out[i], tuple_out[j]):
                if not schema_info.may_contain_alias(
                    SchemaArgument(SchemaArgType.output, i),
                    SchemaArgument(SchemaArgType.output, j),
                ):
                    raise RuntimeError(f"Outputs {i} and {j} alias unexpectedly")

        return out
