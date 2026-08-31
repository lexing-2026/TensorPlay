from __future__ import annotations

from typing import Any

try:
    import z3  # type: ignore[import-not-found]
except ImportError:
    z3 = None

HAS_Z3 = z3 is not None
dyn = None
dyn_type = None
dim = None
tensor_type = None
D = None
z3_dyn = None

if z3 is not None:
    dyn = z3.DeclareSort("DynamicValue")
    dyn_type = z3.Const("dynamic_value", dyn)
    dim = z3.Datatype("Dimension")
    dim.declare("dimension", ("known", z3.IntSort()), ("dynamic", z3.IntSort()))
    dim = dim.create()
    tensor_type = z3.Datatype("TensorShape")
    tensor_type.declare("Dynamic", ("value", dyn))
    for rank in range(1, 5):
        tensor_type.declare(
            f"tensor{rank}",
            *[(str(index), dim) for index in range(rank)],
        )
    tensor_type = tensor_type.create()
    D = dim.dimension
    z3_dyn = tensor_type.Dynamic(dyn_type)

__all__ = ["D", "HAS_Z3", "dim", "dyn", "dyn_type", "tensor_type", "z3_dyn"]
