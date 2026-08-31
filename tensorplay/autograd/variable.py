"""Tensor variable compatibility names for the automatic differentiation API."""

import tensorplay

__all__ = ["VariableMeta", "Variable"]


class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, tensorplay.Tensor)


class Variable(metaclass=VariableMeta):
    _execution_engine = tensorplay._C._autograd
