# mypy: allow-untyped-defs
"""Memory storage classes.

A storage is a one-dimensional, untyped block of memory owned by a
device. Tensors are typed views over storages; use
:meth:`tensorplay.Tensor.untyped_storage` to reach the block behind a
tensor. Only untyped memory is tracked here.
"""

from tensorplay._C import UntypedStorage, is_storage

__all__ = ["UntypedStorage", "is_storage"]
