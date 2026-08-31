from .binary_cmp import allclose, binary_cmp, equal
from .init import constant_, kaiming_uniform_, normal_, uniform_
from .misc_ops import tensor_has_compatible_shallow_copy_type
from .tensor_ops import sharded_clone, sharded_deepcopy, sharded_detach, sharded_inplace_copy

__all__ = ["allclose", "binary_cmp", "equal", "constant_", "kaiming_uniform_", "normal_", "uniform_", "tensor_has_compatible_shallow_copy_type", "sharded_clone", "sharded_deepcopy", "sharded_detach", "sharded_inplace_copy"]
