from .binary import _apply_native_binary, _is_native_binary
from .core import is_masked_tensor, MaskedTensor
from .passthrough import _apply_pass_through_fn, _is_pass_through_fn
from .reductions import _apply_reduction, _is_reduction
from .unary import _apply_native_unary, _is_native_unary

# Importing the operation tables binds the explicit masking-aware methods
# (elementwise ops, reductions, structural ops) onto MaskedTensor.
from . import _ops_refs as _ops_refs
