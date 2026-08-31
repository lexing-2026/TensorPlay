"""Composable function transforms.

A transform takes a function and returns a function, so they compose:
``grad(grad(f))`` is a second derivative, ``vmap(grad(f))`` is a batch of
gradients, and ``jacfwd(jacrev(f))`` is a Hessian.  Everything here works on
plain Python callables over tensors -- no module, no mutable state, no
:attr:`~tensorplay.Tensor.grad` field to clear between calls.

Modules do hold state, so :func:`functional_call` turns one into a function of
its parameters, and :func:`stack_module_state` batches an ensemble of them into
a single set of stacked tensors for :func:`vmap`.
"""

from tensorplay._transforms.apis import chunk_vmap, grad, grad_and_value, vmap
from tensorplay._transforms.batch_norm_replacement import (
    batch_norm_without_running_stats,
    replace_all_batch_norm_modules_,
)
from tensorplay._transforms.eager_transforms import (
    debug_unwrap,
    functionalize,
    hessian,
    jacfwd,
    jacrev,
    jvp,
    linearize,
    vjp,
)
from tensorplay._transforms.einops import rearrange
from tensorplay._transforms.functional_call import functional_call, stack_module_state
from . import _random as _random

__all__ = [
    "vmap",
    "chunk_vmap",
    "grad",
    "grad_and_value",
    "vjp",
    "jvp",
    "jacrev",
    "jacfwd",
    "hessian",
    "linearize",
    "functionalize",
    "debug_unwrap",
    "functional_call",
    "stack_module_state",
    "replace_all_batch_norm_modules_",
    "batch_norm_without_running_stats",
    "rearrange",
    "_random",
]
