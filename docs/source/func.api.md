# tensorplay.func API reference

```{eval-rst}
.. automodule:: tensorplay.func
```

## Function transforms

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    vmap
    chunk_vmap
    grad
    grad_and_value
    vjp
    jvp
    linearize
    jacrev
    jacfwd
    hessian
    functionalize
    rearrange
```

{func}`vmap` maps a function over a new batch dimension and {func}`grad`
differentiates it; the rest are built from or alongside those two.
{func}`grad_and_value` returns the value together with the gradient from a
single evaluation. {func}`vjp` and {func}`jvp` expose the vector-Jacobian
and Jacobian-vector products directly; {func}`jacrev` and {func}`jacfwd`
assemble full Jacobians with reverse- and forward-mode AD; {func}`hessian`
composes the two into a Hessian; and {func}`linearize` evaluates a function
once and returns its value together with a callable forward-mode
linearization at that point. {func}`functionalize` removes mutations from a
function, and {func}`rearrange` reshapes tensors by naming their axes.

{func}`chunk_vmap` is {func}`vmap` with the batch split into a fixed number
of chunks — prefer `vmap(..., chunk_size=...)`, which bounds peak memory in
samples rather than in pieces and so does not change meaning with the batch
size.

## Utilities for working with `tensorplay.nn` modules

A transform needs a function; a module holds state. In general you can
transform over a function that calls a module directly:

```python
import tensorplay as tp
from tensorplay.func import jacrev

model = tp.nn.Linear(3, 3)

def f(x):
    return model(x)

x = tp.randn(3)
jacobian = jacrev(f)(x)
assert jacobian.shape == (3, 3)
```

To differentiate with respect to the module's *parameters*, build a function
whose inputs are the parameters. That is what {func}`functional_call` is
for: it accepts a module, replacement state, and the inputs to the module's
forward pass, and runs the module with the replacement state instead of its
own:

```python
import tensorplay as tp
from tensorplay.func import functional_call, jacrev

model = tp.nn.Linear(3, 3)

def f(params, x):
    return functional_call(model, params, (x,))

x = tp.randn(3)
jacobian = jacrev(f)(dict(model.named_parameters()), x)
assert jacobian["weight"].shape == (3, 3, 3)
assert jacobian["bias"].shape == (3, 3)
```

{func}`stack_module_state` stacks the state of several identical modules
into batched tensors, so {func}`vmap` can evaluate a whole ensemble in one
call instead of looping over the models:

```python
import tensorplay as tp
import tensorplay.nn as nn
from tensorplay.func import functional_call, stack_module_state, vmap

models = [nn.Linear(3, 3) for _ in range(4)]
stacked_params, stacked_buffers = stack_module_state(models)
x = tp.randn(5, 3)

def call_one(params, buffers, x):
    return functional_call(models[0], (params, buffers), (x,))

ensemble_out = vmap(call_one, in_dims=(0, 0, None))(
    stacked_params, stacked_buffers, x
)
assert ensemble_out.shape == (4, 5, 3)
```

For batch norm modules under transforms, see
[patching batch norm](func.batch_norm.md).

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    functional_call
    stack_module_state
    replace_all_batch_norm_modules_
    batch_norm_without_running_stats
```

## Debug utilities

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    debug_unwrap
```

Inside a transformed function, tensors carry an invisible transform level.
{func}`debug_unwrap` removes it so the underlying tensor can be inspected —
printing its shape, checking a value in a debugger — without disturbing the
transform. Continue computing with the original argument, not with the
unwrapped tensor:

```python
import tensorplay as tp
from tensorplay.func import debug_unwrap, vmap

def f(x):
    print(debug_unwrap(x).shape)   # the per-sample tensor, e.g. (3,)
    return x * 2

out = vmap(f)(tp.randn(2, 3))
assert out.shape == (2, 3)
```
