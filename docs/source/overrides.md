# tensorplay.overrides

```{eval-rst}
.. currentmodule:: tensorplay.overrides
```

This module exposes various helper functions for the ``__tensorplay_function__``
protocol. See {ref}`extending-tensorplay` for more details on the
``__tensorplay_function__`` protocol.

The protocol is how non-tensor objects participate in tensor operations: when any
argument of a tensorplay operation implements `__tensorplay_function__`, the operation
delegates to that method instead of running its default path. The functions here are the
plumbing both sides use — the predicates that detect such arguments, the dispatcher that
hands control to them, and the utilities for testing and wrapping.

- {func}`tensorplay.overrides.is_tensor_like` and
  {func}`tensorplay.overrides.is_tensor_method_or_property` classify an object from the
  *tensor side*: does it look like a tensor, and is a given name part of the tensor API?
- {func}`tensorplay.overrides.has_tensorplay_function` (and the `_unary` / `_variadic`
  variants) asks whether any argument in an argument *tuple* will intercept the call.
- {func}`tensorplay.overrides.handle_tensorplay_function` performs the delegation: given
  the public API callable, the relevant args, and the full call arguments, it invokes
  the highest-priority `__tensorplay_function__` implementation.
- {func}`tensorplay.overrides.get_ignored_functions` and
  {func}`tensorplay.overrides.get_overridable_functions` enumerate the dispatchable
  surface — the functions that do and do not route through the protocol.
- {func}`tensorplay.overrides.get_testing_overrides` builds a table of fake overrides
  for every overridable function, which is how the protocol itself is tested.
- {func}`tensorplay.overrides.wrap_tensorplay_function` wraps a public API so custom
  classes are routed through your wrapper first.
- {func}`tensorplay.overrides.resolve_name` renders a function reference as its
  fully qualified name, and {func}`tensorplay.overrides.redispatch_function` re-invokes
  the original implementation for arguments that did not actually need interception.
- {class}`tensorplay.overrides.TensorPlayFunctionMode` is the mode-based form of the
  protocol: a context object whose `__tensorplay_function__` sees every dispatchable
  call made inside its scope, without any argument implementing the protocol.

The gate and the delegation, as an operation's public entry point uses them:

```python
import tensorplay as tp
from tensorplay.overrides import has_tensorplay_function, handle_tensorplay_function

m = tp.ones(2)

assert not has_tensorplay_function((m,))   # plain tensors do not intercept

# inside a public API, the two calls pair up exactly like this:
if has_tensorplay_function((m, m)):        # False here — the default path runs
    result = handle_tensorplay_function(tp.add, (m, m), m, m)
else:
    result = tp.add(m, m)
```

`handle_tensorplay_function` is only meaningful behind the gate: the second argument is
the tuple inspected for interceptors, everything after it is the actual call, and with
no interceptor present it raises `TypeError` — "no implementation found" — rather than
falling back. Operations expose this pairing so one custom argument reroutes the whole
call to your implementation.

## Functions

```{eval-rst}
.. autofunction:: tensorplay.overrides.get_ignored_functions
```

```{eval-rst}
.. autofunction:: tensorplay.overrides.get_overridable_functions
```

```{eval-rst}
.. autofunction:: tensorplay.overrides.resolve_name
```

```{eval-rst}
.. autofunction:: tensorplay.overrides.get_testing_overrides
```

```{eval-rst}
.. autofunction:: tensorplay.overrides.handle_tensorplay_function
```

```{eval-rst}
.. autofunction:: tensorplay.overrides.has_tensorplay_function
```

```{eval-rst}
.. autofunction:: tensorplay.overrides.is_tensor_like
```

```{eval-rst}
.. autofunction:: tensorplay.overrides.is_tensor_method_or_property
```

```{eval-rst}
.. autofunction:: tensorplay.overrides.wrap_tensorplay_function
```

```{eval-rst}
.. autofunction:: tensorplay.overrides.redispatch_function
```

## Modes

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.overrides.TensorPlayFunctionMode
```
