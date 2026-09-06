# tensorplay.stax

```{eval-rst}
.. currentmodule:: tensorplay.stax
```

Static-graph optimization and acceleration. `stax` traces eager execution into
a static graph, applies compiler passes (constant folding, dead code
elimination, operator decomposition), and lowers the result to the available
backends.

This module is TensorPlay-specific.

## Functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    stax
    is_available
```
