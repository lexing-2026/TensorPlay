```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# Automatic differentiation package - tensorplay.autograd

# Automatic differentiation package - torch.autograd

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay._C._autograd.backward
    tensorplay.autograd.grad
```

## Functional higher level API

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.functional.jacobian
    tensorplay.autograd.functional.hessian
    tensorplay.autograd.functional.vjp
    tensorplay.autograd.functional.jvp
    tensorplay.autograd.functional.vhp
    tensorplay.autograd.functional.hvp
```

## {hidden}`Function`

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.function.Function
    tensorplay.autograd.backward
    tensorplay.autograd.jvp
```

## Custom Function utilities

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.function.once_differentiable
```

## Numerical gradient checking

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.gradcheck.gradcheck
    tensorplay.autograd.gradcheck.gradgradcheck
    tensorplay.autograd.gradcheck.GradcheckError
    tensorplay.autograd.gradcheck.get_analytical_jacobian
    tensorplay.autograd.gradcheck.get_numerical_jacobian
    tensorplay.autograd.gradcheck.get_numerical_jacobian_wrt_specific_input
```

## Debugging and anomaly detection

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.anomaly_mode.detect_anomaly
    tensorplay.autograd.anomaly_mode.set_detect_anomaly
```

