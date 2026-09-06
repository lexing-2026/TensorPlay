```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# Automatic differentiation package - tensorplay.autograd


```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.backward
    tensorplay.autograd.grad
```

## Forward-mode Automatic Differentiation

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.forward_ad.make_dual
    tensorplay.autograd.forward_ad.unpack_dual
    tensorplay.autograd.forward_ad.enter_dual_level
    tensorplay.autograd.forward_ad.exit_dual_level
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
    tensorplay.autograd.jvp
```

## Context method mixins

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.function.FunctionMeta
```

## Custom Function utilities

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.function.once_differentiable
    tensorplay.autograd.function.InplaceFunction
    tensorplay.autograd.function.NestedIOFunction
```

## Numerical gradient checking

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.gradcheck.gradcheck
    tensorplay.autograd.gradcheck.gradgradcheck
    tensorplay.autograd.gradcheck.GradcheckError
    tensorplay.autograd.gradcheck.get_numerical_jacobian_wrt_specific_input
```

## Profiler

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.profiler.profiler.profile
    tensorplay.profiler.profiler.record_function
    tensorplay.autograd.profiler_util.EventList
    tensorplay.autograd.profiler_util.FormattedTimesMixin
    tensorplay.autograd.profiler_util.FunctionEvent
    tensorplay.autograd.profiler_util.FunctionEventAvg
    tensorplay.autograd.profiler_util.Interval
    tensorplay.autograd.profiler_util.Kernel
    tensorplay.autograd.profiler_util.MemRecordsAcc
    tensorplay.autograd.profiler_util.StringTable
    tensorplay.profiler.profiler.emit_nvtx
    tensorplay.profiler.profiler.emit_itt
```

## Debugging and anomaly detection

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.anomaly_mode.detect_anomaly
    tensorplay.autograd.anomaly_mode.set_detect_anomaly
```

## Autograd graph

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.graph.saved_tensors_hooks
    tensorplay.autograd.graph.save_on_cpu
    tensorplay.autograd.variable.Variable
    tensorplay.autograd.variable.VariableMeta
```

