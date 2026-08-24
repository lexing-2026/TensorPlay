```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# tensorplay.optim

## Base class

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.optim.optimizer.Optimizer
```

## Module-level hooks

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.optim.optimizer.register_optimizer_step_post_hook
    tensorplay.optim.optimizer.register_optimizer_step_pre_hook
```

## Utilities

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.optim.swap_in_optimizer_params_and_state
```

## Algorithms

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.optim.adadelta.Adadelta
    tensorplay.optim.Adafactor
    tensorplay.optim.adagrad.Adagrad
    tensorplay.optim.adam.Adam
    tensorplay.optim.adamw.AdamW
    tensorplay.optim.sparse_adam.SparseAdam
    tensorplay.optim.adamax.Adamax
    tensorplay.optim.asgd.ASGD
    tensorplay.optim.lbfgs.LBFGS
    tensorplay.optim.Muon
    tensorplay.optim.nadam.NAdam
    tensorplay.optim.radam.RAdam
    tensorplay.optim.rmsprop.RMSprop
    tensorplay.optim.rprop.Rprop
    tensorplay.optim.sgd.SGD
```

## How to adjust learning rate

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.optim.lr_scheduler.LRScheduler
    tensorplay.optim.lr_scheduler.LambdaLR
    tensorplay.optim.lr_scheduler.MultiplicativeLR
    tensorplay.optim.lr_scheduler.StepLR
    tensorplay.optim.lr_scheduler.MultiStepLR
    tensorplay.optim.lr_scheduler.ConstantLR
    tensorplay.optim.lr_scheduler.LinearLR
    tensorplay.optim.lr_scheduler.ExponentialLR
    tensorplay.optim.lr_scheduler.PolynomialLR
    tensorplay.optim.lr_scheduler.CosineAnnealingLR
    tensorplay.optim.lr_scheduler.ChainedScheduler
    tensorplay.optim.lr_scheduler.SequentialLR
    tensorplay.optim.lr_scheduler.ReduceLROnPlateau
    tensorplay.optim.lr_scheduler.CyclicLR
    tensorplay.optim.lr_scheduler.OneCycleLR
    tensorplay.optim.lr_scheduler.CosineAnnealingWarmRestarts
```

### Putting it all together: EMA

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.optim.swa_utils.AveragedModel
    tensorplay.optim.swa_utils.SWALR
    tensorplay.optim.swa_utils.get_ema_avg_fn
    tensorplay.optim.swa_utils.get_swa_avg_fn
    tensorplay.optim.swa_utils.get_swa_multi_avg_fn
    tensorplay.optim.swa_utils.get_ema_multi_avg_fn
    tensorplay.optim.swa_utils.update_bn
```

