# Patching batch norm

```{eval-rst}
.. currentmodule:: tensorplay.func
```

## What's happening?

Batch normalization updates its `running_mean` and `running_var` buffers in
place on every training forward. Under a transform that evaluates a module
more than once — differentiating through it, or mapping an ensemble over it
— those updates would be applied repeatedly and in an order the caller
never asked for. The running statistics are also not inputs of the
function, so no transform can account for them.

## What works today

The operator-coverage picture for normalization, verified on the current
build:

- {func}`vmap` over a batch norm module raises `NotImplementedError`
  (`Kernel not found for op: batch_norm on backend: VmapCPU`) — in
  training mode, in eval mode, and after the patching described below.
  `group_norm` and `layer_norm` are in the same state, so switching the
  norm layer does not help under {func}`vmap` yet.
- {func}`grad` through a batch norm module *does* work: the forward is
  evaluated once, the batch statistics differentiate normally. The
  running-stat buffers are mutated as a side effect — once per evaluation
  of the transformed function.

:::{warning}
The Jacobian transforms do not fail loudly on batch norm.
{func}`jacrev` through a batch norm module returns garbage values (entries
of order 1e33; NaN once running stats are disabled). Do not transform
through normalization layers — check a model for norm modules before
differentiating it more than once.
:::

## The helpers

{func}`replace_all_batch_norm_modules_` drops the running statistics of
every batch-normalization module in a tree, in place, and returns the
root. After the call, `track_running_stats` is `False` and the
`running_mean` / `running_var` / `num_batches_tracked` buffers are `None`:
the module normalizes with the batch's own statistics and no longer
mutates state on the forward path.

```python
import tensorplay as tp
import tensorplay.nn as nn
from tensorplay.func import grad, replace_all_batch_norm_modules_

net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
replace_all_batch_norm_modules_(net)
assert net[1].track_running_stats is False
assert net[1].running_mean is None

x = tp.randn(2, 3, 8, 8)
g = grad(lambda t: net(t).sum())(x)          # mutation-free evaluation
assert g.shape == x.shape
```

{func}`batch_norm_without_running_stats` is the single-module form: it
drops the running statistics of one batch-normalization module, if it is
one.

## When to use them

- Under the single-pass transforms ({func}`grad`, {func}`grad_and_value`,
  {func}`vjp`): patching makes the evaluation free of hidden state — the
  same input always produces the same output, which is what you want when
  differentiating.
- Under {func}`vmap` and the Jacobian transforms: normalization layers are
  not transformable today regardless of the running-stats setting. Strip
  or replace them before mapping; the helpers keep the module ready for
  when the rules land.
