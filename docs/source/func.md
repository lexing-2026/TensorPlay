# tensorplay.func

```{eval-rst}
.. currentmodule:: tensorplay.func
```

`tensorplay.func` is a library of composable function transforms.

- A "function transform" is a higher-order function that accepts a numerical
  function and returns a new function that computes a different quantity.
- It has auto-differentiation transforms ({func}`grad` returns a function
  that computes the gradient of `f`), a vectorization/batching transform
  ({func}`vmap` returns a function that computes `f` over batches of
  inputs), and others.
- The transforms compose with each other: `vmap(grad(f))` computes
  per-sample gradients, `jacrev(jacrev(f))` a Hessian, and `vmap(vmap(f))`
  maps over two independent batch dimensions.

## Why composable function transforms?

A number of use cases are awkward to express with the ordinary
tensor-and-module API alone:

- computing per-sample gradients (or other per-sample quantities)
- running ensembles of models on a single machine
- efficiently computing Jacobians and Hessians

Composing {func}`vmap`, {func}`grad`, {func}`vjp`, and {func}`jvp` covers
all of them without designing a separate subsystem for each use case.

Transforms operate on plain callables, so a module must first be turned into
a function of its state: {func}`functional_call` runs a module with
supplied parameters and buffers, and {func}`stack_module_state` stacks the
state of an ensemble so {func}`vmap` can map over its members.

```{note}
Operator coverage under the transforms is not complete: an operator without
a batching rule raises `NotImplementedError` naming the operator, and a few
known rough edges are collected on the
[UX limitations](func.ux_limitations.md) page.
```

```{eval-rst}
.. toctree::
  :maxdepth: 1

  func.whirlwind_tour
  func.api
  func.ux_limitations
  func.batch_norm
```
