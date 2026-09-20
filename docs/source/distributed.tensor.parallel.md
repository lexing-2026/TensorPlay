# tensorplay.distributed.tensor.parallel

Tensor parallelism shards the parameters of a module across the ranks of a
mesh dimension, so that each rank computes on part of the weights and the
results are combined with collectives. The entry point is
{func}`~tensorplay.distributed.tensor.parallel.parallelize_module`, which
applies a *style* — a plan describing how each submodule is partitioned — to
an existing module. The styles in this module target the classic linear /
embedding layout: column-splitting the output dimension, row-splitting the
input dimension, and sequence-tensor parallelism for transformer blocks.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.tensor.parallel.parallelize_module
    tensorplay.distributed.tensor.parallel.ParallelStyle
    tensorplay.distributed.tensor.parallel.ColwiseParallel
    tensorplay.distributed.tensor.parallel.RowwiseParallel
    tensorplay.distributed.tensor.parallel.SequenceParallel
    tensorplay.distributed.tensor.parallel.PrepareModuleInput
    tensorplay.distributed.tensor.parallel.PrepareModuleOutput
    tensorplay.distributed.tensor.parallel.PrepareModuleInputOutput
    tensorplay.distributed.tensor.parallel.loss_parallel
```

## Parallelizing a module

{func}`~tensorplay.distributed.tensor.parallel.parallelize_module` takes the
module, a [device mesh](distributed.device_mesh.md), and either a single
style or a mapping from submodule names to styles. Inside the transformed
module, parameters and inputs are [distributed tensors](distributed.tensor.md)
with the placements the style chose; the runtime inserts the collectives that
assemble partial results where needed.

## Sharding styles

- {class}`~tensorplay.distributed.tensor.parallel.ColwiseParallel` — split a
  linear layer's weight along its first dimension (the output dimension), so
  each rank produces a partial output column-block. The input stays
  replicated; the output is sharded along its last dimension.
- {class}`~tensorplay.distributed.tensor.parallel.RowwiseParallel` — split the
  weight along its second dimension (the input dimension). The input is
  turned into a sharded tensor, and the local partial output is gathered back
  into a replicated tensor.
- {class}`~tensorplay.distributed.tensor.parallel.SequenceParallel` — keep
  every parameter replicated while sharding the hidden states along the
  sequence dimension. It is meant to be wrapped around the colwise part of a
  transformer block, so the all-gather that restores an unsharded intermediate
  lands at the end of the block.
- {class}`~tensorplay.distributed.tensor.parallel.PrepareModuleInput`,
  {class}`~tensorplay.distributed.tensor.parallel.PrepareModuleOutput` and
  {class}`~tensorplay.distributed.tensor.parallel.PrepareModuleInputOutput` —
  convert inputs and outputs of a submodule to and from the placements a
  style expects, e.g. turning an unsharded input into a sharded one.

## Loss with sharded logits

{func}`~tensorplay.distributed.tensor.parallel.loss_parallel` is a context
manager that activates the distributed cross-entropy path for logits sharded
along the class dimension. When the model's logits stay sharded through the
loss (instead of being gathered into a replicated tensor first), the
cross-entropy reduces the sharded logits directly and the loss gradient is
scaled for the shard size, saving the all-gather that would otherwise
materialize the full logits tensor on every rank. Wrap the loss computation
in ``with loss_parallel():`` to enable it.

## Where to go next

- [distributed tensors](distributed.tensor.md) — the `DTensor` type these
  styles operate on.
- [device mesh](distributed.device_mesh.md) — creating the mesh that
  `parallelize_module` runs on.
- [the distributed package](distributed.md) — process groups and collectives
  underneath.