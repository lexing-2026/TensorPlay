(ddp)=

# Distributed Data Parallel

`tensorplay.nn.parallel.DistributedDataParallel` (DDP) transparently
performs distributed data parallel training: each process holds a replica
of the model, works on its own shard of the batch, and the replicas stay in
lockstep because every gradient is averaged across the group before any
optimizer reads it. This page describes how it works and reveals
implementation details.

## Example

This example creates a process group over the gloo backend, wraps a local
`nn.Linear` with DDP, and runs one forward pass, one
backward pass, and an optimizer step on the DDP model. After that, all
model replicas across processes are exactly the same.

```python
import os

import tensorplay as tp
import tensorplay.distributed as dist
import tensorplay.multiprocessing as mp
import tensorplay.nn as nn
from tensorplay.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10)
    # construct DDP model
    ddp_model = DDP(model)
    # define loss function and optimizer
    optimizer = tp.optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(tp.randn(20, 10))
    # backward pass
    outputs.square().mean().backward()
    # update parameters
    optimizer.step()
    dist.destroy_process_group()


def main():
    world_size = 2
    # Environment variables used by the default env-based rendezvous.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
```

Even when each rank seeds its generator differently before constructing the
model, all replicas end up with the parameters of the process with rank 0:
the constructor broadcasts them. Likewise, after the backward pass the
`grad` fields are identical on every rank — each is the mean of all ranks'
local gradients — so every optimizer step applies the same update and the
replicas cannot drift apart.

## Internal Design

This section reveals how it works under the hood by walking through every
step of one iteration.

- **Prerequisite**: DDP relies on the distributed package's process groups
  for communication. Applications must initialize the process group
  (or pass an explicit `process_group=`) before constructing DDP.

- **Construction**: The DDP constructor takes a reference to the local
  module and first verifies that all ranks agree on the parameter count
  and every parameter's element count, using MIN/MAX all-reduces over the
  size vector. It then broadcasts parameters — and buffers, when
  `broadcast_buffers=True` (the default) — from the process with rank 0
  in coalesced 250 MiB chunks, so every replica starts from exactly the
  same state. Modules listed in the wrapped module's
  `_ddp_params_and_buffers_to_ignore` are excluded.

  The remaining work is building the *reducer*. Parameters are grouped
  into buckets sized by `bucket_cap_mb` (25 MiB by default), and gradients
  are reduced one bucket at a time. Buckets are filled in roughly reverse
  order of `Model.parameters()`: gradients tend to become ready in the
  order the parameters were used, so the last-built bucket is usually the
  first to complete. Each bucket is one flat buffer with a narrowed view
  per parameter. For each parameter DDP registers a post-accumulate
  gradient hook, which the autograd engine invokes the moment that
  parameter's gradient is ready. On CUDA modules there is also a native
  reducer that runs the whole hot path — hook, bucket copy, all-reduce,
  copy-back — in C++ without touching the Python interpreter.

- **Forward pass**: The input is passed to the local module unchanged.
  Before it runs, when `broadcast_buffers` is on and gradients are
  enabled, module buffers are re-broadcast from rank 0, so any local
  divergence (a hand-edited running statistic, say) is repaired at the
  start of every iteration. When `find_unused_parameters=True`, DDP
  additionally walks the autograd graph reachable from the outputs —
  through `grad_fn` and `next_functions` down to the gradient
  accumulators — to learn which parameters will participate in this
  iteration's backward. Parameters that will not participate have their
  bucket slices zero-filled up front and are treated as ready, so their
  buckets can still reduce. With `static_graph=True` the set of
  participating parameters is recorded once and reused, skipping the
  per-iteration traversal.

- **Backward pass**: `backward()` is invoked on the loss, out of DDP's
  control; the hooks registered at construction time do the
  synchronization. When a gradient becomes ready, its hook copies it into
  the parameter's bucket view — or skips the copy entirely under
  `gradient_as_bucket_view`, where `param.grad` *is* the bucket view.
  Once every parameter of a bucket has reported, the bucket is
  all-reduced: averaged directly for float32, or pre-divided and summed
  for other dtypes. The reduced values are copied back into the
  parameters' `grad` fields with one batched copy per bucket. Buckets are
  reduced strictly in bucket-index order (see the note below), and on
  CUDA the all-reduces run on a dedicated communication stream, so
  earlier buckets communicate while the autograd engine is still
  computing later gradients; the compute stream joins the comm stream
  only after the last bucket.

- **Optimizer step**: From the optimizer's perspective it is optimizing
  a local model. Replicas stay in sync because they start from the same
  state and consume identical averaged gradients in every iteration.

:::{note}
DDP requires the reducers on all processes to issue their all-reduces in
exactly the same order. This is done by always reducing buckets in bucket
index order rather than in the order they happen to complete. Mismatched
collective sequences across processes lead to wrong results or hangs.
:::

## When a parameter never receives a gradient

If some parameter is not used in producing the loss, its bucket never
completes, the gradients that did arrive in that bucket are never
averaged, and the failure surfaces one iteration late: the next forward
pass raises

```text
RuntimeError: Expected to have finished reduction in the prior iteration
before starting a new one. ... You can enable find_unused_parameters=True
in the DistributedDataParallel constructor to work around this error.
```

Setting `find_unused_parameters=True` makes DDP traverse the output graph
each iteration, zero-fill and mark the absent gradients as ready, and
reduce normally. The traversal has a cost, so enable it only for models
that genuinely run backward on a subgraph — conditional execution, shared
trunks with unused heads — and prefer `static_graph=True` when the set of
used parameters is fixed after the first iteration.

## Gradient sync you can turn off

Inside the `no_sync()`
context manager, hooks still run but skip the bucket copies and
collectives, so gradients accumulate locally without any communication.
This is the standard way to run several micro-batches per optimizer step:
accumulate under `no_sync()`, then run the final micro-batch (or
explicitly re-enable sync) so its backward averages the fully
accumulated gradients.

## Custom gradient aggregation

`register_comm_hook()`
on the wrapper replaces the default all-reduce with a user callable
`hook(state, bucket) -> Future[Tensor]`: the hook receives each ready
bucket (a `GradBucket` with its index, flat buffer, per-parameter views
and parameters) and returns a future whose resolved value is copied back
into the gradients. The hook can only be registered once and must be
registered before the first backward pass.

## Uneven input counts across ranks

Ranks that run out of batches early can sit in the
the join context (see the Join section of the
distributed algorithms page) so the other ranks can keep
training: the join hook shadows one all-reduce per bucket per shadowed
iteration (contributing zero gradients), and when the last rank finishes,
the final model is broadcast to the ranks that had joined.
