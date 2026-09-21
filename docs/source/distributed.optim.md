# tensorplay.distributed.optim

`tensorplay.distributed.optim` provides optimizers that are distributed-aware:
their internal state is sharded, averaged, or moved, and the update is
communicated across the process group rather than replicated on every rank.
Each class targets a different distribution scenario.

## Zero-redundancy optimizer

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.optim.ZeroRedundancyOptimizer
```

{class}`~tensorplay.distributed.optim.ZeroRedundancyOptimizer` wraps an
arbitrary `Optimizer` class and shards its state across the ranks of a
process group, following the ZeRO idea: each rank keeps only about
`1 / world_size` of the optimizer state (moments, etc.) because it is only
responsible for updating a roughly equal share of the parameters. After the
local update, the rank broadcasts its updated parameter shard to all peers so
every model replica stays in sync. Parameter-to-rank assignment uses a
sorted-greedy packing, and each parameter belongs to exactly one rank — never
split between ranks.

```python
import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.distributed.optim import ZeroRedundancyOptimizer

dist.init_process_group("nccl")
model = tp.nn.Linear(1024, 1024).cuda()

optimizer = ZeroRedundancyOptimizer(  # shards the Adam state across ranks
    model.parameters(),
    optimizer_class=tp.optim.Adam,
    lr=1e-3,
)
```

The per-rank partition can be tuned with `process_group` (which group to shard
across), `parameters_as_bucket_view` (pack parameters into buckets to speed up
communication), and `overlap_with_ddp` (overlap the parameter broadcast with
the gradient all-reduce, which requires registering a DDP communication hook).

## Post-local-SGD optimizer

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.optim.PostLocalSGDOptimizer
```

{class}`~tensorplay.distributed.optim.PostLocalSGDOptimizer` implements
post-local SGD: each rank runs its local optimizer on every step, and after a
warm-up stage the model weights are periodically averaged across the group at
the end of the update. The averaging is delegated to a `ModelAverager` (from
`tensorplay.distributed.algorithms.model_averaging.averagers`), so the period
and warm-up are configured on the averager rather than the optimizer.

```python
from tensorplay.distributed.algorithms.model_averaging.averagers import (
    PeriodicModelAverager,
)
from tensorplay.distributed.optim import PostLocalSGDOptimizer

local_optim = tp.optim.SGD(model.parameters(), lr=0.01)
optimizer = PostLocalSGDOptimizer(
    optim=local_optim,
    averager=PeriodicModelAverager(period=4, warmup_steps=100),
)
```

## Remote optimizer

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.optim.DistributedOptimizer
```

{class}`~tensorplay.distributed.optim.DistributedOptimizer` runs an optimizer
over parameters that live on *remote* workers, referenced through RPC
`RRef`s rather than local tensors. It requires the RPC framework to be
initialized ({func}`tensorplay.distributed.rpc.init_rpc`), and is the
optimizer used with remote modules: the parameters stay on the worker that
owns the module, and the optimizer on the driver drives the updates through
RPC.

## Functional optimizers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.optim.as_functional_optim
```

{func}`~tensorplay.distributed.optim.as_functional_optim` converts an
optimizer class into its *functional* form — an update function that takes the
parameters and the hyperparameters and applies one step, without the optimizer
object itself. Functional optimizers are used by the communication-hook
machinery (e.g. to run an optimizer step inside a gradient all-reduce hook,
which is how `overlap_with_ddp` and the ZeRO/DPP hooks work) and by
`apply_optimizer_in_backward`, which attaches the optimizer step to the
backward pass so parameter updates happen inside `backward()` instead of a
separate `step()` call.

The package also exposes the individual functional updates — `functional_adam`,
`functional_adamw`, `functional_sgd`, `functional_rmsprop`,
`functional_adagrad`, `functional_adadelta`, `functional_adamax`,
`functional_rprop` — under `tensorplay.distributed.optim.functional_*`, which
implement one optimizer step from state tensors passed as arguments.

## Where to go next

- [the distributed package](distributed.md) — process groups and DDP, the
  communication layer these optimizers hook into.
- [RPC](distributed.rpc.md) — the remote-call framework the
  `DistributedOptimizer` and remote modules are built on.
- [FSDP](distributed.fsdp.md) — sharding parameters themselves, one level
  deeper than sharding only the optimizer state.