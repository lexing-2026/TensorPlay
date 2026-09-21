# tensorplay.distributed.fsdp

Fully sharded data parallelism (FSDP) shrinks a model's peak GPU memory by
sharding its parameters across the ranks of a process group. The parameters
of each submodule are flattened into a single buffer and split; only the shard
needed for the current operation is gathered into memory, then freed again
after the forward (depending on `reshard_after_forward`). The two public entry
points differ in how much they hide:

- {func}`~tensorplay.distributed.fsdp.FullyShardedDataParallel` wraps a whole
  module in one call, the classic drop-in style that replaces
  `DistributedDataParallel`.
- {func}`~tensorplay.distributed.fsdp.fully_shard` is the composable style: you
  shard each submodule individually (often applying it to a parent module),
  which lets you interleave sharding with tensor parallelism and customise the
  shard policy per module.

```python
from tensorplay.distributed.fsdp import (
    FullyShardedDataParallel, ShardingStrategy,
)

# requires an initialized process group and a CUDA device per rank
model = FullyShardedDataParallel(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

## Composable `fully_shard` API

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.fsdp.fully_shard
    tensorplay.distributed.fsdp.FSDPModule
    tensorplay.distributed.fsdp.MixedPrecisionPolicy
    tensorplay.distributed.fsdp.CPUOffloadPolicy
    tensorplay.distributed.fsdp.OffloadPolicy
    tensorplay.distributed.fsdp.UnshardHandle
    tensorplay.distributed.fsdp.register_fsdp_forward_method
    tensorplay.distributed.fsdp.share_comm_ctx
```

- {func}`~tensorplay.distributed.fsdp.fully_shard` applies FSDP to one module
  in place. It accepts an optional device `mesh` (the same mesh a
  [distributed tensor](distributed.tensor.md) runs on), a `reshard_after_forward`
  policy for when to free the gathered shards, a `shard_placement_fn` to decide
  how each parameter is split, and the {class}`~tensorplay.distributed.fsdp.MixedPrecisionPolicy`
  / {class}`~tensorplay.distributed.fsdp.CPUOffloadPolicy` policies.
- {class}`~tensorplay.distributed.fsdp.FSDPModule` is the wrapper type that
  `fully_shard` returns; it exposes the sharded state and the
  {class}`~tensorplay.distributed.fsdp.UnshardHandle` to manage the gathered
  shards manually. The module's parameters are concatenated into a single
  `FlatParameter` (a parameter that holds the flattened collection), which is
  what gets sharded across the ranks.
- {class}`~tensorplay.distributed.fsdp.MixedPrecisionPolicy` chooses the data
  types used for the parameters, the reduction, and the module output.
  {class}`~tensorplay.distributed.fsdp.CPUOffloadPolicy` offloads the shards
  to pinned host memory.

## Classic `FullyShardedDataParallel` API

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.fsdp.FullyShardedDataParallel
    tensorplay.distributed.fsdp.ShardingStrategy
    tensorplay.distributed.fsdp.BackwardPrefetch
    tensorplay.distributed.fsdp.MixedPrecision
    tensorplay.distributed.fsdp.CPUOffload
```

- {class}`~tensorplay.distributed.fsdp.ShardingStrategy` selects the sharding
  scheme, e.g. `FULL_SHARD` (split parameters, gradients and optimizer state
  across all ranks) or `NO_SHARD` (replicate, the data-parallel equivalent).
- {class}`~tensorplay.distributed.fsdp.MixedPrecision` controls parameter /
  reduce / buffer dtypes and whether low-precision gradients are kept.
  {class}`~tensorplay.distributed.fsdp.CPUOffload` moves the parameter shards
  to host memory.
- {class}`~tensorplay.distributed.fsdp.BackwardPrefetch` controls how far in
  advance the next shard is prefetched during the backward pass.

## State dictionaries

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.fsdp.StateDictType
    tensorplay.distributed.fsdp.StateDictConfig
    tensorplay.distributed.fsdp.FullStateDictConfig
    tensorplay.distributed.fsdp.LocalStateDictConfig
    tensorplay.distributed.fsdp.ShardedStateDictConfig
    tensorplay.distributed.fsdp.StateDictSettings
    tensorplay.distributed.fsdp.OptimStateDictConfig
    tensorplay.distributed.fsdp.FullOptimStateDictConfig
    tensorplay.distributed.fsdp.LocalOptimStateDictConfig
    tensorplay.distributed.fsdp.ShardedOptimStateDictConfig
    tensorplay.distributed.fsdp.OptimStateKeyType
```

The state-dict classes let a checkpoint be saved and restored in one of three
shapes: full (the standard flat checkpoint), local (per-rank shards only), or
sharded (one checkpoint per rank, the format `FullyShardedDataParallel`
produces). {class}`~tensorplay.distributed.fsdp.StateDictType` is the enum that
selects which; the matching `*Config` classes carry the options, and
{func}`~tensorplay.distributed.fsdp.FullyShardedDataParallel.optim_state_dict`
/ `sharded_optim_state_dict` / `full_optim_state_dict` read and write them.

## Mixed precision

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler
```

{class}`~tensorplay.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler`
is the AMP gradient scaler adapted to sharded optimizers. A plain
`GradScaler` checks for inf/NaN gradients on the tensors it can see — but
with FSDP each rank only unscales its own parameter shard, so a rank whose
shard is healthy would step while another rank's shard overflowed.
`ShardedGradScaler` closes that gap in `unscale_`: after unscaling its
shard, it all-reduces the per-device `found_inf` flags across the process
group, so every rank sees an overflow anywhere in the world, skips the
step, and backs off the scale together.

```python
import tensorplay as tp
from tensorplay.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from tensorplay.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

scaler = ShardedGradScaler()
fully_shard(model, mp_policy=MixedPrecisionPolicy(param_dtype=tp.bfloat16))

for x, y in dataloader:
    loss = model(x).sum()
    scaler.scale(loss).backward()
    scaler.step(optimizer)   # skips together if any rank overflowed
    scaler.update()
```

## Where to go next

- [distributed tensors](distributed.tensor.md) — the sharded `DTensor` type the
  `mesh` argument is built on.
- [device mesh](distributed.device_mesh.md) — creating the mesh for
  `fully_shard(mesh=...)`.
- [the distributed package](distributed.md) — process groups, collectives, and
  the `DistributedDataParallel` wrapper.