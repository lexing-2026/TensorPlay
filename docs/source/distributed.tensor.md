# tensorplay.distributed.tensor

Distributed (sharded) tensors built on top of a
[device mesh](distributed.device_mesh.md). A `DTensor` is a *logical*
tensor whose data is split across the ranks of a mesh: each rank stores a
local shard, and the shards are expected to behave like a single tensor for
the operations the library implements. The layout of a `DTensor` is described
by its mesh together with a *placement* for every mesh dimension.

The module is the natural building block for tensor parallelism: shard the
weights of a module across a mesh with {func}`distribute_module`, then feed it
inputs produced by the distributed factory functions below.

## Creating distributed tensors

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.distributed.tensor.distribute_tensor
    tensorplay.distributed.tensor.distribute_module
    tensorplay.distributed.tensor.from_local
```

- {func}`~tensorplay.distributed.tensor.distribute_tensor` wraps an existing
  tensor into a `DTensor` with the given placements, redistributing the data
  if the current layout on the mesh differs from the requested one.
- {func}`~tensorplay.distributed.tensor.distribute_module` converts an entire
  module into its distributed form: a partition function decides, per
  parameter, whether it is sharded or left replicated, and optional input and
  output functions transform the tensors that enter and leave the module.
- {func}`~tensorplay.distributed.tensor.from_local` builds a `DTensor` out of
  the *local* shards already present on each rank, without any cross-rank
  movement of data (optionally checking that the local shards match the
  requested global shape).

## Factory functions

The factory functions create `DTensor` values directly on a mesh. They accept
the usual construction arguments (`dtype`, `layout`, `requires_grad`) plus
`device_mesh` and `placements`:

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.distributed.tensor.ones
    tensorplay.distributed.tensor.zeros
    tensorplay.distributed.tensor.empty
    tensorplay.distributed.tensor.full
    tensorplay.distributed.tensor.rand
    tensorplay.distributed.tensor.randn
    tensorplay.distributed.tensor.linspace
    tensorplay.distributed.tensor.logspace
```

For example, creating an 8-element replicated vector on a 4-rank mesh keeps a
full copy on every rank, while `Shard(0)` distributes it one element per pair
of ranks:

```python
from tensorplay.distributed.tensor import Shard, ones

# requires an initialized process group, e.g. via dist.init_process_group()
mesh = ...   # a DeviceMesh, see distributed.device_mesh
local = ones(8, device_mesh=mesh, placements=[Shard(0)])
```

## Placements

A placement tells the runtime how one dimension of the mesh participates in
storing the tensor:

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.distributed.tensor.Shard
    tensorplay.distributed.tensor.Replicate
    tensorplay.distributed.tensor.Partial
    tensorplay.distributed.tensor.Placement
```

- {class}`~tensorplay.distributed.tensor.Shard` splits the tensor along a
  logical dimension across the ranks of the corresponding mesh dimension.
- {class}`~tensorplay.distributed.tensor.Replicate` stores a full copy of the
  tensor on every rank of the corresponding mesh dimension.
- {class}`~tensorplay.distributed.tensor.Partial` marks a tensor that has been
  only partially reduced on each rank; the `reduce_op` names the operation
  (e.g. ``"sum"``) that completes the reduction when a full value is needed.

## The DTensor class

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.tensor.DTensor
    tensorplay.distributed.tensor.distribute_tensor
    tensorplay.distributed.tensor.distribute_module
    tensorplay.distributed.tensor.from_local
    tensorplay.distributed.tensor.ones
    tensorplay.distributed.tensor.zeros
    tensorplay.distributed.tensor.empty
    tensorplay.distributed.tensor.full
    tensorplay.distributed.tensor.rand
    tensorplay.distributed.tensor.randn
    tensorplay.distributed.tensor.linspace
    tensorplay.distributed.tensor.logspace
    tensorplay.distributed.tensor.Shard
    tensorplay.distributed.tensor.Replicate
    tensorplay.distributed.tensor.Partial
    tensorplay.distributed.tensor.Placement
```

The class itself exposes the shard on the current rank through
{func}`~tensorplay.distributed.tensor.DTensor.to_local` (moving the tensor
across devices if the shard is on a different accelerator than the local
one), the fully materialized value through
{func}`~tensorplay.distributed.tensor.DTensor.full_tensor`, and a
re-layout operation through
{func}`~tensorplay.distributed.tensor.DTensor.redistribute` that converts
between placements, possibly asynchronously. Its metadata follows a plain
tensor: `shape`, `size`, `stride`, `ndim`, `numel`, `dtype`, `device`, plus
the `device_mesh` and `placements` that describe its layout.

## Where to go next

- [device mesh](distributed.device_mesh.md) — the mesh abstraction that
  `DTensor` runs on.
- [tensor parallelism](distributed.tensor.parallel.md) — sharding entire
  modules with `parallelize_module`.
- [the distributed package](distributed.md) — process groups, collectives,
  and the key-value stores underneath.