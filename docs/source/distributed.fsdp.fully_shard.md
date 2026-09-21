# tensorplay.distributed.fsdp.fully_shard

```{eval-rst}
.. currentmodule:: tensorplay.distributed.fsdp
```

`fully_shard` is the composable entry point of
[FSDP](distributed.fsdp.md): fully sharded data parallelism with
per-parameter sharding for eager-mode usability. Where
{class}`FullyShardedDataParallel` wraps a whole module in one call,
`fully_shard` shards each parameter individually along dim 0 and reuses the
original module — no wrapper object, no flattened parameter buffer.

## The user contract

The contract of `fully_shard(model)`:

- The module is modified **in place**: `type(model)` becomes a new class
  that subclasses both the original class and {class}`FSDPModule` (a
  `Linear` becomes a `LinearFSDPModule`), keeping every method of the
  original while exposing the FSDP methods. `fully_shard` returns the same
  object it was given.
- The parameters are replaced by their local dim-0 shards. In a world of
  size 2, an `(8, 8)` weight becomes a `(4, 8)` parameter on each rank;
  a bias of shape `(8,)` becomes `(4,)`.
- Fully qualified names are unchanged: `model.state_dict()` reports the
  same keys before and after `fully_shard`.
- Call `model(input)`, not `model.forward(input)`. The pre-forward hook
  registered on the module performs the all-gather that materializes the
  full parameters (see the warning below).
- The optimizer is constructed with the *sharded* parameters —
  `tp.optim.SGD(model.parameters(), ...)` after `fully_shard` — and steps
  on them directly.
- After backward, each parameter's gradient is reduce-scattered to its
  shard: `p.grad` has the local shard shape and equals the corresponding
  slice of the full-batch gradient.
- Applying `fully_shard` to the same module twice raises `RuntimeError`.

Without an initialized process group, `fully_shard` falls back to a
one-rank CPU device mesh — sharding is a no-op (each shard is the whole
parameter), which makes the API smoke-testable in a single process:

```python
import tensorplay as tp
import tensorplay.nn as nn
from tensorplay.distributed.fsdp import fully_shard

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
model = fully_shard(model)
print(type(model).__name__)            # SequentialFSDPModule

out = model(tp.randn(2, 4))            # hooks all-gather for the forward
print(tuple(out.shape))                # (2, 4)

opt = tp.optim.SGD(model.parameters(), lr=0.1)
out.sum().backward()
opt.step()

try:
    fully_shard(model)
except RuntimeError as e:
    print(e)                           # fully_shard has already been applied
```

## A sharded step, verified

Two gloo processes shard an `nn.Linear(8, 8)`; both ranks check their
shard, their gradient slice, and one optimizer step against an unsharded
reference computed with identical weights and data:

```python
import multiprocessing as mp

import tensorplay as tp
import tensorplay.distributed as dist
import tensorplay.nn as nn
from tensorplay.distributed.fsdp import fully_shard


def worker(rank, world_size):
    dist.init_process_group(
        "gloo", init_method="tcp://127.0.0.1:29571",
        rank=rank, world_size=world_size,
    )
    tp.manual_seed(42)                    # identical init and data per rank
    model = nn.Linear(8, 8)
    x = tp.randn(4, 8)

    model = fully_shard(model)
    shards = {n: tuple(p.shape) for n, p in model.named_parameters()}
    print(rank, shards)                   # {'weight': (4, 8), 'bias': (4,)}

    out = model(x)                        # all-gather inside the forward
    out.sum().backward()                  # reduce-scatter after backward
    grads = {n: tuple(p.grad.shape) for n, p in model.named_parameters()}
    print(rank, grads)                    # shard shapes again

    tp.optim.SGD(model.parameters(), lr=0.1).step()
    model.unshard()                       # gather to check the result
    print(rank, tuple(next(model.parameters()).shape))   # (8, 8)
    dist.destroy_process_group()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, 2)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
```

In a verified run of this script, each rank's `(4, 8)` weight shard and
`(4,)` bias shard carried exactly the corresponding dim-0 slice of the
full-batch gradients, and after one `SGD` step plus `unshard()` the
parameters matched the unsharded reference run.

## Calling `forward` directly

:::{warning}
`model.forward(input)` bypasses the module hooks, so the parameters stay
sharded and the math runs on the local shard. This does **not** raise: in
the verified two-rank run, `model.forward(x)` silently returned shape
`(4, 4)` instead of `(4, 8)` — each rank computed with its `(4, 8)` weight
shard. Use one of the following instead.
:::

- Call `model(input)` — the normal path.
- {func}`register_fsdp_forward_method(model, "forward")` wraps the named
  method with the same pre/post-forward handling, after which
  `model.forward(x)` works (verified: output identical to the hooked
  call).
- Call `model.unshard()` first, which materializes the full parameters
  until the next `reshard()` (also verified).

`unshard(async_op=True)` returns an {class}`UnshardHandle` whose `wait()`
blocks until the all-gather completes; the synchronous form returns
`None`.

## Communication grouping

Each call to `fully_shard` forms **one communication group** out of the
parameters of the given module that were not already assigned to a group by
an earlier call on a submodule. A group's parameters are all-gathered
together before forward and their gradients reduce-scattered together
after backward.

Apply `fully_shard` bottom-up — each transformer layer before the root —
and the root call groups only what is left (embeddings, the output
projection). More, smaller groups overlap communication with compute the
way smaller DDP buckets do; fewer, larger groups amortize the collectives.
The boundaries are entirely your choice of modules; there is no automatic
bucketing.

The knobs on {class}`FSDPModule` that tune the schedule include:

| Method | Effect |
| --- | --- |
| `unshard()` / `reshard()` | materialize / free the full parameters |
| `set_reshard_after_forward(bool)` | free after forward or keep until backward |
| `set_modules_to_forward_prefetch(mods)` | issue the next all-gather earlier |
| `set_modules_to_backward_prefetch(mods)` | prefetch the next backward all-gather |
| `set_requires_gradient_sync(bool)` | skip the gradient reduce-scatter (gradient accumulation) |
| `set_requires_all_reduce(bool)` | toggle the replicated-dimension reduction |
| `set_gradient_divide_factor(factor)` | scaling applied during the reduce-scatter |
| `reset_iter_state()` | clear the per-iteration bookkeeping |

`fully_shard` also accepts a list of modules (sharding each with its own
group in one call), `ignored_params` to leave parameters unsharded, and a
`mesh` — the same [device mesh](distributed.device_mesh.md) a
[distributed tensor](distributed.tensor.md) runs on — to shard over a
subgroup instead of the whole world. `reshard_after_forward` is `None`
(auto), `bool`, or an `int` number of forwards to keep the parameters
unsharded for (the `int` form is not supported on an SPMD mesh).

## Policies

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.fsdp.fully_shard
    tensorplay.distributed.fsdp.FSDPModule
    tensorplay.distributed.fsdp.UnshardHandle
    tensorplay.distributed.fsdp.MixedPrecisionPolicy
    tensorplay.distributed.fsdp.CPUOffloadPolicy
    tensorplay.distributed.fsdp.OffloadPolicy
    tensorplay.distributed.fsdp.DataParallelMeshDims
    tensorplay.distributed.fsdp.register_fsdp_forward_method
    tensorplay.distributed.fsdp.share_comm_ctx
```

- {class}`MixedPrecisionPolicy` selects the dtypes of the all-gathered
  parameters (`param_dtype`), of the reduce-scattered gradients
  (`reduce_dtype`), and of the module outputs (`output_dtype`), plus
  whether inputs are cast on entry (`cast_forward_inputs`, default `True`).
- {class}`CPUOffloadPolicy` moves the parameter shards to host memory
  (`pin_memory` controls pinned allocation); {class}`OffloadPolicy` is the
  no-op base.
- {class}`DataParallelMeshDims` names which mesh dimensions shard and
  which replicate, for meshes that carry both.

{func}`share_comm_ctx` lets several independently sharded roots reuse one
set of communication resources, and {class}`FSDPModule` deepcopy is not
supported — serialize through state dicts instead.

## Where to go next

- [the FSDP overview](distributed.fsdp.md) — the classic
  `FullyShardedDataParallel` wrapper, state dict types, and the sharded
  gradient scaler.
- [device mesh](distributed.device_mesh.md) — building the mesh the
  `mesh=` argument takes.
- [the distributed package](distributed.md) — process groups and the
  collectives the groups ride on.
