# tensorplay.distributed.algorithms

`tensorplay.distributed.algorithms` collects the pieces that plug *into* a
running data-parallel job: communication hooks that change how gradients are
exchanged between ranks, the join mechanism that lets uneven workloads
participate in the same collectives, and the model averagers that keep
replicas in sync outside the all-reduce path. The two subpackages are
`ddp_comm_hooks` (hooks registered on
`DistributedDataParallel`) and `model_averaging` (averaging strategies).

A DDP communication hook is a callable `(state, bucket) -> future` that
replaces the default gradient all-reduce: when a backward pass fills a
gradient bucket, DDP hands it to your hook instead of reducing it directly.
That single seam is enough to implement gradient compression, fused
optimizer steps, and local-SGD variants without touching the training loop.

```python
import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.distributed.algorithms.ddp_comm_hooks import default_hooks

dist.init_process_group("nccl")
model = tp.nn.Linear(64, 64).cuda()
model = tp.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])

# halve the gradient traffic: compress to fp16 before the all-reduce
model.register_comm_hook(None, default_hooks.fp16_compress_hook)
```

## Default hooks

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook
    tensorplay.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
    tensorplay.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook
    tensorplay.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_wrapper
    tensorplay.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_wrapper
```

- {func}`~tensorplay.distributed.algorithms.ddp_comm_hooks.default_hooks.allreduce_hook`
  is the plain gradient all-reduce — the same communication DDP performs by
  default, useful as the baseline or the inner step of a composed hook.
- `fp16_compress_hook` / `bf16_compress_hook` cast each bucket's gradients
  to half precision before the all-reduce and cast the result back,
  halving the bytes on the wire. The `*_compress_wrapper` variants wrap
  *another* hook, so the compression applies before it runs — for example
  `fp16_compress_wrapper(powerSGD_hook)` compresses the residual that
  PowerSGD feeds to the collective.

## PowerSGD

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState
    tensorplay.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.powerSGD_hook
    tensorplay.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.batched_powerSGD_hook
```

PowerSGD compresses a gradient matrix to the product of two low-rank
factors (rank `matrix_approximation_rank`), all-reduces the factors
instead of the full matrix, and reconstructs on the other side; the
residual (what the factors fail to represent) is carried into the next
iteration when error feedback is on. {class}`~tensorplay.distributed.algorithms.ddp_comm_hooks.powerSGD_hook.PowerSGDState`
carries the per-parameter factors and the tuning knobs — the
approximation rank, the warm-up iteration before compression starts
(`start_powerSGD_iter`), and the minimum compression rate below which a
tensor is left dense. `batched_powerSGD_hook` additionally batches
same-shape tensors into one factorization for better throughput.

## Quantization hooks

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.algorithms.ddp_comm_hooks.quantization_hooks.quantization_pertensor_hook
    tensorplay.distributed.algorithms.ddp_comm_hooks.quantization_hooks.quantization_perchannel_hook
```

The quantization hooks reduce gradient bytes by quantizing each bucket
before the collective — per-tensor (one scale for the whole bucket) or
per-channel (a scale per group of `bucket_size` elements), dequantizing
after. Per-channel keeps more precision at the cost of bookkeeping.

## Optimizer-in-hook (ZeRO-style)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook.hook_with_zero_step
    tensorplay.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook.hook_with_zero_step_interleaved
```

{func}`~tensorplay.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook.hook_with_zero_step`
fuses the {class}`~tensorplay.distributed.optim.ZeroRedundancyOptimizer`
step into the gradient bucket's collective: as each bucket's all-reduce
completes, the hook immediately applies the optimizer update for the
parameters in that bucket, so the optimizer overlaps the backward pass
instead of waiting for it. The `shard_buckets` option partitions buckets
across ranks the way ZeRO partitions parameters. The `_interleaved`
variant schedules the collectives and the parameter shard updates
bucket-by-bucket. Both take the `DistributedDataParallel` module and the
`ZeroRedundancyOptimizer` to drive.

## Post-local SGD

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook.PostLocalSGDState
    tensorplay.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook.post_localSGD_hook
```

{func}`~tensorplay.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook.post_localSGD_hook`
switches the gradient reduction domain over time: before
`start_localSGD_iter` the gradients are all-reduced across the full
`process_group`; afterwards they are only all-reduced within the local
`subgroup` (one node, typically). The per-replica parameters then drift,
which is exactly what the [post-local-SGD
optimizer](distributed.optim.md) compensates for by averaging them
periodically. {class}`~tensorplay.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook.PostLocalSGDState`
carries the two groups, the switch iteration, and whether to keep a
global gradient all-reduce after the switch.

## Model averaging

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.algorithms.model_averaging.ModelAverager
    tensorplay.distributed.algorithms.model_averaging.PeriodicModelAverager
    tensorplay.distributed.algorithms.model_averaging.HierarchicalModelAverager
```

{class}`~tensorplay.distributed.algorithms.model_averaging.ModelAverager`
is the abstract interface the [post-local-SGD
optimizer](distributed.optim.md) consumes: it knows how to average
parameters across a process group. {class}`~tensorplay.distributed.algorithms.model_averaging.PeriodicModelAverager`
averages every `period` steps after `warmup_steps` have elapsed.
{class}`~tensorplay.distributed.algorithms.model_averaging.HierarchicalModelAverager`
averages in a tree of subgroups (given as `period_group_size_dict`),
which reduces the latency of large-scale averaging at the cost of
eventual (rather than immediate) consistency.

## Join

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.algorithms.Join
    tensorplay.distributed.algorithms.JoinHook
    tensorplay.distributed.algorithms.Joinable
```

{class}`~tensorplay.distributed.algorithms.Join` is a context manager for
uneven inputs in data-parallel training: ranks that run out of work early
"join" — their collective participation is emulated so the ranks still
working do not hang on a barrier. A {class}`~tensorplay.distributed.algorithms.Joinable`
class (DDP is one) provides a {class}`~tensorplay.distributed.algorithms.JoinHook`
whose `main_hook` shadows the collectives while some ranks have joined,
and whose `post_hook` runs once everyone has joined, letting the last
ranks propagate any final state.

## Where to go next

- [the distributed package](distributed.md) — DDP and the collectives these
  hooks ride on.
- [the distributed optimizer](distributed.optim.md) — the
  `ZeroRedundancyOptimizer` the zero-step hooks fuse with, and the
  post-local-SGD optimizer the model averagers feed.
- [FSDP](distributed.fsdp.md) — the sharded alternative when optimizer
  state, not gradient traffic, is the bottleneck.