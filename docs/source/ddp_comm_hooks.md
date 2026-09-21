# DDP Communication Hooks

```{eval-rst}
.. currentmodule:: tensorplay.nn.parallel
```

By default, DistributedDataParallel averages each gradient bucket with an all-reduce as
backward fills it: fp32 buckets use a native average collective, other dtypes are
pre-divided by the world size and summed. A **communication hook** replaces that step
with your own function — which is the single seam gradient compression, fused optimizer
overlaps, and local-SGD variants all plug into, without touching the training loop.

The ready-made hooks (allreduce, fp16/bf16 compression, PowerSGD, quantization, the
ZeRO-style optimizer-in-hook, post-local-SGD) are documented in
[distributed.algorithms](distributed.algorithms.md). This page is about writing your
own.

## The contract

Register with {meth}`tensorplay.nn.parallel.DistributedDataParallel.register_comm_hook`:

```python
ddp_model.register_comm_hook(state, hook)
```

- `state` is any object you choose — it is passed back to every hook call untouched.
  Use it for per-process context (a communicator, a compression state, a step counter).
- `hook` is called as `hook(state, bucket)` once per gradient bucket, when the bucket is
  full, on the backward path. It must return the reduced buffer: either the tensor
  itself, or a `Future`-like object whose `wait()` (or `value()`) yields it. The reduced
  values are copied back into the parameters' `.grad`.

The bucket argument is a `GradBucket`:

| member | meaning |
| --- | --- |
| `buffer()` | the flat gradient buffer for the whole bucket |
| `gradients()` | per-parameter gradient views into the buffer |
| `parameters()` | the parameters whose gradients share the bucket |
| `index()` | this bucket's position in the reduction order |
| `is_last()` | whether this is the final bucket of the iteration |
| `set_buffer(t)` | replaces the buffer the reducer will copy back from |

Two rules from the registration check itself: a hook can be registered only **once**
(a second call raises `RuntimeError`), and it must be registered **before** the first
backward pass. Registering any hook also switches DDP off its C++ fast reducer and onto
the Python hook path — the flexibility costs the no-GIL hot path, so hooks are for when
you actually need custom communication.

## A minimal hook

A hook that logs each bucket and does the plain average:

```python
import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.futures import Future


def logging_avg_hook(state, bucket):
    buf = bucket.buffer()
    print(f"[rank {state}] bucket {bucket.index()} reducing "
          f"{buf.numel()} elements")
    dist.all_reduce(buf, op=dist.ReduceOp.AVG)
    fut = Future()
    fut.set_result(buf)
    return fut
```

Registration and use, in the usual two-process gloo setup:

```python
def example(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model = tp.nn.Linear(10, 10)
    ddp_model = tp.nn.parallel.DistributedDataParallel(model)
    ddp_model.register_comm_hook(rank, logging_avg_hook)

    out = ddp_model(tp.randn(4, 10))
    out.square().mean().backward()
    # every parameter's .grad is now the average across the two processes
```

The `Future` return is the asynchronous form — a hook that kicks off a non-blocking
collective can return a future that resolves when the collective completes, and
backwards of later buckets overlap with it. Returning the tensor directly is the
synchronous form and is also accepted.

## Composing with the shipped hooks

The compression wrappers in
{mod}`tensorplay.distributed.algorithms.ddp_comm_hooks` are built on exactly this
contract — `fp16_compress_wrapper(inner_hook)` runs `inner_hook` on a half-precision
copy of the bucket, so a custom hook composed with them stays inside the same
mechanism. Reading their source is the fastest way to see the `state`/`bucket`
discipline in real use.

## Where to go next

- [the distributed algorithms page](distributed.algorithms.md) — every shipped hook and
  its knobs.
- [the DDP note](notes/ddp.md) — how buckets are built and ordered in the first place.
