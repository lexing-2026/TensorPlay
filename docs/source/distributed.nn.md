# tensorplay.distributed.nn

`tensorplay.distributed.nn` contains two independent pieces: autograd-aware
collective *functions* you can drop into any differentiable computation, and
the *remote module* wrapper that places a whole `nn.Module` on another worker
and drives it over RPC.

## Autograd-aware collectives

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.nn.functional.broadcast
    tensorplay.distributed.nn.functional.gather
    tensorplay.distributed.nn.functional.scatter
    tensorplay.distributed.nn.functional.reduce
    tensorplay.distributed.nn.functional.reduce_scatter
    tensorplay.distributed.nn.functional.all_gather
    tensorplay.distributed.nn.functional.all_gather_single
    tensorplay.distributed.nn.functional.all_to_all
    tensorplay.distributed.nn.functional.all_to_all_single
    tensorplay.distributed.nn.functional.all_reduce
```

The functions in `tensorplay.distributed.nn.functional` wrap the blocking
collectives of `tensorplay.distributed` in autograd `Function`s, so they can
appear anywhere in a forward pass and still backpropagate. Their gradients
are the mathematical inverses of the collectives, applied to the gradient
that flows in:

- {func}`~tensorplay.distributed.nn.functional.broadcast` — backward reduces
  the gradients (`SUM`) back to the source rank and zeroes them elsewhere,
  so only the source accumulates.
- {func}`~tensorplay.distributed.nn.functional.reduce` — backward broadcasts
  the reduced result's gradient to every rank.
- {func}`~tensorplay.distributed.nn.functional.gather` /
  {func}`~tensorplay.distributed.nn.functional.scatter` are inverse to each
  other: gather's backward scatters the per-rank gradient slices, and
  scatter's backward gathers them.
- {func}`~tensorplay.distributed.nn.functional.all_reduce` — backward applies
  the same all-reduce to the gradient, since every rank needs the sum.
- {func}`~tensorplay.distributed.nn.functional.all_gather` — backward
  all-reduces the incoming output gradients and keeps the slice this rank
  contributed.
- {func}`~tensorplay.distributed.nn.functional.reduce_scatter` — backward
  all-gathers the scattered gradient.
- {func}`~tensorplay.distributed.nn.functional.all_to_all` (and its
  single-output variant) route gradient slices back along the exchange
  pattern of the forward.

```python
import tensorplay as tp
import tensorplay.distributed as dist
import tensorplay.distributed.nn.functional as df

dist.init_process_group("nccl")

x = tp.randn(4, 8, device="cuda", requires_grad=True)
y = df.all_reduce(x)          # differentiable all-reduce
y.sum().backward()            # gradient flows back through the collective
```

## Remote modules

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.nn.RemoteModule
    tensorplay.distributed.nn.api.interface
```

{class}`~tensorplay.distributed.nn.RemoteModule` constructs a module of your
choosing *on another worker* and hands you a local handle: calling
`forward` (or any generated method) transparently issues an RPC to the
remote module, moves the arguments there, and returns the result. The
`remote_device` string says where the module lives: `"worker_name/device"`
or `"rank:N/device"` — for example `"trainer1/cuda:0"`. The constructor
takes the module class and the arguments its own constructor needs; the
instantiation itself happens on the remote worker.

```python
import tensorplay as tp
import tensorplay.distributed.rpc as rpc
from tensorplay.distributed.nn import RemoteModule

rpc.init_rpc(f"driver", rank=0, world_size=2)

# an nn.Linear now lives on the remote worker, on its cuda:0
remote_linear = RemoteModule(
    "worker1/cuda:0",
    tp.nn.Linear,
    args=(16, 8),
)

out = remote_linear(tp.randn(4, 16))          # runs on worker1

param_rrefs = remote_linear.remote_parameters()  # RRefs, not tensors
module_rref = remote_linear.get_module_rref()

rpc.shutdown()
```

{func}`~tensorplay.distributed.nn.api.interface` is the companion decorator:
mark a class with it to define the exact set of remote methods a remote
module exposes, validated at instantiation time.

`RemoteModule` implements the usual `nn.Module` surface, but the operations
that would mutate local state (`cuda`, `to`, `load_state_dict`,
`register_buffer`, ...) raise, because there is no local module to mutate —
the state lives on the remote worker. Use `remote_parameters()` to get
`RRef`s to the parameters for a
{class}`~tensorplay.distributed.optim.DistributedOptimizer`, or
`get_module_rref()` to pass the module itself into further RPC calls.

## Where to go next

- [RPC](distributed.rpc.md) — the transport remote modules are built on.
- [the distributed package](distributed.md) — the blocking collectives these
  functions differentiate.
- [distributed autograd](distributed.autograd.md) — accumulating gradients
  across workers when the forward pass spans RPC calls.