# tensorplay.distributed.autograd

`tensorplay.distributed.autograd` extends automatic differentiation across
worker boundaries. When a forward pass spans several workers — because
{func}`~tensorplay.distributed.rpc.rpc_sync` / `rpc_async` calls ran parts of
the computation remotely, or a [remote module](distributed.nn.md) executed its
forward on another machine — plain autograd can no longer follow the graph:
each worker saw only its own slice. The distributed autograd runtime records
every operation tagged with a *context id*, stitches the per-worker record
into one logical graph, and can run backward over the whole thing.

The entry point is the {class}`~tensorplay.distributed.autograd.context`
context manager, which returns an integer context id:

```python
import tensorplay as tp
import tensorplay.distributed.rpc as rpc
import tensorplay.distributed.autograd as dist_autograd

rpc.init_rpc("worker", rank=0, world_size=2)

def remote_double(t):
    return t * 2

with dist_autograd.context() as ctx_id:
    t = tp.randn(4, requires_grad=True)
    # part of the graph executes on the remote worker
    out = rpc.rpc_sync("worker", remote_double, args=(t,))
    loss = out.sum()
    dist_autograd.backward(ctx_id, loss)

grads = dist_autograd.get_gradients(ctx_id)
print(grads[t])   # the gradient accumulates back onto t
```

Inside the `with` block, every operation this worker executes — local or
RPC-dispatched — is recorded under `ctx_id`. The remote worker records its
own slice under the same context id (propagated with the RPC call), so the
two record streams describe one graph.

## API

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.autograd.context
    tensorplay.distributed.autograd.backward
    tensorplay.distributed.autograd.get_gradients
    tensorplay.distributed.autograd.is_available
    tensorplay.distributed.autograd.is_initialized
    tensorplay.distributed.autograd.DistAutogradContext
```

- {class}`~tensorplay.distributed.autograd.context` enters a new distributed
  autograd context and yields its id; leaving the block releases the
  context, so collect the gradients you need before exiting.
- {func}`~tensorplay.distributed.autograd.backward` runs the backward pass
  rooted at the given tensors. Every worker that recorded operations under
  the context participates: each sends its local gradients to wherever they
  are needed, following the recorded send/recv edges between workers.
  Pass `retain_graph=True` to keep the recorded graph for a second
  backward.
- {func}`~tensorplay.distributed.autograd.get_gradients` returns the mapping
  from each tensor that required gradients to the gradient accumulated under
  that context id.
- {func}`~tensorplay.distributed.autograd.is_available` /
  {func}`~tensorplay.distributed.autograd.is_initialized` report whether the
  native distributed-autograd runtime is compiled in and ready.
- {class}`~tensorplay.distributed.autograd.DistAutogradContext` is the native
  context object the context manager holds; the id it yields is the handle
  the other functions take.

Note that RPC must be initialized (`rpc.init_rpc`) before entering a
context — the runtime propagates context ids with RPC metadata.

## Where to go next

- [RPC](distributed.rpc.md) — the remote calls whose graphs this package
  stitches together.
- [remote modules](distributed.nn.md) — module-level sugar over RPC that
  composes with distributed autograd.
- [the distributed optimizer](distributed.optim.md) — stepping the remote
  parameters that distributed autograd produces gradients for.