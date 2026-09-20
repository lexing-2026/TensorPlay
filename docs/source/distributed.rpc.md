# tensorplay.distributed.rpc

`tensorplay.distributed.rpc` lets you call functions on a worker running in
another process — possibly on another machine — and get the result back. It
is the primitive behind the higher-level distributed model APIs (remote
module, remote tensors); the source of truth, however, is the worker-to-worker
call.

RPC is set up separately from the process group. You call
{func}`~tensorplay.distributed.rpc.init_rpc` on every worker; each worker is
identified by a name, and the workers announce themselves to each other via
the rendezvous store or environment variables so that a call targeting
`"worker1"` can be routed to the right process. Once initialized you make
three kinds of calls to a target worker:

- {func}`~tensorplay.distributed.rpc.rpc_sync` — fire the function and block
  until the result is returned.
- {func}`~tensorplay.distributed.rpc.rpc_async` — fire the function and get a
  {class}`~tensorplay.distributed.rpc.Future` back immediately; the result is
  produced later.
- {func}`~tensorplay.distributed.rpc.remote` — run the function on the target
  and get an {class}`~tensorplay.distributed.rpc.RRef` handle to its result,
  which lives on the remote worker and can be passed to further remote calls
  without copying the value over the wire.

```python
import os
import tensorplay.distributed.rpc as rpc

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"

# run on both workers, each with its own name:
rpc.init_rpc("trainer", rank=int(os.environ["RANK"]), world_size=2)

def add(a, b):
    return a + b

# on rank 0, ask the other worker to compute
result = rpc.rpc_sync("trainer", add, args=(3, 4))
print(result)  # 7

rpc.shutdown()
```

## Initialization and shutdown

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.rpc.init_rpc
    tensorplay.distributed.rpc.shutdown
    tensorplay.distributed.rpc.get_worker_info
    tensorplay.distributed.rpc.get_rpc_timeout
    tensorplay.distributed.rpc.is_available
```

{func}`~tensorplay.distributed.rpc.init_rpc` initializes the RPC agent: it
takes the worker `name`, a `backend` (the only built-in backend is
{class}`~tensorplay.distributed.rpc.BackendType.TENSORPIPE`), the worker's
`rank` and `world_size`, and backend-specific options. It blocks until all
workers in the world have joined, then performs an all-gather so every worker
can map names to the others. Calling it a second time, or calling any RPC
function before it, raises. {func}`~tensorplay.distributed.rpc.shutdown` tears
the agent down, waiting for in-flight calls first when `graceful=True`.
{func}`~tensorplay.distributed.rpc.get_worker_info` returns the
{class}`~tensorplay.distributed.rpc.WorkerInfo` for a named worker (or for the
current worker when no name is given), and {func}`~tensorplay.distributed.rpc.is_available`
reports whether RPC is compiled into this build.

## Remote calls

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.rpc.rpc_sync
    tensorplay.distributed.rpc.rpc_async
    tensorplay.distributed.rpc.remote
    tensorplay.distributed.rpc.RRef
    tensorplay.distributed.rpc.Future
    tensorplay.distributed.rpc.AllGatherStates
    tensorplay.distributed.rpc.method_factory
    tensorplay.distributed.rpc.new_method
```

- {func}`~tensorplay.distributed.rpc.rpc_sync(to, func, args=None, kwargs=None, timeout=-1)`
  calls `func` on the worker `to` and returns its result. `to` is a worker name
  (or {class}`~tensorplay.distributed.rpc.WorkerInfo`).
- {func}`~tensorplay.distributed.rpc.rpc_async` does the same but returns a
  {class}`~tensorplay.distributed.rpc.Future`. The future is the only handle
  you need: it resolves to the result (or raises the remote exception) when
  the call completes, and it also carries the stream/tensor asynchrony of the
  remote worker.
- {func}`~tensorplay.distributed.rpc.remote` creates a remote reference. The
  function runs on the target worker and the
  {class}`~tensorplay.distributed.rpc.RRef` that comes back points at a value
  that lives on that worker. An `RRef` can be a call argument, so you can pass
  a remote object into another remote call; the worker that owns it keeps it
  alive until the last reference is gone, which is what makes this building
  the foundation for distributed objects.
- {func}`~tensorplay.distributed.rpc.method_factory` (and its alias
  {func}`~tensorplay.distributed.rpc.new_method`) creates a bound callable to a
  remote method so you can write `obj.method(x)` style calls after resolving
  the method once.

{func}`~tensorplay.distributed.rpc.rpc_async` and `rpc_sync` take a `timeout`
in seconds; if the target worker does not respond within it the call raises.

## Backends and options

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.rpc.BackendType
    tensorplay.distributed.rpc.BackendValue
    tensorplay.distributed.rpc.TensorPipeRpcBackendOptions
    tensorplay.distributed.rpc.register_backend
    tensorplay.distributed.rpc.backend_registered
    tensorplay.distributed.rpc.init_backend
    tensorplay.distributed.rpc.construct_rpc_backend_options
```

{class}`~tensorplay.distributed.rpc.BackendType` is the enum of built-in RPC
backends; the supported value is `TENSORPIPE`, a gRPC-based transport. Each
backend has a corresponding options object — {class}`~tensorplay.distributed.rpc.TensorPipeRpcBackendOptions`
for `TENSORPIPE` — that carries settings passed to {func}`~tensorplay.distributed.rpc.init_rpc`.
Registering a custom backend is done with {func}`~tensorplay.distributed.rpc.register_backend`,
which associates a backend name with the two handlers that build its options
and initialize it; {func}`~tensorplay.distributed.rpc.init_backend` then
constructs a concrete agent from a backend and a name. 
{func}`~tensorplay.distributed.rpc.construct_rpc_backend_options` builds the
default options object for a backend.

## Where to go next

- [the distributed package](distributed.md) — process groups and collectives,
  the data-parallel counterpart to RPC.
- [distributed tensors](distributed.tensor.md) — the sharded tensor type you
  can exchange over RPC.
- [device mesh](distributed.device_mesh.md) — defining the rank topology you
  may want to share with RPC workers.