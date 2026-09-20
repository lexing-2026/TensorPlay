# tensorplay.distributed.elastic

`tensorplay.distributed.elastic` runs a training job across a set of nodes
that can change size while the job is live. The mental model has three moving
parts:

- A **rendezvous** decides who is in the world. Each node asks a shared
  rendezvous store "how many nodes and which roles have joined?" and the
  answer changes over time: nodes can join and leave between training steps,
  so the world size and per-rank world mapping are renegotiated rather than
  fixed at launch.
- An **agent** runs on every node. It reads the rendezvous result, launches
  the local workers (processes for a subprocess entrypoint, workers for a
  callable), monitors their health, restarts them on failure up to
  `max_restarts`, and re-rendezvous when the peer set changes.
- The **workers** are your actual training processes. The elastic agent gives
  each one the standard environment variables (`RANK`, `WORLD_SIZE`,
  `MASTER_ADDR`, `MASTER_PORT`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`) so the
  training script initializes the distributed package as it would under any
  launcher.

The entry point is {func}`~tensorplay.distributed.run.main`, the
`python -m tensorplay.distributed.run` command's implementation, which parses
the CLI into a {class}`~tensorplay.distributed.launcher.LaunchConfig` and
hands the job to the elastic agent.

```python
# train.py  -- runs on every worker
import tensorplay
import tensorplay.distributed as dist

dist.init_process_group("nccl", rank=dist.get_rank(), world_size=dist.get_world_size())
# ... training loop ...
```

```bash
python -m tensorplay.distributed.run \
  --nproc-per-node=2 --nnodes=1 \
  --rdzv-backend=static --rdzv-endpoint=localhost:29500 \
  train.py
```

When a worker process later calls `dist.monitored_barrier`/`dist.barrier`,
the elastic agent can observe a stuck or dead worker and either restart it or
fail the job, depending on the remaining restart budget.

## The launcher

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.launcher.LaunchConfig
    tensorplay.distributed.launcher.elastic_launch
    tensorplay.distributed.launcher.launch_agent
```

{class}`~tensorplay.distributed.launcher.LaunchConfig` carries the job
definition: the min/max node count (`min_nodes`/`max_nodes`), workers per node
(`nproc_per_node`), the rendezvous backend and endpoint, the restart and
monitor settings, and the start method (`spawn` by default). When `min_nodes`
and `max_nodes` differ the job is elastic and ranks can be added or removed
while it runs. {func}`~tensorplay.distributed.launcher.elastic_launch` wraps
`launch_agent` and is the programmatic form of the CLI: build a
`LaunchConfig`, pass your entrypoint callable or script path, then call the
resulting function with any trailing CLI arguments.

## The agent

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.elastic.agent.server.WorkerSpec
    tensorplay.distributed.elastic.agent.server.Worker
    tensorplay.distributed.elastic.agent.server.WorkerGroup
    tensorplay.distributed.elastic.agent.server.WorkerState
    tensorplay.distributed.elastic.agent.server.RunResult
    tensorplay.distributed.elastic.agent.server.ElasticAgent
    tensorplay.distributed.elastic.agent.server.SimpleElasticAgent
    tensorplay.distributed.elastic.agent.server.LocalElasticAgent
```

{class}`~tensorplay.distributed.elastic.agent.server.WorkerSpec` is the
blueprint of the local worker group: the `role` name, `local_world_size`, the
entrypoint (`fn` callable or `entrypoint` command plus `args`), and the
settings that bound restarts (`max_restarts`), the rendezvous handler, and how
workers are launched (`start_method`, `redirects`, `tee`, `log_dir`, and the
environment variables derived from the spec). Every node runs the same spec,
so the resulting world arithmetic is consistent across nodes.

{class}`~tensorplay.distributed.elastic.agent.server.ElasticAgent` is the
abstract agent interface over one worker-group role. The
{class}`~tensorplay.distributed.elastic.agent.server.SimpleElasticAgent` run
loop is the reusable implementation: it alternates between rendezvous (getting
the current membership) and worker monitoring, restarting workers that exit
with a non-terminal state and re-rendezvousing when other nodes are waiting.
{class}`~tensorplay.distributed.elastic.agent.server.LocalElasticAgent`
manages the workers on one node, launching them through `start_processes`
(with per-rank environments) and letting the base-class loop handle failures
and restarts. {class}`~tensorplay.distributed.elastic.agent.server.WorkerState`
is the lifecycle of a single worker slot (`INIT`, `HEALTHY`, `UNHEALTHY`,
`SUCCEEDED`, `FAILED`, `STOPPED`); {class}`~tensorplay.distributed.elastic.agent.server.RunResult`
records the terminal outcome per role.

## Rendezvous

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.elastic.rendezvous.RendezvousHandler
    tensorplay.distributed.elastic.rendezvous.RendezvousParameters
    tensorplay.distributed.elastic.rendezvous.RendezvousInfo
    tensorplay.distributed.elastic.rendezvous.RendezvousStoreInfo
    tensorplay.distributed.elastic.rendezvous.RendezvousSettings
    tensorplay.distributed.elastic.rendezvous.DynamicRendezvousHandler
    tensorplay.distributed.elastic.rendezvous.StaticTCPRendezvous
    tensorplay.distributed.elastic.rendezvous.C10dRendezvousBackend
    tensorplay.distributed.elastic.rendezvous.P10dRendezvousBackend
    tensorplay.distributed.elastic.rendezvous.create_handler
```

{class}`~tensorplay.distributed.elastic.rendezvous.RendezvousHandler` is the
algorithmic interface behind one rendezvous backend. The main method is
{func}`~tensorplay.distributed.elastic.rendezvous.RendezvousHandler.next_rendezvous`,
which blocks until the world reaches a consistent membership and returns a
{class}`~tensorplay.distributed.elastic.rendezvous.RendezvousInfo` describing
who joined and their rank-to-world mapping. The handler also reports how many
nodes are waiting (`num_nodes_waiting`) which the agent uses to trigger
scale-up, and supports closing and shutting down the rendezvous.

{class}`~tensorplay.distributed.elastic.rendezvous.RendezvousParameters`
describes one rendezvous request: the backend name, the store `endpoint`,
the `run_id`, the `min_nodes`/`max_nodes` bounds, and backend-specific options
in `config`. {class}`~tensorplay.distributed.elastic.rendezvous.StaticTCPRendezvous`
implements the simplest backend over a static TCP store (fixed membership),
{class}`~tensorplay.distributed.elastic.rendezvous.DynamicRendezvousHandler`
is the dynamic variant used by the etcd/tcp backends that allow membership to
change, and {class}`~tensorplay.distributed.elastic.rendezvous.C10dRendezvousBackend`
backed by a process-group store and its XPU counterpart
{class}`~tensorplay.distributed.elastic.rendezvous.P10dRendezvousBackend`.
{func}`~tensorplay.distributed.elastic.rendezvous.create_handler` builds a
handler from a backend name and parameters.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.elastic.rendezvous.RendezvousError
    tensorplay.distributed.elastic.rendezvous.RendezvousClosedError
    tensorplay.distributed.elastic.rendezvous.RendezvousTimeoutError
    tensorplay.distributed.elastic.rendezvous.RendezvousConnectionError
    tensorplay.distributed.elastic.rendezvous.RendezvousStateError
    tensorplay.distributed.elastic.rendezvous.RendezvousGracefulExitError
    tensorplay.distributed.elastic.rendezvous.RendezvousExhaustedError
```

The rendezvous error types let a caller distinguish why membership could not
be formed: the rendezvous was closed, it timed out, a connection was lost, the
world reached an inconsistent state, it was asked to exit gracefully, or the
maximum node count was exhausted.

## Multiprocessing

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.elastic.multiprocessing.start_processes
    tensorplay.distributed.elastic.multiprocessing.PContext
    tensorplay.distributed.elastic.multiprocessing.RunProcsResult
    tensorplay.distributed.elastic.multiprocessing.ProcessFailure
    tensorplay.distributed.elastic.multiprocessing.ChildFailedError
    tensorplay.distributed.elastic.multiprocessing.SignalException
    tensorplay.distributed.elastic.multiprocessing.Redirects
    tensorplay.distributed.elastic.multiprocessing.Std
    tensorplay.distributed.elastic.multiprocessing.to_map
```

{func}`~tensorplay.distributed.elastic.multiprocessing.start_processes`
launches `len(envs)` workers and returns the managing
{class}`~tensorplay.distributed.elastic.multiprocessing.PContext`. The
`entrypoint` is either a command string (subprocess workers, where `args` is a
shared argument list) or a picklable callable (multiprocessing workers, where
`args` holds one tuple per rank). Output redirection and logging are configured
through {class}`~tensorplay.distributed.elastic.multiprocessing.Redirects` and
{class}`~tensorplay.distributed.elastic.multiprocessing.Std`, so a worker's
`stdout`/`stderr` can be teed to the log directory or suppressed. When a child
fails, the exception that propagates to the manager is a
{class}`~tensorplay.distributed.elastic.multiprocessing.ChildFailedError` (for
multiprocessing workers) or a
{class}`~tensorplay.distributed.elastic.multiprocessing.ProcessFailure` (for
subprocesses) carrying which worker failed and its output.

## Data utilities

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.elastic.utils.data.CyclingIterator
    tensorplay.distributed.elastic.utils.data.ElasticDistributedSampler
    tensorplay.distributed.elastic.utils.get_env_variable_or_raise
    tensorplay.distributed.elastic.utils.get_socket_with_port
    tensorplay.distributed.elastic.utils.macros
```

{class}`~tensorplay.distributed.elastic.utils.data.ElasticDistributedSampler`
is the data-parallel sampler you use when the world size can change: it
shards the dataset across the current ranks and, crucially, produces a
world-sized sample stream that stays stable as nodes join or leave, so the
resumed epochs do not reorder the data. {class}`~tensorplay.distributed.elastic.utils.data.CyclingIterator`
wraps an iterator and repopulates it from a generator function whenever it is
exhausted, so a training loop can run for an arbitrary number of steps even
when a node's local dataset is finite. {func}`~tensorplay.distributed.elastic.utils.get_env_variable_or_raise`
reads a required environment variable and raises a helpful error when missing,
{class}`~tensorplay.distributed.elastic.utils.macros` is the collection of
elastic-injected macro functions (such as the world-size and local-rank
helpers) that the agent makes available to workers.

## Events, metrics, and timers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.elastic.events.Event
    tensorplay.distributed.elastic.events.NodeState
    tensorplay.distributed.elastic.events.RdzvEvent
    tensorplay.distributed.elastic.events.record
    tensorplay.distributed.elastic.metrics.configure
    tensorplay.distributed.elastic.metrics.MetricsConfig
    tensorplay.distributed.elastic.metrics.prof
    tensorplay.distributed.elastic.timer.TimerClient
    tensorplay.distributed.elastic.timer.TimerServer
    tensorplay.distributed.elastic.timer.configure
    tensorplay.distributed.elastic.timer.expires
```

The instrumentation is pluggable so the same agent loop can emit events to a
logging handler, publish metrics to a stream, or enforce deadlines.
{class}`~tensorplay.distributed.elastic.events.Event` describes one agent or
worker lifecycle event; {class}`~tensorplay.distributed.elastic.events.NodeState`
is the set of node lifecycle states, and {func}`~tensorplay.distributed.elastic.events.record`
dispatches an event to the handler configured for a destination.
{class}`~tensorplay.distributed.elastic.metrics.MetricsConfig` /
{func}`~tensorplay.distributed.elastic.metrics.configure` select where metrics
go, and {func}`~tensorplay.distributed.elastic.metrics.prof` times a block and
publishes the duration. {class}`~tensorplay.distributed.elastic.timer.TimerClient`
sends deadline requests to a {class}`~tensorplay.distributed.elastic.timer.TimerServer`,
letting a worker register a "this step must finish before X" deadline; 
{func}`~tensorplay.distributed.elastic.timer.expires` is the context manager
that raises if the deadline is not met.

## Where to go next

- [the distributed package](distributed.md) — process groups, collectives, and
  initialization, which the launched workers call in their training loop.
- [device mesh](distributed.device_mesh.md) — multi-dimensional group layout
  for tensor and FSDP parallelism once the elastic world is formed.
- [FSDP](distributed.fsdp.md) — the sharding strategy for the model the
  elastic job trains.