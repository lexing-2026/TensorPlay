(cuda-semantics)=

# CUDA semantics

`tensorplay.cuda` is a general-purpose GPU compute package built around the
CUDA runtime. General-purpose means that it is not tuned to one workload:
the same primitives serve a linear algebra kernel, a convolution, a
reduction, or a pointwise transform, on data of any shape and dtype the
backends cover. The CUDA execution model — kernels are queued to a device
and run asynchronously — is visible through the whole package: from how
memory is allocated, to how operations are ordered, to how errors surface.

## Best practices

### Device-agnostic code

Code that should run on a CUDA-enabled build and a CPU-only build alike
should branch on availability rather than on build flags:

```python
if tensorplay.cuda.is_available():
    model = model.cuda()
    data = data.cuda()
```

`is_available()` performs a cheap check once the runtime is initialized.
By default the check itself can touch the driver; if it may run before a
`fork`, set the environment variable `TENSORPLAY_NVML_BASED_CUDA_CHECK=1`
so the check avoids initializing the runtime and cannot poison the child.

### Keep the runtime out of forked children

CUDA contexts cannot be re-initialized in a process forked after the parent
initialized them; see {ref}`multiprocessing-best-practices` for the start
methods to use when subprocesses need the device.

(cuda-memory-management-note)=

## Memory management

TensorPlay uses a caching memory allocator to speed up device allocations:
freed blocks are kept in the process's pools and reused for later
allocations of the same size instead of being returned to the driver. This
makes allocation latency largely independent of the driver, at the cost of
holding memory that looks "freed" from the operating system's point of
view.

The distinction shows up in two counters:

- {func}`tensorplay.cuda.memory_allocated` — bytes currently occupied by
  live tensors.
- {func}`tensorplay.cuda.memory_reserved` — bytes currently held by the
  caching allocator, including cached-but-unoccupied blocks.

`memory_reserved` is therefore always at least `memory_allocated`, and the
gap is memory you can get back. {func}`tensorplay.cuda.empty_cache`
releases all unoccupied cached blocks to the driver; it does not free
memory that tensors still occupy, so it is safe to call at any point, but
it is rarely a fix for out-of-memory errors — the memory it releases is
memory the process was not using anyway.

Peak statistics are tracked alongside: {func}`tensorplay.cuda.max_memory_allocated`
and {func}`tensorplay.cuda.max_memory_reserved` report the high-water marks,
and {func}`tensorplay.cuda.reset_peak_memory_stats` moves the high-water
mark down to the current value. For a full breakdown — allocation counts,
segment and block sizes, per-capacity splits — use
{func}`tensorplay.cuda.memory_stats`; {func}`tensorplay.cuda.memory_summary`
renders the same data as a human-readable table.

:::{note}
Host (pinned) memory has its own family of statistics:
{func}`tensorplay.cuda.host_memory_stats` and its peak/accumulated
counters, reset with the corresponding `reset_peak_*` /
`reset_accumulated_*` functions.
:::

A per-process cap can be declared with
{func}`tensorplay.cuda.set_per_process_memory_fraction`; in this build the
fraction is recorded for API compatibility but not enforced by the
allocator, so treat it as documentation for tooling rather than a limit.

## Asynchronous execution

A CUDA device runs operations asynchronously with the host: calling a
TensorPlay CUDA operation enqueues a kernel (or a host-device copy) on the
current stream and returns immediately. The host then races ahead, and
results are only guaranteed to be ready when something waits on them —
either a {func}`tensorplay.cuda.synchronize` (whole device), a
`stream.synchronize()` (one stream), or an `event.synchronize()`.

The practical consequences:

- Timing Python-side code around CUDA calls measures queueing, not
  execution. Measure with two {class}`tensorplay.cuda.streams.Event`
  objects created with `enable_timing=True` — `start.record()`, run,
  `end.record()` — and read `start.elapsed_time(end)` *after*
  `end.synchronize()`; events created without `enable_timing` refuse the
  query.
- Errors raised by the device surface at the next synchronization point in
  the host code, not at the enqueueing call. When debugging, enabling
  {func}`tensorplay.cuda.set_sync_debug_mode` makes the runtime check for
  synchronizing calls and warn about them.
- Copies between host and device are also queued. `Tensor.cuda(...)` /
  `Tensor.to(...)` accept a `non_blocking` flag, which applies to
  host-device transfers. The overlap only becomes meaningful when the host
  memory is pinned (see {func}`tensorplay.pin_memory`); a transfer out of
  ordinary pageable memory may still block or stage through a buffer.

## Streaming parallelism

A stream is a queue of device work: operations submitted to the same stream
run in submission order, and operations in different streams can run
concurrently, sharing the device's compute units. All work TensorPlay
submits goes to the {func}`current stream <tensorplay.cuda.current_stream>`,
which defaults to the device's {func}`default stream <tensorplay.cuda.default_stream>`.

To run independent work concurrently, submit it to side streams:

```python
s1 = tensorplay.cuda.Stream()
s2 = tensorplay.cuda.Stream()
# Capture some intermediate tensors for later
with tensorplay.cuda.stream(s1):
    # work submitted to s1
    ...
with tensorplay.cuda.stream(s2):
    # work submitted to s2, may overlap with s1's kernels
    ...
```

`with tensorplay.cuda.stream(s):` (or {class}`tensorplay.cuda.StreamContext`
directly) makes `s` the current stream for the duration of the block; the
previous stream is restored on exit. A {class}`tensorplay.cuda.streams.Stream`
takes a `priority` — a lower number is a higher priority, 0 by default.

There is no implicit ordering between streams. When later work must see
earlier work's results, order the streams explicitly:

- `later.wait_stream(earlier)` — makes all future work in `later` wait for
  everything already submitted to `earlier`.
- `later.wait_event(e)` — the same, but waiting on a recorded point instead
  of a whole stream's history: `e = earlier.record_event()` first, then
  `later.wait_event(e)`.

This is the pattern for data parallelism over one device (each stream owns
a slice of a batch), double-buffering host copies against compute, and
building custom pipelines whose stages should overlap.

## CUDA Graphs

Whole sequences of enqueued work can be captured once and replayed many
times, skipping per-launch queueing overhead. Capture happens inside the
{func}`tensorplay.cuda.graphs.graph` context manager, which records the
work its body submits:

```python
g = tensorplay.cuda.CUDAGraph()
# warmup on a side stream, then capture
with tensorplay.cuda.graphs.graph(g):
    out = model(x)
...
g.replay()
```

Capture requires a side stream — the legacy default stream cannot capture —
and the graph context manager handles that by default by using the
runtime's dedicated per-device capture stream. Memory allocated during
capture comes from the graph's memory pool so that replay finds the
tensors in place; separate captures that should share scratch space can do
so by passing the same {func}`pool handle <tensorplay.cuda.graphs.graph_pool_handle>`.
To check whether the current stream is in the middle of a capture, use
{func}`tensorplay.cuda.graphs.is_current_stream_capturing`.
