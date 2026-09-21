# tensorplay.distributed.pipelining

`tensorplay.distributed.pipelining` splits one model so its layers run on
different workers, one stage of the pipeline per worker, and orchestrates the
forward and backward passes as streams of microbatches flowing through the
stages. It is a form of model parallelism: instead of replicating the whole
model on every rank, each rank holds a different slice of the layers, so the
model whose parameters do not fit on one device still fits across the group.

Training with a pipeline split has three parts:

- **Splitting** — describe where the model graph is cut into stages. The
  {func}`~tensorplay.distributed.pipelining.pipeline` function traces the
  module with example micro-batched inputs and produces a
  {class}`~tensorplay.distributed.pipelining.Pipe` — the intermediate
  representation of the split model. You choose the cut points either with a
  `split_spec` (a mapping of submodule names to
  {class}`~tensorplay.distributed.pipelining.SplitPoint` markers) or with a
  `split_policy` callable, or by placing {func}`~tensorplay.distributed.pipelining.pipe_split`
  calls inside the module's forward.
- **Building stages** — {func}`~tensorplay.distributed.pipelining.build_stage`
  turns the `Pipe` and a stage index into a
  {class}`~tensorplay.distributed.pipelining.PipelineStage`: the worker-local
  module plus the send/recv channels for the tensors crossing its two
  boundaries. Each rank builds its own stage.
- **Scheduling** — a schedule class runs the training step over the pipeline,
  chunking each full batch into `n_microbatches` microbatches so the stages
  can work on different microbatches at the same time.

```python
import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.distributed.pipelining import (
    SplitPoint, pipeline, build_stage, Schedule1F1B,
)

dist.init_process_group("nccl")

class Block(tp.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = tp.nn.Sequential(tp.nn.Linear(dim, dim), tp.nn.ReLU())

    def forward(self, x):
        return self.net(x)

class Model(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = Block(128)   # stage 0 on rank 0
        self.layer1 = Block(128)   # stage 1 on rank 1

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        return x

model = Model()
mb_args = (tp.randn(8, 128),)          # one microbatch of inputs
pipe = pipeline(model, mb_args, split_spec={"layer1": SplitPoint.BEGINNING})

stage = build_stage(
    pipe, stage_index=dist.get_rank(),
    device=tp.device(f"cuda:{dist.get_rank()}"),
)
schedule = Schedule1F1B(stage, n_microbatches=4)

for x, y in dataloader:
    out = schedule.step(x.to(dev))
    loss = out.sum()
    loss.backward()
    stage.optimizer.step()
```

## Describing the split

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.pipelining.pipeline
    tensorplay.distributed.pipelining.Pipe
    tensorplay.distributed.pipelining.pipe_split
    tensorplay.distributed.pipelining.SplitPoint
```

- {func}`~tensorplay.distributed.pipelining.pipeline` traces the module with
  `mb_args`/`mb_kwargs` shaped like one microbatch, applies the split
  specification, and returns a {class}`~tensorplay.distributed.pipelining.Pipe`.
  Pass `split_spec` (submodule name → `SplitPoint`) or `split_policy` (a
  callable that rewrites the traced graph), but not both.
- {class}`~tensorplay.distributed.pipelining.Pipe` is the split
  representation: a module that knows the stage count, the per-stage
  submodules and their parameters, and the shapes of the tensors that cross
  stage boundaries. It is what `build_stage` consumes.
- {func}`~tensorplay.distributed.pipelining.pipe_split` is a marker you call
  inside a module's `forward`; tracing records "cut here". Combined with a
  nested call to `pipeline`, this is the least intrusive way to annotate split
  points.
- {class}`~tensorplay.distributed.pipelining.SplitPoint` is the enum
  describing where a stage boundary falls relative to a submodule:
  `BEGINNING` (before the submodule runs) or `END` (after it).

## Building a stage

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.pipelining.build_stage
    tensorplay.distributed.pipelining.PipelineStage
```

{func}`~tensorplay.distributed.pipelining.build_stage` takes the `Pipe`, a
stage index, the active device and the process group, and returns the
{class}`~tensorplay.distributed.pipelining.PipelineStage` for this rank. The
stage owns the submodule slice, holds the input/output buffers and gradient
buffers that bridge the stage boundaries, and can wrap a weight-grad
communication callback (`dw_builder`) for overlapping all-reduce with
computation. For the device-mesh style of scheduling (a 2D mesh where the
pipeline dimension is separate from the data-parallel dimension), build the
stage with `get_mesh` so the schedule routes communication over the mesh's
pipeline groups.

## Scheduling

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.pipelining.PipelineScheduleSingle
    tensorplay.distributed.pipelining.PipelineScheduleMulti
    tensorplay.distributed.pipelining.ScheduleGPipe
    tensorplay.distributed.pipelining.Schedule1F1B
    tensorplay.distributed.pipelining.ScheduleInterleaved1F1B
    tensorplay.distributed.pipelining.ScheduleLoopedBFS
    tensorplay.distributed.pipelining.ScheduleInterleavedZeroBubble
    tensorplay.distributed.pipelining.ScheduleZBVZeroBubble
    tensorplay.distributed.pipelining.ScheduleDualPipeV
```

The schedule is what actually runs the step: it chunks the input into
microbatches, walks them through the stage(s), and returns the per-microbatch
losses. {class}`~tensorplay.distributed.pipelining.PipelineScheduleSingle`
drives one stage (used when every rank runs one stage); for interleaved
schedules where a rank owns several stages, use the multi-stage
{class}`~tensorplay.distributed.pipelining.PipelineScheduleMulti` base, which
schedules the same microbatch over the local stages.

The concrete schedules differ in how forward and backward microbatches are
ordered:

- {class}`~tensorplay.distributed.pipelining.ScheduleGPipe` runs all forward
  microbatches first, then drains their backwards — simple, but leaves the
  pipeline idle until the last forward finishes.
- {class}`~tensorplay.distributed.pipelining.Schedule1F1B` (one forward, one
  backward) overlaps a microbatch's backward with the next microbatch's
  forward after a warm-up phase, keeping more stages busy.
- {class}`~tensorplay.distributed.pipelining.ScheduleInterleaved1F1B` extends
  1F1B to interleaved stage assignment (several stages per rank), so a rank
  alternates between its chunks of the model.
- {class}`~tensorplay.distributed.pipelining.ScheduleLoopedBFS` and the
  ZeroBubble family ({class}`~tensorplay.distributed.pipelining.ScheduleInterleavedZeroBubble`,
  {class}`~tensorplay.distributed.pipelining.ScheduleZBVZeroBubble`) apply
  increasingly aggressive recomputation and bubble-elimination schemes to
  approach the theoretical pipeline throughput.
- {class}`~tensorplay.distributed.pipelining.ScheduleDualPipeV` runs the
  bidirectional local-stage schedule for the DualPipe V variant.

All schedules take `n_microbatches` (how finely the batch is chunked), an
optional `loss_fn` applied to the stage's output, and chunk specs describing
which input dimensions are sharded per microbatch (`args_chunk_spec` /
`kwargs_chunk_spec`), so a tensor of shape `(16, 128)` with
`n_microbatches=4` can be split along dim 0 into four `(4, 128)` chunks.

## Where to go next

- [the distributed package](distributed.md) — process groups and collectives,
  the messaging layer pipeline communication runs on.
- [device mesh](distributed.device_mesh.md) — 2D meshes for combining the
  pipeline dimension with data or tensor parallelism.
- [FSDP](distributed.fsdp.md) — sharding the parameter memory within each
  stage's rank, complementary to splitting the model across stages.