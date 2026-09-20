# tensorplay.distributed.checkpoint

`tensorplay.distributed.checkpoint` saves and restores the state of a
distributed training job — one or more `state_dict` structures split across
the ranks of a process group — and makes the checkpoint look like a single
artifact on the storage medium. Unlike a naive per-rank save, the package
coordinates the write across ranks: the individual pieces are gathered into a
global plan, every rank writes its share of the data, and a metadata file is
committed once all ranks have finished, so a partially written checkpoint can
be detected by the missing or incomplete metadata.

A checkpoint on the file system is a directory with one or more per-rank data
files plus a `.metadata` file describing the layout — which tensors live where
and how the pieces map back to the original state dictionary. Loading reads
that metadata and restores each value into the (already allocated) state
dictionary it was loaded from, so the shape and dtype of the objects you load
into come from the caller, not from the checkpoint.

```python
import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.distributed.checkpoint import save, load

dist.init_process_group("nccl")

model = tp.nn.Linear(512, 512).cuda(dist.get_rank())
model = tp.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])

state_dict = {"model": model.state_dict()}
save(state_dict, checkpoint_id="file:///checkpoints/run-1")

# later, on every rank:
model2 = tp.nn.Linear(512, 512).cuda(dist.get_rank())
load({"model": model2.state_dict()}, checkpoint_id="file:///checkpoints/run-1")
```

## Saving and loading

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.checkpoint.save
    tensorplay.distributed.checkpoint.async_save
    tensorplay.distributed.checkpoint.save_state_dict
    tensorplay.distributed.checkpoint.load
    tensorplay.distributed.checkpoint.load_state_dict
    tensorplay.distributed.checkpoint.AsyncCheckpointerType
    tensorplay.distributed.checkpoint.AsyncSaveResponse
```

- {func}`~tensorplay.distributed.checkpoint.save` snapshots the input state
  dictionary (detaching and cloning tensors so the checkpoint never aliases
  live training state), flattens it into a write plan, coordinates the plan
  across ranks, writes the data through a storage writer, and commits the
  metadata once every rank has written its share. Pass `checkpoint_id` (a
  storage URI such as `file:///path`) or an explicit `storage_writer`.
- {func}`~tensorplay.distributed.checkpoint.async_save` returns control to the
  caller immediately: the input is staged (copied to pin/shared memory where
  configured) and the write runs on a background thread or process, chosen by
  {class}`~tensorplay.distributed.checkpoint.AsyncCheckpointerType`. It returns
  a future (or an {class}`~tensorplay.distributed.checkpoint.AsyncSaveResponse`
  when a stager is used) that you can wait on at a safe point, typically after
  the next training step.
- {func}`~tensorplay.distributed.checkpoint.save_state_dict` / {func}`~tensorplay.distributed.checkpoint.load_state_dict`
  are the lower-level entry points that take an explicit storage
  writer/reader and a `coordinator_rank`, used when embedding the checkpoint
  logic inside a custom pipeline.
- {func}`~tensorplay.distributed.checkpoint.load` reads the metadata, builds a
  load plan, and restores values in place into the existing state dictionary.
  On failure it rolls back the state dictionary to the snapshot taken before
  the load, so a failed load never leaves the model half-modified.

## State dictionary helpers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.checkpoint.get_state_dict
    tensorplay.distributed.checkpoint.set_state_dict
    tensorplay.distributed.checkpoint.get_model_state_dict
    tensorplay.distributed.checkpoint.set_model_state_dict
    tensorplay.distributed.checkpoint.get_optimizer_state_dict
    tensorplay.distributed.checkpoint.set_optimizer_state_dict
    tensorplay.distributed.checkpoint.StateDictOptions
    tensorplay.distributed.checkpoint.load_sharded_optimizer_state_dict
```

For sharded training (FSDP, DTensor, or a plain distributed model), the state
dict you would normally assemble yourself differs across ranks. The
{func}`~tensorplay.distributed.checkpoint.get_model_state_dict` /
{func}`~tensorplay.distributed.checkpoint.get_optimizer_state_dict` family
walks the model and optimizer for you, producing the per-rank view that
`checkpoint.save` expects, and the matching `set_*` functions apply a saved
state dict back onto the live model and optimizer. {class}`~tensorplay.distributed.checkpoint.StateDictOptions`
controls the details: `full_state_dict` collapses shards into an
all-parameters view, `cpu_offload` moves CPU tensors to pinned memory before
saving, `ignore_frozen_params` skips parameters that do not require
gradients, and `broadcast_from_rank0` lets rank 0 distribute the full state
dict instead of each rank loading its own shard. The optimizer state
dictionaries produced and consumed by the `*_optimizer_state_dict` functions
are `OptimizerStateType`, a plain `dict[str, Any]` keyed by parameter-qualified
names.
{func}`~tensorplay.distributed.checkpoint.load_sharded_optimizer_state_dict` is
a companion helper for loading a *sharded* optimizer state dict (the format
FSDP produces for its optimizer) against the planner, so a checkpoint saved by
FSDP can be resumed by a job with a different world size.

## Storages

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.checkpoint.FileSystemWriter
    tensorplay.distributed.checkpoint.FileSystemReader
    tensorplay.distributed.checkpoint.FileSystemBase
    tensorplay.distributed.checkpoint.FileSystem
    tensorplay.distributed.checkpoint.SerializationFormat
    tensorplay.distributed.checkpoint.HuggingFaceStorageWriter
    tensorplay.distributed.checkpoint.HuggingFaceStorageReader
    tensorplay.distributed.checkpoint.QuantizedHuggingFaceStorageReader
    tensorplay.distributed.checkpoint.MegaStorageWriter
    tensorplay.distributed.checkpoint.MegaStorageReader
    tensorplay.distributed.checkpoint.StorageWriter
    tensorplay.distributed.checkpoint.StorageReader
```

The storage layer decides where the bytes land. {class}`~tensorplay.distributed.checkpoint.FileSystemWriter`
writes one file per rank into the checkpoint directory by default, optionally
with metadata after every rank finishes; {class}`~tensorplay.distributed.checkpoint.FileSystemReader`
reads such a directory back. {class}`~tensorplay.distributed.checkpoint.SerializationFormat`
chooses the on-disk tensor encoding (`torch_save`, the default, or
`safetensors`). The HuggingFace storages write checkpoints in the layout used
by Hugging Face model repositories (config + sharded weight files), letting
you save a model that can be loaded by the Hugging Face ecosystem directly.
{class}`~tensorplay.distributed.checkpoint.StorageWriter` /
{class}`~tensorplay.distributed.checkpoint.StorageReader` are the abstract
interfaces a custom backend implements.

## Planners

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.checkpoint.DefaultSavePlanner
    tensorplay.distributed.checkpoint.DefaultLoadPlanner
    tensorplay.distributed.checkpoint.SavePlanner
    tensorplay.distributed.checkpoint.LoadPlanner
    tensorplay.distributed.checkpoint.WriteItem
    tensorplay.distributed.checkpoint.WriteItemType
    tensorplay.distributed.checkpoint.ReadItem
    tensorplay.distributed.checkpoint.LoadItemType
    tensorplay.distributed.checkpoint.SavePlan
    tensorplay.distributed.checkpoint.LoadPlan
    tensorplay.distributed.checkpoint.TensorWriteData
    tensorplay.distributed.checkpoint.BytesIOWriteData
```

The planner turns a state dictionary into a list of concrete data items and
back. On save it walks the state dict, assigns each tensor a `WriteItem` that
records where the tensor lives (`TensorWriteData`) and how its shards map into
the checkpoint (`ChunkStorageMetadata`), then assembles a global `SavePlan`
from the per-rank local plans. On load it turns the metadata back into
`ReadItem`s and a `LoadPlan`. The default planners handle tensors, sharded
tensors, and stateful objects out of the box; implementing the abstract
{class}`~tensorplay.distributed.checkpoint.SavePlanner` /
{class}`~tensorplay.distributed.checkpoint.LoadPlanner` interfaces lets you
customize which values are stored and how.

## Metadata and supporting types

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.checkpoint.Metadata
    tensorplay.distributed.checkpoint.TensorStorageMetadata
    tensorplay.distributed.checkpoint.BytesStorageMetadata
    tensorplay.distributed.checkpoint.ChunkStorageMetadata
    tensorplay.distributed.checkpoint.MetadataIndex
    tensorplay.distributed.checkpoint.StorageMeta
    tensorplay.distributed.checkpoint.TensorProperties
    tensorplay.distributed.checkpoint.Stateful
    tensorplay.distributed.checkpoint.CheckpointableTensor
    tensorplay.distributed.checkpoint.CheckpointException
```

{class}`~tensorplay.distributed.checkpoint.Metadata` is the description of a
whole checkpoint: for each key in the state dictionary the
{class}`~tensorplay.distributed.checkpoint.TensorStorageMetadata` (or
{class}`~tensorplay.distributed.checkpoint.BytesStorageMetadata` for
non-tensor bytes) says what is stored, at which `MetadataIndex`, and how the
stored chunks reassemble into the full tensor. {class}`~tensorplay.distributed.checkpoint.StorageMeta`
carries the storage-level properties, and {class}`~tensorplay.distributed.checkpoint.TensorProperties`
the dtype/layout/device of the tensor being stored.

The {class}`~tensorplay.distributed.checkpoint.Stateful` protocol marks objects
that know how to serialize themselves: a value with `state_dict()` and
`load_state_dict()` methods is saved through them instead of being pickled
directly. {class}`~tensorplay.distributed.checkpoint.CheckpointableTensor` is
the interface a tensor-like type implements to participate in the checkpoint
(being shardable and reloadable by shape/dtype). Errors raised while saving or
loading surface as {class}`~tensorplay.distributed.checkpoint.CheckpointException`.

## Where to go next

- [FSDP](distributed.fsdp.md) — the sharding strategy whose state dicts
  `get_model_state_dict` / `get_optimizer_state_dict` are designed to collect.
- [distributed tensors](distributed.tensor.md) — the sharded tensor type that
  checkpoint stores and restores transparently.
- [the distributed package](distributed.md) — process groups, init, and
  collectives that the checkpoint machinery uses to coordinate ranks.