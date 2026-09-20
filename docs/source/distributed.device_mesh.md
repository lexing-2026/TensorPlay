# tensorplay.distributed.device_mesh

A *device mesh* is the execution context for
[distributed tensors](distributed.tensor.md). It is an n-dimensional array
whose entries are global ranks: the value at coordinates
`(i, j, ...)` is the rank of the process holding that position of the mesh.
The mesh creates one process group per dimension, so collective operations
can run independently along each dimension — for example a 2D mesh
`(2, 4)` can do data parallelism along its first dimension and tensor
parallelism along the second.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.device_mesh.init_device_mesh
    tensorplay.distributed.device_mesh.DeviceMesh
```

## Creating a mesh

{func}`~tensorplay.distributed.device_mesh.init_device_mesh` creates a
`DeviceMesh` from a device type and a tuple of mesh dimensions:

```python
from tensorplay.distributed.device_mesh import init_device_mesh

# requires an initialized process group, e.g. via dist.init_process_group()
mesh = init_device_mesh("cuda", mesh_shape=(2, 4),
                        mesh_dim_names=("dp", "tp"))
```

Semantics follow the SPMD model: `mesh_shape` (and, when given,
`mesh_dim_names`) must be identical across all ranks, and the function blocks
until every rank has joined. Each mesh dimension can be given a name via
`mesh_dim_names`; the names are reflected in
{attr}`~tensorplay.distributed.device_mesh.DeviceMesh.mesh_dim_names` and can
be used to address that dimension when sharding a
[distributed tensor](distributed.tensor.md).

## Querying the mesh

A `DeviceMesh` answers the usual questions about rank placement:

- {func}`~tensorplay.distributed.device_mesh.DeviceMesh.get_rank` — the global
  rank at a given coordinate (or the local rank of this process).
- {func}`~tensorplay.distributed.device_mesh.DeviceMesh.get_local_rank` — the
  ordinal of this process *within* a mesh dimension.
- {func}`~tensorplay.distributed.device_mesh.DeviceMesh.get_coordinate` — the
  coordinates of this process in the mesh.
- {func}`~tensorplay.distributed.device_mesh.DeviceMesh.get_group` /
  {func}`~tensorplay.distributed.device_mesh.DeviceMesh.get_all_groups` — the
  underlying process groups, per mesh dimension.

## Where to go next

- [distributed tensors](distributed.tensor.md) — `DTensor` values that live
  on a mesh.
- [tensor parallelism](distributed.tensor.parallel.md) — sharding module
  weights across the mesh.
- [the distributed package](distributed.md) — the process-group model and
  collectives the mesh is built on.