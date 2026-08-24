```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# Distributed communication package - tensorplay.distributed

# Distributed communication package - torch.distributed

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.rendezvous.register_rendezvous_handler
    tensorplay.distributed.algorithms.model_averaging.utils.average_parameters
    tensorplay.distributed.algorithms.model_averaging.utils.average_parameters_or_parameter_groups
    tensorplay.distributed.algorithms.model_averaging.utils.get_params_to_average
```

## Initialization

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.distributed_c10d.is_available
    tensorplay.distributed.distributed_c10d.init_process_group
    tensorplay.distributed.device_mesh.init_device_mesh
    tensorplay.distributed.distributed_c10d.is_initialized
    tensorplay.distributed.distributed_c10d.is_mpi_available
    tensorplay.distributed.distributed_c10d.is_nccl_available
    tensorplay.distributed.distributed_c10d.is_gloo_available
    tensorplay.distributed.distributed_c10d.batch_isend_irecv
    tensorplay.distributed.distributed_c10d.destroy_process_group
    tensorplay.distributed.distributed_c10d.irecv
```

## Post-Initialization

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.distributed_c10d.Backend
    tensorplay.distributed.distributed_c10d.get_backend
    tensorplay.distributed.distributed_c10d.get_rank
    tensorplay.distributed.distributed_c10d.get_world_size
```

## Groups

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.distributed_c10d.new_group
    tensorplay.distributed.distributed_c10d.get_group_rank
    tensorplay.distributed.distributed_c10d.get_global_rank
    tensorplay.distributed.distributed_c10d.get_process_group_ranks
```

## DeviceMesh

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.device_mesh.DeviceMesh
```

## Point-to-point communication

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.distributed_c10d.send
    tensorplay.distributed.distributed_c10d.recv
    tensorplay.distributed.distributed_c10d.isend
    tensorplay.distributed.distributed_c10d.send_object_list
    tensorplay.distributed.distributed_c10d.recv_object_list
    tensorplay.distributed.distributed_c10d.P2POp
```

## Collective functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.distributed_c10d.broadcast
    tensorplay.distributed.distributed_c10d.broadcast_object_list
    tensorplay.distributed.distributed_c10d.all_reduce
    tensorplay.distributed.distributed_c10d.reduce
    tensorplay.distributed.distributed_c10d.all_gather
    tensorplay.distributed.distributed_c10d.all_gather_object
    tensorplay.distributed.distributed_c10d.gather
    tensorplay.distributed.distributed_c10d.gather_object
    tensorplay.distributed.distributed_c10d.scatter
    tensorplay.distributed.distributed_c10d.scatter_object_list
    tensorplay.distributed.distributed_c10d.reduce_scatter
    tensorplay.distributed.distributed_c10d.all_to_all_single
    tensorplay.distributed.distributed_c10d.all_to_all
    tensorplay.distributed.distributed_c10d.barrier
    tensorplay.distributed.distributed_c10d.Work
    tensorplay.distributed.distributed_c10d.ReduceOp
```

## Distributed Key-Value Store

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed._store.Store
    tensorplay.distributed._store.TCPStore
    tensorplay.distributed._store.FileStore
```

## Launch utility

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.launch.launch
    tensorplay.distributed.launch.main
    tensorplay.distributed.launch.parse_args
```

## Watchdog (Experimental)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.collective_utils.all_gather_object_enforce_type
```

