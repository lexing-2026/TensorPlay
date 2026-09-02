```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# Distributed communication package - tensorplay.distributed

# Distributed communication package - process groups

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

    tensorplay.distributed.distributed_core.is_available
    tensorplay.distributed.distributed_core.init_process_group
    tensorplay.distributed.device_mesh.init_device_mesh
    tensorplay.distributed.distributed_core.is_initialized
    tensorplay.distributed.distributed_core.is_mpi_available
    tensorplay.distributed.distributed_core.is_nccl_available
    tensorplay.distributed.distributed_core.is_gloo_available
    tensorplay.distributed.distributed_core.batch_isend_irecv
    tensorplay.distributed.distributed_core.destroy_process_group
    tensorplay.distributed.distributed_core.irecv
```

## Post-Initialization

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.distributed_core.Backend
    tensorplay.distributed.distributed_core.get_backend
    tensorplay.distributed.distributed_core.get_rank
    tensorplay.distributed.distributed_core.get_world_size
```

## Groups

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.distributed_core.new_group
    tensorplay.distributed.distributed_core.get_group_rank
    tensorplay.distributed.distributed_core.get_global_rank
    tensorplay.distributed.distributed_core.get_process_group_ranks
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

    tensorplay.distributed.distributed_core.send
    tensorplay.distributed.distributed_core.recv
    tensorplay.distributed.distributed_core.isend
    tensorplay.distributed.distributed_core.send_object_list
    tensorplay.distributed.distributed_core.recv_object_list
    tensorplay.distributed.distributed_core.P2POp
```

## Collective functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.distributed_core.broadcast
    tensorplay.distributed.distributed_core.broadcast_object_list
    tensorplay.distributed.distributed_core.all_reduce
    tensorplay.distributed.distributed_core.reduce
    tensorplay.distributed.distributed_core.all_gather
    tensorplay.distributed.distributed_core.all_gather_object
    tensorplay.distributed.distributed_core.gather
    tensorplay.distributed.distributed_core.gather_object
    tensorplay.distributed.distributed_core.scatter
    tensorplay.distributed.distributed_core.scatter_object_list
    tensorplay.distributed.distributed_core.reduce_scatter
    tensorplay.distributed.distributed_core.all_to_all_single
    tensorplay.distributed.distributed_core.all_to_all
    tensorplay.distributed.distributed_core.barrier
    tensorplay.distributed.distributed_core.Work
    tensorplay.distributed.distributed_core.ReduceOp
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
