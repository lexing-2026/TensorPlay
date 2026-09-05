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

    tensorplay.distributed.elastic.utils.api.get_env_variable_or_raise
    tensorplay.distributed.elastic.utils.distributed.get_free_port
    tensorplay.distributed.elastic.utils.log_level.get_log_level
    tensorplay.distributed.elastic.utils.logging.get_logger
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
    tensorplay.distributed.distributed_core.get_default_backend_for_device
```

## Post-Initialization

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed.distributed_core.Backend
    tensorplay.distributed.distributed_core.get_backend
    tensorplay.distributed.distributed_core.get_backend_config
    tensorplay.distributed.distributed_core.get_rank
    tensorplay.distributed.distributed_core.get_world_size
    tensorplay.distributed.distributed_core.set_timeout
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
    tensorplay.distributed.distributed_core.split_group
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
    tensorplay.distributed.distributed_core.irecv
    tensorplay.distributed.distributed_core.send_object_list
    tensorplay.distributed.distributed_core.recv_object_list
    tensorplay.distributed.distributed_core.batch_isend_irecv
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
    tensorplay.distributed.distributed_core.all_reduce_coalesced
    tensorplay.distributed.distributed_core.reduce
    tensorplay.distributed.distributed_core.all_gather
    tensorplay.distributed.distributed_core.all_gather_single
    tensorplay.distributed.distributed_core.all_gather_object
    tensorplay.distributed.distributed_core.all_gather_coalesced
    tensorplay.distributed.distributed_core.gather
    tensorplay.distributed.distributed_core.gather_single
    tensorplay.distributed.distributed_core.gather_object
    tensorplay.distributed.distributed_core.scatter
    tensorplay.distributed.distributed_core.scatter_object_list
    tensorplay.distributed.distributed_core.reduce_scatter
    tensorplay.distributed.distributed_core.reduce_scatter_single
    tensorplay.distributed.distributed_core.all_to_all_single
    tensorplay.distributed.distributed_core.all_to_all
    tensorplay.distributed.distributed_core.barrier
    tensorplay.distributed.distributed_core.monitored_barrier
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
    tensorplay.distributed._store.HashStore
    tensorplay.distributed._store.FileStore
    tensorplay.distributed._store.PrefixStore
```

## Watchdog (Experimental)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.distributed._watchdog.shutdown
    tensorplay.distributed._watchdog.stream_timeout
    tensorplay.distributed._watchdog.cpu_timeout
    tensorplay.distributed._watchdog.op_timeout
    tensorplay.distributed.collective_utils.all_gather_object_enforce_type
    tensorplay.distributed.launcher.api.launch_agent
```
