```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# tensorplay.cuda


```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.cuda.StreamContext
    tensorplay.cuda.can_device_access_peer
    tensorplay.cuda.check_error
    tensorplay.cuda.current_blas_handle
    tensorplay.cuda.current_solver_handle
    tensorplay.cuda.current_device
    tensorplay.cuda.current_stream
    tensorplay.cuda.cudart
    tensorplay.cuda.default_stream
    tensorplay.cuda.device
    tensorplay.cuda.device_count
    tensorplay.cuda.device_memory_used
    tensorplay.cuda.device_of
    tensorplay.cuda.get_arch_list
    tensorplay.cuda.get_device_capability
    tensorplay.cuda.get_device_name
    tensorplay.cuda.get_device_properties
    tensorplay.cuda.get_gencode_flags
    tensorplay.cuda.get_stream_from_external
    tensorplay.cuda.get_sync_debug_mode
    tensorplay.cuda.init
    tensorplay.cuda.ipc_collect
    tensorplay.cuda.is_available
    tensorplay.cuda.is_bf16_supported
    tensorplay.cuda.is_initialized
    tensorplay.cuda.is_tf32_supported
    tensorplay.cuda.memory_usage
    tensorplay.cuda.set_device
    tensorplay.cuda.set_stream
    tensorplay.cuda.set_sync_debug_mode
    tensorplay.cuda.stream
    tensorplay.cuda.synchronize
    tensorplay.cuda.utilization
    tensorplay.cuda.temperature
    tensorplay.cuda.power_draw
    tensorplay.cuda.clock_rate
    tensorplay.cuda.AcceleratorError
    tensorplay.cuda.OutOfMemoryError
```

## Random Number Generator

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.cuda.random.get_rng_state
    tensorplay.cuda.random.get_rng_state_all
    tensorplay.cuda.random.set_rng_state
    tensorplay.cuda.random.set_rng_state_all
    tensorplay.cuda.random.manual_seed
    tensorplay.cuda.random.manual_seed_all
    tensorplay.cuda.random.seed
    tensorplay.cuda.random.seed_all
    tensorplay.cuda.random.initial_seed
```

## Streams and events

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.cuda.streams.Stream
    tensorplay.cuda.streams.ExternalStream
    tensorplay.cuda.streams.Event
```

## Graphs (beta)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.cuda.graphs.is_current_stream_capturing
    tensorplay.cuda.graphs.graph_pool_handle
    tensorplay.cuda.graphs.CUDAGraph
    tensorplay.cuda.graphs.graph
    tensorplay.cuda.graphs.make_graphed_callables
    tensorplay.cuda.graphs.export_dot
```

## Graph Kernel Annotations (prototype)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.cuda.graph_annotations.is_available
    tensorplay.cuda.graph_annotations.mark_kernels
    tensorplay.cuda.graph_annotations.get_kernel_annotations
    tensorplay.cuda.graph_annotations.clear_kernel_annotations
```

(cuda-memory-management)=

## Memory management

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.cuda.memory.caching_allocator_disabled
    tensorplay.cuda.memory.caching_allocator_enable
    tensorplay.cuda.memory.use_mem_pool
    tensorplay.cuda.nccl.version
    tensorplay.cuda.profiler.profile
    tensorplay.cuda.profiler.start
    tensorplay.cuda.profiler.stop
```

## NVIDIA Tools Extension (NVTX)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.cuda.nvtx.mark
    tensorplay.cuda.nvtx.range_push
    tensorplay.cuda.nvtx.range_pop
    tensorplay.cuda.nvtx.range
    tensorplay.cuda.nvtx.range_end
    tensorplay.cuda.nvtx.range_start
```

## GPUDirect Storage (prototype)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.cuda.gds.GdsFile
```

## Green Contexts (experimental)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.cuda.green_contexts.GreenContext
    tensorplay.cuda.nccl.is_available
```

## TensorPlay-specific additions

```{eval-rst}
.. currentmodule:: tensorplay.cuda
```
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    CUDAPluggableAllocator
    CudaError
    DeferredCudaCallError
    Device
    MemPool
    caching_allocator_alloc
    caching_allocator_delete
    change_current_allocator
    classproperty
    empty_cache
    get_allocator_backend
    get_per_process_memory_fraction
    host_memory_stats
    host_memory_stats_as_nested_dict
    is_gds_available
    list_gpu_processes
    max_memory_allocated
    max_memory_reserved
    mem_get_info
    memory_allocated
    memory_reserved
    memory_snapshot
    memory_stats
    memory_stats_as_nested_dict
    memory_summary
    reset_accumulated_host_memory_stats
    reset_accumulated_memory_stats
    reset_peak_host_memory_stats
    reset_peak_memory_stats
    set_per_process_memory_fraction
```

