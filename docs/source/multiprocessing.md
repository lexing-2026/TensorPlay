```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

(multiprocessing-doc)=

# Multiprocessing package - tensorplay.multiprocessing

:::{warning}
If the main process exits abruptly (e.g. because of an incoming signal),
Python's `multiprocessing` sometimes fails to clean up its children.
It's a known caveat, so if you're seeing any resource leaks after
interrupting the interpreter, it probably means that this has just happened
to you.
:::

## Strategy management

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.multiprocessing.set_sharing_strategy
    tensorplay.multiprocessing.get_sharing_strategy
    tensorplay.multiprocessing.get_all_sharing_strategies
```

(multiprocessing-cuda-note)=

## CUDA in multiprocessing

Cross-process sharing is implemented for CPU tensors only: pickling a tensor
on any other device raises a `RuntimeError`, so move tensors to the CPU
before sending them through a queue, and move them back on the device in the
receiving process.

Using CUDA inside subprocesses additionally requires the `spawn` or
`forkserver` start method. The runtime cannot be re-initialized in a process
that was forked after initialization; such a child fails with
`RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA
with multiprocessing, you must use the 'spawn' start method`.

## Sharing strategies

This section provides a brief overview into how different sharing strategies
work. They apply to CPU tensors only; tensors on other devices cannot be
shared between processes.

### File descriptor - `file_descriptor`

:::{note}
This is the default strategy (except for macOS and Windows where it's not
supported).
:::
This strategy passes an anonymous memory segment to the receiving process as
a duplicated file descriptor. When a tensor is put on a queue, the sender
copies the data into a private memory-backed segment created with
`memfd_create` and transfers the descriptor; the receiver maps it and
rebuilds a tensor aliasing those pages. The sending tensor keeps its
original (private) storage, so the two sides are connected only through the
snapshot taken at send time. Tensors that already live in shared memory skip
this path and are re-sent by name instead.
Note that if there will be a lot of tensors shared, this strategy will keep a
large number of file descriptors open most of the time. If your system has low
limits for the number of open file descriptors, and you can't raise them, you
should use the `file_system` strategy.

### File system - `file_system`

This strategy will use file names given to `shm_open` to identify the shared
memory regions. When a tensor is sent, its storage is moved into a named
shared segment in place, and only the name travels to the receiving process.
This has a benefit of not requiring the implementation to cache
the file descriptors, but at the same time is prone to shared
memory leaks. The file can't be deleted right after its creation, because other
processes need to access it to open their views. If the processes fatally
crash, or are killed, and don't call the storage destructors, the files will
remain in the system. This is very serious, because they keep using up the
memory until the system is restarted, or they're freed manually.
To counter the problem of shared memory file leaks, {mod}`tensorplay.multiprocessing`
will keep track of all shared memory allocations so they can be deallocated
when all processes connected to them exit.

## Spawning subprocesses

:::{note}
Available for Python >= 3.4.
This depends on the `spawn` start method in Python's
`multiprocessing` package.
:::
Spawning a number of subprocesses to perform some function can be done
by creating `Process` instances and calling `join` to wait for
their completion. This approach works fine when dealing with a single
subprocess but presents potential issues when dealing with multiple
processes.
Namely, joining processes sequentially implies they will terminate
sequentially. If they don't, and the first process does not terminate,
the process termination will go unnoticed. Also, there are no native
facilities for error propagation.
The `spawn` function below addresses these concerns and takes care
of error propagation, out of order termination, and will actively
terminate processes upon detecting an error in one of them.
A `ProcessContext` is returned by `spawn` when called with `join=False`.
% This module needs to be documented. Adding here in the meantime
% for tracking purposes

## tensorplay.multiprocessing.pool

```{eval-rst}
.. py:module:: tensorplay.multiprocessing.pool
```
```{eval-rst}
.. py:module:: tensorplay.multiprocessing.queue
```
```{eval-rst}
.. py:module:: tensorplay.multiprocessing.reductions
```

