(multiprocessing-best-practices)=

# Multiprocessing best practices

{mod}`tensorplay.multiprocessing` is a drop-in replacement for Python's
{mod}`python:multiprocessing` module. It supports the exact same operations,
but extends it, so that all tensors sent through a
{class}`python:multiprocessing.Queue` will have their data moved into shared
memory and will only send a handle to another process. Change the import from
`multiprocessing` to `tensorplay.multiprocessing` and every tensor put on a
queue, or otherwise pickled to another process, is transparently converted to
a shared-memory view.

:::{note}
When a {class}`~tensorplay.Tensor` is sent to another process, only the data
is shared. The tensor's autograd state is not: a tensor that requires grad
must be a leaf, and the receiving process rebuilds it from the shared data
with `requires_grad` preserved but an empty grad accumulator. Non-leaf
tensors that require grad are refused outright with a `RuntimeError`, since
autograd does not support crossing process boundaries — call `.detach()` on
them before sending if you only want to transfer the data.
:::

This allows implementation of various training methods, like Hogwild, A3C,
or any others that require asynchronous operation. See the
{ref}`strategy and spawn documentation <multiprocessing-doc>` for the full
reference of the sharing strategies.

## CPU tensors only

Cross-process sharing is implemented for CPU tensors. Sending a tensor on any
other device through a queue raises a `RuntimeError`; move such tensors to
the CPU (and usually `.detach()` them) before sending, and move them back on
the receiving side.

## CUDA in multiprocessing

The CUDA runtime cannot be re-initialized in a process forked after the
runtime was initialized: the child raises
`RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA
with multiprocessing, you must use the 'spawn' start method`. Either
initialize the accelerator only after forking, or use the `spawn` or
`forkserver` start methods, which give every child a clean initialization.

:::{note}
The start method can be set via either creating a context with
`multiprocessing.get_context(...)` or directly using
`multiprocessing.set_start_method(...)`.
:::

## Best practices and tips

### Avoiding and fighting deadlocks

There are a lot of things that can go wrong when a new process is spawned, with
the most common cause of deadlocks being background threads. If there's any
thread that holds a lock or imports a module, and `fork` is called, it's very
likely that the subprocess will be in a corrupted state and will deadlock or
fail in a different way. Note that even if you don't, Python built-in
libraries do - no need to look further than {mod}`python:multiprocessing`.
{class}`python:multiprocessing.Queue` is actually a very complex class, that
spawns multiple threads used to serialize, send and receive objects, and they
can cause aforementioned problems too. If you find yourself in such situation
try using a {class}`~python:multiprocessing.queues.SimpleQueue`, that doesn't
use any additional threads.

### Reuse buffers passed through a Queue

Remember that each time you put a {class}`~tensorplay.Tensor` into a
{class}`python:multiprocessing.Queue`, it has to be moved into shared memory.
If it's already shared, it is a no-op, otherwise it will incur an additional
memory copy that can slow down the whole process. Even if you have a pool of
processes sending data to a single one, make it send the buffers back - this
is nearly free and will let you avoid a copy when sending next batch.

### Asynchronous multiprocess training (e.g. Hogwild)

Using {mod}`tensorplay.multiprocessing`, it is possible to train a model
asynchronously, with parameters either shared all the time, or being
periodically synchronized. In the first case, we recommend sending over the
whole model object, while in the latter, we advise to only send the
`state_dict`.

We recommend using {class}`python:multiprocessing.Queue` for passing all kinds
of TensorPlay objects between processes. It is possible to e.g. inherit the
tensors and storages already in shared memory, when using the `fork` start
method, however it is very bug prone and should be used with care, and only by
advanced users. Queues, even though they're sometimes a less elegant solution,
will work properly in all cases.

:::{warning}
You should be careful about having global statements, that are not guarded
with an `if __name__ == '__main__'`. If a different start method than
`fork` is used, they will be executed in all subprocesses.
:::

#### Hogwild

A minimal Hogwild structure: the model's parameters live in one shared-memory
allocation, and every process runs its own training loop against them without
any locking.

```python
import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.multiprocessing as mp

def train(model):
    # Construct data_loader, optimizer, etc.
    for data, labels in data_loader:
        optimizer.zero_grad()
        loss_fn(model(data), labels).backward()
        optimizer.step()  # This will update the shared parameters

if __name__ == '__main__':
    num_processes = 4
    model = nn.Linear(784, 10)
    # NOTE: this is required for the `fork` method to work
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
```

## CPU in multiprocessing

Inappropriate multiprocessing can lead to CPU oversubscription, causing
different processes to compete for CPU resources, resulting in low
efficiency.

### CPU oversubscription

CPU oversubscription is a technical term that refers to a situation
where the total number of vCPUs allocated to a system exceeds the total
number of vCPUs available on the hardware.

This leads to severe contention for CPU resources. In such cases, there is
frequent switching between processes, which increases process switching
overhead and decreases overall system efficiency.

When running a training script on CPU using 4 processes, and there are
N vCPUs available on the machine, each subprocess will allocate N vCPUs
for itself, resulting in a requirement of 4*N vCPUs. However, the machine
only has N vCPUs available. Consequently, the different processes will
compete for resources, leading to frequent process switching.

The following observations indicate the presence of CPU oversubscription:

- High CPU utilization: by using the `htop` command, you can observe
  that the CPU utilization is consistently high, often reaching or
  exceeding its maximum capacity. This indicates that the demand for
  CPU resources exceeds the available physical cores, causing
  contention and competition among processes for CPU time.
- Frequent context switching with low system efficiency: processes
  compete for CPU time, and the operating system needs to rapidly
  switch between processes to allocate resources fairly. This frequent
  context switching adds overhead and reduces the overall system
  efficiency.

### Avoid CPU oversubscription

A good way to avoid CPU oversubscription is proper resource allocation.
Ensure that the number of processes or threads running concurrently does
not exceed the available CPU resources.

In this case, a solution would be to specify the appropriate number of
threads in the subprocesses. This can be achieved by setting the number
of threads for each process using the
{func}`tensorplay.set_num_threads` function in each subprocess.

Assuming there are N vCPUs on the machine and M processes will be
generated, the maximum `num_threads` value used by each process would
be `floor(N/M)`. As a general guideline, the maximum value for
`num_threads` should be `floor(N/M)` to avoid CPU oversubscription.
