"""Single-process, multi-device data parallelism.

Single-process, multi-device data parallelism: the input batch is scattered
along ``dim`` across devices, a replica of the module runs on each device in
its own thread (each with its device guard and current stream), and outputs
are gathered back to ``output_device``. Replication uses ``copy.deepcopy``
"""

import copy
import itertools
import threading
from collections import OrderedDict

import tensorplay as tp

from ..modules.module import Module

__all__ = ["DataParallel", "data_parallel", "gather", "replicate", "scatter"]


def _get_device_index(device, optional=False, allow_cpu=False) -> int:
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        dev = tp.device(device)
        dtype, index = dev.type, dev.index
    else:
        dtype, index = device.type, device.index
    if isinstance(index, int) and index >= 0:
        return index
    if not allow_cpu and dtype == "cpu":
        raise ValueError(f"Expected a non cpu device, but got: {device}")
    return 0


def scatter(inputs, target_gpus, dim=0):
    r"""Slices tensors into approximately equal chunks and moves them across ``target_gpus``."""
    def scatter_map(obj):
        if isinstance(obj, tp.Tensor):
            return [
                obj[start:end].to(tp.device("cuda", target_gpus[j]))
                for j, (start, end) in enumerate(_chunks(obj.size(dim), len(target_gpus)))
            ]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for _ in range(len(target_gpus))]

    return list(itertools.chain.from_iterable(zip(*map(scatter_map, inputs))))


def _chunks(nclasses, chunks_no):
    chunk_size = (nclasses + chunks_no - 1) // chunks_no
    divs = list(range(0, nclasses, min(chunk_size, nclasses)))
    divs.append(nclasses)
    return list(zip(divs[:-1], divs[1:]))


def replicate(network, devices):
    r"""Replicates a module on several devices (deepcopy + per-device move)."""
    devices = [_get_device_index(x) for x in devices]
    replicas = [network]
    for _ in devices[1:]:
        replicas.append(copy.deepcopy(network))
    for replica, device in zip(replicas, devices):
        replica.to(tp.device("cuda", device))
    return replicas


def gather(outputs, target_device, dim=0):
    r"""Gathers tensors from different GPUs on a specified device."""
    target_device = _get_device_index(target_device, allow_cpu=True)

    def gather_map(objs):
        out = objs[0]
        if isinstance(out, tp.Tensor):
            moved = [o.to(tp.device("cuda", target_device)) for o in objs]
            if len(moved) == 1:
                return moved[0]
            return tp.cat(moved, dim)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in objs):
                raise ValueError("All dicts must have the same number of keys.")
            return type(out)(((k,), gather_map([d[k] for d in objs])) for k in out)
        if isinstance(out, (list, tuple)):
            if not all(len(out) == len(d) for d in objs):
                raise ValueError("All lists must have the same length.")
            return type(out)(map(gather_map, zip(*objs)))
        return type(out)(objs)

    return gather_map(outputs)


def parallel_apply(modules, inputs, devices=None):
    r"""Applies each `modules[i]` in parallel on separate devices.

    Arguments:
        modules (list): modules to be replicated
        inputs (list): inputs to corresponding modules
    """
    assert len(modules) == len(inputs)
    if devices is None:
        devices = [
            tp.cuda.current_device() if hasattr(tp.cuda, "current_device") else 0
            for _ in range(len(modules))
        ]
    devices = [_get_device_index(x, optional=True) for x in devices]

    lock = threading.Lock()
    results = {}
    grad_enabled = tp.is_grad_enabled()

    def _worker(i, module, input, device=None):
        try:
            with lock:
                if device is None:
                    device = tp.cuda.current_device() if hasattr(tp.cuda, "current_device") else -1
            with tp.cuda.device(device), tp.cuda.stream(
                tp.cuda.current_stream(device)
            ), tp.set_grad_enabled(grad_enabled):
                if not isinstance(input, (tuple, list)):
                    input = (input,)
                output = module(*input)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [
            threading.Thread(target=_worker, args=(i, module, input, devices[i]))
            for i, (module, input) in enumerate(zip(modules, inputs))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given by ``device_ids``.

    This is the functional version of :class:`~tensorplay.nn.DataParallel`.

    Args:
        module: the module to evaluate in parallel
        inputs: inputs to the module
        dim (int): dimension along which to split the input
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,) if inputs is not None else ()

    if device_ids is None:
        device_ids = list(range(tp.cuda.device_count())) if hasattr(tp, "cuda") else []

    if not device_ids:
        return module(*inputs, **(module_kwargs or {}))

    used_devices = [_get_device_index(x, optional=True) for x in device_ids]
    replicate_devices = used_devices[:len(used_devices)]

    if hasattr(tp, "cuda") and tp.cuda.is_available():
        src_device = replicate_devices[0]
        for param in module.parameters():
            if param is not None:
                if _get_device_index(param.device, True) != src_device:
                    raise RuntimeError(
                        f"module must have its parameters and buffers on device "
                        f"{src_device} but found one of them on device {_get_device_index(param.device, True)}"
                    )
                break

    if output_device is None:
        output_device = replicate_devices[0]

    if len(replicate_devices) == 1:
        local_inputs = [x.to(tp.device("cuda", replicate_devices[0])) for x in inputs]
        return module(*local_inputs, **(module_kwargs or {}))

    inputs_scattered = scatter(inputs, replicate_devices, dim)
    replicas = replicate(module, replicate_devices)
    outputs = parallel_apply(replicas, inputs_scattered, replicate_devices)
    return gather(outputs, output_device, dim)


class DataParallel(Module):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given :attr:`module` by
    splitting the input along the batch dimension (dimension 0). In the forward
    pass, the module is replicated on each device, each replica handles a slice
    of the input, and outputs are gathered on :attr:`output_device`.

    The batch size should be larger than the number of GPUs used. It is also
    recommended to use ``lr`` linearly scaled with the effective world size.

    Args:
        module: module to be parallelized
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0) -> None:
        super().__init__()
        if not tp.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(tp.cuda.device_count()))
        if isinstance(device_ids, int):
            device_ids = [device_ids]
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = [_get_device_index(x, optional=True) for x in device_ids]
        self.output_device = _get_device_index(output_device, allow_cpu=True)

        if len(self.device_ids) == 0:
            raise ValueError("device_ids can not be empty when cuda is available")
        if len(self.device_ids) == 1:
            self.module.to(tp.device("cuda", self.device_ids[0]))

    def forward(self, *inputs, **kwargs):
        if not self.device_ids or len(self.device_ids) == 1:
            return self.module(*inputs, **kwargs)

        inputs_scattered = scatter(inputs, self.device_ids, self.dim)
        replicas = replicate(self.module, self.device_ids[:len(inputs_scattered[0])])
        outputs = parallel_apply(replicas, inputs_scattered, self.device_ids[:len(replicas)])
        return gather(outputs, self.output_device, self.dim)

    def __getitem__(self, idx: int) -> Module:
        return self.module[idx]

    def __len__(self) -> int:
        return len(self.module)

    def train(self, mode: bool = True):
        self.module.train(mode)
        return super().train(mode)
