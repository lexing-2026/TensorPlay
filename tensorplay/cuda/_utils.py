# mypy: allow-untyped-defs
r"""Internal helpers for the ``tensorplay.cuda`` package.

``_dummy_type``, ``_ClassPropertyDescriptor``, ``classproperty`` and
``_LazySeedTracker`` are ported verbatim from ``torch._utils`` so that this
package keeps the same graceful-degradation behaviour as torch when a feature
is unavailable in the current build.
"""

from typing import Any, Callable, Optional


def _dummy_type(name: str) -> type:
    def get_err_fn(is_init: bool):
        def err_fn(obj, *args, **kwargs):
            if is_init:
                class_name = obj.__class__.__name__
            else:
                class_name = obj.__name__
            raise RuntimeError(f"Tried to instantiate dummy base class {class_name}")

        return err_fn

    return type(
        name, (object,), {"__init__": get_err_fn(True), "__new__": get_err_fn(False)}
    )


class _ClassPropertyDescriptor:
    def __init__(self, fget, fset=None):
        self.fget = fget

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        return self.fget.__get__(instance, owner)()


def classproperty(func: Callable) -> _ClassPropertyDescriptor:
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)


class _LazySeedTracker:
    # Since seeding is memory-less, only track the latest seed.
    # Note: `manual_seed_all` followed by `manual_seed` overwrites
    # the seed on current device. We track the order of **latest**
    # calls between these two API.
    def __init__(self):
        self.manual_seed_all_cb = None
        self.manual_seed_cb = None
        self.call_order = []

    def queue_seed_all(self, cb, traceback):
        self.manual_seed_all_cb = (cb, traceback)
        # update seed_all to be latest
        self.call_order = [self.manual_seed_cb, self.manual_seed_all_cb]

    def queue_seed(self, cb, traceback):
        self.manual_seed_cb = (cb, traceback)
        # update seed to be latest
        self.call_order = [self.manual_seed_all_cb, self.manual_seed_cb]

    def get_calls(self) -> list:
        return self.call_order


def _get_device_index(
    device: Any,
    optional: bool = False,
    allow_cpu: bool = False,
) -> int:
    r"""Get the device index from :attr:`device`.

    Accepts a ``tensorplay.Device``, a Python integer, a device string such as
    ``"cuda:0"``, any object with a CUDA ``device`` attribute (e.g. tensors),
    or ``None``. When :attr:`optional` is True and no index can be inferred,
    returns the index of the current CUDA device; otherwise raises.
    """
    from .._C import Device, DeviceType

    if isinstance(device, int):
        return device
    if isinstance(device, Device):
        if not allow_cpu and not device.is_cuda():
            raise ValueError(f"Expected a cuda device, but got: {device}")
        idx = device.index
        if idx is None or idx < 0:
            from . import current_device

            if optional:
                return current_device()
            if idx is None or idx < 0 and device.is_cuda():
                raise ValueError(
                    f"Expected a torch.device with a specified index or an integer, but got:{device}"
                )
        return int(idx)
    if isinstance(device, str):
        device = Device(device)
        return _get_device_index(device, optional=optional, allow_cpu=allow_cpu)
    if hasattr(device, "device"):
        return _get_device_index(
            device.device, optional=optional, allow_cpu=allow_cpu
        )
    if hasattr(device, "index"):
        return int(device.index)
    if optional:
        from . import current_device

        return current_device()
    raise ValueError(f"Expected a cuda device, but got: {device}")
