"""Registration helpers for an out-of-tree device backend.

TensorPlay keeps one device slot, ``PrivateUse1``, for a backend that lives
outside this repository.  Bringing up such a backend takes four steps:

1. In C++, against the ``p10`` headers (see
   :func:`tensorplay.utils.cpp_extension.load`): register an allocator with
   ``REGISTER_ALLOCATOR(DeviceType::PrivateUse1, ...)``, device hooks with
   ``REGISTER_PRIVATEUSE1_HOOKS(...)``, and kernels with
   ``TENSORPLAY_LIBRARY_IMPL(PrivateUse1, ...)``.  The autograd, autocast
   and vmap layers of the slot already exist, so gradients flow through the
   backend's kernels as soon as the forward kernels are registered.
2. In Python, :func:`rename_privateuse1_backend` gives the device its name
   (``"npu"``, say); ``tensorplay.device("npu:0")`` then parses, prints and
   compares like any other device.
3. Optionally, :func:`generate_methods_for_privateuse1_backend` adds the
   convenience spellings ``Tensor.npu()``, ``Tensor.is_npu`` and
   ``Module.npu()``.
4. Optionally, ``tensorplay._register_device_module("npu", module)``
   installs a runtime module as ``tensorplay.npu``.  Without one, the
   runtime serves a default module built on the device hooks
   (:class:`PrivateUse1Module`), so ``tensorplay.accelerator`` queries work
   from step 1 on.

A backend packaged as a wheel can run steps 1-4 from an entry point in the
``tensorplay.backends`` group; ``import tensorplay`` loads every such entry
point unless ``TENSORPLAY_DEVICE_BACKEND_AUTOLOAD=0``.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import tensorplay
from tensorplay import _C

__all__ = [
    "PrivateUse1Module",
    "generate_methods_for_privateuse1_backend",
    "get_privateuse1_backend_name",
    "rename_privateuse1_backend",
]

def get_privateuse1_backend_name() -> str:
    """The out-of-tree device's name (``"privateuseone"`` until renamed)."""
    return _C._get_privateuse1_backend_name()


def rename_privateuse1_backend(backend_name: str) -> None:
    """Give the out-of-tree device slot the name ``backend_name``.

    After the call, ``tensorplay.device(backend_name)`` and
    ``f"{backend_name}:1"`` parse to the slot, tensors on it report
    ``device.type == backend_name``, and ``tensorplay.autocast`` accepts the
    name as a ``device_type``.  The name is fixed for the rest of the
    process: renaming again to the same name is a no-op, to another name an
    error.  The name must be an identifier and must not be an in-tree device
    type.

    Example::

        >>> tensorplay.utils.rename_privateuse1_backend("npu")
        >>> tensorplay.device("npu:1")
        npu:1
    """
    if not isinstance(backend_name, str):
        raise TypeError(f"backend_name must be a str, got {type(backend_name).__name__}")
    _C._rename_privateuse1_backend(backend_name)


def _require_renamed(what: str) -> str:
    name = get_privateuse1_backend_name()
    if not _C._is_privateuse1_backend_registered():
        raise RuntimeError(
            f"{what} needs the out-of-tree backend to have a name; call "
            "tensorplay.utils.rename_privateuse1_backend(name) first"
        )
    return name


def _normalize_device(name: str, device: Union[int, str, "tensorplay.device", None]) -> "tensorplay.device":
    """The device of the slot that ``device`` names (current one for None)."""
    if device is None:
        index = _C._privateuse1_current_device() if _C._is_privateuse1_hooks_registered() else 0
        return tensorplay.device(name, index)
    if isinstance(device, int):
        return tensorplay.device(name, device)
    parsed = tensorplay.device(device) if isinstance(device, str) else device
    if not isinstance(parsed, tensorplay.device):
        raise TypeError(f"invalid {name} device: {device!r}")
    if parsed.type != name:
        raise ValueError(f"expected a {name} device, got {parsed}")
    return parsed


def _check_unclaimed(owner: Any, attr: str, name: str) -> None:
    if hasattr(owner, attr):
        raise RuntimeError(
            f"{owner.__name__}.{attr} already exists; the methods of the "
            f"'{name}' backend were generated before, or the name collides "
            "with an existing attribute"
        )


def _generate_tensor_methods(name: str) -> None:
    Tensor = tensorplay.Tensor
    is_attr = f"is_{name}"
    _check_unclaimed(Tensor, is_attr, name)
    _check_unclaimed(Tensor, name, name)

    def is_on_backend(self) -> bool:
        return self.device.type == name

    def to_backend(self, device=None, non_blocking: bool = False, **kwargs):
        """Copy of the tensor on the backend's device (``self`` when it
        already lives there)."""
        return self.to(_normalize_device(name, device), non_blocking=non_blocking, **kwargs)

    is_on_backend.__name__ = is_attr
    to_backend.__name__ = name
    setattr(Tensor, is_attr, property(is_on_backend))
    setattr(Tensor, name, to_backend)


def _generate_module_methods(name: str) -> None:
    Module = tensorplay.nn.Module
    _check_unclaimed(Module, name, name)

    def module_to_backend(self, device=None):
        """Move every parameter and buffer to the backend's device."""
        target = _normalize_device(name, device)
        return self._apply(lambda t: t.to(target))

    module_to_backend.__name__ = name
    setattr(Module, name, module_to_backend)


def _generate_packed_sequence_methods(name: str) -> None:
    from tensorplay.nn.utils.rnn import PackedSequence

    is_attr = f"is_{name}"
    _check_unclaimed(PackedSequence, is_attr, name)
    _check_unclaimed(PackedSequence, name, name)

    def is_on_backend(self) -> bool:
        return self.data.device.type == name

    def to_backend(self, device=None, non_blocking: bool = False):
        target = _normalize_device(name, device)
        return self.to(target, non_blocking=non_blocking)

    is_on_backend.__name__ = is_attr
    to_backend.__name__ = name
    setattr(PackedSequence, is_attr, property(is_on_backend))
    setattr(PackedSequence, name, to_backend)


def generate_methods_for_privateuse1_backend(
    for_tensor: bool = True,
    for_module: bool = True,
    for_packed_sequence: bool = True,
) -> None:
    """Add ``Tensor.<name>()``, ``Tensor.is_<name>``, ``Module.<name>()``
    and the ``PackedSequence`` counterparts for the renamed backend.

    These are spellings of ``.to(device)`` and ``.device.type == name``;
    generic code reads better written that way.  Generating twice, or
    generating a name that collides with an existing attribute, raises.
    """
    name = _require_renamed("generate_methods_for_privateuse1_backend")
    if for_tensor:
        _generate_tensor_methods(name)
    if for_module:
        _generate_module_methods(name)
    if for_packed_sequence:
        _generate_packed_sequence_methods(name)


class PrivateUse1Module:
    """Default runtime module of the out-of-tree device.

    Served as ``tensorplay.<name>`` when the backend registers no module of
    its own; every query goes to the device hooks the backend registered in
    C++ (``REGISTER_PRIVATEUSE1_HOOKS``).
    """

    def __init__(self, name: str) -> None:
        self.__name__ = f"tensorplay.{name}"
        self._name = name

    def __repr__(self) -> str:
        return f"<default device module for '{self._name}'>"

    def is_available(self) -> bool:
        return bool(_C._privateuse1_is_available())

    def is_initialized(self) -> bool:
        return bool(_C._is_privateuse1_hooks_registered())

    def device_count(self) -> int:
        return int(_C._privateuse1_device_count())

    def current_device(self) -> int:
        return int(_C._privateuse1_current_device())

    def set_device(self, device: Union[int, str, "tensorplay.device"]) -> None:
        index = _normalize_device(self._name, device).index
        _C._privateuse1_set_device(index if index >= 0 else 0)

    def synchronize(self, device: Optional[Union[int, str, "tensorplay.device"]] = None) -> None:
        index = -1 if device is None else _normalize_device(self._name, device).index
        _C._privateuse1_synchronize(index)

    def _is_in_bad_fork(self) -> bool:
        return False


def _default_device_module(name: str) -> Optional[PrivateUse1Module]:
    """The default module for the renamed slot, or None for other names."""
    if not _C._is_privateuse1_backend_registered() or name != get_privateuse1_backend_name():
        return None
    return PrivateUse1Module(name)
