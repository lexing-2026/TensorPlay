from __future__ import annotations

import collections
import sys
import types
from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeVar

import tensorplay as tp
import tensorplay.distributed.rpc as rpc
from tensorplay.distributed import _remote_device
from tensorplay.distributed.nn.jit import instantiator
from tensorplay.distributed.rpc.internal import _internal_rpc_pickler
from tensorplay.nn import Module, Parameter

__all__ = ["RemoteModule", "interface"]

T = TypeVar("T", bound="Module")


def interface(cls: type) -> type:
    """Mark a class as a remote-method interface.

    A remote module exposes exactly the ``forward`` signature of its
    interface class; instantiation validates the marker before generating
    the remote-method bindings.
    """

    setattr(cls, "__tensorplay_interface__", True)
    return cls

_REMOTE_MODULE_PICKLED_ATTRIBUTES = (
    "on",
    "device",
    "is_device_map_set",
    "is_scriptable",
    "generated_methods",
    "module_rref",
)
_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING = (
    "training",
    "_parameters",
    "_buffers",
    "_non_persistent_buffers_set",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_is_full_backward_hook",
    "_forward_hooks",
    "_forward_pre_hooks",
    "_forward_hooks_always_called",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_modules",
    "forward_async",
    "forward",
)

_SerializedRemoteModule = collections.namedtuple(
    "_SerializedRemoteModule", _REMOTE_MODULE_PICKLED_ATTRIBUTES
)


def _create_module(module_cls: type[Module], args: tuple[Any, ...], kwargs: dict[str, Any], device: str) -> Module:
    module = module_cls(*args, **kwargs)
    if not isinstance(module, Module):
        raise ValueError(
            "module_cls(*args, **kwargs) must return a tensorplay.nn.Module, "
            f"got {type(module)!r}"
        )
    module.to(device)
    return module


def _create_module_with_interface(
    module_cls: type[Module],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    device: str,
    module_interface_cls: Any,
) -> rpc.RRef[Module]:
    del module_interface_cls
    return rpc.RRef(_create_module(module_cls, args, kwargs, device))


def _param_rrefs(module_rref: rpc.RRef[Module], recurse: bool) -> list[rpc.RRef[Parameter]]:
    return [rpc.RRef(parameter) for parameter in module_rref.local_value().parameters(recurse)]


def _remote_forward(
    module_rref: rpc.RRef[Module],
    device: str,
    is_device_map_set: bool,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
) -> Any:
    del device, is_device_map_set
    return module_rref.local_value().forward(*call_args, **call_kwargs)


def _set_training(module_rref: rpc.RRef[Module], mode: bool) -> None:
    module_rref.local_value().train(mode)


def _raise_not_supported(name: str) -> None:
    raise ValueError(f"Method {name!r} is not supported by RemoteModule")


class _RemoteModule(Module):
    def __init__(
        self,
        remote_device: str,
        module_cls: type[Module],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        _module_interface_cls: Any = None,
    ) -> None:
        super().__init__()
        enable_moving_cpu_tensors_to_cuda = self._prepare_init(remote_device)
        call_args = tuple(args or ())
        call_kwargs = dict(kwargs or {})

        if _module_interface_cls is not None:
            self.is_scriptable = True
            self._init_template(
                _module_interface_cls, enable_moving_cpu_tensors_to_cuda
            )
            rpc.rpc_async(
                self.on,
                _instantiate_template,
                (_module_interface_cls, enable_moving_cpu_tensors_to_cuda),
            ).wait()
            self.module_rref = rpc.remote(
                self.on,
                _create_module,
                (module_cls, call_args, call_kwargs, self.device),
            )
        else:
            self.is_scriptable = False
            self.generated_methods = (
                instantiator.instantiate_non_scriptable_remote_module_template()._generated_methods
            )
            self.module_rref = rpc.remote(
                self.on,
                _create_module,
                (module_cls, call_args, call_kwargs, self.device),
            )

        self._install_generated_methods()
        self._check_attribute_picklability()

    def remote_parameters(self, recurse: bool = True) -> list[rpc.RRef[Parameter]]:
        return rpc.rpc_sync(self.on, _param_rrefs, args=(self.module_rref, recurse))

    def get_module_rref(self) -> rpc.RRef[Module]:
        return self.module_rref

    def __getstate__(self) -> None:
        raise RuntimeError("RemoteModule can only be serialized by the RPC pickler")

    def __setstate__(self, state: Any) -> None:
        del state
        raise RuntimeError("RemoteModule can only be restored by the RPC pickler")

    def register_buffer(self, name: str, tensor: Any, persistent: bool = True) -> None:
        del name, tensor, persistent
        _raise_not_supported("register_buffer")

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        del name, param
        _raise_not_supported("register_parameter")

    def add_module(self, name: str, module: Module | None) -> None:
        del name, module
        _raise_not_supported("add_module")

    def apply(self, fn: Callable[[Module], None]) -> T:
        del fn
        _raise_not_supported("apply")

    def cuda(self, device: Any = None) -> T:
        del device
        _raise_not_supported("cuda")

    def ipu(self, device: Any = None) -> T:
        del device
        _raise_not_supported("ipu")

    def xpu(self, device: Any = None) -> T:
        del device
        _raise_not_supported("xpu")

    def cpu(self) -> T:
        _raise_not_supported("cpu")

    def type(self, dst_type: Any) -> T:
        del dst_type
        _raise_not_supported("type")

    def float(self) -> T:
        _raise_not_supported("float")

    def double(self) -> T:
        _raise_not_supported("double")

    def half(self) -> T:
        _raise_not_supported("half")

    def bfloat16(self) -> T:
        _raise_not_supported("bfloat16")

    def to(self, *args: Any, **kwargs: Any) -> T:
        del args, kwargs
        _raise_not_supported("to")

    def register_forward_pre_hook(self, hook: Any, *args: Any, **kwargs: Any) -> Any:
        del hook, args, kwargs
        _raise_not_supported("register_forward_pre_hook")

    def register_forward_hook(self, hook: Any, *args: Any, **kwargs: Any) -> Any:
        del hook, args, kwargs
        _raise_not_supported("register_forward_hook")

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        _raise_not_supported("state_dict")

    def load_state_dict(self, state_dict: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        del state_dict, args, kwargs
        _raise_not_supported("load_state_dict")

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        del recurse
        raise ValueError("Use remote_parameters() to access remote parameters")

    def named_parameters(self, *args: Any, **kwargs: Any) -> Iterator[tuple[str, Parameter]]:
        del args, kwargs
        _raise_not_supported("named_parameters")

    def buffers(self, recurse: bool = True) -> Iterator[Any]:
        del recurse
        _raise_not_supported("buffers")

    def named_buffers(self, *args: Any, **kwargs: Any) -> Iterator[tuple[str, Any]]:
        del args, kwargs
        _raise_not_supported("named_buffers")

    def children(self) -> Iterator[Module]:
        _raise_not_supported("children")

    def named_children(self) -> Iterator[tuple[str, Module]]:
        _raise_not_supported("named_children")

    def modules(self) -> Iterator[Module]:
        _raise_not_supported("modules")

    def named_modules(self, *args: Any, **kwargs: Any) -> Iterator[tuple[str, Module]]:
        del args, kwargs
        _raise_not_supported("named_modules")

    def train(self, mode: bool = True) -> T:
        rpc.rpc_sync(self.on, _set_training, args=(self.module_rref, bool(mode)))
        return self  # type: ignore[return-value]

    def eval(self) -> T:
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True) -> T:
        del requires_grad
        _raise_not_supported("requires_grad_")

    def zero_grad(self, set_to_none: bool = True) -> None:
        del set_to_none
        _raise_not_supported("zero_grad")

    def share_memory(self) -> T:
        _raise_not_supported("share_memory")

    def extra_repr(self) -> str:
        _raise_not_supported("extra_repr")

    def _prepare_init(self, remote_device_str: str) -> bool:
        if not rpc._is_current_rpc_agent_set():
            raise AssertionError("RemoteModule requires an initialized RPC runtime")
        descriptor = _remote_device(remote_device_str)
        current = rpc.get_worker_info()
        self.on = descriptor.worker_name() or descriptor.rank()
        if self.on is None:
            self.on = current.name
        self.device = str(descriptor.device())
        agent = rpc._get_current_rpc_agent()
        options = getattr(agent, "options", None)
        maps = getattr(options, "device_maps", {}) if options is not None else {}
        self.is_device_map_set = bool(maps.get(str(self.on), {}))
        return tp.device(self.device).type == "cuda"

    def _init_template(self, module_interface_cls: Any, enable_moving_cpu_tensors_to_cuda: bool) -> None:
        generated = instantiator.instantiate_scriptable_remote_module_template(
            module_interface_cls, enable_moving_cpu_tensors_to_cuda
        )
        self.generated_methods = generated._generated_methods

    def _check_attribute_picklability(self) -> None:
        for name in self.__dict__:
            if (
                name not in _REMOTE_MODULE_PICKLED_ATTRIBUTES
                and name not in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING
            ):
                raise AttributeError(
                    f"Attribute {name!r} must be explicitly classified for RPC serialization"
                )

    def _install_generated_methods(self) -> None:
        for method in self.generated_methods:
            method_name = method.__name__
            setattr(self, method_name, types.MethodType(method, self))

    @staticmethod
    def init_from_module_rref(
        remote_device: str,
        module_rref: rpc.RRef[Module],
        _module_interface_cls: Any = None,
    ) -> "RemoteModule":
        remote_module = object.__new__(RemoteModule)
        enable_moving = remote_module._prepare_init(remote_device)
        if _module_interface_cls is None:
            remote_module.is_scriptable = False
            remote_module.generated_methods = (
                instantiator.instantiate_non_scriptable_remote_module_template()._generated_methods
            )
        else:
            remote_module.is_scriptable = True
            remote_module._init_template(_module_interface_cls, enable_moving)
        remote_module.module_rref = module_rref
        remote_module._install_generated_methods()
        remote_module._check_attribute_picklability()
        return remote_module


class RemoteModule(_RemoteModule):
    def __init__(
        self,
        remote_device: str,
        module_cls: type[Module],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(remote_device, module_cls, args, kwargs)


def _instantiate_template(module_interface_cls: Any, enable_moving_cpu_tensors_to_cuda: bool) -> None:
    instantiator.instantiate_scriptable_remote_module_template(
        module_interface_cls, enable_moving_cpu_tensors_to_cuda
    )


def _remote_module_receiver(*attributes: Any) -> RemoteModule:
    values = _SerializedRemoteModule._make(attributes)
    module = object.__new__(RemoteModule)
    module.__dict__.update(values._asdict())
    module.module_rref = rpc.RRef._deserialize(module.module_rref)
    module._install_generated_methods()
    return module


def _remote_module_reducer(remote_module: RemoteModule) -> tuple[Any, tuple[Any, ...]]:
    values: dict[str, Any] = {}
    for name, value in remote_module.__dict__.items():
        if name == "module_rref":
            values[name] = value._serialize()
        elif name in _REMOTE_MODULE_PICKLED_ATTRIBUTES:
            values[name] = value
        elif name not in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING:
            print(
                f"RPC serialization ignored unclassified RemoteModule attribute {name!r}",
                file=sys.stderr,
            )
    return (
        _remote_module_receiver,
        tuple(values[name] for name in _REMOTE_MODULE_PICKLED_ATTRIBUTES),
    )


_internal_rpc_pickler._register_reducer(RemoteModule, _remote_module_reducer)
