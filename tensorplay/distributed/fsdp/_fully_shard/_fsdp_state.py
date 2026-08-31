"""State containers and hooks for composable sharding."""

from contextlib import contextmanager
from typing import Any, Iterable

from .._common_utils import TrainingState
from ...device_mesh import DeviceMesh, init_device_mesh
from ._fsdp_common import DataParallelMeshInfo
from ._fsdp_param import FSDPParam, ParamModuleInfo
from ._fsdp_param_group import FSDPParamGroup

__all__ = [
    "FSDPStateContext",
    "FSDPState",
    "RegisterPreBackwardFunction",
    "_get_module_fsdp_state",
    "_register_group_forward_hooks",
]


class FSDPStateContext:
    def __init__(self) -> None:
        self.states: list[FSDPState] = []
        self.is_root = False


class FSDPState:
    """Per-module communication and parameter state."""

    def __init__(self, module: Any = None) -> None:
        self.module = module
        self._state = self
        self._fsdp_state = self
        self._training_state = TrainingState.IDLE
        self._param_group: FSDPParamGroup | None = None
        self._handle = None
        self._requires_gradient_sync = True
        self._requires_all_reduce = True
        self._reshard_after_forward = True
        self._reshard_after_backward = True
        self._mp_policy = None
        self._forward_hooks_registered = False

    def _get_state_for_module(self, module: Any) -> "FSDPState":
        return getattr(module, "_fsdp_state", self)

    def _fsdp_param_group(self) -> FSDPParamGroup:
        if self._param_group is None:
            raise RuntimeError("FSDP state has not been initialized")
        return self._param_group

    def init(self, modules: Iterable[Any], device: Any, mp_policy: Any, auto_reshard_after_forward: bool | int | None) -> None:
        modules = list(modules)
        if self.module is None:
            self.module = modules[0]
        mesh = getattr(self, "mesh", None)
        if mesh is None:
            mesh = init_device_mesh("cpu", (1,))
        self.mesh = mesh
        mesh_info = DataParallelMeshInfo(mesh, 0)
        params: list[FSDPParam] = []
        for module in modules:
            for name, param in module.named_parameters(recurse=False):
                params.append(FSDPParam(param, ParamModuleInfo(module, name, name), mesh_info, device=device, mp_policy=mp_policy))
        self._param_group = FSDPParamGroup(params, modules, mesh_info, None, device, None, mp_policy, getattr(self, "offload_policy", None))
        self._handle = self._param_group
        self._mp_policy = mp_policy
        self._reshard_after_forward = bool(auto_reshard_after_forward) if auto_reshard_after_forward is not None else True
        self._param_group._reshard_after_forward_enabled = self._reshard_after_forward
        self._register_hooks(modules)

    def _register_hooks(self, modules: Iterable[Any]) -> None:
        if self._forward_hooks_registered:
            return
        for module in modules:
            module.register_forward_pre_hook(self._pre_forward_hook, with_kwargs=True)
            module.register_forward_hook(self._post_forward_hook, with_kwargs=True)
        self._forward_hooks_registered = True

    def _pre_forward_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        self._training_state = TrainingState.FORWARD
        return self._pre_forward(module, args, kwargs)

    def _post_forward_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> Any:
        del kwargs
        return self._post_forward(module, args, output)

    def _root_pre_forward(self, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
        return self._pre_forward(module, args, kwargs)

    def _lazy_init(self) -> None:
        self._fsdp_param_group().lazy_init()

    def _validate_no_duplicate_params(self) -> None:
        seen: set[int] = set()
        for param in self._fsdp_param_group().params:
            identity = id(param.param)
            if identity in seen:
                raise ValueError("a parameter cannot occur twice in one sharded group")
            seen.add(identity)

    def _init_shared_state(self) -> None:
        self._shared_state = FSDPStateContext()

    def _init_fqns(self) -> None:
        self._fqns = [param.module_info.fqn for param in self._fsdp_param_group().params]

    def _pre_forward(self, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
        del module
        self._lazy_init()
        self._training_state = TrainingState.FORWARD
        return self._fsdp_param_group().pre_forward(self.module, args, kwargs)

    def _post_forward(self, module: Any, input: Any, output: Any) -> Any:
        del module
        result = self._fsdp_param_group().post_forward(self.module, input, output)
        self._training_state = TrainingState.IDLE
        return result

    def _cast_forward_inputs(self, args: Any, kwargs: Any) -> tuple[Any, Any]:
        return self._fsdp_param_group().pre_forward(self.module, args, kwargs)

    def _cast_output_dtype(self, output: Any) -> Any:
        dtype = getattr(self._mp_policy, "output_dtype", None)
        if dtype is None:
            return output
        return _cast_tree(output, dtype)

    def _force_complete_incomplete_states(self, output: Any) -> Any:
        del output
        return None

    def _pre_backward(self, grad: Any) -> Any:
        del grad
        self._training_state = TrainingState.PRE_BACKWARD
        self._fsdp_param_group().pre_backward(None)
        return None

    def _root_post_backward_final_callback(self) -> None:
        self._training_state = TrainingState.IDLE
        self._fsdp_param_group().finalize_backward()

    def _register_pre_backward_hook(self, output: Any) -> Any:
        if isinstance(output, tuple):
            return tuple(self._register_pre_backward_hook(value) for value in output)
        hook = getattr(output, "register_hook", None)
        if hook is not None and getattr(output, "requires_grad", False):
            hook(self._pre_backward)
        return output

    def _register_root_post_backward_final_callback(self) -> None:
        return None

    def _reset_iter_state(self) -> None:
        self._training_state = TrainingState.IDLE
        if self._param_group is not None:
            self._param_group._reset_iter_state()


def _cast_tree(value: Any, dtype: Any) -> Any:
    if hasattr(value, "is_floating_point") and value.is_floating_point():
        return value.to(dtype=dtype)
    if isinstance(value, tuple):
        return tuple(_cast_tree(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast_tree(item, dtype) for item in value]
    if isinstance(value, dict):
        return {key: _cast_tree(item, dtype) for key, item in value.items()}
    return value


def _get_module_fsdp_state(module: Any) -> FSDPState | None:
    return getattr(module, "_fsdp_state", None)


def _register_group_forward_hooks(modules: Iterable[Any], pre_hook: Any, post_hook: Any, modules_to_run: Any = None, cast_output_dtype: Any = None) -> None:
    selected = set(modules_to_run or modules)
    for module in modules:
        if module not in selected:
            continue
        module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        module.register_forward_hook(post_hook, with_kwargs=True)


class RegisterPreBackwardFunction:
    @staticmethod
    def forward(state: FSDPState, output: Any) -> Any:
        state._pre_backward(None)
        return output

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> None:
        ctx.state = inputs[0]

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[None, ...]:
        ctx.state._root_post_backward_final_callback()
        return (None,) * (len(grad_outputs) + 1)

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> tuple[Any, ...]:
        del ctx
        return grad_inputs
