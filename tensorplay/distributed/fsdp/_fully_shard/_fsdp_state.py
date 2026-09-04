"""State containers and hooks for composable sharding."""

import functools
import warnings
from contextlib import contextmanager
from typing import Any, Iterable

import tensorplay as tp
from tensorplay.autograd import Function, Variable

from .._common_utils import TrainingState, collect_grad_tensors
from ...device_mesh import DeviceMesh
from ...utils import _apply_to_tensors, _to_kwargs
from ._fsdp_common import DataParallelMeshInfo, _cast_fp_tensor, _dynamo_disable
from ._fsdp_collectives import _current_stream, _wait_event, _wait_stream
from ._fsdp_init import _init_default_mesh, _init_param_group
from ._fsdp_param import FSDPParam, ParamModuleInfo
from ._fsdp_param_group import FSDPCommContext, FSDPParamGroup

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
        self.iter_forward_root: FSDPState | None = None
        self.post_backward_final_callback_queued = False
        self.is_last_backward = True
        self.post_optim_event = None
        self.is_root = False


class FSDPState:
    """Per-module communication and parameter state."""

    def __init__(self, module: Any = None) -> None:
        self.module = module
        self._root_modules: tuple[Any, ...] = (module,) if module is not None else ()
        self._state = self
        self._fsdp_state = self
        self._training_state = TrainingState.IDLE
        self._param_group: FSDPParamGroup | None = None
        self._fsdp_param_groups: list[FSDPParamGroup] = []
        self._handle = None
        self._requires_gradient_sync = True
        self._requires_all_reduce = True
        self._reshard_after_forward = True
        self._reshard_after_backward = True
        self._auto_reshard_after_forward: bool | None = True
        self._is_root: bool | None = None
        self._initialized = False
        self._mp_policy = None
        self._forward_hooks_registered = False
        self._pre_forward_handles: list[Any] = []
        self._post_forward_handles: list[Any] = []
        self._root_pre_forward_handles: list[Any] = []
        self._post_backward_final_callback_queued = False
        self._post_optim_event = None
        self._state_ctx = FSDPStateContext()
        self._comm_ctx = FSDPCommContext()
        self._states_to_forward_prefetch: list[FSDPState] = []
        self._states_to_backward_prefetch: list[FSDPState] = []
        self._modules_to_run_forward: set[Any] = set()

    def _get_state_for_module(self, module: Any) -> "FSDPState | None":
        return getattr(module, "_fsdp_state", None)

    def _fsdp_param_group(self) -> FSDPParamGroup:
        if len(self._fsdp_param_groups) > 1:
            raise AssertionError(
                "a single parameter group is required; use _all_param_groups()"
            )
        if self._param_group is None:
            raise RuntimeError("FSDP state has not been initialized")
        return self._param_group

    def _all_param_groups(self) -> list[FSDPParamGroup]:
        if self._fsdp_param_groups:
            return self._fsdp_param_groups
        if self._param_group is not None:
            return [self._param_group]
        if self._initialized:
            return []
        raise RuntimeError("FSDP state has not been initialized")

    def init(
        self,
        modules: Iterable[Any],
        device: Any,
        mp_policy: Any,
        auto_reshard_after_forward: bool,
        shard_placement_fn: Any = None,
        post_forward_mesh_info: DataParallelMeshInfo | None = None,
        reshard_after_forward: Any = True,
        managed_modules: Iterable[Any] | None = None,
    ) -> None:
        modules = tuple(modules)
        if not modules:
            raise ValueError("FSDP state requires at least one root module")
        managed_modules = tuple(
            managed_modules if managed_modules is not None else modules
        )
        self._device = device
        if self.module is None:
            self.module = modules[0]
        self._root_modules = modules
        mesh = getattr(self, "mesh", None)
        if mesh is None:
            mesh = _init_default_mesh("cpu")
        self.mesh = mesh
        mesh_info = getattr(self, "mesh_info", None)
        if mesh_info is None:
            mesh_info = DataParallelMeshInfo(mesh, 0)
        module_prefixes: dict[int, str] = {}
        for root_module in modules:
            module_prefixes.update(
                {
                    id(candidate): prefix
                    for prefix, candidate in root_module.named_modules()
                }
            )
        params: list[FSDPParam] = []
        param_infos: dict[int, ParamModuleInfo] = {}
        seen_occurrences: set[tuple[int, str]] = set()
        ignored_params = getattr(self, "ignored_params", set())
        for module in managed_modules:
            for name, param in module.named_parameters(
                recurse=False, remove_duplicate=False
            ):
                if param in ignored_params:
                    continue
                occurrence = (id(module), str(name))
                if occurrence in seen_occurrences:
                    continue
                seen_occurrences.add(occurrence)
                prefix = module_prefixes.get(id(module), "")
                fqn = f"{prefix}.{name}" if prefix else name
                module_info = param_infos.get(id(param))
                if module_info is not None:
                    module_info.shared_modules.append(module)
                    module_info.shared_param_names.append(str(name))
                    continue
                module_info = ParamModuleInfo(module, fqn, str(name))
                param_infos[id(param)] = module_info
                params.append(
                    FSDPParam(
                        param,
                        module_info,
                        mesh_info,
                        post_forward_mesh_info=post_forward_mesh_info,
                        device=device,
                        shard_placement_fn=shard_placement_fn,
                        mp_policy=mp_policy,
                        offload_policy=getattr(self, "offload_policy", None),
                    )
                )
        self._fsdp_param_groups = []
        _init_param_group(
            self,
            params,
            modules,
            mesh_info,
            post_forward_mesh_info,
            device,
            shard_placement_fn,
            mp_policy,
            getattr(self, "offload_policy", None),
            reshard_after_forward,
        )
        self._param_group = (
            self._fsdp_param_groups[0] if self._fsdp_param_groups else None
        )
        self._handle = self._param_group
        self._mp_policy = mp_policy
        self._auto_reshard_after_forward = bool(auto_reshard_after_forward)
        self._reshard_after_forward = post_forward_mesh_info is not None
        for group in self._fsdp_param_groups:
            group._fsdp_state = self
            group._reshard_after_forward_enabled = self._reshard_after_forward
            group._reshard_after_backward_enabled = self._reshard_after_backward
        self._initialized = True
        self._register_hooks(modules)

    def _register_hooks(self, modules: Iterable[Any]) -> None:
        if self._forward_hooks_registered:
            return
        modules = list(modules)
        if len(modules) > 1:
            _register_group_forward_hooks(
                modules,
                self._pre_forward_hook,
                self._post_forward_hook,
                self._modules_to_run_forward,
                self._cast_output_dtype,
            )
        else:
            for module in modules:
                self._pre_forward_handles.append(module.register_forward_pre_hook(
                    self._pre_forward_hook, prepend=True, with_kwargs=True
                ))
                self._post_forward_handles.append(module.register_forward_hook(
                    self._post_forward_hook,
                    with_kwargs=True,
                    always_call=True,
                ))
        self._forward_hooks_registered = True

    def _pre_forward_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return self._pre_forward(module, args, kwargs)

    def _post_forward_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> Any:
        del kwargs
        return self._post_forward(module, args, output)

    def _root_pre_forward(self, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
        self._lazy_init()
        if self._state_ctx.iter_forward_root is not None:
            return args, kwargs
        self._state_ctx.iter_forward_root = self
        post_optim_event = self._state_ctx.post_optim_event
        if post_optim_event is None:
            post_optim_event = self._post_optim_event
        if post_optim_event is not None:
            groups = self._all_param_groups()
            if groups:
                groups[0]._wait_all_gather_streams_on_event(post_optim_event)
            else:
                _wait_event(post_optim_event, self._comm_ctx.all_gather_copy_in_stream)
                _wait_event(post_optim_event, self._comm_ctx.all_gather_stream)
            self._state_ctx.post_optim_event = None
            self._post_optim_event = None
        else:
            comm_ctx = self._comm_ctx
            groups = self._all_param_groups()
            device = groups[0].device if groups else self._device
            current_stream = _current_stream(device)
            _wait_stream(comm_ctx.all_gather_copy_in_stream, current_stream)
            _wait_stream(comm_ctx.all_gather_stream, current_stream)
        groups = self._all_param_groups()
        device = groups[0].device if groups else self._device
        device_type = str(getattr(device, "type", device)).split(":", 1)[0].lower()
        if device_type in {"cuda", "hpu", "xpu", "mtia"} and (args or kwargs):
            args_tuple, kwargs_tuple = _to_kwargs(args, kwargs, device, False)
            args = args_tuple[0]
            kwargs = kwargs_tuple[0]
        return args, kwargs

    def _lazy_init(self) -> None:
        if self._is_root is not None:
            return
        self._is_root = True
        self._state_ctx.is_root = True
        if self.module is None:
            raise RuntimeError("FSDP state has no root module")
        visited: set[int] = set()
        root_modules = self._root_modules or (self.module,)
        root_set = set(root_modules)
        for root_module in root_modules:
            for _, candidate in root_module.named_modules():
                state = self._get_state_for_module(candidate)
                if state is None or id(state) in visited:
                    continue
                if candidate not in root_set and state is not self:
                    if state._is_root is not None:
                        raise RuntimeError(
                            "nested FSDP state was initialized before its root state"
                        )
                    state._is_root = False
                visited.add(id(state))
                self._state_ctx.states.append(state)
                state._state_ctx = self._state_ctx
                state._shared_state = self._state_ctx
        if self._auto_reshard_after_forward:
            for root_group in self._all_param_groups():
                root_group.post_forward_mesh_info = None
                root_group._reshard_after_forward_enabled = False
            self._reshard_after_forward = False
        self._init_fqns()
        self._init_shared_state()
        self._validate_no_duplicate_params()
        for state in self._state_ctx.states or [self]:
            for group in state._all_param_groups():
                group.lazy_init()

    def _validate_no_duplicate_params(self) -> None:
        seen: set[int] = set()
        for state in self._state_ctx.states or [self]:
            for group in state._all_param_groups():
                for param in group.params:
                    identity = param._orig_param_uid
                    if identity in seen:
                        raise ValueError(
                            f"Parameter '{param._param_fqn}' is shared with a parameter "
                            "already managed by another FSDP group"
                        )
                    seen.add(identity)

    def _init_shared_state(self) -> None:
        self._shared_state = self._state_ctx
        self._comm_ctx.lazy_init(self._device)
        shared_context = self._comm_ctx
        self._comm_ctx = shared_context
        for state in self._state_ctx.states or [self]:
            state._state_ctx = self._state_ctx
            state._shared_state = self._state_ctx
            state._comm_ctx = shared_context
            groups = state._all_param_groups()
            for index, group in enumerate(groups):
                group.comm_ctx = shared_context
                group._param_group_index = index
                group._num_param_groups = len(groups)
        shared_context.reduce_scatter_max_input_buffers = max(
            (
                group.reduce_scatter_max_input_buffers
                for state in self._state_ctx.states or [self]
                for group in state._all_param_groups()
            ),
            default=1,
        )

    def _init_fqns(self) -> None:
        if self._is_root is not True:
            raise AssertionError("expected the root state to initialize parameter names")
        param_to_fsdp_param: dict[Any, FSDPParam] = {}
        module_to_groups: dict[Any, list[FSDPParamGroup]] = {}
        states = self._state_ctx.states or [self]
        for state in states:
            for group in state._all_param_groups():
                for param in group.params:
                    value = getattr(param, "sharded_param", None)
                    if value is None:
                        value = param.param
                    param_to_fsdp_param[value] = param
                    owner = param.module_info.module
                    current = getattr(owner, "_parameters", {}).get(
                        param.module_info.name
                    )
                    if current is not None:
                        param_to_fsdp_param[current] = param
                for module in group.modules:
                    module_to_groups.setdefault(module, []).append(group)
        for root_module in self._root_modules or (self.module,):
            for name, param in root_module.named_parameters():
                fsdp_param = param_to_fsdp_param.get(param)
                if fsdp_param is not None:
                    fsdp_param.module_info.fqn = name
            for module_name, module in root_module.named_modules():
                for group in module_to_groups.get(module, ()):
                    if group._module_fqn is None:
                        group._module_fqn = module_name
                    elif module_name not in group._module_fqn.split(", "):
                        group._module_fqn = f"{group._module_fqn}, {module_name}"
        self._fqns = [
            param.module_info.fqn
            for state in states
            for group in state._all_param_groups()
            for param in group.params
        ]

    @_dynamo_disable
    def _pre_forward(self, module: Any, args: Any, kwargs: Any) -> tuple[Any, Any]:
        if self._training_state == TrainingState.PRE_BACKWARD:
            for group in self._all_param_groups():
                if not group.is_unsharded():
                    group.unshard()
                    group.wait_for_unshard()
            return self._cast_forward_inputs(args, kwargs)
        self._lazy_init()
        entering_forward_pass = self._training_state != TrainingState.FORWARD
        self._training_state = TrainingState.FORWARD
        if entering_forward_pass:
            args, kwargs = self._root_pre_forward(module, args, kwargs)
        for group in self._all_param_groups():
            args, kwargs = group.pre_forward(module, args, kwargs)
        if entering_forward_pass:
            for state in self._states_to_forward_prefetch:
                for group in state._all_param_groups():
                    group._prefetch_unshard(group, "forward")
        return args, kwargs

    @_dynamo_disable
    def _post_forward(self, module: Any, input: Any, output: Any) -> Any:
        if self._training_state == TrainingState.PRE_BACKWARD:
            return self._cast_output_dtype(output)
        result = output
        for group in self._all_param_groups():
            result = group.post_forward(module, input, result)
        result = self._register_pre_backward_hook(result)
        self._training_state = TrainingState.IDLE
        if self._state_ctx.iter_forward_root is self:
            result = self._force_complete_incomplete_states(result)
            for state in self._state_ctx.states or [self]:
                for group in state._all_param_groups():
                    group.finalize_forward()
            all_gather_state = self._comm_ctx.all_gather_state
            if all_gather_state is not None:
                _wait_event(
                    all_gather_state.event,
                    self._comm_ctx.all_gather_copy_in_stream,
                )
                _wait_event(
                    all_gather_state.event,
                    self._comm_ctx.all_gather_stream,
                )
                self._comm_ctx.all_gather_state = None
            self._state_ctx.iter_forward_root = None
        return self._cast_output_dtype(result)

    def _cast_forward_inputs(self, args: Any, kwargs: Any) -> tuple[Any, Any]:
        policy = self._mp_policy
        if getattr(policy, "cast_forward_inputs", False):
            dtype = getattr(policy, "param_dtype", None)
            if dtype is not None:
                args = _cast_tree(args, dtype)
                kwargs = _cast_tree(kwargs, dtype)
        return args, kwargs

    def _cast_output_dtype(self, output: Any) -> Any:
        dtype = getattr(self._mp_policy, "output_dtype", None)
        if dtype is None:
            return output
        return _cast_tree(output, dtype)

    def _force_complete_incomplete_states(self, output: Any) -> Any:
        for state in reversed(self._state_ctx.states):
            if state is self or not state._modules_to_run_forward:
                continue
            for group in state._all_param_groups():
                output = group.post_forward(None, None, output)
            output = state._register_pre_backward_hook(output)
            state._training_state = TrainingState.IDLE
            output = state._cast_output_dtype(output)
            state._modules_to_run_forward.clear()
        return output

    @_dynamo_disable
    def _pre_backward(self, grad: Any) -> Any:
        if self._training_state == TrainingState.PRE_BACKWARD:
            return grad
        self._training_state = TrainingState.PRE_BACKWARD
        self._register_root_post_backward_final_callback()
        default_prefetch = not self._states_to_backward_prefetch
        for group in self._all_param_groups():
            group.pre_backward(default_prefetch)
        for state in self._states_to_backward_prefetch:
            for group in reversed(state._all_param_groups()):
                with group.use_training_state(TrainingState.PRE_BACKWARD):
                    group.unshard(group.unshard_async_op)
        return grad

    @_dynamo_disable
    def _root_post_backward_final_callback(self) -> None:
        state_ctx = self._state_ctx
        state_ctx.iter_forward_root = None
        states = state_ctx.states or [self]
        for state in states:
            state._modules_to_run_forward.clear()
            for group in reversed(state._all_param_groups()):
                if group._training_state != TrainingState.POST_BACKWARD:
                    group.post_backward()
                group._training_state = TrainingState.IDLE
            state._training_state = TrainingState.IDLE
            if state_ctx.is_last_backward:
                for group in state._all_param_groups():
                    group.finalize_backward()
        if state_ctx.is_last_backward:
            comm_ctx = self._comm_ctx
            comm_ctx.post_forward_order.clear()
            groups = self._all_param_groups()
            device = groups[0].device if groups else self._device
            current_stream = _current_stream(device)
            for reduce_scatter_state in comm_ctx.reduce_scatter_states:
                _wait_event(reduce_scatter_state.event, current_stream)
            comm_ctx.reduce_scatter_states.clear()
        state_ctx.post_backward_final_callback_queued = False
        for state in states:
            state._post_backward_final_callback_queued = False

    def _register_pre_backward_hook(self, output: Any) -> Any:
        if not getattr(tp, "is_grad_enabled", lambda: True)():
            return output
        for value in collect_grad_tensors(output):
            is_view = getattr(value, "_is_view", None)
            if callable(is_view) and is_view():
                warnings.warn(
                    "A sharded module returned a view tensor. An in-place operation "
                    "on this view can remove the pre-backward hook and skip parameter "
                    "gathering. Use an out-of-place operation or clone the output "
                    "before mutation.",
                    UserWarning,
                    stacklevel=2,
                )
            hook = getattr(value, "register_hook", None)
            if callable(hook):
                hook(self._pre_backward)
        active_check = getattr(
            getattr(tp, "_C", None), "_are_functorch_transforms_active", None
        )
        if callable(active_check) and active_check():
            return _apply_to_tensors(
                lambda value: RegisterPreBackwardFunction.apply(self, value)
                if not getattr(value, "requires_grad", False)
                and (
                    getattr(value, "is_floating_point", lambda: False)()
                    or getattr(value, "is_complex", lambda: False)()
                )
                else value,
                output,
            )
        return output

    def _post_backward_output(self, grad: Any) -> Any:
        return grad

    def _register_root_post_backward_final_callback(self) -> None:
        if self._state_ctx.post_backward_final_callback_queued:
            return
        self._state_ctx.post_backward_final_callback_queued = True
        self._post_backward_final_callback_queued = True
        Variable._execution_engine.queue_callback(
            self._root_post_backward_final_callback
        )

    def _reset_lazy_init(self) -> None:
        self._is_root = None

    def _reset_iter_state(self) -> None:
        if self._is_root is False:
            raise RuntimeError("reset_iter_state must be called on the root FSDP module")
        groups = self._all_param_groups()
        device = groups[0].device if groups else self._device
        current_stream = _current_stream(device)
        all_gather_state = self._comm_ctx.all_gather_state
        if all_gather_state is not None:
            _wait_event(all_gather_state.event, current_stream)
            self._comm_ctx.all_gather_state = None
        for reduce_scatter_state in self._comm_ctx.reduce_scatter_states:
            _wait_event(reduce_scatter_state.event, current_stream)
        self._comm_ctx.reduce_scatter_states.clear()
        for event in self._comm_ctx._last_post_reduce_events.values():
            _wait_event(event, current_stream)
        self._comm_ctx._last_post_reduce_events.clear()
        self._comm_ctx.post_forward_order.clear()
        states = self._state_ctx.states or [self]
        for state in states:
            state._modules_to_run_forward.clear()
            state._training_state = TrainingState.IDLE
            state._post_backward_final_callback_queued = False
            for group in state._all_param_groups():
                group._reset_iter_state()
        self._state_ctx.iter_forward_root = None
        self._state_ctx.post_backward_final_callback_queued = False


def _cast_tree(value: Any, dtype: Any) -> Any:
    return _apply_to_tensors(functools.partial(_cast_fp_tensor, dtype), value)


def _get_module_fsdp_state(module: Any) -> FSDPState | None:
    return getattr(module, "_fsdp_state", None)


def _register_group_forward_hooks(
    modules: Iterable[Any],
    pre_hook: Any,
    post_hook: Any,
    modules_to_run: Any = None,
    cast_output_dtype: Any = None,
) -> None:
    modules = list(modules)
    modules_set = set(modules)
    modules_to_run = modules_to_run if modules_to_run is not None else set()

    @_dynamo_disable
    @functools.wraps(pre_hook)
    def wrapped_pre_hook(module: Any, args: Any, kwargs: Any) -> Any:
        if not modules_to_run:
            modules_to_run.update(modules_set)
        return pre_hook(module, args, kwargs)

    def get_wrapped_post_hook(module: Any) -> Any:
        @_dynamo_disable
        @functools.wraps(post_hook)
        def wrapped_post_hook(
            hook_module: Any, input: Any, kwargs: Any, output: Any
        ) -> Any:
            if module in modules_to_run:
                modules_to_run.remove(module)
                if not modules_to_run:
                    return post_hook(hook_module, input, kwargs, output)
            if callable(cast_output_dtype):
                return cast_output_dtype(output)
            return output

        return wrapped_post_hook

    for module in modules:
        module.register_forward_pre_hook(
            wrapped_pre_hook, prepend=True, with_kwargs=True
        )
        module.register_forward_hook(
            get_wrapped_post_hook(module),
            with_kwargs=True,
            always_call=True,
        )


class RegisterPreBackwardFunction(Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(state: FSDPState, output: Any) -> Any:
        return output

    @staticmethod
    def setup_context(ctx: Any, inputs: Any, output: Any) -> None:
        ctx.state = inputs[0]

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[None, ...]:
        if len(grad_outputs) != 1:
            raise RuntimeError("pre-backward marker expects one gradient")
        return (None, ctx.state._pre_backward(grad_outputs[0]))

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        tangent = grad_inputs[1]
        if tangent is None:
            return None
        return RegisterPreBackwardFunction.apply(ctx.state, tangent)
