"""Base optimizer."""

import functools
import inspect
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from copy import deepcopy
from itertools import chain
from typing import Any, ParamSpec, TypeAlias, TypeVar, cast, overload

import tensorplay as tp

from tensorplay.utils._foreach_utils import (
    _get_foreach_kernels_supported_devices,
    _get_fused_kernels_supported_devices,
    _group_tensors_by_device_and_dtype as _foreach_group_tensors_by_device_and_dtype,
    Indices,
    TensorListList,
)
from tensorplay.utils.hooks import RemovableHandle


# These helpers intentionally keep the names and the contracts used by
# torch.optim.optimizer.  Optimizer implementations can therefore be
# mechanically ported from the installed Torch source; backend-specific work
# stays below this layer.
_T = TypeVar("_T")
_P = ParamSpec("_P")
R = TypeVar("R")

Args: TypeAlias = tuple[Any, ...]
Kwargs: TypeAlias = dict[str, Any]
StateDict: TypeAlias = dict[str, Any]
DeviceDict: TypeAlias = dict[tp.Device | None, tp.Tensor]
DeviceDtypeDict: TypeAlias = dict[
    tuple[tp.Device, tp.dtype] | None, tp.Tensor
]

GlobalOptimizerPreHook: TypeAlias = Callable[
    ["Optimizer", Args, Kwargs], tuple[Args, Kwargs] | None
]
GlobalOptimizerPostHook: TypeAlias = Callable[["Optimizer", Args, Kwargs], None]

__all__ = [
    "Optimizer",
    "register_optimizer_step_pre_hook",
    "register_optimizer_step_post_hook",
]
_global_optimizer_pre_hooks: dict[int, GlobalOptimizerPreHook] = OrderedDict()
_global_optimizer_post_hooks: dict[int, GlobalOptimizerPostHook] = OrderedDict()
_foreach_supported_types = [tp.Tensor]

ParamsT: TypeAlias = (
    Iterable[tp.Tensor]
    | Iterable[dict[str, Any]]
    | Iterable[tuple[str, tp.Tensor]]
)

_params_doc = r"""params (iterable): iterable of parameters or named_parameters to optimize
            or iterable of dicts defining parameter groups. When using named_parameters,
            all parameters in all groups should be named"""

_maximize_doc = r"""maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)"""

_foreach_doc = r"""foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. Note that the foreach implementation uses
            ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
            being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
            parameters through the optimizer at a time or switch this flag to False (default: None)"""

_fused_doc = r"""fused (bool, optional): whether the fused implementation is used.
            Currently, tensorplay.float64, tensorplay.float32, tensorplay.float16, and
            tensorplay.bfloat16 are supported. (default: None)"""

_capturable_doc = r"""capturable (bool, optional): whether this instance is safe to
            capture in a graph, whether for CUDA graphs or for tensorplay.compile support.
            Tensors are only capturable when on supported accelerators. Passing True can
            impair ungraphed performance, so if you don't intend to graph capture this
            instance, leave it False (default: False)"""

_differentiable_doc = r"""differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step() function
            runs in a tensorplay.no_grad() context. Setting to True can impair performance,
            so leave it False if you don't intend to run autograd through this instance
            (default: False)"""


def _record_function(name):
    """Use TensorPlay's profiler record function when the backend exposes it."""

    profiler = getattr(tp, "profiler", None)
    if profiler is None:
        profiler = getattr(getattr(tp, "_C", None), "profiler", None)
    record_function = getattr(profiler, "record_function", None)
    if record_function is None:
        return nullcontext()
    return record_function(name)


class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


def _get_value(value):
    # item is significantly faster than a CPU tensor in eager mode.
    if isinstance(value, tp.Tensor) and tp.compiler.is_compiling():
        return value
    return value.item() if isinstance(value, tp.Tensor) else value


def _stack_if_compiling(value):
    if tp.compiler.is_compiling():
        return tp.stack(cast(list[tp.Tensor], value))
    return value


def _disable_dynamo(func):
    """Keep a stateful optimizer helper outside the active Stax capture."""

    @functools.wraps(func)
    def disabled(*args, **kwargs):
        disable_capture = getattr(tp.compiler, "_disable_capture", None)
        if disable_capture is None:
            return func(*args, **kwargs)
        with disable_capture():
            return func(*args, **kwargs)

    disabled._tensorplay_disable_stax = True
    return disabled


def _disable_dynamo_if_unsupported(single_tensor_fn=None):
    # Keep the single-tensor function in the module globals for TorchScript-
    # compatible callers that still inspect this decorator's closure.
    if single_tensor_fn is not None:
        globals()[single_tensor_fn.__name__] = single_tensor_fn

    def decorator(func):
        disabled_func = _disable_dynamo(func)
        parameters = inspect.signature(func).parameters
        has_state_steps = True
        try:
            state_steps_index = list(parameters.keys()).index("state_steps")
        except ValueError:
            has_state_steps = False
            state_steps_index = -1

        @functools.wraps(func)
        def maybe_fallback(*args, **kwargs):
            if (
                tp.compiler.is_compiling()
                and (
                    (
                        not kwargs.get("capturable", False)
                        and has_state_steps
                        and (arg := args[state_steps_index])
                        and isinstance(arg, Sequence)
                        and arg[0].device.type in {"cuda", "xpu"}
                    )
                    or (
                        "state_steps" in kwargs
                        and (kwarg := kwargs["state_steps"])
                        and isinstance(kwarg, Sequence)
                        and kwarg[0].device.type in {"cuda", "xpu"}
                    )
                )
            ):
                return disabled_func(*args, **kwargs)
            return func(*args, **kwargs)

        return maybe_fallback

    return decorator


def _default_to_fused_or_foreach(params, differentiable, use_fused=False):
    if differentiable:
        return False, False
    fused_supported_devices = _get_fused_kernels_supported_devices()
    # Match Torch's default selection exactly: foreach is an implicit
    # accelerator path, while CPU falls back to the single-tensor route unless
    # the optimizer has an explicit native CPU kernel of its own.
    foreach_supported_devices = _get_foreach_kernels_supported_devices()
    fused = bool(use_fused) and all(
        param is None
        or (
            isinstance(param, tuple(_foreach_supported_types))
            and param.device.type in fused_supported_devices
            and param.is_floating_point()
        )
        for param in params
    )
    foreach = not fused and all(
        param is None
        or (
            isinstance(param, tuple(_foreach_supported_types))
            and param.device.type in foreach_supported_devices
        )
        for param in params
    )
    return fused, foreach


def _device_dtype_check_for_fused(param, cuda_unsupported=False):
    supported = _get_fused_kernels_supported_devices()
    if cuda_unsupported and "cuda" in supported:
        supported.remove("cuda")
    if param.device.type not in supported or not param.is_floating_point():
        raise RuntimeError(
            "`fused=True` requires all the params to be floating point Tensors of "
            f"supported devices: {supported} but {param.dtype} and {param.device.type}"
        )


def _view_as_real(params, *state_and_grads):
    for index, param in enumerate(params):
        if param.is_complex():
            params[index] = tp.view_as_real(param)
            for state in state_and_grads:
                state[index] = tp.view_as_real(state[index])


def _get_scalar_dtype(is_fused=None):
    if is_fused:
        return tp.float32
    return tp.float64 if tp.get_default_dtype() == tp.float64 else tp.float32


def _get_capturable_supported_devices(supports_xla=True):
    """Return the device type list that supports capturable optimizer."""

    # TensorPlay currently has CUDA as its only graph-capturable backend.
    return ["cuda"]


def _to_scalar(value):
    if isinstance(value, tp.Tensor) and value.dim() != 0:
        return value.squeeze()
    return value


def _group_tensors_by_device_and_dtype(tensorlists, with_indices=False):
    if tp.compiler.is_compiling():
        return {
            (None, None): (
                tensorlists,
                list(range(len(tensorlists[0])))
            )
        }
    return _foreach_group_tensors_by_device_and_dtype(tensorlists, with_indices)


def register_optimizer_step_pre_hook(hook: GlobalOptimizerPreHook) -> RemovableHandle:
    handle = RemovableHandle(_global_optimizer_pre_hooks)
    _global_optimizer_pre_hooks[handle.id] = hook
    return handle


def register_optimizer_step_post_hook(hook: GlobalOptimizerPostHook) -> RemovableHandle:
    handle = RemovableHandle(_global_optimizer_post_hooks)
    _global_optimizer_post_hooks[handle.id] = hook
    return handle


def _use_grad_for_differentiable(func):
    """Mirror torch.optim's grad-mode wrapper around ``step``.

    Torch runs the complete optimizer step under ``no_grad`` by default and
    enables grad recording only when ``differentiable=True``.  The closure is
    always evaluated with grad enabled.  Keeping this policy in one wrapper
    avoids individual optimizers accidentally mixing the two modes.
    """

    @functools.wraps(func)
    def _use_grad(self, closure=None):
        differentiable = bool(self.defaults.get("differentiable", False))
        with tp.set_grad_enabled(differentiable):
            if closure is not None:
                def grad_closure():
                    with tp.enable_grad():
                        return closure()
            else:
                grad_closure = None
            disable_capture = getattr(tp.compiler, "_disable_capture", None)
            if disable_capture is None:
                return func(self, grad_closure)
            # Torch places graph breaks immediately around optimizer.step so
            # AOTAutograd does not functionalize the in-place optimizer state
            # mutation into the model graph.  Stax uses the same boundary.
            with disable_capture():
                return func(self, grad_closure)

    return _use_grad


class Optimizer:
    r"""
    Base class for optimizers.

    Args:
        params (iterable): an iterable of :class:`Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        self.defaults = defaults
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()
        self._patch_step_function()
        if isinstance(params, tp.Tensor):
            raise TypeError(
                "params argument given to the optimizer should be an iterable "
                f"of Tensors or dicts, but got {type(params)}"
            )
        self.state = defaultdict(dict)
        self.param_groups = []
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)
        self._warned_capturable_if_run_uncaptured = True

    @_disable_dynamo
    def add_param_group(self, param_group):
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, but got {type(param_group)}")

        params = param_group['params']
        if isinstance(params, tp.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in a set will change between runs. '
                            'Please use a list instead.')
        else:
            param_group['params'] = list(params)

        extracted_param_tensors = []
        extracted_param_names = []
        for param in param_group['params']:
            if isinstance(param, tuple) and len(param) == 2:
                extracted_param_names.append(param[0])
                extracted_param_tensors.append(param[1])
            else:
                extracted_param_tensors.append(param)
        param_group['params'] = extracted_param_tensors
        if extracted_param_names:
            if len(extracted_param_names) != len(extracted_param_tensors):
                raise ValueError(
                    "all optimizer params should be with/without names. Some param names are missing"
                )
            param_group['param_names'] = extracted_param_names

        for param in param_group['params']:
            if not isinstance(param, tp.Tensor):
                raise TypeError("optimizer can only optimize Tensors, but one of the params is " + str(type(param)))
            if not self.defaults.get("differentiable", None) and not (
                param.is_leaf or getattr(param, "retains_grad", False)
            ):
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    f"parameter group didn't specify a value of required optimization parameter {name}"
                )
            param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            raise ValueError("optimizer contains a parameter group with duplicate parameters")

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))
            if ('param_names' in param_group) != ('param_names' in group):
                raise ValueError(
                    "all optimizer param groups should be with/without names"
                )

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    @staticmethod
    def _new_handle(hooks, hook_id):
        class _Handle:
            id = hook_id

            def remove(self):
                hooks.pop(hook_id, None)

        return _Handle()

    def register_step_pre_hook(self, hook):
        hook_id = max(self._optimizer_step_pre_hooks.keys(), default=-1) + 1
        self._optimizer_step_pre_hooks[hook_id] = hook
        return self._new_handle(self._optimizer_step_pre_hooks, hook_id)

    def register_step_post_hook(self, hook):
        hook_id = max(self._optimizer_step_post_hooks.keys(), default=-1) + 1
        self._optimizer_step_post_hooks[hook_id] = hook
        return self._new_handle(self._optimizer_step_post_hooks, hook_id)

    def register_state_dict_pre_hook(self, hook, prepend=False):
        hook_id = max(self._optimizer_state_dict_pre_hooks.keys(), default=-1) + 1
        self._optimizer_state_dict_pre_hooks[hook_id] = hook
        if prepend:
            self._optimizer_state_dict_pre_hooks.move_to_end(hook_id, last=False)
        return self._new_handle(self._optimizer_state_dict_pre_hooks, hook_id)

    def register_state_dict_post_hook(self, hook, prepend=False):
        hook_id = max(self._optimizer_state_dict_post_hooks.keys(), default=-1) + 1
        self._optimizer_state_dict_post_hooks[hook_id] = hook
        if prepend:
            self._optimizer_state_dict_post_hooks.move_to_end(hook_id, last=False)
        return self._new_handle(self._optimizer_state_dict_post_hooks, hook_id)

    def register_load_state_dict_pre_hook(self, hook, prepend=False):
        hook_id = max(self._optimizer_load_state_dict_pre_hooks.keys(), default=-1) + 1
        self._optimizer_load_state_dict_pre_hooks[hook_id] = hook
        if prepend:
            self._optimizer_load_state_dict_pre_hooks.move_to_end(hook_id, last=False)
        return self._new_handle(self._optimizer_load_state_dict_pre_hooks, hook_id)

    def register_load_state_dict_post_hook(self, hook, prepend=False):
        hook_id = max(self._optimizer_load_state_dict_post_hooks.keys(), default=-1) + 1
        self._optimizer_load_state_dict_post_hooks[hook_id] = hook
        if prepend:
            self._optimizer_load_state_dict_post_hooks.move_to_end(hook_id, last=False)
        return self._new_handle(self._optimizer_load_state_dict_post_hooks, hook_id)

    @_disable_dynamo
    def zero_grad(self, set_to_none=True):
        foreach = bool(
            self.defaults.get("foreach", False)
            or self.defaults.get("fused", False)
        )
        grouped_grads = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        if foreach and not p.grad.is_sparse:
                            grouped_grads.setdefault(
                                (p.grad.device, p.grad.dtype), []
                            ).append(p.grad)
                        else:
                            p.grad.zero_()
        if foreach:
            for grads in grouped_grads.values():
                tp._foreach_zero_(grads)

    def step(self, closure=None):
        raise NotImplementedError

    @_disable_dynamo
    def state_dict(self):
        for hook in self._optimizer_state_dict_pre_hooks.values():
            hook(self)
        # Pack state and param_groups into a dictionary
        # We need to map parameters to ids because parameters are objects,
        # and we need to store ids in the state dict

        param_mappings = {}
        start_index = 0

        def pack_group(group):
            packed = {k: v for k, v in group.items() if k != 'params'}
            packed['params'] = []
            nonlocal start_index
            for index, param in enumerate(group['params'], start_index):
                if param is None:
                    continue
                param_mappings.setdefault(id(param), index)
                packed['params'].append(param_mappings[id(param)])
            start_index += len(packed['params'])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        
        packed_state = {}
        for p, p_state in self.state.items():
            if isinstance(p, tp.Tensor) and id(p) in param_mappings:
                packed_state[param_mappings[id(p)]] = p_state
            elif not isinstance(p, tp.Tensor):
                packed_state[p] = p_state
        
        state_dict = {
            'state': packed_state,
            'param_groups': param_groups,
        }
        for hook in self._optimizer_state_dict_post_hooks.values():
            result = hook(self, state_dict)
            if result is not None:
                state_dict = result
        return state_dict

    @_disable_dynamo
    def load_state_dict(self, state_dict):
        # Shallow copy to avoid modifying the input
        state_dict = state_dict.copy()

        for hook in self._optimizer_load_state_dict_pre_hooks.values():
            result = hook(self, state_dict)
            if result is not None:
                state_dict = result
        
        # Validate state_dict
        groups = self.param_groups
        # Torch deep-copies the group metadata because it rewrites the saved
        # ``params`` lists in place below.  TensorPlay tensors support the
        # same deepcopy contract as tensors used by torch optimizers.
        saved_groups = copy.deepcopy(state_dict['param_groups'])

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group that doesn't match the size of optimizer's group")

        # Update parameter groups
        id_map = dict(
            zip(
                (p_id for group in saved_groups for p_id in group['params']),
                (p for group in groups for p in group['params']),
            )
        )

        def _cast(param, value, param_id=None, key=None):
            if isinstance(value, tp.Tensor):
                if key == "step":
                    capturable = False
                    fused = False
                    for group in state_dict["param_groups"]:
                        if param_id in group["params"]:
                            capturable = group.get("capturable", False)
                            fused = group.get("fused", False)
                            break
                    if capturable or fused:
                        return value.to(dtype=tp.float32, device=param.device)
                    return value
                if param.is_floating_point():
                    return value.to(dtype=param.dtype, device=param.device)
                return value.to(device=param.device)
            if isinstance(value, dict):
                return {
                    k: _cast(param, v, param_id=param_id, key=k)
                    for k, v in value.items()
                }
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                return type(value)(
                    _cast(param, v, param_id=param_id) for v in value
                )
            return value

        # Update state
        new_state = defaultdict(dict)
        for p_id, s in state_dict['state'].items():
            if p_id in id_map:
                param = id_map[p_id]
                new_state[param] = _cast(
                    param, s, param_id=p_id
                )
            else:
                new_state[p_id] = s

        def update_group(group, new_group):
            new_group['params'] = group['params']
            if 'param_names' in group and 'param_names' not in new_group:
                new_group['param_names'] = group['param_names']
            return new_group

        param_groups = [
            update_group(group, saved_group)
            for group, saved_group in zip(groups, saved_groups)
        ]
        self.__setstate__({
            'state': new_state,
            'param_groups': param_groups,
        })
        for hook in self._optimizer_load_state_dict_post_hooks.values():
            hook(self)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._optimizer_step_pre_hooks = getattr(
            self, '_optimizer_step_pre_hooks', OrderedDict()
        )
        self._optimizer_step_post_hooks = getattr(
            self, '_optimizer_step_post_hooks', OrderedDict()
        )
        self._optimizer_state_dict_pre_hooks = getattr(
            self, '_optimizer_state_dict_pre_hooks', OrderedDict()
        )
        self._optimizer_state_dict_post_hooks = getattr(
            self, '_optimizer_state_dict_post_hooks', OrderedDict()
        )
        self._optimizer_load_state_dict_pre_hooks = getattr(
            self, '_optimizer_load_state_dict_pre_hooks', OrderedDict()
        )
        self._optimizer_load_state_dict_post_hooks = getattr(
            self, '_optimizer_load_state_dict_post_hooks', OrderedDict()
        )
        self._patch_step_function()
        self.defaults.setdefault('differentiable', False)

    def _accelerator_graph_capture_health_check(self):
        """Compatibility hook for Torch's graph-capture health check.

        TensorPlay validates the eager capturable contract when each optimizer
        resolves its parameter group.  The backend does not expose a public
        ``current_stream().is_capturing()`` query at this layer, so there is no
        additional host-side probe to perform here.
        """

    _cuda_graph_capture_health_check = _accelerator_graph_capture_health_check

    def _optimizer_step_code(self):
        """Profiler integration point matching ``torch.optim.Optimizer``."""

    @staticmethod
    def profile_hook_step(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self = cast(Optimizer, args[0])
            profile_name = f"Optimizer.step#{self.__class__.__name__}.step"
            with _record_function(profile_name):
                for pre_hook in chain(
                    _global_optimizer_pre_hooks.values(),
                    self._optimizer_step_pre_hooks.values(),
                ):
                    result = pre_hook(self, args, kwargs)
                    if result is not None:
                        if isinstance(result, tuple) and len(result) == 2:
                            args, kwargs = result
                        else:
                            raise RuntimeError(
                                f"{func} must return None or a tuple of "
                                f"(new_args, new_kwargs), but got {result}."
                            )

                output = func(*args, **kwargs)
                self._optimizer_step_code()
                for post_hook in chain(
                    self._optimizer_step_post_hooks.values(),
                    _global_optimizer_post_hooks.values(),
                ):
                    post_hook(self, args, kwargs)
                return output

        return wrapper

    @staticmethod
    def _group_tensors_by_device_and_dtype(tensorlistlist, with_indices=False):
        if not tensorlistlist:
            return {}
        if tp.compiler.is_compiling():
            return {
                (None, None): (
                    tensorlistlist,
                    list(range(len(tensorlistlist[0]))),
                )
            }
        return _foreach_group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=with_indices
        )

    @staticmethod
    def _process_value_according_to_param_policy(
        param, value, param_id, param_groups, key=None
    ):
        if not isinstance(value, tp.Tensor):
            return value
        fused = False
        capturable = False
        for group in param_groups:
            if param_id in group['params']:
                fused = group.get('fused', False)
                capturable = group.get('capturable', False)
                break
        if key == 'step':
            if capturable or fused:
                return value.to(dtype=tp.float32, device=param.device)
            return value
        if param.is_floating_point():
            return value.to(dtype=param.dtype, device=param.device)
        return value.to(device=param.device)

    def _patch_step_function(self):
        self._zero_grad_profile_name = (
            f"Optimizer.zero_grad#{self.__class__.__name__}.zero_grad"
        )
        hooked = getattr(self.__class__.step, "hooked", None)
        if not hooked:
            self.__class__.step = self.profile_hook_step(self.__class__.step)
            self.__class__.step.hooked = True

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += f'\nParameter Group {i}\n'
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += f'    {key}: {group[key]}\n'
        format_string += ')'
        return format_string
