import tensorplay as tp
import copy
from collections import defaultdict
from collections.abc import Iterable
from functools import wraps
from itertools import chain
from collections import OrderedDict


# These helpers intentionally keep the names and the contracts used by
# torch.optim.optimizer.  Optimizer implementations can therefore be
# mechanically ported from the installed Torch source; backend-specific work
# stays below this layer.
_global_optimizer_pre_hooks = OrderedDict()
_global_optimizer_post_hooks = OrderedDict()


class _RequiredParameter:
    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


def _get_value(value):
    return value.item() if isinstance(value, tp.Tensor) else value


def _stack_if_compiling(value):
    # TensorPlay's eager compiler has no optimizer-specific graph-stack path;
    # preserving the input is the same behavior as Torch eager mode.
    return value


def _disable_dynamo_if_unsupported(single_tensor_fn=None):
    def decorator(func):
        return func
    return decorator


def _default_to_fused_or_foreach(params, differentiable, use_fused=False):
    if differentiable:
        return False, False
    if not params:
        return bool(use_fused), not bool(use_fused)
    # TensorPlay currently exposes explicit foreach kernels only for the
    # optimizers that already have native implementations.  The optimizer
    # module may still choose its scalar source path when this returns False.
    all_cuda = all(
        p is None or (isinstance(p, tp.Tensor) and p.is_cuda)
        for p in params
    )
    return bool(use_fused and all_cuda), bool(all_cuda and not use_fused)


def _device_dtype_check_for_fused(param, cuda_unsupported=False):
    supported = ["cuda"] if not cuda_unsupported else []
    if not param.is_cuda or not param.is_floating_point():
        raise RuntimeError(
            "`fused=True` requires all the params to be floating point "
            f"Tensors of supported devices: {supported}"
        )


def _view_as_real(params, *state_and_grads):
    for index, param in enumerate(params):
        if param.is_complex():
            params[index] = tp.view_as_real(param)
            for state in state_and_grads:
                state[index] = tp.view_as_real(state[index])


def _get_scalar_dtype(is_fused=None):
    # TensorPlay has no global default-dtype API exposed to Python; its
    # optimizer scalar state follows Torch's float32 default in this build.
    return tp.float32


def _get_capturable_supported_devices(supports_xla=True):
    # TensorPlay currently has CUDA as its only graph-capturable backend.
    return ["cuda"]


def _to_scalar(value):
    if isinstance(value, tp.Tensor) and value.ndim != 0:
        return value.squeeze()
    return value


def _group_tensors_by_device_and_dtype(tensorlists):
    """Torch-compatible grouping result for foreach implementations.

    The second item in each value is the original index list, exactly as
    ``torch.utils._foreach_utils._group_tensors_by_device_and_dtype`` returns.
    """

    grouped = {}
    if not tensorlists:
        return grouped
    for index, values in enumerate(zip(*tensorlists)):
        defined = next((value for value in values if value is not None), None)
        key = None if defined is None else (defined.device, defined.dtype)
        if key not in grouped:
            grouped[key] = ([[] for _ in tensorlists], [])
        lists, indices = grouped[key]
        for target, value in zip(lists, values):
            target.append(value)
        indices.append(index)
    return grouped


def register_optimizer_step_pre_hook(hook):
    handle_id = max(_global_optimizer_pre_hooks.keys(), default=-1) + 1
    _global_optimizer_pre_hooks[handle_id] = hook

    class _Handle:
        id = handle_id

        def remove(self):
            _global_optimizer_pre_hooks.pop(handle_id, None)

    return _Handle()


def register_optimizer_step_post_hook(hook):
    handle_id = max(_global_optimizer_post_hooks.keys(), default=-1) + 1
    _global_optimizer_post_hooks[handle_id] = hook

    class _Handle:
        id = handle_id

        def remove(self):
            _global_optimizer_post_hooks.pop(handle_id, None)

    return _Handle()


def _use_grad_for_differentiable(func):
    """Mirror torch.optim's grad-mode wrapper around ``step``.

    Torch runs the complete optimizer step under ``no_grad`` by default and
    enables grad recording only when ``differentiable=True``.  The closure is
    always evaluated with grad enabled.  Keeping this policy in one wrapper
    avoids individual optimizers accidentally mixing the two modes.
    """

    @wraps(func)
    def _use_grad(self, closure=None):
        args = (self, closure)
        kwargs = {}
        for hook in chain(
            _global_optimizer_pre_hooks.values(),
            getattr(self, "_optimizer_step_pre_hooks", {}).values(),
        ):
            result = hook(self, args, kwargs)
            if result is not None:
                if not (isinstance(result, tuple) and len(result) == 2):
                    raise RuntimeError(
                        f"{func} must return None or a tuple of (new_args, new_kwargs), but got {result}."
                    )
                args, kwargs = result
        if len(args) < 2:
            raise RuntimeError("optimizer step pre-hook must preserve the closure argument")
        closure = args[1]
        differentiable = bool(self.defaults.get("differentiable", False))
        with tp.set_grad_enabled(differentiable):
            if closure is not None:
                def grad_closure():
                    with tp.enable_grad():
                        return closure()
            else:
                grad_closure = None
            result = func(args[0], grad_closure, **kwargs)
        for hook in chain(
            getattr(self, "_optimizer_step_post_hooks", {}).values(),
            _global_optimizer_post_hooks.values(),
        ):
            hook(self, args, kwargs)
        return result

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
        self.state = defaultdict(dict)
        self.param_groups = []
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

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

    def zero_grad(self, set_to_none=True):
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
                        p.grad.zero_()

    def step(self, closure=None):
        raise NotImplementedError

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
        self.defaults.setdefault('differentiable', False)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += f'\nParameter Group {i}\n'
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += f'    {key}: {group[key]}\n'
        format_string += ')'
        return format_string
