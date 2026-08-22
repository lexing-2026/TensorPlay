# Ported from torch/distributed/_composable/replicate.py.
import copy
import weakref
from collections.abc import Iterable
from typing import Any

from tensorplay.nn.modules.module import Module

from tensorplay.distributed._composable.contract import (
    RegistryItem,
    _get_registry,
    contract,
)
from tensorplay.distributed._composable_state import _State, _insert_module_state
from tensorplay.nn.parallel.distributed import DistributedDataParallel

__all__: list[str] = []

_ROOT_MODULE_PREFIX = ""


class _ReplicateState(_State):
    def __init__(self) -> None:
        super().__init__()
        self.module: Module = None  # type: ignore[assignment]
        self.has_initialized: bool = False
        self._param_names: list[str] = []
        self._no_sync: bool = False
        self._init_args: tuple[Any, ...] | None = None
        self._init_kwargs: dict[str, Any] = {}
        self._comm_hook_args: list[Any] = []

    def _collect_params(self, module, ignored_modules, ignored_params,
                        prefix=_ROOT_MODULE_PREFIX):
        # skip if managed by fully_shard API
        if _is_fully_sharded(module):
            return

        # if a module is ignored, all descendants of the module are ignored.
        if module in ignored_modules:
            return

        recurse_prefix = (
            f"{prefix}." if prefix != _ROOT_MODULE_PREFIX else _ROOT_MODULE_PREFIX
        )

        for n, p in module.named_parameters(recurse=False):
            if p not in ignored_params:
                self._param_names.append(f"{recurse_prefix}{n}")

        for name, child_module in module.named_children():
            self._collect_params(
                child_module,
                ignored_modules,
                ignored_params,
                prefix=f"{recurse_prefix}{name}",
            )

    def lazy_init(self) -> None:
        if self._init_args is None:
            raise AssertionError
        self.init(*self._init_args, **self._init_kwargs)
        self.register_comm_hook()
        self._init_args = ()
        self._init_kwargs = {}

    def init(
        self,
        module: Module,
        ignored_modules: set[Module],
        **kwargs,
    ) -> None:
        if self.has_initialized:
            return

        self.has_initialized = True
        self.module = module
        ignored_params = {p for m in ignored_modules for p in m.parameters()}
        for submodule in module.modules():
            if _is_fully_sharded(submodule):
                ignored_params.update(submodule.parameters())
        self._collect_params(module, ignored_modules, ignored_params)

        if "device_id" in kwargs:
            # replicate() supports passing device_id as Union[int, device]
            # even for CPU devices so users don't have to change code for
            # CPU/GPU runs.
            if kwargs["device_id"] is not None:
                device_id = kwargs["device_id"]
                if isinstance(device_id, str) and "cpu" in str(device_id):
                    kwargs["device_ids"] = None
                else:
                    kwargs["device_ids"] = [device_id]
            else:
                kwargs["device_ids"] = None
            kwargs.pop("device_id")

        self._ddp = DistributedDataParallel(self.module, **kwargs)
        # Weakref to the DDP instance is currently only used for testing.
        replicate.state(self.module)._ddp_weakref = weakref.ref(self._ddp)

    def register_comm_hook(self) -> None:
        for comm_args, comm_kwargs in self._comm_hook_args:
            self._ddp.register_comm_hook(*comm_args, **comm_kwargs)
        self._comm_hook_args.clear()

    def record_init_args(self, *args, **kwargs) -> None:
        self._init_args = args
        self._init_kwargs = kwargs

    def forward_pre_hook(self, module, args, kwargs):
        if self._init_args or self._init_kwargs:
            self.lazy_init()
        self._ddp.require_backward_grad_sync = not self._no_sync
        DistributedDataParallel._active_ddp_module = self._ddp \
            if hasattr(DistributedDataParallel, "_active_ddp_module") else None
        return self._ddp._pre_forward(*args, **kwargs)

    def forward_post_hook(self, module, input, output):
        DistributedDataParallel._active_ddp_module = None \
            if hasattr(DistributedDataParallel, "_active_ddp_module") else None
        return self._ddp._post_forward(output)


def unimplemented_deepcopy(*args: Any, **kwarg: Any):
    raise NotImplementedError("Deepcopy of replicated modules is not supported.")


@contract(_state_key := "__replicate_state_key__")
def replicate(
    module: Module,
    ignored_modules: Iterable[Module] | None = None,
    **kwargs,
) -> Module:
    r"""Replicates a module (torch composable-API parity).

    Args:
        module (nn.Module): module to replicate

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """

    if _is_fully_sharded(module):
        raise RuntimeError(
            "Cannot apply `replicate()` on a Module already managed by "
            "`fully_shard`"
        )

    if ignored_modules is None:
        ignored_modules = {}
    else:
        ignored_modules = set(ignored_modules)

    state = replicate.state(module)
    module.register_forward_pre_hook(state.forward_pre_hook, with_kwargs=True)
    module.register_forward_hook(state.forward_post_hook)

    state.record_init_args(module, ignored_modules, **kwargs)
    _insert_module_state(module, state)

    # Place DDP leftmost for highest priority in the method resolution order
    cls = module.__class__
    dct = {"__deepcopy__": unimplemented_deepcopy}
    new_cls = type(f"DDP{cls.__name__}", (DDP, cls), dct)
    module.__class__ = new_cls
    return module


# Attach torch's ``state`` classmethod pattern (registry-backed).
def _replicate_state_factory():
    return _ReplicateState()


def _get_state(module: Module) -> _ReplicateState:
    registry = _get_registry(module)
    state = registry.get(_state_key)
    if state is None:
        state = _ReplicateState()
        registry[_state_key] = state
    return state


replicate.state = staticmethod(_get_state)


def _is_fully_sharded(module: Module) -> bool:
    r"""Check if module is marked with fully_shard."""
    registry = _get_registry(module)
    if registry is None:
        return False
    return "fully_shard" in registry


class DDP:
    """Mixin installed by :func:`replicate` marking a replicated module."""
