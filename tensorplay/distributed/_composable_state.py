import weakref
from typing import cast

from tensorplay.nn.modules.module import Module


class _State:
    pass


_module_state_mapping: weakref.WeakKeyDictionary[
    Module, weakref.ReferenceType[_State]
] = weakref.WeakKeyDictionary()


def _insert_module_state(module: Module, state: _State) -> None:
    global _module_state_mapping
    if module in _module_state_mapping:
        raise AssertionError(f"Inserting {module} more than once.")
    _module_state_mapping[module] = weakref.ref(state)


def _get_module_state(module: Module) -> _State | None:
    """
    Return the ``_State`` in ``model``.

    Given a ``module``, this API finds out if the module is also a ``_State``
    instance or if the module is managed by a composable API. If the module
    is also a ``_State``, ``module`` will be casted to ``_State` and returned.
    If it is managed by a composable API, the corresponding ``_State`` will
    be returned.
    """
    global _module_state_mapping
    if isinstance(module, _State):
        return cast(_State, module)
    else:
        if module in _module_state_mapping:
            state_ref = _module_state_mapping[module]
            state = state_ref()
            if state is None:
                raise AssertionError("State has already been garbage collected")
            return state
        else:
            return None
