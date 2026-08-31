from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any

from tensorplay.nn.modules import module as module_api

__all__ = ["ModTracker"]


class ModTracker:
    """Track the active module path while a model executes."""

    def __init__(self) -> None:
        self.parents: set[str] = {"Global"}
        self._active_module_cnt: dict[str, int] = {}
        self._known_modules: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()
        self._seen_modules: weakref.WeakSet[Any] = weakref.WeakSet()
        self._handles: list[Any] = []
        self._stack: list[str] = []
        self._is_bw = False
        self._user_pre_fw_hook: Callable[..., Any] | None = None
        self._user_post_fw_hook: Callable[..., Any] | None = None
        self._user_pre_bw_hook: Callable[..., Any] | None = None
        self._user_post_bw_hook: Callable[..., Any] | None = None

    @property
    def is_bw(self) -> bool:
        return self._is_bw

    def get_known_fqn(self, mod: Any) -> str | None:
        return self._known_modules.get(mod)

    def register_user_hooks(
        self,
        pre_fw_hook: Callable[..., Any] | None = None,
        post_fw_hook: Callable[..., Any] | None = None,
        pre_bw_hook: Callable[..., Any] | None = None,
        post_bw_hook: Callable[..., Any] | None = None,
    ) -> None:
        hooks = (
            ("pre_fw_hook", "_user_pre_fw_hook", pre_fw_hook),
            ("post_fw_hook", "_user_post_fw_hook", post_fw_hook),
            ("pre_bw_hook", "_user_pre_bw_hook", pre_bw_hook),
            ("post_bw_hook", "_user_post_bw_hook", post_bw_hook),
        )
        for label, attr, hook in hooks:
            if hook is not None and getattr(self, attr) is not None:
                raise AssertionError(f"only one {label} can be registered")
            if hook is not None:
                setattr(self, attr, hook)

    def clear_user_hooks(self) -> None:
        self._user_pre_fw_hook = None
        self._user_post_fw_hook = None
        self._user_pre_bw_hook = None
        self._user_post_bw_hook = None

    def _get_mod_name(self, mod: Any) -> str:
        known = self._known_modules.get(mod)
        if known is not None:
            return known
        if self._stack:
            name = f"{self._stack[-1]}.{type(mod).__name__}"
        else:
            name = type(mod).__name__
        self._known_modules[mod] = name
        if mod not in self._seen_modules:
            for child_name, child in mod.named_children():
                self._known_modules[child] = f"{name}.{child_name}"
                self._get_mod_name(child)
            self._seen_modules.add(mod)
        return name

    def _fw_pre_hook(self, mod: Any, inputs: Any) -> None:
        name = self._get_mod_name(mod)
        self._active_module_cnt[name] = self._active_module_cnt.get(name, 0) + 1
        self.parents.add(name)
        self._stack.append(name)
        if self._user_pre_fw_hook is not None:
            self._user_pre_fw_hook(mod, inputs)

    def _fw_post_hook(self, mod: Any, inputs: Any, output: Any) -> None:
        name = self._get_mod_name(mod)
        if self._user_post_fw_hook is not None:
            self._user_post_fw_hook(mod, inputs, output)
        count = self._active_module_cnt.get(name, 1) - 1
        if count <= 0:
            self._active_module_cnt.pop(name, None)
            self.parents.discard(name)
        else:
            self._active_module_cnt[name] = count
        if self._stack and self._stack[-1] == name:
            self._stack.pop()

    def _bw_pre_hook(self, mod: Any, grad_input: Any) -> None:
        self._is_bw = True
        name = self._get_mod_name(mod)
        self.parents.add(name)
        if self._user_pre_bw_hook is not None:
            self._user_pre_bw_hook(mod, grad_input)

    def _bw_post_hook(self, mod: Any, grad_input: Any, grad_output: Any) -> None:
        if self._user_post_bw_hook is not None:
            self._user_post_bw_hook(mod, grad_input)
        name = self._get_mod_name(mod)
        self.parents.discard(name)

    def __enter__(self) -> "ModTracker":
        self._handles = [
            module_api.register_module_forward_pre_hook(self._fw_pre_hook),
            module_api.register_module_forward_hook(self._fw_post_hook),
        ]
        return self

    def __exit__(self, *args: Any) -> None:
        del args
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.parents = {"Global"}
        self._active_module_cnt.clear()
        self._stack.clear()
        self._is_bw = False
