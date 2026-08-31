from __future__ import annotations

import importlib.abc
import importlib.util
import inspect
import sys
from typing import Any

from .templates.remote_module_template import get_remote_module_template

_FILE_PREFIX = "_remote_module_"


class _StringLoader(importlib.abc.SourceLoader):
    def __init__(self, data: str) -> None:
        self.data = data

    def get_source(self, fullname: str) -> str:
        del fullname
        return self.data

    def get_data(self, path: str) -> bytes:
        del path
        return self.data.encode("utf-8")

    def get_filename(self, fullname: str) -> str:
        return f"<{fullname}>.py"

    def path_stats(self, path: str) -> dict[str, int]:
        del path
        raise OSError("generated module has no filesystem stats")

    def exec_module(self, module: Any) -> None:
        source = self.get_source(module.__name__)
        code = compile(source, self.get_filename(module.__name__), "exec", dont_inherit=True)
        exec(code, module.__dict__)


def _interface_method(module_interface: Any) -> inspect.Signature:
    if not getattr(module_interface, "__tensorplay_interface__", False):
        raise AssertionError("interface must be marked with the interface marker")
    method = getattr(module_interface, "forward", None)
    if method is None:
        raise AssertionError("interface must define forward")
    signature = inspect.signature(method)
    parameters = list(signature.parameters.values())
    if not parameters or parameters[0].name != "self":
        raise AssertionError("interface forward must define self first")
    return signature.replace(parameters=parameters[1:])


def get_arg_return_types_from_interface(module_interface: Any) -> tuple[str, str, str]:
    signature = _interface_method(module_interface)
    source_parameters: list[str] = []
    has_varargs = any(
        parameter.kind is parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    inserted_keyword_separator = False
    for parameter in signature.parameters.values():
        if parameter.kind is parameter.KEYWORD_ONLY and not has_varargs and not inserted_keyword_separator:
            source_parameters.append("*")
            inserted_keyword_separator = True
        source_parameters.append(str(parameter))
    arg_types = ", ".join(source_parameters)
    positional: list[str] = []
    varargs: list[str] = []
    keyword_values: list[str] = []
    varkwargs: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind is parameter.VAR_POSITIONAL:
            varargs.append(parameter.name)
        elif parameter.kind is parameter.VAR_KEYWORD:
            varkwargs.append(parameter.name)
        elif parameter.kind is parameter.KEYWORD_ONLY:
            keyword_values.append(f"{parameter.name!r}: {parameter.name}")
        else:
            positional.append(parameter.name)
    if positional:
        call_args = f"tuple([{', '.join(positional)}])"
    else:
        call_args = "tuple()"
    for name in varargs:
        call_args = f"{call_args} + tuple({name})"
    if keyword_values:
        call_kwargs = "{" + ", ".join(keyword_values) + "}"
    else:
        call_kwargs = "dict()"
    for name in varkwargs:
        call_kwargs = f"dict({call_kwargs}, **{name})"
    return arg_types, call_args, call_kwargs


def _do_instantiate_remote_module_template(
    generated_module_name: str,
    values: dict[str, str],
    enable_moving_cpu_tensors_to_cuda: bool,
) -> Any:
    existing = sys.modules.get(generated_module_name)
    if existing is not None:
        return existing
    source = get_remote_module_template(enable_moving_cpu_tensors_to_cuda).format(**values)
    loader = _StringLoader(source)
    spec = importlib.util.spec_from_loader(generated_module_name, loader)
    if spec is None:
        raise AssertionError("unable to create generated module specification")
    module = importlib.util.module_from_spec(spec)
    module.__loader__ = loader
    sys.modules[generated_module_name] = module
    loader.exec_module(module)
    return module


def instantiate_scriptable_remote_module_template(
    module_interface_cls: Any,
    enable_moving_cpu_tensors_to_cuda: bool = True,
) -> Any:
    arg_types, call_args, call_kwargs = get_arg_return_types_from_interface(module_interface_cls)
    qualified_name = f"{module_interface_cls.__module__}.{module_interface_cls.__qualname__}"
    generated_name = f"{_FILE_PREFIX}{qualified_name.replace('.', '_')}"
    return _do_instantiate_remote_module_template(
        generated_name,
        {
            "arg_types": arg_types,
            "args": call_args,
            "kwargs": call_kwargs,
        },
        enable_moving_cpu_tensors_to_cuda,
    )


def instantiate_non_scriptable_remote_module_template() -> Any:
    return _do_instantiate_remote_module_template(
        f"{_FILE_PREFIX}non_scriptable",
        {
            "arg_types": "*args, **kwargs",
            "args": "tuple(args)",
            "kwargs": "dict(kwargs)",
        },
        True,
    )
