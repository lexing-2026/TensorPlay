import types
from typing import Any

import tensorplay._C


class _ClassNamespace(types.ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__("torch.classes" + name)
        self.name = name

    def __getattr__(self, attr: str) -> Any:
        proxy = tensorplay._C._get_custom_class_python_wrapper(self.name, attr)
        if proxy is None:
            raise RuntimeError(f"Class {self.name}.{attr} not registered!")
        return proxy


class _Classes(types.ModuleType):
    __file__ = "_classes.py"

    def __init__(self) -> None:
        super().__init__("torch.classes")

    def __getattr__(self, name: str) -> _ClassNamespace:
        namespace = _ClassNamespace(name)
        setattr(self, name, namespace)
        return namespace

    @property
    def loaded_libraries(self) -> Any:
        return tensorplay.ops.loaded_libraries

    def load_library(self, path: str) -> None:
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom classes with the TensorPlay JIT runtime. This allows dynamically
        loading custom classes. For this, you should compile your class
        and the static registration code into a shared library object, and then
        call ``tensorplay.classes.load_library('ops/custom.lib')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``tensorplay.classes.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Args:
            path (str): A path to a shared library to load.
        """
        tensorplay.ops.load_library(path)


# The classes "namespace"
classes = _Classes()