"""Helpers shared by compiler backends.

* :func:`aot_autograd` builds a backend from forward/backward compilers: the
  captured graph is traced at dispatcher level into forward and backward
  graphs (:mod:`tensorplay._stax.aot_autograd`) and each is handed to its
  compiler.
* :func:`device_from_inputs` / :func:`dtype_from_inputs` pick the device and
  dtype a backend should target from its example inputs.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Callable, Iterable
from typing import Any

import tensorplay

log = logging.getLogger(__name__)

__all__ = [
    "AotAutograd",
    "aot_autograd",
    "counters",
    "device_from_inputs",
    "dtype_from_inputs",
]

#: Per-component event counts (``counters["aot_autograd"]["ok"]``).
counters: collections.defaultdict[str, collections.Counter[str]] = collections.defaultdict(
    collections.Counter
)

_AOT_KWARGS = frozenset(
    {
        "fw_compiler",
        "bw_compiler",
        "inference_compiler",
        "partition_fn",
        "decompositions",
        "keep_inference_input_mutations",
    }
)


class AotAutograd:
    """Backend callable that compiles through ahead-of-time autograd."""

    def __init__(self, **kwargs: Any) -> None:
        unknown = set(kwargs) - _AOT_KWARGS
        if unknown:
            raise TypeError(f"unexpected aot_autograd option(s): {sorted(unknown)}")
        if "fw_compiler" not in kwargs:
            raise TypeError("aot_autograd requires fw_compiler")
        self.__name__ = "compiler_fn"
        self.kwargs = kwargs

    def __call__(self, gm: Any, example_inputs: list[Any], **kwargs: Any) -> Callable[..., Any]:
        if kwargs:
            log.warning("aot_autograd-based backend ignoring extra kwargs %s", kwargs)

        from .aot_autograd import aot_module_simplified

        options = dict(self.kwargs)
        decompositions = options.get("decompositions")
        if callable(decompositions):
            # A zero-argument loader keeps heavy tables out of import time.
            options["decompositions"] = decompositions()
        bw_compiler = options.get("bw_compiler") or options["fw_compiler"]
        options["bw_compiler"] = _without_capture(bw_compiler)
        options["inference_compiler"] = options.get("inference_compiler") or options["fw_compiler"]

        counters["aot_autograd"]["total"] += 1
        try:
            compiled = aot_module_simplified(gm, example_inputs, **options)
        except Exception:
            counters["aot_autograd"]["not_ok"] += 1
            raise
        counters["aot_autograd"]["ok"] += 1
        return compiled


def _without_capture(compiler: Callable[..., Any]) -> Callable[..., Any]:
    """Keep the backward compiler and its product out of graph capture."""

    if getattr(compiler, "_tensorplay_without_capture", False):
        return compiler

    def compile_backward(*args: Any, **kwargs: Any) -> Any:
        compiled = tensorplay.compiler.disable(compiler)(*args, **kwargs)
        wrapped = tensorplay.compiler.disable(compiled)
        if getattr(compiled, "_boxed_call", False):
            wrapped._boxed_call = True  # type: ignore[attr-defined]
        return wrapped

    compile_backward._tensorplay_without_capture = True  # type: ignore[attr-defined]
    return compile_backward


def aot_autograd(**kwargs: Any) -> AotAutograd:
    """Backend from ``fw_compiler`` (plus optional ``bw_compiler``,
    ``inference_compiler``, ``partition_fn``, ``decompositions``,
    ``keep_inference_input_mutations``)."""

    return AotAutograd(**kwargs)


def device_from_inputs(example_inputs: Iterable[Any]) -> Any:
    for value in example_inputs:
        if isinstance(value, tensorplay.Tensor):
            return value.device
    return tensorplay.device("cpu")


def dtype_from_inputs(example_inputs: Iterable[Any]) -> Any:
    for value in example_inputs:
        if isinstance(value, tensorplay.Tensor):
            return value.dtype
    return tensorplay.float32
