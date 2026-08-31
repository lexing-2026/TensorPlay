"""Lazy registration of built-in TensorPlay compiler backends."""

from __future__ import annotations

from .registry import register_backend


def register() -> None:
    from .stax import stax

    register_backend(stax, name="stax")
    # Apache-TVM is an optional dependency; the backend module itself stays
    # import-light and only validates availability at compile time.
    from .tvm import tvm as tvm_backend

    register_backend(tvm_backend, name="tvm")
