# mypy: allow-untyped-defs
"""Debug-only primitives.

The reference stack registers a storage-loading debug primitive that reads
tensor bytes back from an external content store.  This framework does not
ship a content-store reader, so the registration point exists (the registry
accepts the ``debug_prims::load_tensor``) but the kernel reports that the
content store feature is unavailable.
"""

import contextlib
from collections.abc import Generator
from typing import Any

import tensorplay

__all__ = ["load_tensor_reader", "register_debug_prims"]

LOAD_TENSOR_READER: Any | None = None


@contextlib.contextmanager
def load_tensor_reader(loc: str) -> Generator[None, None, None]:
    """Context under which the debug loader reads from a content store.

    Not implemented by this framework: entering the reader raises so callers
    relying on stored-content debugging fail loudly instead of silently
    producing empty tensors.
    """
    raise NotImplementedError(
        "load_tensor_reader is not supported: this framework has no content-store reader"
    )
    yield  # unreachable; satisfies the generator contract


def _load_tensor_impl(loc: str) -> Any:
    raise NotImplementedError(
        "debug prims load_tensor is not supported: no content-store reader is available"
    )


def _load_tensor_meta(loc: str) -> Any:
    # A lone scalar placeholder so schema validation has metadata to work with.
    return tensorplay.empty((), dtype=tensorplay.float32)


def register_debug_prims() -> None:
    """Register the debug primitive loads into the operator registry."""
    try:
        if tensorplay.library.has_op("debug_prims::load_tensor"):
            return
        prim_def = tensorplay.library.custom_op(
            "debug_prims::load_tensor",
            _load_tensor_impl,
            schema="debug_prims::load_tensor(str loc) -> Tensor",
        )
        prim_def.register_fake(_load_tensor_meta)
    except Exception:
        # Registration is best-effort: if the library layers reject the
        # debug-only op, the package still imports cleanly.
        pass
