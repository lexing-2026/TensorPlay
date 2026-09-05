# mypy: allow-untyped-defs
"""Graph-compiler marking helpers.

Thin facade over :mod:`tensorplay.compiler`: the marking entry points record
which callables and tensor dimensions the tracer may treat as fixed.  Under
eager execution the markers are inert metadata — the semantics of the marked
callables never change.
"""

from tensorplay.compiler import allow_in_graph  # noqa: F401


def disallow_in_graph(fn):
    """Mark ``fn`` as opaque: the tracer emits one call node for it."""
    return fn


def barriers_are_compiled() -> bool:
    return False


def is_dynamo_supported() -> bool:
    """Whether the graph tracer is available in this build."""
    try:
        import tensorplay.compiler as _compiler

        return hasattr(_compiler, "compile")
    except ImportError:
        return False


def mark_static(tensor, dim=None):
    """Mark ``tensor`` (or one dimension of it) as static for shape policies.

    Eager execution has no dynamic shape environment, so this records
    nothing; the marker stays a no-op that mirrors the compile-time API.
    """
    return tensor
