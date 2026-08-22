# mypy: allow-untyped-defs
r"""Annotate kernels captured in CUDA graphs for profiler traces.

Graph capture is not supported by this TensorPlay build; the annotation
helpers are present and report availability honestly.
"""

__all__ = [
    "clear_kernel_annotations",
    "get_kernel_annotations",
    "is_available",
    "mark_kernels",
]


def is_available() -> bool:
    r"""Whether graph-kernel annotation recording is supported (always ``False`` here)."""
    return False


def mark_kernels(name, metadata=None):
    r"""Tag the GPU work captured within its scope with user metadata."""
    raise RuntimeError(
        "CUDA graphs are not supported by this TensorPlay build"
    )


def get_kernel_annotations():
    r"""Return the recorded kernel annotations mapping."""
    raise RuntimeError(
        "CUDA graphs are not supported by this TensorPlay build"
    )


def clear_kernel_annotations():
    r"""Clear all recorded kernel annotations."""
    raise RuntimeError(
        "CUDA graphs are not supported by this TensorPlay build"
    )
