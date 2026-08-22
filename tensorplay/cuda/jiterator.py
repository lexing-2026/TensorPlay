# mypy: allow-untyped-defs
r"""jiterator-based elementwise kernels, mirroring :mod:`torch.cuda.jiterator`.

Requires NVRTC-backed jiterator support this TensorPlay build does not
expose; instantiation raises.
"""

__all__ = ["_JiteratorFunction"]


class _JiteratorFunction:
    r"""Callable that launches a jiterator-generated kernel."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "jiterator kernels are not supported by this TensorPlay build"
        )


def _create_jiterator_fn(*args, **kwargs):
    raise RuntimeError(
        "jiterator kernels are not supported by this TensorPlay build"
    )


def _create_multi_output_jiterator_fn(*args, **kwargs):
    raise RuntimeError(
        "jiterator kernels are not supported by this TensorPlay build"
    )
