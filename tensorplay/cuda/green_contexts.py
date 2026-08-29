# mypy: allow-untyped-defs
r"""

Green context partitioning requires driver-level support this TensorPlay
build does not expose.
"""

from ._utils import _dummy_type


__all__ = ["GreenContext"]


GreenContext = _dummy_type("GreenContext")
