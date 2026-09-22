"""Availability flag for the compiled ops bundled with the core library.

The vision ops (sampling, pooling, detection helpers) ship inside the core
extension and are always importable, so this flag is a constant.
"""

_HAS_OPS = True

__all__ = ["_HAS_OPS"]
