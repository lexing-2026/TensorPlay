import tensorplay._C as _C

__all__ = ["is_available"]


def is_available() -> bool:
    """Return whether OpenMP support was included in this build."""
    return bool(_C.has_openmp())
