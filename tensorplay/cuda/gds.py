# mypy: allow-untyped-defs
r"""GPU Direct Storage (GDS) support, mirroring :mod:`torch.cuda.gds`."""

__all__ = ["GdsFile", "is_gds_available"]


def is_gds_available() -> bool:
    r"""Return whether GDS is available. Always ``False`` in this build."""
    return False


class GdsFile:
    r"""A GDS file handle (not supported by this TensorPlay build)."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("GDS is not supported by this TensorPlay build")
