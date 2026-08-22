# mypy: allow-untyped-defs
r"""NCCL collectives on CUDA tensors, mirroring :mod:`torch.cuda.nccl`.

Requires NCCL bindings this TensorPlay build does not expose; every public
name reports availability honestly via :func:`is_available`.
"""

import warnings


__all__ = ["all_reduce", "reduce", "broadcast", "all_gather", "reduce_scatter"]

SUM = 0  # ncclRedOp_t
PRODUCT = 1
MAX = 2
MIN = 3


def is_available(tensors) -> bool:
    if not hasattr(_nccl_module(), "_nccl_all_reduce"):
        return False

    devices = set()
    for tensor in tensors:
        if tensor.is_sparse:
            return False
        if not tensor.is_contiguous():
            return False
        if not tensor.is_cuda:
            return False
        device = tensor.get_device()
        if device in devices:
            return False
        devices.add(device)

    return True


def version():
    r"""
    Returns the version of the NCCL.

    This function returns a tuple containing the major, minor, and patch version numbers of the NCCL.
    The suffix is also included in the tuple if a version suffix exists.
    Returns:
        tuple: The version information of the NCCL.
    """
    ver = _nccl_module()._nccl_version()
    major = ver >> 32
    minor = (ver >> 16) & 65535
    patch = ver & 65535
    suffix = _nccl_module()._nccl_version_suffix().decode("utf-8")
    if suffix == "":
        return (major, minor, patch)
    else:
        return (major, minor, patch, suffix)


def _nccl_module():
    import tensorplay._C as C

    nccl = getattr(C, "_nccl", None)
    if nccl is None:
        warnings.warn("TensorPlay is not compiled with NCCL support", stacklevel=2)
    return nccl if nccl is not None else _MissingNCCL()


class _MissingNCCL:
    def __getattr__(self, name):
        raise RuntimeError("TensorPlay is not compiled with NCCL support")


def unique_id():
    return _nccl_module()._nccl_unique_id()


def init_rank(num_ranks, uid=None, rank=None):
    if uid is None:
        uid = unique_id()
    if rank is None:
        rank = num_ranks
    return _nccl_module()._nccl_init_rank(num_ranks, uid, rank)


def all_reduce(tensors, op=SUM):
    _check_tensors(tensors, "all_reduce")
    _nccl_module()._nccl_all_reduce(tensors)


def reduce(tensors, root=0, op=SUM):
    _check_tensors(tensors, "reduce")
    _nccl_module()._nccl_reduce(tensors, root, op)


def broadcast(tensors, root=0):
    _check_tensors(tensors, "broadcast")
    _nccl_module()._nccl_broadcast(tensors, root)


def all_gather(tensors):
    _check_tensors(tensors, "all_gather")
    _nccl_module()._nccl_all_gather(tensors)


def reduce_scatter(outputs, inputs_, op=SUM):
    if not is_available(inputs_):
        raise RuntimeError("invalid input to reduce_scatter")
    _nccl_module()._nccl_reduce_scatter(outputs, inputs_, op)


def _check_tensors(tensors, api_name):
    if not is_available(tensors):
        raise RuntimeError(f"invalid input to {api_name}")
