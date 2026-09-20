"""Shared helpers for the suites that check tensorplay against the
reference framework.

The comparison suites convert both sides to numpy and assert elementwise
agreement; these helpers centralize the conversion, the device list and
the tensor constructor so every suite agrees on the semantics. They are
duck-typed on the expected side: anything with ``detach``/``numpy``
methods (the reference framework's tensors) flows through the same
conversion path.
"""

import numpy as np

import tensorplay as tp
from tensorplay import Tensor

__all__ = [
    "assert_reference_close",
    "from_reference",
    "reference_devices",
    "to_numpy",
]


def to_numpy(t):
    """Convert a tensor from either framework, or an array-like, to ndarray.

    Both frameworks' tensors are detached and moved to the host first;
    ndarray input passes through unchanged and other values fall back to
    :func:`numpy.asarray`.
    """
    if isinstance(t, np.ndarray):
        return t
    if isinstance(t, Tensor):
        return t.detach().cpu().numpy()
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def assert_reference_close(actual, expected, rtol=1e-5, atol=1e-6, msg=""):
    """Assert elementwise agreement between the two frameworks' outputs.

    Both sides are converted with :func:`to_numpy` and compared with
    :func:`numpy.testing.assert_allclose`, so the expected side may be a
    reference-framework tensor or a plain array-like.
    """
    np.testing.assert_allclose(to_numpy(actual), to_numpy(expected),
                               rtol=rtol, atol=atol, err_msg=msg)


def reference_devices():
    """Host-first device list: CPU plus CUDA whenever the runtime has it."""
    devs = ["cpu"]
    if tp.cuda.is_available():
        devs.append("cuda")
    return devs


def from_reference(ref_t, device, requires_grad=False):
    """Copy a reference-framework tensor, or an array-like, into a fresh
    tensorplay tensor."""
    if hasattr(ref_t, "detach"):
        ref_t = ref_t.detach().numpy()
    t = tp.tensor(np.asarray(ref_t), device=device)
    if requires_grad:
        t = t.requires_grad_(True)
    return t
