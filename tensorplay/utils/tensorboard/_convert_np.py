"""Convert user inputs into numpy arrays for summary construction."""

import numpy as np

import tensorplay


def make_np(x) -> np.ndarray:
    """Coerce scalars, numpy arrays or TensorPlay tensors into numpy arrays."""
    if isinstance(x, np.ndarray):
        return x
    if np.isscalar(x):
        return np.array([x])
    if hasattr(x, "device") and hasattr(x, "detach"):
        if x.device.type == "meta":
            return np.random.randn(1)
        return _prepare_tensor(x)
    raise NotImplementedError(
        f"Got {type(x)}, but numpy array or tensorplay tensor are expected."
    )


def _prepare_tensor(x) -> np.ndarray:
    if x.dtype == tensorplay.bfloat16:
        x = x.to(tensorplay.float16)
    return x.detach().cpu().numpy()
