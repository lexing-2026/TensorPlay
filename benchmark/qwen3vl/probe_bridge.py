"""How a framework tensor reaches an accelerator kernel written elsewhere.

A kernel takes raw device addresses, not a tensor type, so the framework's
tensor only has to give up the address of its storage for the kernel to write
into it.  Three ways to get there are tried in order of how much they assume:
handing the tensor to the kernel as it is, exchanging it through the standard
description, and describing it by hand against its own storage.

    python probe_bridge.py
"""

from __future__ import annotations

import ctypes

import numpy as np


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    pass


_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))
DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DELETER),
]


def _keep_alive(*_args):
    """The storage belongs to the framework tensor; the view owns nothing."""


_KEEPALIVE = _DELETER(_keep_alive)
_PYCAPSULE_NEW = ctypes.pythonapi.PyCapsule_New
_PYCAPSULE_NEW.restype = ctypes.py_object
_PYCAPSULE_NEW.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_CODES = {
    np.dtype("float16"): (2, 16),
    np.dtype("float32"): (2, 32),
    np.dtype("int8"): (0, 8),
    np.dtype("int32"): (0, 32),
    np.dtype("int64"): (0, 64),
}


def capsule_for(data_ptr, shape, dtype, device_type, device_id, keep):
    """A description of storage a framework tensor already owns."""
    code, bits = _CODES[np.dtype(dtype)]
    shape_arr = (ctypes.c_int64 * len(shape))(*shape)
    managed = DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(data_ptr)
    managed.dl_tensor.device = DLDevice(device_type, device_id)
    managed.dl_tensor.ndim = len(shape)
    managed.dl_tensor.dtype = DLDataType(code, bits, 1)
    managed.dl_tensor.shape = shape_arr
    managed.dl_tensor.strides = None
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None
    managed.deleter = _KEEPALIVE
    keep.append((managed, shape_arr))
    return _PYCAPSULE_NEW(ctypes.cast(ctypes.pointer(managed), ctypes.c_void_p),
                          b"dltensor", None)


import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def scale_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(y_ptr + i, tl.load(x_ptr + i, mask=m) * 2.0, mask=m)


def main():
    import torch
    import torch_mlu  # noqa: F401
    import tensorplay as tp

    import tensorplay_mlu

    tensorplay_mlu.load()

    print("=== 1. the framework tensor handed to the kernel as it is ===")
    n = 256
    x = tp.arange(0, n, device="mlu", dtype=tp.float16)
    y = tp.zeros(n, device="mlu", dtype=tp.float16)
    print(f"  framework tensor: {x.shape} {x.dtype} data_ptr=0x{x.data_ptr():x}")

    # The kernel reads an address, so what matters is whether it accepts the
    # framework's tensor where it expects a tensor-like argument.
    try:
        scale_kernel[(1,)](x, y, n, BLOCK=n)
        tp.mlu.synchronize()
        got = y.float().cpu().numpy()
        expect = np.arange(n, dtype=np.float32) * 2.0
        print(f"  accepted: values {got[:3].tolist()} ... {got[-1]}")
        print(f"  matches: {np.allclose(got, expect)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  rejected: {type(exc).__name__}: {str(exc)[:160]}")

    print("\n=== 2. the code each library stamps on its own accelerator tensors ===")
    peer = torch.empty(4, dtype=torch.float16, device="mlu")
    try:
        print(f"  peer __dlpack_device__: {peer.__dlpack_device__()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  peer __dlpack_device__ raised: {type(exc).__name__} {str(exc)[:100]}")
    try:
        print(f"  peer __dlpack__        : {peer.__dlpack__()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  peer __dlpack__ raised : {type(exc).__name__} {str(exc)[:100]}")
    try:
        own = x.__dlpack_device__() if hasattr(x, "__dlpack_device__") else None
        print(f"  framework __dlpack_device__: {own}")
    except Exception as exc:  # noqa: BLE001
        print(f"  framework __dlpack_device__ raised: {type(exc).__name__} {str(exc)[:100]}")

    print("\n=== 3. a description built by hand, read by the peer library ===")
    for device_type in (0, 1, 2, 12, 13, 15, 16, 256, 257):
        keep = []
        cap = capsule_for(x.data_ptr(), tuple(x.shape), "float16", device_type, 0, keep)
        try:
            view = torch.from_dlpack(cap)
            print(f"  device_type={device_type:4d}: accepted -> {view.device} "
                  f"dtype={view.dtype} ptr_match={view.data_ptr() == x.data_ptr()}")
        except Exception as exc:  # noqa: BLE001
            print(f"  device_type={device_type:4d}: {type(exc).__name__} "
                  f"{str(exc)[:90]}")

    print("\n=== 4. a description built by hand, read back by the framework ===")
    keep = []
    cap = capsule_for(peer.data_ptr(), tuple(peer.shape), "float16", 13, 0, keep)
    try:
        back = tp.from_dlpack(cap)
        print(f"  framework accepted: {back.shape} {back.dtype} {back.device} "
              f"ptr_match={back.data_ptr() == peer.data_ptr()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  framework raised: {type(exc).__name__} {str(exc)[:120]}")


if __name__ == "__main__":
    main()
