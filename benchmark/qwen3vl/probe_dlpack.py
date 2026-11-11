"""Check that accelerator tensors can be handed to a foreign operator.

An operator published outside the framework takes its own tensor type and
writes into its own tensor type.  For the framework's tensors to be usable with
it, the two must be able to share the same device memory rather than copy
through the host.  This checks both directions of that exchange and what it
costs.

    python probe_dlpack.py
"""

from __future__ import annotations

import time

import numpy as np


def main():
    import torch
    import tensorplay as tp

    import tensorplay_mlu

    tensorplay_mlu.load()

    a = tp.randn(1, 8, 16, 128, device="mlu", dtype=tp.float16)
    print("tensorplay tensor:", type(a).__name__, a.shape, a.dtype, a.device)

    print("\ntensorplay -> foreign")
    try:
        t = torch.from_dlpack(a)
        print(f"  ok: {type(t).__name__} {tuple(t.shape)} {t.dtype} {t.device}")
        print(f"  shares storage: {t.data_ptr() == a.data_ptr()}")
    except Exception as exc:  # noqa: BLE001
        print("  failed:", type(exc).__name__, exc)
        return

    print("\nforeign -> tensorplay")
    try:
        b = tp.from_dlpack(t.__dlpack__())
        print(f"  ok: {type(b).__name__} {b.shape} {b.dtype} {b.device}")
    except Exception as exc:  # noqa: BLE001
        print("  failed:", type(exc).__name__, exc)

    print("\nround trip value")
    a2 = tp.from_dlpack(a.__dlpack__())
    t2 = torch.from_dlpack(a2)
    t2.view(-1)[0] = 3.5
    got = float(a.reshape(-1)[0].detach().cpu().numpy())
    print(f"  write through the view, read back through the original: {got}")

    print("\ncost of the bridge")
    repeat = 2000
    for _ in range(50):
        torch.from_dlpack(a)
        tp.from_dlpack(a.__dlpack__())
    tp.mlu.synchronize()
    start = time.time()
    for _ in range(repeat):
        torch.from_dlpack(a)
    duration = time.time() - start
    print(f"  tensorplay -> foreign: {duration / repeat * 1e6:7.2f} us")
    start = time.time()
    for _ in range(repeat):
        tp.from_dlpack(a.__dlpack__())
    duration = time.time() - start
    print(f"  foreign -> tensorplay: {duration / repeat * 1e6:7.2f} us")


if __name__ == "__main__":
    main()
