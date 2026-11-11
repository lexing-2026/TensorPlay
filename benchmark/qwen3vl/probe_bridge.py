"""Check that a device tensor can be handed to a peer library and taken back.

The vendor's resident-weight product is only reachable from its own tensor
type, so a projection written against it hands the activation over and takes
the result back.  Both directions must be views of the same memory rather than
copies, and each side's values must be the ones the other wrote.

    python probe_bridge.py
"""

from __future__ import annotations

import torch  # noqa: F401
import torch_mlu  # noqa: F401
import torch_mlu_ops

import tensorplay as tp
import tensorplay_mlu


def main():
    tensorplay_mlu.load()

    source = tp.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype=tp.float16, device="mlu",
    )
    print("tensorplay", source.device, source.dtype, tuple(source.shape))

    exported = torch.from_dlpack(tp.to_dlpack(source))
    torch.mlu.synchronize()
    print("as a peer tensor:", exported.device, exported.dtype,
          tuple(exported.shape))
    print("same memory:", exported.data_ptr() == source.data_ptr())
    print("same values:",
          bool(torch.equal(exported.float().cpu(),
                           torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8],
                                         [9, 10, 11, 12]], dtype=torch.float32))))

    # A value written on the peer side must be visible on the other one.
    exported.mul_(10.0)
    torch.mlu.synchronize()
    reading = source.cpu()
    print("a write on the peer side reaches tensorplay:",
          float(reading[2, 3]) == 120.0)

    # And back again: a peer tensor adopted by tensorplay shares its memory.
    returned = tp.from_dlpack(exported.contiguous())
    print("taken back:", returned.device, tuple(returned.shape),
          float(returned.cpu()[0, 0]) == 10.0)

    # The product itself on a real shape, with the activation handed over.
    activation = tp.randn(1, 4096, device="mlu", dtype=tp.float16)
    codes = torch.randint(-8, 8, (4096, 2048), dtype=torch.int8, device="mlu")
    scale = torch.full((4096, 32), 0.01, dtype=torch.float16, device="mlu")
    result = torch_mlu_ops.weight_only_quant_matmul(
        torch.from_dlpack(tp.to_dlpack(activation)), codes, scale,
        quant_bit_size=4)
    adopted = tp.from_dlpack(result)
    torch.mlu.synchronize()
    print("product result:", adopted.device, adopted.dtype,
          tuple(adopted.shape), "finite:",
          bool(torch.isfinite(adopted.to(torch.float32).cpu()).all()))


if __name__ == "__main__":
    main()
