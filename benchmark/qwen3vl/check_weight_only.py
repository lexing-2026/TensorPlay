"""Check the native weight-only path against a directly recovered matrix.

Reads a quantized layer from the checkpoint twice — once by widening the
groups to single precision, once by repacking them into the framework's
weight-only linear — and compares the two matrix products.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp  # noqa: E402
from tensorplay.ao.quantization.intx_weight_only import WeightOnlyIntxLinear  # noqa: E402

from qwen3vl.ct_weights import Checkpoint, dequantize_linear, weight_only_group  # noqa: E402
from qwen3vl.model import QUANT_BITS, QUANT_GROUP_SIZE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--layer",
        default="model.language_model.layers.0.self_attn.q_proj.weight",
    )
    ap.add_argument("--rows", type=int, default=8)
    args = ap.parse_args()

    ckpt = Checkpoint(args.model)
    dense = dequantize_linear(ckpt, args.layer)
    packed, scale, zero = weight_only_group(ckpt, args.layer, QUANT_BITS)
    out_dim, in_dim = dense.shape
    print(f"{args.layer}: {dense.shape}  packed {packed.shape}  groups {scale.shape}")

    module = WeightOnlyIntxLinear(
        in_dim, out_dim, bits=QUANT_BITS, groupsize=QUANT_GROUP_SIZE
    )
    module.weight.copy_(tp.from_numpy(packed))
    module.scales_and_zeros.copy_(
        tp.stack([tp.from_numpy(scale.T).contiguous(), tp.from_numpy(zero.T).contiguous()], dim=-1)
    )

    rng = np.random.default_rng(0)
    x = rng.standard_normal((args.rows, in_dim)).astype(np.float32)
    ref = x @ dense.T
    got = module(tp.from_numpy(x)).numpy()
    diff = np.abs(got - ref)
    print(f"max abs {diff.max():.3e}  rel {diff.max() / (np.abs(ref).max() + 1e-9):.3e}")
    print("matrices identical:", np.array_equal(
        np.round(got, 4), np.round(ref, 4)))


if __name__ == "__main__":
    main()
