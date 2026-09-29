#!/usr/bin/env python3
"""Emit one translation unit per fused-attention leaf, as the reference does.

The reference splits its forward schedule into one file per leaf -- per element
type, head width, causal choice and split -- and each file holds a single
explicit instantiation.  The reason is memory, not tidiness: the kernels are
large, and a translation unit that held every leaf would need the compiler to
hold all of them at once.  This emits the same set for the element types and
head widths this tree builds the schedule for.
"""
from __future__ import annotations

import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "src/backend/cuda/transformers"

# The element types and head widths the schedule is instantiated for.  A width
# absent here is not built, and a call needing it does not reach the schedule.
DTYPES = {"fp16": "cutlass::half_t", "bf16": "cutlass::bfloat16_t"}
HEAD_DIMS = (32, 64, 96, 128)

HEADER = """// One leaf of the fused attention schedule, in a file of its own.
//
// Splitting the schedule this way is what makes the set affordable to build:
// the kernels are large, so a file that held every instantiation would need the
// compiler to hold all of them at once.  Each file below holds exactly one.
#include "FlashFwdLauncher.h"

namespace {ns} {{

{body}

}}  // namespace {ns}
"""


def leaf(body: str) -> tuple[str, str]:
    return ("FlashFwdLauncher.h", body)


def emit() -> list[pathlib.Path]:
    written: list[pathlib.Path] = []
    ns = "tensorplay_native_flash"
    for width in HEAD_DIMS:
        for tag, ctype in DTYPES.items():
            for causal in (False, True):
                suffix = "_causal" if causal else ""
                flag = "true" if causal else "false"

                body = (
                    f"template<>\n"
                    f"void run_mha_fwd_<{ctype}, {width}, {flag}>(\n"
                    f"    Flash_fwd_params& params, cudaStream_t stream) {{\n"
                    f"  run_mha_fwd_hdim{width}<{ctype}, {flag}>(params, stream);\n"
                    f"}}\n"
                )
                path = OUT / f"flash_fwd_hdim{width}_{tag}{suffix}_sm80.cu"
                path.write_text(HEADER.format(ns=ns, body=body))
                written.append(path)

                body = (
                    f"template void run_mha_fwd_splitkv_align<{ctype}, {width}, "
                    f"{flag}>(Flash_fwd_params& params, cudaStream_t stream);\n"
                )
                path = OUT / f"flash_fwd_split_align_hdim{width}_{tag}{suffix}_sm80.cu"
                path.write_text(HEADER.format(ns=ns, body=body))
                written.append(path)

                # The aligned split kernel is instantiated in its own file so
                # the two compile in parallel; declaring it extern here keeps
                # the dispatch below from instantiating it a second time.
                body = (
                    f"// The num_splits==1 blocksize-aligned tree is instantiated in\n"
                    f"// its own translation unit so it compiles in parallel; declaring it\n"
                    f"// extern here keeps the dispatch below from re-instantiating it.\n"
                    f"extern template void run_mha_fwd_splitkv_align<{ctype}, "
                    f"{width}, {flag}>(\n"
                    f"    Flash_fwd_params& params, cudaStream_t stream);\n\n"
                    f"template void run_mha_fwd_splitkv_dispatch<{ctype}, "
                    f"{width}, {flag}>(\n"
                    f"    Flash_fwd_params& params, cudaStream_t stream);\n"
                )
                path = OUT / f"flash_fwd_split_hdim{width}_{tag}{suffix}_sm80.cu"
                path.write_text(HEADER.format(ns=ns, body=body))
                written.append(path)
    return written


if __name__ == "__main__":
    for p in emit():
        print(p.name)
