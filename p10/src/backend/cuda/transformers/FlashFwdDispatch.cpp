// The fused attention schedule's dispatcher.
//
// This is a host translation unit on purpose, and the reason is the same one
// the reference keeps its dispatcher in a host file: the leaves are explicit
// instantiations of a kernel template, and a host compiler emits no device
// code for them at all.  Compiled as a device unit this file would instantiate
// the whole schedule again in one place -- every head width, every element
// type, every mask choice -- and the kernels are large enough that holding
// them all at once is what runs the compiler out of memory.  Read from here as
// a host file, it names the leaves and calls them.

#include "FlashFwdParams.h"

namespace tensorplay_native_flash {

// Read the element type, the head width and the causal choice off the
// parameters and call the leaf for them.  A call that asks for the key axis to
// be split -- because it named a number of splits, or because the keys are a
// paged store -- goes to the splitting leaf instead.
void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream,
                 bool force_split_kernel) {
  // The wide precision has leaves of its own, one per head width it serves.
  // Its split-kv path shares the reduced precisions' dispatch, but with a
  // narrower key tile whose shared-memory footprint fits the on-chip budget.
  if (params.is_fp32) {
    if (params.d == 128) {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        if (params.num_splits <= 1 && !force_split_kernel) {
          run_mha_fwd_<float, 128, Is_causal>(params, stream);
        } else {
          run_mha_fwd_splitkv_dispatch<float, 128, Is_causal>(params, stream);
        }
      });
    } else if (params.d == 64) {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_mha_fwd_<float, 64, Is_causal>(params, stream);
      });
    }
    return;
  }
  FP16_SWITCH(!params.is_bf16, [&] {
    HEADDIM_SWITCH(params.d, [&] {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        if (params.num_splits <= 1 && !force_split_kernel) {
          run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
        } else {
          run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params,
                                                                        stream);
        }
      });
    });
  });
}

}  // namespace tensorplay_native_flash
