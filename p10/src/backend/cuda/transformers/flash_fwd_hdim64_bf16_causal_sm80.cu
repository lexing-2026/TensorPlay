// One leaf of the fused attention schedule, in a file of its own.
//
// Splitting the schedule this way is what makes the set affordable to build:
// the kernels are large, so a file that held every instantiation would need the
// compiler to hold all of them at once.  Each file below holds exactly one.
#include "FlashFwdLauncher.h"

namespace tensorplay_native_flash {

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 64, true>(
    Flash_fwd_params& params, cudaStream_t stream) {
  run_mha_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
}


}  // namespace tensorplay_native_flash
