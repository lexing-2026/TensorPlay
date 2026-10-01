// One leaf of the fused attention schedule, in a file of its own.
//
// Splitting the schedule this way is what makes the set affordable to build:
// the kernels are large, so a file that held every instantiation would need the
// compiler to hold all of them at once.  Each file below holds exactly one.
#include "FlashFwdLauncher.h"

namespace tensorplay_native_flash {

template void run_mha_fwd_splitkv_align<float, 128, true>(Flash_fwd_params& params, cudaStream_t stream);


}  // namespace tensorplay_native_flash