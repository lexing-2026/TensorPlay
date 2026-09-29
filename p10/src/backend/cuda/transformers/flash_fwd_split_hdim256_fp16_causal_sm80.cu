// One leaf of the fused attention schedule, in a file of its own.
//
// Splitting the schedule this way is what makes the set affordable to build:
// the kernels are large, so a file that held every instantiation would need the
// compiler to hold all of them at once.  Each file below holds exactly one.
#include "FlashFwdLauncher.h"

namespace tensorplay_native_flash {
// The num_splits==1 blocksize-aligned tree is instantiated in
// its own translation unit so it compiles in parallel; declaring it
// extern here keeps the dispatch below from re-instantiating it.
extern template void run_mha_fwd_splitkv_align<cutlass::half_t, 256, true>(
    Flash_fwd_params& params, cudaStream_t stream);

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 256, true>(
    Flash_fwd_params& params, cudaStream_t stream);


}  // namespace tensorplay_native_flash
