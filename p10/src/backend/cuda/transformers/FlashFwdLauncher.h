#pragma once

// The fused attention schedule's launcher, taken from the reference as it is
// written there rather than restated here.
//
// What this header settles before the reference's own header is read:
//
//   - the reference checks a CUDA result through the two names its error header
//     defines, and that header is supplied at the path it is included by, so
//     this one reads as written;
//   - the header lives in whatever namespace its configuration names, so the
//     namespace is fixed before the read and released after it;
//
// The reference keeps its dispatcher over element type, head width and causal
// choice in the same file as its tensor entry points, not in the launch
// header, and it is reproduced the same way here: in a translation unit of its
// own, so that the leaves below are not compiled after a use of themselves.
//
// Everything else -- the three kernel definitions, the choice between causal
// and windowed masking, the choice of the aligned kernel, the head-width and
// element-type selection -- is the reference's, and is deliberately not
// duplicated.  A packed batch is not a separate call: the kernels read the
// sequence table out of the parameters, so the same entry point serves it.

#include "FlashCheck.h"

#ifndef FLASH_NAMESPACE
#define FLASH_NAMESPACE tensorplay_native_flash
#endif

// The features this build does not have, named the way the reference names
// them, so that its own conditionals remove the code for them rather than this
// header having to know where it is.  Dropping, the bias slopes and the score
// cap are all absent: every launch here passes the keep probability of one and
// leaves the other two null, so the code for them would be instantiated and
// never run.
//
// Local masking is deliberately not in that list.  The reference's launcher
// chooses between causal and windowed masking at the launch site, and a build
// that turns local masking off cannot express a window at all; that is the
// switch the sliding-window path turns on.
#define FLASHATTENTION_DISABLE_DROPOUT
#define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_SOFTCAP

#include "flash/flash_fwd_launch_template.h"

#undef FLASHATTENTION_DISABLE_SOFTCAP
#undef FLASHATTENTION_DISABLE_ALIBI
#undef FLASHATTENTION_DISABLE_DROPOUT


#undef FLASH_NAMESPACE

namespace tensorplay {
namespace cuda {

using ::tensorplay_native_flash::Flash_fwd_params;

// The dispatcher, defined in the translation unit of its own for the reason
// given above and declared here for the launchers to call.
namespace tensorplay_native_flash {
void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream,
                 bool force_split_kernel = false);
}  // namespace tensorplay_native_flash

}  // namespace cuda
}  // namespace tensorplay
