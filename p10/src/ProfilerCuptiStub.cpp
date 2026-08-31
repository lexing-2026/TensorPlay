// No-op GPU profiler hooks for HIP builds.
//
// The HIP runtime path reuses the shared profiler arena, but the GPU-activity
// collector is not part of this build, so the entry points degrade to a
// disabled state: g_gpu_trace stays false and sessions record zero GPU rows.
// Keeping the symbols defined lets the extension import under RTLD_NOW
// without embedding conditional logic at the call sites.

#include <atomic>
#include <string>
#include <vector>

#include "Profiler.h"

namespace tensorplay {
namespace prof {

TENSORPLAY_API std::atomic<bool> g_gpu_trace{false};
TENSORPLAY_API bool cupti_available() { return false; }
TENSORPLAY_API std::string cupti_last_error() {
    return "GPU trace collector unavailable in this build";
}
TENSORPLAY_API bool cupti_start() { return false; }
TENSORPLAY_API void cupti_stop_and_collect(std::vector<GpuActivity>&) {}
TENSORPLAY_API bool cupti_push_ext(uint64_t) { return false; }
TENSORPLAY_API void cupti_pop_ext() {}
TENSORPLAY_API uint32_t cupti_version() { return 0; }

} // namespace prof
} // namespace tensorplay
