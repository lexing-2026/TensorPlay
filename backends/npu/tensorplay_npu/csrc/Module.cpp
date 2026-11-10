// C entry points the Python package calls through ctypes.

#include <cstdint>

#include "AscendAllocator.h"
#include "AscendRuntime.h"
#include "CpuFallback.h"

using namespace tensorplay;

extern "C" {

// Called once after the library is loaded: every operator that still has
// no kernel for the device runs through the host.
int tp_npu_init() {
    try {
        cpu_fallback::register_host_fallbacks(DispatchKey::PrivateUse1);
        return 0;
    } catch (...) {
        return 1;
    }
}

int tp_npu_device_count() { return npu::device_count(); }

// allocated, reserved, peak allocated, peak reserved (bytes).
void tp_npu_memory_stats(int index, int64_t* out) {
    const npu::MemoryStats stats = npu::memory_stats(index);
    out[0] = stats.allocated;
    out[1] = stats.reserved;
    out[2] = stats.peak_allocated;
    out[3] = stats.peak_reserved;
}

void tp_npu_reset_peak_memory_stats(int index) { npu::reset_peak_memory_stats(index); }

void tp_npu_empty_cache() { npu::empty_cache(); }

} // extern "C"
