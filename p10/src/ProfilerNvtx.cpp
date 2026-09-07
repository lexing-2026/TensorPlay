// NVTX (NVIDIA Tools Extension) bridge -- runtime-loaded, zero build deps.
//
// Loads libnvtx via dlopen on first use (the CUDA toolkit ships it; nsight
// systems consumes the emitted ranges).  When the library is absent every
// entry point degrades gracefully:
//   * profiler ranges become no-ops (emit_nvtx() stays safe everywhere);
//   * the tensorplay.cuda.nvtx surface raises the same RuntimeError as the
//     historical stub, preserving its documented contract.
//
// This gives nsight-systems users op-name annotated timelines through
// pulling CUPTI in-process (kernel-level tracing remains nsys's job via its
// own injection -- it already sees TensorPlay's real CUDA API calls).

#include "Profiler.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <mutex>
#include <string>

#include "Exception.h"

namespace tensorplay {
namespace prof {

namespace {

std::mutex g_nvtx_mutex;
bool g_loaded = false;
bool g_available = false;

int (*nv_range_push_a)(const char*) = nullptr;
int (*nv_range_pop)() = nullptr;
void (*nv_mark_a)(const char*) = nullptr;
unsigned long long (*nv_range_start_a)(const char*) = nullptr;
void (*nv_range_end)(unsigned long long) = nullptr;

#if defined(_WIN32)
// The NVTX runtime ships inside the CUDA toolkit's bin directory (and
// system32 when the toolkit installer updates PATH), so probing runs
// through the Win32 loader with the same symbol set the .so probes use.
using NvtxHandle = HMODULE;
static NvtxHandle nvtx_open(const char* name) { return LoadLibraryA(name); }
static void* nvtx_sym(NvtxHandle h, const char* n) {
    return reinterpret_cast<void*>(GetProcAddress(h, n));
}
#else
using NvtxHandle = void*;
static NvtxHandle nvtx_open(const char* name) {
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}
static void* nvtx_sym(NvtxHandle h, const char* n) { return dlsym(h, n); }
#endif

void ensure_loaded() {
    std::lock_guard<std::mutex> lock(g_nvtx_mutex);
    if (g_loaded) return;
    g_loaded = true;
    // Probe NVIDIA first, then the ROCm twin.  ROCTx (libroctx64) exports
    // the exact NVTX signatures under the roctx* prefix, so omniperf /
    // rocprof consume the same emit_nvtx() ranges on AMD GPUs.
    struct Candidate {
        const char* lib;
        const char* push;
        const char* pop;
        const char* mark;
        const char* start;
        const char* end;
    };
#if defined(_WIN32)
    const Candidate candidates[] = {
        {"nvToolsExt64_1.dll", "nvtxRangePushA", "nvtxRangePop", "nvtxMarkA",
         "nvtxRangeStartA", "nvtxRangeEnd"},
        {"nvToolsExt64.dll", "nvtxRangePushA", "nvtxRangePop", "nvtxMarkA",
         "nvtxRangeStartA", "nvtxRangeEnd"},
    };
#else
    const Candidate candidates[] = {
        {"libnvtx.so", "nvtxRangePushA", "nvtxRangePop", "nvtxMarkA",
         "nvtxRangeStartA", "nvtxRangeEnd"},
        {"libnvtx_dynamic.so", "nvtxRangePushA", "nvtxRangePop", "nvtxMarkA",
         "nvtxRangeStartA", "nvtxRangeEnd"},
        {"libNVToolsExt.so", "nvtxRangePushA", "nvtxRangePop", "nvtxMarkA",
         "nvtxRangeStartA", "nvtxRangeEnd"},
        {"libroctx64.so", "roctxRangePushA", "roctxRangePop", "roctxMarkA",
         "roctxRangeStartA", "roctxRangeEnd"},
    };
#endif
    for (const auto& c : candidates) {
        if (NvtxHandle h = nvtx_open(c.lib)) {
            nv_range_push_a =
                reinterpret_cast<int (*)(const char*)>(nvtx_sym(h, c.push));
            nv_range_pop =
                reinterpret_cast<int (*)()>(nvtx_sym(h, c.pop));
            nv_mark_a = reinterpret_cast<void (*)(const char*)>(
                nvtx_sym(h, c.mark));
            nv_range_start_a =
                reinterpret_cast<unsigned long long (*)(const char*)>(
                    nvtx_sym(h, c.start));
            nv_range_end = reinterpret_cast<void (*)(unsigned long long)>(
                nvtx_sym(h, c.end));
            g_available = nv_range_push_a && nv_range_pop && nv_mark_a &&
                          nv_range_start_a && nv_range_end;
            return;
        }
    }
}

inline bool usable() {
    ensure_loaded();
    return g_available;
}

inline bool emitting() {
    return g_emit_nvtx.load(std::memory_order_relaxed) && usable();
}

} // namespace

TENSORPLAY_API std::atomic<bool> g_emit_nvtx{false};

TENSORPLAY_API bool nvtx_available() { return usable(); }

TENSORPLAY_API int nvtx_range_push(const char* msg) {
    if (!usable()) {
        TP_THROW(RuntimeError,
                 "NVTX functions not installed. Are you sure you have a CUDA "
                 "build?");
    }
    return nv_range_push_a(msg);
}

TENSORPLAY_API int nvtx_range_pop() {
    if (!usable()) {
        TP_THROW(RuntimeError,
                 "NVTX functions not installed. Are you sure you have a CUDA "
                 "build?");
    }
    return nv_range_pop();
}

TENSORPLAY_API void nvtx_mark(const char* msg) {
    if (!usable()) {
        TP_THROW(RuntimeError,
                 "NVTX functions not installed. Are you sure you have a CUDA "
                 "build?");
    }
    nv_mark_a(msg);
}

TENSORPLAY_API unsigned long long nvtx_range_start(const char* msg) {
    if (!usable()) {
        TP_THROW(RuntimeError,
                 "NVTX functions not installed. Are you sure you have a CUDA "
                 "build?");
    }
    return nv_range_start_a(msg);
}

TENSORPLAY_API void nvtx_range_end(unsigned long long id) {
    if (!usable()) {
        TP_THROW(RuntimeError,
                 "NVTX functions not installed. Are you sure you have a CUDA "
                 "build?");
    }
    nv_range_end(id);
}

// Internal no-throw hooks used by OpRecord/user-span lifecycles.
TENSORPLAY_API void nvtx_span_begin(const char* name) {
    if (emitting()) nv_range_push_a(name);
}

TENSORPLAY_API void nvtx_span_end() {
    if (emitting()) nv_range_pop();
}

} // namespace prof
} // namespace tensorplay
