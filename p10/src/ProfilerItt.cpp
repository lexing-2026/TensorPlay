// Intel ITT (Instrumentation and Tracing Technology) bridge -- runtime-loaded,
// zero build deps.  Counterpart of ProfilerNvtx.cpp for the VTune/Advisor family:
//
//   * profile(..., emit_itt=True) (or the emit_itt() context manager) makes
//     every op / user-span / backward-node span emit an ITT task under the
//     "tensorplay" domain, so VTune's timeline shows TensorPlay operator
//     names;
//   * when libittnotify is absent every hook degrades to a silent no-op and
//     the raw entry points raise the documented stub error.
//
// Kernel/CPU sampling itself stays the external tool's job (perf/VTune
// attach to the live process; symbols are exported by libp10).

#include "Profiler.h"

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <mutex>
#include <string>

#include "Exception.h"

namespace tensorplay {
namespace prof {

// The libittnotify distribution probes .so sonames only; there is no wired-up
// Windows runtime, so the bridge compiles to no-ops there.
#if defined(_WIN32)

TENSORPLAY_API std::atomic<bool> g_emit_itt{false};

TENSORPLAY_API bool itt_available() { return false; }

TENSORPLAY_API void itt_task_begin_name(const char*) {}
TENSORPLAY_API void itt_task_end() {}
TENSORPLAY_API void itt_span_begin(const char*) {}
TENSORPLAY_API void itt_span_end() {}

} // namespace prof
} // namespace tensorplay

#else

namespace {

std::mutex g_itt_mutex;
bool g_loaded = false;
bool g_available = false;

// Opaque handle types mirror <ittnotify.h> layouts; we never touch fields
// except constructing zeroed __itt_id values.
struct IttId {
    void* a = nullptr;
    unsigned long long b = 0;
};
struct IttDomain;
struct IttStringHandle;

void* (*fn_domain_create)(const char*) = nullptr;
void* (*fn_string_handle_create)(const char*) = nullptr;
void (*fn_task_begin)(void*, IttId, IttId, void*) = nullptr;
void (*fn_task_end)(void*) = nullptr;

void* g_domain = nullptr;

const IttId kNullId{};

void ensure_loaded() {
    std::lock_guard<std::mutex> lock(g_itt_mutex);
    if (g_loaded) return;
    g_loaded = true;
    for (const char* name : {"libittnotify.so", "libittnotify64.so"}) {
        if (void* h = dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
            fn_domain_create = reinterpret_cast<void* (*)(const char*)>(
                dlsym(h, "__itt_domain_create"));
            fn_string_handle_create =
                reinterpret_cast<void* (*)(const char*)>(
                    dlsym(h, "__itt_string_handle_create"));
            fn_task_begin = reinterpret_cast<
                void (*)(void*, IttId, IttId, void*)>(
                dlsym(h, "__itt_task_begin"));
            fn_task_end =
                reinterpret_cast<void (*)(void*)>(dlsym(h, "__itt_task_end"));
            g_available = fn_domain_create && fn_string_handle_create &&
                          fn_task_begin && fn_task_end;
            if (g_available) {
                g_domain = fn_domain_create("tensorplay");
            }
            return;
        }
    }
}

inline bool usable() {
    ensure_loaded();
    return g_available;
}

} // namespace

TENSORPLAY_API std::atomic<bool> g_emit_itt{false};

TENSORPLAY_API bool itt_available() { return usable(); }

TENSORPLAY_API void itt_task_begin_name(const char* name) {
    if (!usable() || !g_domain) return;
    void* sh = fn_string_handle_create(name);
    fn_task_begin(g_domain, kNullId, kNullId, sh);
}

TENSORPLAY_API void itt_task_end() {
    if (!usable() || !g_domain) return;
    fn_task_end(g_domain);
}

// Internal no-throw hooks used by OpRecord lifecycles.
TENSORPLAY_API void itt_span_begin(const char* name) {
    if (!g_emit_itt.load(std::memory_order_relaxed)) return;
    itt_task_begin_name(name);
}

TENSORPLAY_API void itt_span_end() {
    if (!g_emit_itt.load(std::memory_order_relaxed)) return;
    itt_task_end();
}

} // namespace prof
} // namespace tensorplay

#endif // _WIN32
