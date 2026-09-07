// CUPTI kernel-level GPU tracing for the native profiler (USE_CUDA builds;
// the activity API is driven with
// CONCURRENT_KERNEL + MEMCPY + MEMSET + RUNTIME + EXTERNAL_CORRELATION and
// buffer-request/completion callbacks, and CUDA dispatch brackets push/pop
// the op's slot id as an external correlation id so kernel records join
// back to the op that launched them (op -> runtime API -> kernel).
//
// libcupti is dlopen'd (major-line soname candidates first) so the core
// library never hard-depends on it and sessions on machines without the
// toolkit degrade gracefully.  Struct definitions come from the toolkit
// headers the build already ships (CUDAToolkit_INCLUDE_DIRS), so record
// parsing stays ABI-correct for the CUDA the extension was compiled against.
//
// Hot path: outside a gpu_trace session the only per-op cost is the
// g_gpu_trace load in GpuTimerPair::arm (one acquire-load, same class as
// g_gpu_timing).  While a session runs, CUPTI buffers are recycled from a
// pool and parsed incrementally on CUPTI's own completion thread, so stop
// only has to flush and snapshot.

#include "Profiler.h"

#ifdef USE_CUDA

#include <cupti.h>

// ---- Activity-record revisions ---------------------------------------------
// Toolkit headers ship a subset of record revisions: 12.8+ headers replace
// Memcpy5 with Memcpy6, 12.0+ headers add Kernel9, and older headers lack
// the newest revisions entirely.  The dlopen'd runtime in turn delivers
// whichever revision its own toolkit writes.  Across the 11.x/12.x line
// every revision shares the layout of the prefix fields read below
// (start/end/correlationId/deviceId/streamId/name/bytes/value/copyKind):
// revisions only append or replace trailing fields.  So the alias picks the
// newest revision the build headers know, and parsing stays correct for any
// runtime of the same major line.
#if defined(CUPTI_API_VERSION) && CUPTI_API_VERSION >= 21
using ActivityKernel = CUpti_ActivityKernel9;
#elif defined(CUPTI_API_VERSION) && CUPTI_API_VERSION >= 18
using ActivityKernel = CUpti_ActivityKernel8;
#else
using ActivityKernel = CUpti_ActivityKernel4;
#endif
#if defined(CUPTI_API_VERSION) && CUPTI_API_VERSION >= 26
using ActivityMemcpy = CUpti_ActivityMemcpy6;
#elif defined(CUPTI_API_VERSION) && CUPTI_API_VERSION >= 18
using ActivityMemcpy = CUpti_ActivityMemcpy5;
#else
using ActivityMemcpy = CUpti_ActivityMemcpy4;
#endif
#if defined(CUPTI_API_VERSION) && CUPTI_API_VERSION >= 18
using ActivityMemset = CUpti_ActivityMemset4;
#else
using ActivityMemset = CUpti_ActivityMemset3;
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tensorplay {
namespace prof {

namespace {

// ---- dlopen'd CUPTI entry points ------------------------------------------
using Fn_GetTimestamp = CUptiResult (*)(uint64_t*);
using Fn_RegisterCallbacks = CUptiResult (*)(CUpti_BuffersCallbackRequestFunc,
                                             CUpti_BuffersCallbackCompleteFunc);
using Fn_Enable = CUptiResult (*)(CUpti_ActivityKind);
using Fn_FlushAll = CUptiResult (*)(uint32_t);
using Fn_GetNextRecord = CUptiResult (*)(uint8_t*, size_t, CUpti_Activity**);
using Fn_GetNumLostRecords = CUptiResult (*)(uint8_t*, size_t, size_t*);
using Fn_PushExt = CUptiResult (*)(CUpti_ExternalCorrelationKind, uint64_t);
using Fn_PopExt = CUptiResult (*)(CUpti_ExternalCorrelationKind, uint64_t*);
using Fn_GetCallbackName = CUptiResult (*)(CUpti_CallbackDomain, uint32_t,
                                           const char**);
using Fn_GetVersion = CUptiResult (*)(uint32_t*);
using Fn_Disable = Fn_Enable;
using Fn_PushExternalCorrelationId = Fn_PushExt;
using Fn_PopExternalCorrelationId = Fn_PopExt;

struct CuptiFns {
    Fn_GetTimestamp GetTimestamp = nullptr;
    Fn_RegisterCallbacks RegisterCallbacks = nullptr;
    Fn_Enable Enable = nullptr;
    Fn_Enable Disable = nullptr;
    Fn_FlushAll FlushAll = nullptr;
    Fn_GetNextRecord GetNextRecord = nullptr;
    Fn_GetNumLostRecords GetNumLostRecords = nullptr;
    Fn_PushExt PushExternalCorrelationId = nullptr;
    Fn_PopExt PopExternalCorrelationId = nullptr;
    Fn_GetCallbackName GetCallbackName = nullptr;
    Fn_GetVersion GetVersion = nullptr;
};

std::mutex g_cupti_mutex;      // guards everything below (session scope)
CuptiFns* g_fns = nullptr;     // dlsym'd entry points (null until start)
void* g_cupti_lib = nullptr;   // dlopen handle (kept for the process life)
std::string g_last_error;      // init/dlopen diagnostic for the binding
bool g_callbacks_registered = false;
bool g_kinds_enabled = false;
int64_t g_time_offset_ns = 0;  // steady_ns ~= cupti_ns + offset

// 1 MiB activity buffers recycled between CUPTI and our parser.
constexpr size_t kBufSize = 1u << 20;
std::deque<uint8_t*>* g_free_bufs = nullptr;

// Parsed activities of the running session.
std::vector<GpuActivity>* g_acts = nullptr;  // kernels + memcpy + memset
std::vector<GpuActivity>* g_api_acts = nullptr;  // runtime/driver rows
// correlationId -> external (OpRecord slot) id from EXTERNAL_CORRELATION
// records; applied to kernels/memcpy/memset when the snapshot is taken.
std::unordered_map<uint32_t, uint64_t>* g_corr2ext = nullptr;

// Session-lifetime name arena (kernel names point into CUPTI's buffer and
// must be copied before the buffer is recycled).
std::deque<std::string>* g_name_arena = nullptr;
std::unordered_map<std::string, const char*>* g_name_index = nullptr;

uint64_t steady_now_ns() {
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

const char* intern_name_locked(const char* name, size_t len) {
    if (name == nullptr || len == 0) return "unknown";
    std::string key(name, len);
    auto it = g_name_index->find(key);
    if (it != g_name_index->end()) return it->second;
    g_name_arena->push_back(std::move(key));
    const char* stored = g_name_arena->back().c_str();
    (*g_name_index)[std::string(name, len)] = stored;
    return stored;
}

const char* memcpy_name(uint8_t kind) {
    switch (kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "Memcpy HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "Memcpy DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "Memcpy DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "Memcpy HtoH";
#ifdef CUPTI_ACTIVITY_MEMCPY_KIND_P2P
        case CUPTI_ACTIVITY_MEMCPY_KIND_P2P: return "Memcpy P2P";
#endif
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA: return "Memcpy HtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH: return "Memcpy AtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA: return "Memcpy AtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD: return "Memcpy AtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA: return "Memcpy DtoA";
        default: return "Memcpy";
    }
}

// API rows get proper names through cuptiGetCallbackName (falls back to a
// generic spelling when the runtime cannot resolve one).
const char* api_name(CUpti_CallbackDomain domain, uint32_t cbid) {
    if (g_fns && g_fns->GetCallbackName) {
        const char* name = nullptr;
        if (g_fns->GetCallbackName(domain, cbid, &name) == CUPTI_SUCCESS &&
            name != nullptr && name[0] != '\0') {
            return intern_name_locked(name, std::strlen(name));
        }
    }
    std::string fallback = domain == CUPTI_CB_DOMAIN_RUNTIME_API
        ? "cuda_runtime " : "cuda_driver ";
    fallback += std::to_string(cbid);
    return intern_name_locked(fallback.data(), fallback.size());
}

void ensure_state_locked() {
    if (!g_acts) g_acts = new std::vector<GpuActivity>();
    if (!g_api_acts) g_api_acts = new std::vector<GpuActivity>();
    if (!g_corr2ext) g_corr2ext = new std::unordered_map<uint32_t, uint64_t>();
    if (!g_free_bufs) g_free_bufs = new std::deque<uint8_t*>();
    if (!g_name_arena) g_name_arena = new std::deque<std::string>();
    if (!g_name_index) g_name_index =
        new std::unordered_map<std::string, const char*>();
}

void reset_session_locked() {
    if (g_acts) g_acts->clear();
    if (g_api_acts) g_api_acts->clear();
    if (g_corr2ext) g_corr2ext->clear();
    if (g_name_arena) g_name_arena->clear();
    if (g_name_index) g_name_index->clear();
}

uint64_t ext_for_locked(uint32_t correlation) {
    auto it = g_corr2ext->find(correlation);
    if (it == g_corr2ext->end()) return GpuActivity::kNoExt;
    return it->second;
}

GpuActivity base_act(const char* name, char kind, uint64_t start,
                     uint64_t end, uint32_t corr, int32_t device,
                     int32_t stream) {
    GpuActivity a;
    a.name = name;
    a.start_ns = start;
    a.end_ns = end > start ? end : start;
    a.correlation = corr;
    a.external_id = GpuActivity::kNoExt;
    a.thread_id = 0;
    a.cbid = 0;
    a.device = device;
    a.stream = stream;
    a.kind = kind;
    a.bytes = 0;
    a.copy_kind = 0;
    a.value = 0;
    return a;
}

// ---- CUPTI callback thread entry points -----------------------------------
void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size,
                               size_t* maxNumRecords) {
    uint8_t* buf = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_cupti_mutex);
        if (g_free_bufs && !g_free_bufs->empty()) {
            buf = g_free_bufs->back();
            g_free_bufs->pop_back();
        }
    }
    if (buf == nullptr) buf = static_cast<uint8_t*>(std::malloc(kBufSize));
    if (buf == nullptr) {
        // CUPTI drops records for this round; profiling keeps running.
        *buffer = nullptr;
        *size = 0;
        *maxNumRecords = 0;
        return;
    }
    *buffer = buf;
    *size = kBufSize;
    *maxNumRecords = 0;  // fill until full
}

void parse_buffer_locked(uint8_t* buffer, size_t valid_size) {
    CUpti_Activity* record = nullptr;
    while (g_fns->GetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
        switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                // Revision aliased above; only prefix fields are touched.
                const auto* k = reinterpret_cast<const ActivityKernel*>(record);
                const size_t len = k->name ? std::strlen(k->name) : 0;
                g_acts->push_back(base_act(
                    intern_name_locked(k->name, len), 'k',
                    k->start + g_time_offset_ns, k->end + g_time_offset_ns,
                    k->correlationId,
                    static_cast<int32_t>(k->deviceId),
                    static_cast<int32_t>(k->streamId)));
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                const auto* m =
                    reinterpret_cast<const ActivityMemcpy*>(record);
                GpuActivity a = base_act(
                    memcpy_name(m->copyKind), 'm',
                    m->start + g_time_offset_ns, m->end + g_time_offset_ns,
                    m->correlationId,
                    static_cast<int32_t>(m->deviceId),
                    static_cast<int32_t>(m->streamId));
                a.bytes = m->bytes;
                a.copy_kind = m->copyKind;
                g_acts->push_back(a);
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMSET: {
                const auto* s =
                    reinterpret_cast<const ActivityMemset*>(record);
                GpuActivity a = base_act(
                    "Memset (Device)", 's',
                    s->start + g_time_offset_ns, s->end + g_time_offset_ns,
                    s->correlationId,
                    static_cast<int32_t>(s->deviceId),
                    static_cast<int32_t>(s->streamId));
                a.bytes = s->bytes;
                a.value = s->value;
                g_acts->push_back(a);
                break;
            }
            case CUPTI_ACTIVITY_KIND_RUNTIME:
            case CUPTI_ACTIVITY_KIND_DRIVER: {
                const auto* r =
                    reinterpret_cast<const CUpti_ActivityAPI*>(record);
                const CUpti_CallbackDomain domain =
                    record->kind == CUPTI_ACTIVITY_KIND_RUNTIME
                        ? CUPTI_CB_DOMAIN_RUNTIME_API
                        : CUPTI_CB_DOMAIN_DRIVER_API;
                GpuActivity a = base_act(
                    api_name(domain, static_cast<uint32_t>(r->cbid)),
                    record->kind == CUPTI_ACTIVITY_KIND_RUNTIME ? 'r' : 'd',
                    r->start + g_time_offset_ns, r->end + g_time_offset_ns,
                    r->correlationId, -1, -1);
                a.thread_id = r->threadId;
                a.cbid = static_cast<uint32_t>(r->cbid);
                g_api_acts->push_back(a);
                break;
            }
            case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
                const auto* e = reinterpret_cast<
                    const CUpti_ActivityExternalCorrelation*>(record);
                if (e->externalKind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) {
                    (*g_corr2ext)[e->correlationId] = e->externalId;
                }
                break;
            }
            default:
                break;
        }
    }
    size_t lost = 0;
    if (g_fns->GetNumLostRecords &&
        g_fns->GetNumLostRecords(buffer, valid_size, &lost) == CUPTI_SUCCESS &&
        lost > 0) {
        // Records can be lost under extreme launch rates; surface once per
        // session instead of per buffer (count only, names unaffected).
        g_last_error = std::to_string(lost) + " CUPTI records lost";
    }
}

void CUPTIAPI buffer_completed(CUcontext /*context*/, uint32_t /*streamId*/,
                               uint8_t* buffer, size_t /*size*/,
                               size_t validSize) {
    if (buffer == nullptr) return;
    std::lock_guard<std::mutex> lock(g_cupti_mutex);
    if (g_fns != nullptr && validSize > 0) parse_buffer_locked(buffer, validSize);
    if (g_free_bufs) g_free_bufs->push_back(buffer);
    else std::free(buffer);
}

void disable_kinds_locked() {
    if (!g_fns || !g_kinds_enabled) return;
    static const CUpti_ActivityKind kinds[] = {
        CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI_ACTIVITY_KIND_MEMSET,
        CUPTI_ACTIVITY_KIND_RUNTIME,
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
    };
    for (CUpti_ActivityKind kind : kinds) {
        (void)g_fns->Disable(kind);
    }
    g_kinds_enabled = false;
}

} // namespace

TENSORPLAY_API std::atomic<bool> g_gpu_trace{false};

TENSORPLAY_API bool cupti_available() {
    if (g_fns != nullptr) return true;
    std::lock_guard<std::mutex> lock(g_cupti_mutex);
    if (g_fns != nullptr) return true;
    if (g_last_error.empty()) {
        // Probe the library without enabling anything.
        const char* candidates[] = {"libcupti.so.13", "libcupti.so.12",
                                    "libcupti.so.11", "libcupti.so.1",
                                    "libcupti.so"};
        for (const char* soname : candidates) {
            void* lib = dlopen(soname, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
            if (lib != nullptr) {
                dlclose(lib);
                return true;
            }
        }
        g_last_error = "libcupti not found (tried libcupti.so.13/.12/.11/.1/.so)";
    }
    return false;
}

TENSORPLAY_API std::string cupti_last_error() {
    std::lock_guard<std::mutex> lock(g_cupti_mutex);
    return g_last_error;
}

// CUPTI library version (cuptiGetVersion), or 0 when unavailable.  Unlike
// cupti_start this does not enable any activity kind; it only resolves the
// entry point so trace export can stamp the schema metadata.
TENSORPLAY_API uint32_t cupti_version() {
    std::lock_guard<std::mutex> lock(g_cupti_mutex);
    if (g_fns != nullptr && g_fns->GetVersion != nullptr) {
        uint32_t ver = 0;
        if (g_fns->GetVersion(&ver) == CUPTI_SUCCESS) return ver;
        return 0;
    }
    void* lib = g_cupti_lib;
    if (lib == nullptr) {
        const char* candidates[] = {"libcupti.so.13", "libcupti.so.12",
                                    "libcupti.so.11", "libcupti.so.1",
                                    "libcupti.so"};
        for (const char* soname : candidates) {
            lib = dlopen(soname, RTLD_LAZY | RTLD_LOCAL);
            if (lib != nullptr) {
                g_cupti_lib = lib;
                break;
            }
        }
    }
    if (lib == nullptr) return 0;
    auto fn = reinterpret_cast<Fn_GetVersion>(dlsym(lib, "cuptiGetVersion"));
    if (fn == nullptr) return 0;
    uint32_t ver = 0;
    if (fn(&ver) != CUPTI_SUCCESS) return 0;
    return ver;
}

TENSORPLAY_API bool cupti_start() {
    std::lock_guard<std::mutex> lock(g_cupti_mutex);
    ensure_state_locked();
    reset_session_locked();
    g_last_error.clear();

    if (g_fns == nullptr) {
        if (g_cupti_lib == nullptr) {
            const char* candidates[] = {"libcupti.so.13", "libcupti.so.12",
                                        "libcupti.so.11", "libcupti.so.1",
                                        "libcupti.so"};
            for (const char* soname : candidates) {
                g_cupti_lib = dlopen(soname, RTLD_LAZY | RTLD_LOCAL);
                if (g_cupti_lib != nullptr) break;
            }
            if (g_cupti_lib == nullptr) {
                g_last_error = std::string("dlopen libcupti failed: ") +
                               dlerror();
                return false;
            }
        }
        auto* fns = new CuptiFns();
        #define TP_CUPTI_DLSYM(field, name)                                     \
            fns->field = reinterpret_cast<Fn_##field>(                          \
                dlsym(g_cupti_lib, name));                                      \
            if (fns->field == nullptr) {                                        \
                g_last_error = std::string("dlsym ") + name + " failed";        \
                delete fns;                                                     \
                return false;                                                   \
            }
        TP_CUPTI_DLSYM(GetTimestamp, "cuptiGetTimestamp")
        TP_CUPTI_DLSYM(RegisterCallbacks, "cuptiActivityRegisterCallbacks")
        TP_CUPTI_DLSYM(Enable, "cuptiActivityEnable")
        TP_CUPTI_DLSYM(Disable, "cuptiActivityDisable")
        TP_CUPTI_DLSYM(FlushAll, "cuptiActivityFlushAll")
        TP_CUPTI_DLSYM(GetNextRecord, "cuptiActivityGetNextRecord")
        // Optional diagnostics/name helpers: absent from some libcupti
        // exports; call sites null-check.
        fns->GetNumLostRecords = reinterpret_cast<Fn_GetNumLostRecords>(
            dlsym(g_cupti_lib, "cuptiActivityGetNumLostRecords"));
        fns->GetCallbackName = reinterpret_cast<Fn_GetCallbackName>(
            dlsym(g_cupti_lib, "cuptiGetCallbackName"));
        fns->GetVersion = reinterpret_cast<Fn_GetVersion>(
            dlsym(g_cupti_lib, "cuptiGetVersion"));
        TP_CUPTI_DLSYM(PushExternalCorrelationId,
                       "cuptiActivityPushExternalCorrelationId")
        TP_CUPTI_DLSYM(PopExternalCorrelationId,
                       "cuptiActivityPopExternalCorrelationId")
        #undef TP_CUPTI_DLSYM
        g_fns = fns;
    }

    // Calibrate the CUPTI/steady timebase once per start; activity
    // timestamps share cuptiGetTimestamp's epoch, so a single offset maps
    // every record onto the op timeline's clock.
    uint64_t cupti_now = 0;
    if (g_fns->GetTimestamp(&cupti_now) == CUPTI_SUCCESS) {
        g_time_offset_ns = static_cast<int64_t>(steady_now_ns()) -
                           static_cast<int64_t>(cupti_now);
    }

    if (!g_callbacks_registered) {
        if (g_fns->RegisterCallbacks(buffer_requested, buffer_completed) !=
            CUPTI_SUCCESS) {
            g_last_error = "cuptiActivityRegisterCallbacks failed";
            return false;
        }
        g_callbacks_registered = true;
    }

    static const CUpti_ActivityKind kinds[] = {
        CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI_ACTIVITY_KIND_MEMSET,
        CUPTI_ACTIVITY_KIND_RUNTIME,
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
    };
    CUptiResult rc = CUPTI_SUCCESS;
    for (CUpti_ActivityKind kind : kinds) {
        rc = g_fns->Enable(kind);
        if (rc != CUPTI_SUCCESS) {
            g_last_error = std::string("cuptiActivityEnable(") +
                           std::to_string(static_cast<int>(kind)) +
                           ") failed: " + std::to_string(static_cast<int>(rc));
            disable_kinds_locked();
            return false;
        }
    }
    g_kinds_enabled = true;
    return true;
}

TENSORPLAY_API void cupti_stop_and_collect(std::vector<GpuActivity>& out) {
    std::vector<GpuActivity> acts;
    std::vector<GpuActivity> api_acts;
    {
        std::lock_guard<std::mutex> lock(g_cupti_mutex);
        if (g_fns == nullptr) return;
        disable_kinds_locked();
    }
    // FlushAll delivers every buffered record through buffer_completed
    // synchronously on the calling thread; it must run with g_cupti_mutex
    // released because the completion callback takes that mutex to parse
    // and recycle the buffer.
    (void)g_fns->FlushAll(0);
    // A short bounded grace period absorbs completion callbacks that land
    // on CUPTI's internal thread just after the flush.
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    {
        std::lock_guard<std::mutex> lock(g_cupti_mutex);
        if (g_acts) acts.swap(*g_acts);
        if (g_api_acts) api_acts.swap(*g_api_acts);
        if (g_corr2ext) {
            for (auto& a : acts) {
                auto it = g_corr2ext->find(a.correlation);
                if (it != g_corr2ext->end()) a.external_id = it->second;
            }
        }
        std::stable_sort(acts.begin(), acts.end(),
                         [](const GpuActivity& x, const GpuActivity& y) {
                             return x.start_ns < y.start_ns;
                         });
        std::stable_sort(api_acts.begin(), api_acts.end(),
                         [](const GpuActivity& x, const GpuActivity& y) {
                             return x.start_ns < y.start_ns;
                         });
    }
    out.reserve(out.size() + acts.size() + api_acts.size());
    out.insert(out.end(), acts.begin(), acts.end());
    out.insert(out.end(), api_acts.begin(), api_acts.end());
}

TENSORPLAY_API bool cupti_push_ext(uint64_t id) {
    if (!g_gpu_trace.load(std::memory_order_acquire)) return false;
    // g_fns is stable for the process life after the first successful
    // start; reading it without the mutex is safe (set-once pointer).
    CuptiFns* fns = g_fns;
    if (fns == nullptr || fns->PushExternalCorrelationId == nullptr) {
        return false;
    }
    return fns->PushExternalCorrelationId(
               CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id) == CUPTI_SUCCESS;
}

TENSORPLAY_API void cupti_pop_ext() {
    CuptiFns* fns = g_fns;
    if (fns == nullptr || fns->PopExternalCorrelationId == nullptr) return;
    uint64_t last = 0;
    (void)fns->PopExternalCorrelationId(
        CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &last);
}

} // namespace prof
} // namespace tensorplay

#elif defined(_WIN32) // USE_CUDA but Windows: no wired-up CUPTI loader yet

#include <atomic>
#include <string>
#include <vector>

namespace tensorplay {
namespace prof {

// Keep the symbols so the binding links without ifdef noise; GPU rows are
// simply absent from gpu_trace sessions.
TENSORPLAY_API std::atomic<bool> g_gpu_trace{false};
TENSORPLAY_API bool cupti_available() { return false; }
TENSORPLAY_API std::string cupti_last_error() {
    return "CUPTI tracing is not available in Windows builds";
}
TENSORPLAY_API bool cupti_start() { return false; }
TENSORPLAY_API void cupti_stop_and_collect(std::vector<GpuActivity>&) {}
TENSORPLAY_API bool cupti_push_ext(uint64_t) { return false; }
TENSORPLAY_API void cupti_pop_ext() {}
TENSORPLAY_API uint32_t cupti_version() { return 0; }

} // namespace prof
} // namespace tensorplay

#else  // !USE_CUDA

#include <atomic>
#include <string>
#include <vector>

namespace tensorplay {
namespace prof {

// CPU builds: keep the symbols so the binding links without ifdef noise.
TENSORPLAY_API std::atomic<bool> g_gpu_trace{false};
TENSORPLAY_API bool cupti_available() { return false; }
TENSORPLAY_API std::string cupti_last_error() {
    return "TensorPlay built without CUDA";
}
TENSORPLAY_API bool cupti_start() { return false; }
TENSORPLAY_API void cupti_stop_and_collect(std::vector<GpuActivity>&) {}
TENSORPLAY_API bool cupti_push_ext(uint64_t) { return false; }
TENSORPLAY_API void cupti_pop_ext() {}
TENSORPLAY_API uint32_t cupti_version() { return 0; }

} // namespace prof
} // namespace tensorplay

#endif // USE_CUDA
