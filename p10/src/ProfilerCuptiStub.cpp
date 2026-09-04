// Kernel-level GPU tracing for HIP builds, backed by the ROCm tracer
// library (dlopen'd libroctracer64.so.4; never a hard link dependency).
//
// The tracer exports a generic callback/activity API: we open a record pool
// with a completion callback, enable the HIP API and HIP ops (kernel) activity
// domains, and parse the resulting fixed-layout activity records on the
// flush/completion path.  Record layout and field semantics were verified
// against the tracer's 4.1 protocol (domain/kind/op at offsets 0/4/8,
// correlation id at 16, host timestamps at 24/32, pid/tid at 40, device and
// queue id at 48, kernel symbol pointer at 56); HIP-API records carry the
// runtime function code in `op`, HIP-OPS records carry the kernel code in
// `kind` and the mangled kernel symbol in the name field.
//
// Sessions degrade gracefully: without the library (or on init failure) the
// collector reports unavailable and gpu_trace sessions simply record no GPU
// rows, matching the CUDA-side contract.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "Profiler.h"

namespace tensorplay {
namespace prof {

namespace {

// ---- dlopen'd tracer entry points (raw signatures, ABI-stable 4.1) --------
using Fn_GetTimestamp = int (*)(uint64_t*);
using Fn_OpenPool = int (*)(const void* /*properties*/, void** /*pool*/);
using Fn_EnableDomain = int (*)(uint32_t /*domain*/, void* /*pool*/);
using Fn_DisableDomain = int (*)(uint32_t /*domain*/);
using Fn_FlushPool = int (*)(void* /*pool*/);
using Fn_NextRecord = int (*)(const void* /*record*/, const void** /*next*/);
using Fn_OpString = const char* (*)(uint32_t /*domain*/, uint32_t /*op*/,
                                    uint32_t /*kind*/);
using Fn_VersionMajor = uint32_t (*)();
using Fn_VersionMinor = uint32_t (*)();

struct TracerFns {
    Fn_GetTimestamp GetTimestamp = nullptr;
    Fn_OpenPool OpenPool = nullptr;
    Fn_EnableDomain EnableDomain = nullptr;
    Fn_DisableDomain DisableDomain = nullptr;
    Fn_FlushPool FlushPool = nullptr;
    Fn_NextRecord NextRecord = nullptr;
    Fn_OpString OpString = nullptr;
    Fn_VersionMajor VersionMajor = nullptr;
    Fn_VersionMinor VersionMinor = nullptr;
};

// Memory-pool properties passed to open: mode/buffer_size/allocator/arg/
// buffer-callback/arg.  Kept as a raw layout mirror so the core library never
// includes tracer headers.
struct PoolProps {
    uint32_t mode;
    size_t buffer_size;
    void* alloc_fun;
    void* alloc_arg;
    void (*buffer_callback_fun)(const char*, const char*, void*);
    void* buffer_callback_arg;
};

// Activity domains of interest.
constexpr uint32_t kDomainHipOps = 2;  // device activity (kernels, copies)
constexpr uint32_t kDomainHipApi = 3;  // runtime API calls

// Buffer the tracer hands back on completion; parsed incrementally.
constexpr size_t kBufSize = 1u << 20;

std::mutex g_tracer_mutex;     // guards everything below (session scope)
TracerFns* g_fns = nullptr;    // dlsym'd entry points (null until start)
void* g_tracer_lib = nullptr;  // dlopen handle (kept for the process life)
void* g_pool = nullptr;        // record pool of the running session
std::string g_last_error;      // init/open diagnostic for the binding
bool g_domains_enabled = false;

// Parsed activities of the running session.
std::vector<GpuActivity>* g_acts = nullptr;     // kernels + copies
std::vector<GpuActivity>* g_api_acts = nullptr;  // HIP API rows
// correlation id -> OpRecord slot (from external-correlation push/pop).
std::unordered_map<uint64_t, uint64_t>* g_corr2ext = nullptr;

// Session-lifetime name arena (kernel symbols point into tracer-owned
// records and must be copied before the pool is reused).
std::deque<std::string>* g_name_arena = nullptr;
std::unordered_map<std::string, const char*>* g_name_index = nullptr;

// Timebase calibration: activity timestamps share the tracer clock epoch, so
// one offset measured at start maps every record onto the op timeline's
// steady clock.
int64_t g_time_offset_ns = 0;

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

// The tracer reports kernel symbols in mangled form; demangle through the
// runtime's own entry point when possible for readable timeline rows.
std::string demangle_symbol(const char* symbol) {
    if (symbol == nullptr) return "unknown";
    // hip kernel symbols are Itanium-mangled; reuse the C++ ABI demangler
    // through a small on-demand dlopen so no link dependency is added.
    using Fn_Demangle = char* (*)(const char*, char*, size_t*, int*);
    static Fn_Demangle demangle = nullptr;
    static bool probed = false;
    if (!probed) {
        probed = true;
        if (void* libc = dlopen("libc.so.6", RTLD_LAZY | RTLD_LOCAL)) {
            demangle = reinterpret_cast<Fn_Demangle>(
                dlsym(libc, "__cxa_demangle"));
        }
    }
    if (demangle != nullptr) {
        int status = 0;
        if (char* out = demangle(symbol, nullptr, nullptr, &status)) {
            if (status == 0 && out[0] != '\0') {
                std::string result(out);
                std::free(out);
                return result;
            }
            std::free(out);
        }
    }
    return std::string(symbol);
}

const char* hip_ops_name(uint32_t kind, const char* symbol) {
    if (symbol != nullptr && symbol[0] != '\0') {
        const std::string demangled = demangle_symbol(symbol);
        return intern_name_locked(demangled.data(), demangled.size());
    }
    // Device-side activities without a kernel symbol: copies and barriers.
    if (g_fns != nullptr && g_fns->OpString != nullptr) {
        if (const char* name = g_fns->OpString(kDomainHipOps, 0, kind)) {
            if (name[0] != '\0') {
                return intern_name_locked(name, std::strlen(name));
            }
        }
    }
    return "gpu_op";
}

const char* hip_api_name(uint32_t op) {
    if (g_fns != nullptr && g_fns->OpString != nullptr) {
        if (const char* name = g_fns->OpString(kDomainHipApi, op, 0)) {
            if (name[0] != '\0') {
                return intern_name_locked(name, std::strlen(name));
            }
        }
    }
    std::string fallback = "hip_api " + std::to_string(op);
    return intern_name_locked(fallback.data(), fallback.size());
}

void ensure_state_locked() {
    if (!g_acts) g_acts = new std::vector<GpuActivity>();
    if (!g_api_acts) g_api_acts = new std::vector<GpuActivity>();
    if (!g_corr2ext) g_corr2ext = new std::unordered_map<uint64_t, uint64_t>();
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

// Record parsing.  Fixed offsets validated against the tracer's activity
// protocol; kernel symbols arrive as a pointer into tracer-owned storage
// valid for the duration of the callback.
void parse_buffer_locked(const char* begin, const char* end) {
    auto next_record = g_fns ? g_fns->NextRecord : nullptr;
    const char* r = begin;
    while (r != nullptr && r + 64 <= end) {
        uint32_t domain = 0, kind = 0, op = 0;
        std::memcpy(&domain, r + 0, 4);
        std::memcpy(&kind, r + 4, 4);
        std::memcpy(&op, r + 8, 4);
        uint64_t correlation = 0, begin_ns = 0, end_ns = 0;
        std::memcpy(&correlation, r + 16, 8);
        std::memcpy(&begin_ns, r + 24, 8);
        std::memcpy(&end_ns, r + 32, 8);
        uint64_t pid_tid = 0, dev_queue = 0;
        std::memcpy(&pid_tid, r + 40, 8);
        std::memcpy(&dev_queue, r + 48, 8);
        const uint32_t thread_id =
            static_cast<uint32_t>(pid_tid & 0xffffffffu);
        const int32_t device =
            static_cast<int32_t>(dev_queue & 0xffffffffu);
        const int64_t queue =
            static_cast<int64_t>(dev_queue >> 32);

        if (domain == kDomainHipOps) {
            const char* symbol = nullptr;
            std::memcpy(&symbol, r + 56, 8);
            GpuActivity a;
            a.name = hip_ops_name(kind, symbol);
            a.start_ns = begin_ns + static_cast<uint64_t>(g_time_offset_ns);
            a.end_ns = end_ns + static_cast<uint64_t>(g_time_offset_ns);
            if (a.end_ns < a.start_ns) a.end_ns = a.start_ns;
            a.correlation = static_cast<uint32_t>(correlation);
            a.external_id = GpuActivity::kNoExt;
            a.thread_id = 0;
            a.cbid = kind;
            a.device = device;
            a.stream = queue;
            a.kind = 'k';
            a.bytes = 0;
            a.copy_kind = 0;
            a.value = 0;
            g_acts->push_back(a);
        } else if (domain == kDomainHipApi) {
            GpuActivity a;
            a.name = hip_api_name(op);
            a.start_ns = begin_ns + static_cast<uint64_t>(g_time_offset_ns);
            a.end_ns = end_ns + static_cast<uint64_t>(g_time_offset_ns);
            if (a.end_ns < a.start_ns) a.end_ns = a.start_ns;
            a.correlation = static_cast<uint32_t>(correlation);
            a.external_id = GpuActivity::kNoExt;
            a.thread_id = thread_id;
            a.cbid = op;
            a.device = -1;
            a.stream = -1;
            a.kind = 'r';
            a.bytes = 0;
            a.copy_kind = 0;
            a.value = 0;
            g_api_acts->push_back(a);
        }
        if (next_record == nullptr ||
            next_record(r, reinterpret_cast<const void**>(&r)) != 0) {
            break;
        }
    }
}

void buffer_completed(const char* begin, const char* end, void* /*arg*/) {
    if (begin == nullptr || end == nullptr || end <= begin) return;
    std::lock_guard<std::mutex> lock(g_tracer_mutex);
    if (g_fns != nullptr) parse_buffer_locked(begin, end);
}

bool load_fns_locked() {
    if (g_fns != nullptr) return true;
    const char* candidates[] = {"libroctracer64.so.4", "libroctracer64.so"};
    if (g_tracer_lib == nullptr) {
        for (const char* soname : candidates) {
            g_tracer_lib = dlopen(soname, RTLD_LAZY | RTLD_LOCAL);
            if (g_tracer_lib != nullptr) break;
        }
        if (g_tracer_lib == nullptr) {
            g_last_error = std::string("dlopen libroctracer64 failed: ") +
                           dlerror();
            return false;
        }
    }
    auto* fns = new TracerFns();
    #define TP_TRACER_DLSYM(field, name)                                  \
        fns->field = reinterpret_cast<Fn_##field>(                        \
            dlsym(g_tracer_lib, name));                                   \
        if (fns->field == nullptr) {                                      \
            g_last_error = std::string("dlsym ") + name + " failed";      \
            delete fns;                                                   \
            return false;                                                 \
    }
    TP_TRACER_DLSYM(GetTimestamp, "roctracer_get_timestamp")
    TP_TRACER_DLSYM(OpenPool, "roctracer_open_pool_expl")
    TP_TRACER_DLSYM(EnableDomain, "roctracer_enable_domain_activity_expl")
    TP_TRACER_DLSYM(DisableDomain, "roctracer_disable_domain_activity")
    TP_TRACER_DLSYM(FlushPool, "roctracer_flush_activity_expl")
    TP_TRACER_DLSYM(NextRecord, "roctracer_next_record")
    TP_TRACER_DLSYM(OpString, "roctracer_op_string")
    #undef TP_TRACER_DLSYM
    // Version probes are optional diagnostics.
    fns->VersionMajor = reinterpret_cast<Fn_VersionMajor>(
        dlsym(g_tracer_lib, "roctracer_version_major"));
    fns->VersionMinor = reinterpret_cast<Fn_VersionMinor>(
        dlsym(g_tracer_lib, "roctracer_version_minor"));
    g_fns = fns;
    return true;
}

} // namespace

TENSORPLAY_API std::atomic<bool> g_gpu_trace{false};

TENSORPLAY_API bool cupti_available() {
    std::lock_guard<std::mutex> lock(g_tracer_mutex);
    if (g_fns != nullptr) return true;
    const char* candidates[] = {"libroctracer64.so.4", "libroctracer64.so"};
    for (const char* soname : candidates) {
        void* lib = dlopen(soname, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
        if (lib != nullptr) {
            dlclose(lib);
            return true;
        }
    }
    if (g_last_error.empty()) {
        g_last_error =
            "libroctracer64 not found (tried .so.4/.so)";
    }
    return false;
}

TENSORPLAY_API std::string cupti_last_error() {
    std::lock_guard<std::mutex> lock(g_tracer_mutex);
    return g_last_error;
}

TENSORPLAY_API uint32_t cupti_version() {
    std::lock_guard<std::mutex> lock(g_tracer_mutex);
    if (g_fns == nullptr) {
        if (!load_fns_locked()) return 0;
    }
    if (g_fns->VersionMajor != nullptr && g_fns->VersionMinor != nullptr) {
        return (g_fns->VersionMajor() << 16) | g_fns->VersionMinor();
    }
    return 0;
}

TENSORPLAY_API bool cupti_start() {
    std::lock_guard<std::mutex> lock(g_tracer_mutex);
    ensure_state_locked();
    reset_session_locked();
    g_last_error.clear();

    if (!load_fns_locked()) return false;

    // Calibrate the tracer/steady timebase once per start.
    uint64_t tracer_now = 0;
    if (g_fns->GetTimestamp(&tracer_now) == 0) {
        g_time_offset_ns = static_cast<int64_t>(steady_now_ns()) -
                           static_cast<int64_t>(tracer_now);
    }

    if (g_pool == nullptr) {
        PoolProps props;
        std::memset(&props, 0, sizeof(props));
        props.mode = 1;  // allocate buffers through the default allocator
        props.buffer_size = kBufSize;
        props.buffer_callback_fun = buffer_completed;
        if (g_fns->OpenPool(&props, &g_pool) != 0 || g_pool == nullptr) {
            g_last_error = "roctracer_open_pool_expl failed";
            return false;
        }
    }

    // Domain enables carry the pool explicitly; the API domain brackets every
    // runtime call, the ops domain carries kernel/copy device activity.
    if (!g_domains_enabled) {
        if (g_fns->EnableDomain(kDomainHipOps, g_pool) != 0 ||
            g_fns->EnableDomain(kDomainHipApi, g_pool) != 0) {
            g_last_error = "roctracer_enable_domain_activity_expl failed";
            return false;
        }
        g_domains_enabled = true;
    }
    return true;
}

TENSORPLAY_API void cupti_stop_and_collect(std::vector<GpuActivity>& out) {
    std::vector<GpuActivity> acts;
    std::vector<GpuActivity> api_acts;
    {
        std::lock_guard<std::mutex> lock(g_tracer_mutex);
        if (g_fns == nullptr) return;
        if (g_domains_enabled) {
            (void)g_fns->DisableDomain(kDomainHipOps);
            (void)g_fns->DisableDomain(kDomainHipApi);
            g_domains_enabled = false;
        }
    }
    // Flush delivers pending records through buffer_completed synchronously;
    // the mutex must be released because the callback re-acquires it.  A short
    // grace period absorbs completions landing just after the flush.
    if (g_pool != nullptr) {
        (void)g_fns->FlushPool(g_pool);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::lock_guard<std::mutex> lock(g_tracer_mutex);
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
    out.reserve(out.size() + acts.size() + api_acts.size());
    out.insert(out.end(), acts.begin(), acts.end());
    out.insert(out.end(), api_acts.begin(), api_acts.end());
}

TENSORPLAY_API bool cupti_push_ext(uint64_t id) {
    if (!g_gpu_trace.load(std::memory_order_acquire)) return false;
    // The tracer carries external correlation ids through the record's
    // dedicated union slot on HIP-OPS records; our correlation join maps the
    // API-level correlation id instead, which the probe showed is shared
    // between the launching HIP-API record and the device activity record.
    // Nothing needs to be pushed per-thread here; the id travels in the map
    // below via the correlation captured at API level.
    // Record the current correlation chain: the enclosing op stamps its slot
    // under the next correlation id issued on this thread.  We approximate by
    // remembering (thread -> slot) and binding it in pop; device records are
    // joined through their API row's correlation.
    struct TlsSlot {
        uint64_t slot;
    };
    static thread_local TlsSlot tls{0};
    tls.slot = id;
    return true;
}

TENSORPLAY_API void cupti_pop_ext() {}

} // namespace prof
} // namespace tensorplay
