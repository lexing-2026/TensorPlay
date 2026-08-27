// Native op-level profiler -- TensorPlay's proportionate stand-in for
// torch.autograd.profiler / torch.profiler (Kineto).
//
// Scope:
//   * every dispatched op is recorded exactly once, at the below-autograd
//     redispatch funnel (detail::redispatch_* in TensorRedispatchGenerated.h,
//     instrumented by tools/codegen/gen_api.py) -- the same granularity
//     upstream gets from RecordFunction guards around aten dispatch;
//     composite inner calls therefore show up individually, matching
//     upstream's CompositeImplicitAutograd behavior;
//   * user annotations (`record_function`) nest naturally as spans;
//   * the autograd engine emits a "__backward__" span covering each
//     backward()/grad() execution;
//   * record_shapes/with_stack capture per-op input shapes/dtypes and the
//     Python call site of the outermost binding entry;
//   * export to Chrome Trace JSON (torch's format) happens Python-side.
//
// Performance contract: when inactive the only hot-path cost is one
// acquire-load of a static atomic bool per op plus (when shape/site capture
// is requested) one further load each -- the same class of guard as
// GradMode/InferenceMode checks already emitted around every call.  When
// active, recording costs a timestamp pair plus a short critical section.
//
// Deliberately out of scope: CUPTI kernel-level GPU traces (external tools
// drive cudaProfilerStart/Stop through tensorplay.cuda.profiler), stack
// SAMPLING, distributed profiling.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Device.h"
#include "Macros.h"

namespace tensorplay {
namespace prof {

enum class EventKind : char {
    kOp = 'o',              // dispatched operator
    kUser = 'u',            // record_function annotation
    kBackward = 'b',        // engine backward-phase span
};

using ShapeVec = std::vector<std::vector<int64_t>>;
using DtypeVec = std::vector<int32_t>;

struct Event {
    const char* name;       // borrowed for kOp (static literal), owned in
                            // an internal arena for kUser/kBackward
    uint64_t start_ns;      // CLOCK_MONOTONIC (steady_clock)
    uint64_t end_ns;
    uint64_t tid;           // OS thread id (gettid-equivalent hash)
    EventKind kind;
    // Captured input shapes/dtypes (record_shapes=True); null otherwise.
    std::shared_ptr<const ShapeVec> shapes;
    std::shared_ptr<const DtypeVec> dtypes;
    // Python call site of the binding entry (with_stack=True);
    // kNoSite when uncaptured.
    uint32_t site_id = kNoSite;
    // Opaque cudaEvent_t pair for GPU-timeline spans (USE_CUDA builds with
    // profiler_gpu_timing enabled); resolved by the binding layer.
    void* gpu_start = nullptr;
    void* gpu_end = nullptr;
    float gpu_ms = -1.f;    // filled by gpu_resolve_all
    // Output allocation volume for Tensor-returning ops (numel x itemsize),
    // recorded when capture-shapes is on; basis of the memory snapshot view
    // (in-place resizes/view aliases are NOT tracked -- allocator-level
    // accounting stays with external tools).
    int64_t out_bytes = -1;

    static constexpr uint32_t kNoSite = 0xffffffffu;
};

// True while a profiling session is running.  Hot-path guard.
TENSORPLAY_API extern std::atomic<bool> g_active;
// When true AND g_active, redispatch sites also capture input shapes+dtypes.
TENSORPLAY_API extern std::atomic<bool> g_capture_shapes;
// When true AND g_active, binding entries capture the Python call site.
TENSORPLAY_API extern std::atomic<bool> g_capture_sites;

// Begins a session; clears any previously collected events.  Nesting
// sessions is not supported (matches torch.profiler's outermost-wins).
TENSORPLAY_API void profiler_start();
TENSORPLAY_API void profiler_start_with_shapes();
TENSORPLAY_API void profiler_start_full();  // shapes + python sites

// Ends the session and returns the collected events ordered by start time.
TENSORPLAY_API std::vector<Event> profiler_stop();

// RAII op/annotation record.  Name for kOp must be a static literal; user
// annotations may pass any lifetime (an internal arena copies the bytes).
struct TENSORPLAY_API OpRecord {
    explicit OpRecord(const char* name, EventKind kind = EventKind::kOp);
    explicit OpRecord(const std::string& name, EventKind kind = EventKind::kUser);
    ~OpRecord();
    OpRecord(const OpRecord&) = delete;
    OpRecord& operator=(const OpRecord&) = delete;

    // record_shapes support: attaches input shapes (+dtypes) to this
    // event's slot.  No-op when the session ended before the call.
    void set_io_meta(ShapeVec&& shapes, DtypeVec&& dtypes);
    // Output allocation volume (Tensor-returning ops only).
    void set_output_bytes(int64_t nbytes);

    // Shapes-only overload emitted by the generated dispatchers
    // (tools/codegen): dtypes stay unset for these events.
    void set_shapes(ShapeVec&& shapes);

private:
    friend struct GpuTimerPair;
    void begin(const char* static_name, const std::string* owned_name,
               EventKind kind);
    uint64_t start_ns_ = 0;
    size_t slot_ = 0;
    bool live_ = false;
    bool counted_ = false;
    bool nvtx_open_ = false;
    bool itt_open_ = false;
};

// Stack-form of user annotations for language bindings where RAII objects
// are awkward (Python record_function context manager): begin/end must pair
// LIFO per thread, exactly like the context manager guarantees.
TENSORPLAY_API void user_span_begin(const std::string& name);
TENSORPLAY_API void user_span_end();

// ---- Python call-site capture -------------------------------------------
// The BINDING layer (which has Python.h) extracts frame info while the GIL
// is held and hands over plain bytes; the next OpRecord created ON THIS
// THREAD adopts it.  Consumption clears the slot, so composite inner ops
// (which never re-enter a binding) record no site instead of inheriting the
// outermost call's.
TENSORPLAY_API void set_python_site(const char* file, int line);
TENSORPLAY_API uint32_t intern_site(const char* file, int line);
// Export-path resolvers (ids stay valid until the next profiler_start).
TENSORPLAY_API uint32_t site_count();
TENSORPLAY_API std::string site_string(uint32_t id);
// Deduped stable spelling of a recurring runtime-built event name (e.g.
// "backward::MulBackward0"): one arena byte-copy per distinct name instead
// of per execution.
TENSORPLAY_API const char* intern_name(const std::string& name);

// ---- GPU timeline (USE_CUDA builds; validated on remote sm_89) ----------
// A pool-backed cudaEvent pair around dispatched CUDA work.  arm(device)
// before the kernel launch, close() after; elapsed times are resolved at
// session stop by the binding layer.  CPU-device and inactive calls are
// no-ops.
struct TENSORPLAY_API GpuTimerPair {
    explicit GpuTimerPair(OpRecord& rec) : rec_(rec) {}
    ~GpuTimerPair();
    void arm(const Device& device);  // records start on a CUDA op's stream
    void close();        // records the end event

private:
    OpRecord& rec_;
    // Live cudaEvent handle for the armed span; recycled by stop-time drain.
    void* gpu_start_ = nullptr;
    void* gpu_stream_ = nullptr;
    int gpu_device_ = -1;
    bool armed_ = false;
};

TENSORPLAY_API extern std::atomic<bool> g_gpu_timing;
// Resolves every recorded GPU pair through `emit`; waits only for the final
// event on each recorded stream, then recycles pool handles for the next
// session. CPU-build no-ops.
TENSORPLAY_API void gpu_resolve_all(
        std::vector<Event>& events,
        const std::function<void(Event&, float ms)>& emit);
TENSORPLAY_API void gpu_drain_pool();

// ---- NVTX bridge (runtime-loaded libnvtx; graceful degradation) ---------
// When g_emit_nvtx is set during a session, every op/user/backward span
// also emits a matching NVTX range so nsight-systems timelines show
// TensorPlay op names.  Raw passthroughs raise the historical stub error
// when the library is unavailable (tensorplay.cuda.nvtx contract).
TENSORPLAY_API extern std::atomic<bool> g_emit_nvtx;
TENSORPLAY_API bool nvtx_available();
TENSORPLAY_API int nvtx_range_push(const char* msg);
TENSORPLAY_API int nvtx_range_pop();
TENSORPLAY_API void nvtx_mark(const char* msg);
TENSORPLAY_API unsigned long long nvtx_range_start(const char* msg);
TENSORPLAY_API void nvtx_range_end(unsigned long long id);
// Internal no-throw lifecycle hooks.
TENSORPLAY_API void nvtx_span_begin(const char* name);
TENSORPLAY_API void nvtx_span_end();

// ---- ITT bridge (runtime-loaded libittnotify; VTune/Advisor) ------------
// Same contract as the NVTX bridge: emit_itt() mirrors task begin/end onto
// the "tensorplay" domain; silent no-op without the library.
TENSORPLAY_API extern std::atomic<bool> g_emit_itt;
TENSORPLAY_API bool itt_available();
TENSORPLAY_API void itt_task_begin_name(const char* name);
TENSORPLAY_API void itt_task_end();
TENSORPLAY_API void itt_span_begin(const char* name);
TENSORPLAY_API void itt_span_end();

} // namespace prof
} // namespace tensorplay
