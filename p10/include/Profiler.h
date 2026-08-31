// Profiler scope:
//   * every dispatched op is recorded exactly once at the below-autograd
//     redispatch funnel (detail::redispatch_* in TensorRedispatchGenerated.h,
//     instrumented by tools/codegen/gen_api.py);
//   * user annotations (`record_function`) nest naturally as spans;
//   * the autograd engine emits a "__backward__" span covering each
//     backward()/grad() execution;
//   * record_shapes/with_stack capture per-op input shapes/dtypes and the
//     full Python frame chain of the outermost binding entry;
//   * allocator-level memory events (CPU + CUDA caching allocators) when
//     memory capture is requested -- real alloc/free accounting, not the
//     factory-op estimate the snapshot view falls back to;
//   * CUPTI kernel-level GPU tracing (USE_CUDA builds): kernel/memcpy/
//     memset activity records plus CUDA runtime/driver API rows, with
//     op->runtime->kernel correlation via CUPTI external correlation ids
//     (see ProfilerCupti.cpp);
//   * Python stack sampling runs binding-side (tensorplay/profiler.py);
//
// Performance contract: when inactive the only hot-path cost is one
// acquire-load of a static atomic bool per op plus (when shape/site capture
// is requested) one further load each -- the same class of guard as
// GradMode/InferenceMode checks already emitted around every call. When
// active, recording costs a timestamp pair plus a short critical section.
//
// The CUPTI collector is dlopen'd (never a hard link dependency) and only
// runs during a gpu_trace session.

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

// Estimated FLOPs for one op invocation from its input shapes
// (multiply-accumulate counted as two operations), or 0 when the arithmetic
// is not inferable from operand shapes alone.  Convolution estimates assume
// stride 1 / padding 0 / dilation 1 (op attributes are not captured).  Used
// by the binding layer to stamp each collected event at session stop.
TENSORPLAY_API int64_t estimate_flops(const char* name, const ShapeVec& shapes);

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
    // Full Python frame chain of the binding entry (with_stack=True);
    // kNoSite when uncaptured.  Frames resolve through stack_frames().
    uint32_t stack_id = kNoSite;
    // Opaque cudaEvent_t pair for GPU-timeline spans (USE_CUDA builds with
    // profiler_gpu_timing enabled); resolved by the binding layer.
    void* gpu_start = nullptr;
    void* gpu_end = nullptr;
    float gpu_ms = -1.f;    // filled by gpu_resolve_all, or -- in gpu_trace
                            // mode -- the summed kernel duration (ms) of
                            // the kernels correlated to this op
    // Number of GPU activities correlated to this op in gpu_trace mode.
    int32_t kernel_count = 0;
    // Output allocation volume for Tensor-returning ops (numel x itemsize),
    // recorded when capture-shapes is on; basis of the memory snapshot view
    // (allocator-level accounting lives in MemEvent below).
    int64_t out_bytes = -1;

    static constexpr uint32_t kNoSite = 0xffffffffu;
};

    // Allocator-level memory event (profile_memory sessions). One per user
    // allocation or user free observed at the caching-allocator boundary.
struct MemEvent {
    uint64_t ts_ns;
    void* ptr;
    int64_t bytes;          // requested bytes; frees repeat the block size
    bool alloc;             // true=alloc, false=free
    bool cuda;              // device class (CPU vs CUDA)
    int32_t device;         // device index (-1 for CPU)
    int64_t stream;         // owning stream (-1 for CPU / unknown)
    uint64_t tid;
};

// GPU activity record produced by the CUPTI collector (gpu_trace mode).
//   'k' kernel ('kernel'), 'm' memcpy ('gpu_memcpy'), 's' memset
//   ('gpu_memset'), 'r' runtime API ('cuda_runtime'), 'd' driver API
//   ('cuda_driver').
struct GpuActivity {
    const char* name;       // arena-owned for kernels/memcpy/memset;
                            // synthesized API spelling for 'r'/'d'
    uint64_t start_ns;      // session steady-clock base (calibrated offset)
    uint64_t end_ns;
    uint32_t correlation;   // CUPTI correlation id
    uint64_t external_id;   // OpRecord slot of the enclosing op, or kNoExt
    uint32_t thread_id;     // OS tid (API records; 0 otherwise)
    uint32_t cbid;          // runtime/driver callback id (API records)
    int32_t device;
    int32_t stream;
    char kind;
    uint64_t bytes;         // memcpy/memset volumes
    uint8_t copy_kind;      // CUPTI_ACTIVITY_MEMCPY_KIND_* (memcpy only)
    uint32_t value;         // memset fill value

    static constexpr uint64_t kNoExt = 0xffffffffffffffffull;
};

// True while a profiling session is running.  Hot-path guard.
TENSORPLAY_API extern std::atomic<bool> g_active;
// When true AND g_active, redispatch sites also capture input shapes+dtypes.
TENSORPLAY_API extern std::atomic<bool> g_capture_shapes;
// When true AND g_active, binding entries capture the Python call site.
TENSORPLAY_API extern std::atomic<bool> g_capture_sites;

// Begins a session; clears any previously collected events.  Nesting
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
    bool trace_pushed_ = false;  // external-correlation id on the CUPTI stack
};

// Stack-form of user annotations for language bindings where RAII objects
// are awkward (Python record_function context manager): begin/end must pair
// LIFO per thread, exactly like the context manager guarantees.
TENSORPLAY_API void user_span_begin(const std::string& name);
TENSORPLAY_API void user_span_end();

// ---- Python call-site / stack capture ------------------------------------
// The BINDING layer (which has Python.h) extracts frame info while the GIL
// is held and hands over plain bytes; the next OpRecord created ON THIS
// THREAD adopts it.  Consumption clears the slot, so composite inner ops
// (which never re-enter a binding) record no site instead of inheriting the
// outermost call's.
struct ProfFrame {
    std::string file;
    std::string func;
    int line;
};
TENSORPLAY_API void set_python_site(const char* file, int line);
// Full frame chain: the frames' front is the outermost user frame (same
// site the single-frame variant would record).  Copies into the intern
// table; the caller's vector may be discarded afterwards.
TENSORPLAY_API void set_python_stack(std::vector<ProfFrame>&& frames);
TENSORPLAY_API uint32_t intern_site(const char* file, int line);
// Export-path resolvers (ids stay valid until the next profiler_start).
TENSORPLAY_API uint32_t site_count();
TENSORPLAY_API std::string site_string(uint32_t id);
// Deduped stable spelling of a recurring runtime-built event name (e.g.
// "backward::MulBackward0"): one arena byte-copy per distinct name instead
// of per execution.
TENSORPLAY_API const char* intern_name(const std::string& name);
// Interned stack table (session lifetime, cleared on next start).
TENSORPLAY_API uint32_t intern_stack(std::vector<ProfFrame>&& frames);
TENSORPLAY_API std::vector<ProfFrame> stack_frames(uint32_t id);

// ---- Allocator-level memory capture --------------------------------------
// Toggled per session (profile_memory=True).  Hot-path cost when inactive:
// one acquire-load per allocation/free.
TENSORPLAY_API extern std::atomic<bool> g_mem_capture;
TENSORPLAY_API void mem_record_alloc(void* ptr, int64_t bytes,
                                     bool cuda, int32_t device,
                                     int64_t stream);
TENSORPLAY_API void mem_record_free(void* ptr, int64_t bytes,
                                    bool cuda, int32_t device,
                                    int64_t stream);
// Moves out the collected memory events (binding calls at session stop).
TENSORPLAY_API std::vector<MemEvent> mem_take();

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

// ---- CUPTI kernel-level tracing (USE_CUDA builds; dlopen'd libcupti) -----
// gpu_trace sessions enable the CUPTI activity API: CONCURRENT_KERNEL,
// MEMCPY, MEMSET, RUNTIME, DRIVER and EXTERNAL_CORRELATION.  Every CUDA-
// targeting op pushes its OpRecord slot as an external correlation id for
// the duration of its dispatch (GpuTimerPair), so kernel activity records
// join back to the op that launched them (op -> runtime API -> kernel).
// All CUPTI entry points are runtime-loaded; unavailable libraries degrade
// to cupti_available() == false and no GPU rows.
TENSORPLAY_API extern std::atomic<bool> g_gpu_trace;
TENSORPLAY_API bool cupti_available();
// Human-readable dlopen/init failure reason ("" when healthy).
TENSORPLAY_API std::string cupti_last_error();
// CUPTI library version (cuptiGetVersion), or 0 when unavailable.  Does not
// enable any activity kind; used to stamp trace-export schema metadata.
TENSORPLAY_API uint32_t cupti_version();
// Registers callbacks, calibrates the CUPTI/steady timebase and enables
// activity kinds.  Returns false (and records the reason) on failure.
TENSORPLAY_API bool cupti_start();
// Flushes, disables kinds and hands back the parsed activities ordered by
// start time.  Safe to call without a prior successful start.
TENSORPLAY_API void cupti_stop_and_collect(std::vector<GpuActivity>& out);
// External-correlation push/pop around CUDA dispatch (no-throw no-ops when
// the collector is not running).
TENSORPLAY_API bool cupti_push_ext(uint64_t id);
TENSORPLAY_API void cupti_pop_ext();

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
// The ITT bridge emits task begin/end events onto
// the "tensorplay" domain; silent no-op without the library.
TENSORPLAY_API extern std::atomic<bool> g_emit_itt;
TENSORPLAY_API bool itt_available();
TENSORPLAY_API void itt_task_begin_name(const char* name);
TENSORPLAY_API void itt_task_end();
TENSORPLAY_API void itt_span_begin(const char* name);
TENSORPLAY_API void itt_span_end();

} // namespace prof
} // namespace tensorplay
