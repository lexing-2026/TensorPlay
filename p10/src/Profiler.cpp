#include "Profiler.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace tensorplay {
namespace prof {

namespace {

std::mutex g_mutex;
// All events of the current session, appended under g_mutex.  Slots are
// stable for the whole session (append-only), so OpRecord can hold an index.
std::vector<Event>* g_events = nullptr;
// Allocator-level memory events of the current session (profile_memory).
std::vector<MemEvent>* g_mem_events = nullptr;
// Arenas keeping session-lifetime bytes alive (user names, deduped sites).
std::deque<std::string>* g_name_arena = nullptr;
std::vector<std::pair<std::string, int>>* g_site_table = nullptr;
std::unordered_map<std::string, uint32_t>* g_site_index = nullptr;
// Interned stacks: frames and full frame chains dedupe across the process
// lifetime (source locations repeat for every op), so keep one arena for
// frames and one id->chain table per session.
std::deque<std::string>* g_frame_arena = nullptr;
std::unordered_map<std::string, uint32_t>* g_frame_index = nullptr;
std::vector<std::vector<uint32_t>>* g_stack_table = nullptr;
std::deque<ProfFrame>* g_frame_store = nullptr;

// A redispatch record can outlive the Python call that is stopping a
// session (for example, a CUDA/CPU worker may still be unwinding).  Stop
// waits for these records before moving the event vector, so GPU pairs cannot
// be appended after gpu_resolve_all has taken its snapshot.
std::atomic<uint64_t> g_inflight{0};
std::mutex g_inflight_mutex;
std::condition_variable g_inflight_cv;

void release_inflight() noexcept {
    if (g_inflight.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        g_inflight_cv.notify_all();
    }
}

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

uint64_t this_thread_id() {
    // Stable per-thread identifier without depending on platform gettid.
    // Chrome traces only need grouping consistency, not OS semantics.
    thread_local const uint64_t tid = std::hash<std::thread::id>{}(
        std::this_thread::get_id());
    return tid;
}

// Pending Python call site of this thread's outermost binding entry.
thread_local struct {
    bool valid = false;
    uint32_t site_id = Event::kNoSite;
    uint32_t stack_id = Event::kNoSite;
} t_pending_site;

void clear_session_locked() {
    if (!g_events) g_events = new std::vector<Event>();
    if (!g_mem_events) g_mem_events = new std::vector<MemEvent>();
    if (!g_name_arena) g_name_arena = new std::deque<std::string>();
    if (!g_site_table) g_site_table = new std::vector<std::pair<std::string, int>>();
    if (!g_site_index) g_site_index = new std::unordered_map<std::string, uint32_t>();
    if (!g_stack_table) g_stack_table = new std::vector<std::vector<uint32_t>>();
    g_events->clear();
    g_mem_events->clear();
    g_name_arena->clear();
    // Frame arena / stacks dedupe process-lifetime; the id->chain table is
    // session-scoped (ids are resolved before the next start).
    g_stack_table->clear();
}

} // namespace

TENSORPLAY_API std::atomic<bool> g_active{false};
TENSORPLAY_API std::atomic<bool> g_capture_shapes{false};
TENSORPLAY_API std::atomic<bool> g_capture_sites{false};
TENSORPLAY_API std::atomic<bool> g_mem_capture{false};

void profiler_start() {
    std::lock_guard<std::mutex> lock(g_mutex);
    clear_session_locked();
    g_active.store(true, std::memory_order_release);
}

void profiler_start_with_shapes() {
    profiler_start();
    // Set after the buffer is ready so no early event misses its shapes.
    g_capture_shapes.store(true, std::memory_order_release);
}

void profiler_start_full() {
    profiler_start();
    g_capture_shapes.store(true, std::memory_order_release);
    g_capture_sites.store(true, std::memory_order_release);
}

std::vector<Event> profiler_stop() {
    g_active.store(false, std::memory_order_release);
    g_capture_shapes.store(false, std::memory_order_release);
    g_capture_sites.store(false, std::memory_order_release);
    g_mem_capture.store(false, std::memory_order_release);
    {
        std::unique_lock<std::mutex> wait_lock(g_inflight_mutex);
        g_inflight_cv.wait(wait_lock, [] {
            return g_inflight.load(std::memory_order_acquire) == 0;
        });
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    std::vector<Event> out;
    if (g_events) {
        out = std::move(*g_events);
        g_events->clear();
    }
    // Keep the name arena alive until the next start.  Event::name is a
    // borrowed pointer for user/backward annotations, and the Python binding
    // copies it while converting this returned vector into tuples.  Clearing
    // here leaves those pointers dangling exactly when an optimizer step is
    // wrapped in record_function (plain op-only profiles do not expose it).
    // Site tables likewise stay resolvable until the next start.
    std::stable_sort(out.begin(), out.end(),
                     [](const Event& a, const Event& b) {
                         return a.start_ns < b.start_ns;
                     });
    return out;
}

void set_python_site(const char* file, int line) {
    if (!g_active.load(std::memory_order_acquire)) return;
    const uint32_t id = intern_site(file, line);
    t_pending_site.valid = true;
    t_pending_site.site_id = id;
}

void set_python_stack(std::vector<ProfFrame>&& frames) {
    if (!g_active.load(std::memory_order_acquire)) return;
    if (frames.empty()) return;
    // The outermost frame doubles as the single-line site id, so the
    // legacy site accessor (Event::site_id -> site_string) keeps working
    // alongside the full chain.
    const uint32_t site_id = intern_site(frames.front().file.c_str(),
                                         frames.front().line);
    const uint32_t stack_id = intern_stack(std::move(frames));
    if (stack_id == Event::kNoSite) return;
    t_pending_site.valid = true;
    t_pending_site.site_id = site_id;
    t_pending_site.stack_id = stack_id;
}

uint32_t intern_stack(std::vector<ProfFrame>&& frames) {
    if (frames.empty()) return Event::kNoSite;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_frame_arena) {
        g_frame_arena = new std::deque<std::string>();
        g_frame_index = new std::unordered_map<std::string, uint32_t>();
        g_frame_store = new std::deque<ProfFrame>();
        g_stack_table = new std::vector<std::vector<uint32_t>>();
    }
    std::vector<uint32_t> ids;
    ids.reserve(frames.size());
    for (auto& f : frames) {
        // "file:line (func)" dedup key; line varies per call site, so the
        // cache keys on the exact frame tuple.
        std::string key = f.file;
        key += ":";
        key += std::to_string(f.line);
        key += " (";
        key += f.func;
        key += ")";
        auto it = g_frame_index->find(key);
        uint32_t fid;
        if (it != g_frame_index->end()) {
            fid = it->second;
        } else {
            fid = static_cast<uint32_t>(g_frame_store->size());
            g_frame_store->push_back(std::move(f));
            g_frame_index->emplace(std::move(key), fid);
        }
        ids.push_back(fid);
    }
    const uint32_t id = static_cast<uint32_t>(g_stack_table->size());
    g_stack_table->push_back(std::move(ids));
    return id;
}

std::vector<ProfFrame> stack_frames(uint32_t id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::vector<ProfFrame> out;
    if (!g_stack_table || !g_frame_store || id >= g_stack_table->size()) {
        return out;
    }
    out.reserve(g_stack_table->at(id).size());
    for (uint32_t fid : g_stack_table->at(id)) {
        out.push_back(g_frame_store->at(fid));
    }
    return out;
}

// ---- Allocator-level memory capture ---------------------------------------
void mem_record_alloc(void* ptr, int64_t bytes, bool cuda, int32_t device,
                      int64_t stream) {
    if (!g_mem_capture.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_mem_capture.load(std::memory_order_relaxed)) return;
    if (!g_mem_events) g_mem_events = new std::vector<MemEvent>();
    MemEvent e;
    e.ts_ns = now_ns();
    e.ptr = ptr;
    e.bytes = bytes;
    e.alloc = true;
    e.cuda = cuda;
    e.device = device;
    e.stream = stream;
    e.tid = this_thread_id();
    g_mem_events->push_back(e);
}

void mem_record_free(void* ptr, int64_t bytes, bool cuda, int32_t device,
                     int64_t stream) {
    if (!g_mem_capture.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_mem_capture.load(std::memory_order_relaxed)) return;
    if (!g_mem_events) g_mem_events = new std::vector<MemEvent>();
    MemEvent e;
    e.ts_ns = now_ns();
    e.ptr = ptr;
    e.bytes = bytes;
    e.alloc = false;
    e.cuda = cuda;
    e.device = device;
    e.stream = stream;
    e.tid = this_thread_id();
    g_mem_events->push_back(e);
}

std::vector<MemEvent> mem_take() {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::vector<MemEvent> out;
    if (g_mem_events) {
        out = std::move(*g_mem_events);
        g_mem_events->clear();
    }
    return out;
}

uint32_t intern_site(const char* file, int line) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_site_table) g_site_table = new std::vector<std::pair<std::string, int>>();
    if (!g_site_index) g_site_index = new std::unordered_map<std::string, uint32_t>();
    std::string key(file);
    key += ":";
    key += std::to_string(line);
    auto it = g_site_index->find(key);
    if (it != g_site_index->end()) return it->second;
    const uint32_t id = static_cast<uint32_t>(g_site_table->size());
    g_site_table->emplace_back(std::move(key), line);
    (*g_site_index)[key] = id;
    return id;
}

const char* intern_name(const std::string& name) {
    // Session-independent dedup: backward-node class names repeat for the
    // lifetime of the process, so keep one stable copy per distinct string.
    static std::mutex m;
    static std::deque<std::string>* arena = nullptr;
    static std::unordered_map<std::string, size_t>* index = nullptr;
    std::lock_guard<std::mutex> lock(m);
    if (!arena) {
        arena = new std::deque<std::string>();
        index = new std::unordered_map<std::string, size_t>();
    }
    auto it = index->find(name);
    if (it != index->end()) return arena->at(it->second).c_str();
    arena->push_back(name);
    const size_t pos = arena->size() - 1;
    (*index)[name] = pos;
    return arena->back().c_str();
}

void OpRecord::begin(const char* static_name, const std::string* owned_name,
                     EventKind kind) {
    // NVTX ranges can fire without a profiling session; user-annotation bytes
    // are retained only during a session, so spans still require one.
    const bool nvtx_on =
        g_emit_nvtx.load(std::memory_order_relaxed) && kind == EventKind::kOp;
    const bool itt_on =
        g_emit_itt.load(std::memory_order_relaxed) && kind == EventKind::kOp;
    if (!g_active.load(std::memory_order_acquire)) return;
    g_inflight.fetch_add(1, std::memory_order_acq_rel);
    counted_ = true;
    const uint64_t start = now_ns();
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_active.load(std::memory_order_relaxed)) {
        counted_ = false;
        release_inflight();
        return;
    }
    Event e;
    e.name = static_name;
    e.kind = kind;
    e.tid = this_thread_id();
    e.start_ns = start;
    e.end_ns = start;
    if (owned_name != nullptr) {
        g_name_arena->push_back(*owned_name);
        e.name = g_name_arena->back().c_str();
    }
    if (t_pending_site.valid) {
        e.site_id = t_pending_site.site_id;
        e.stack_id = t_pending_site.stack_id;
        t_pending_site.valid = false;  // consume: inner ops record no site
    }
    if ((nvtx_on || itt_on) && e.name != nullptr) {
        // static literal or interned spelling -- durable beyond this call
        nvtx_span_begin(e.name);
        itt_span_begin(e.name);
        nvtx_open_ = nvtx_on;
        itt_open_ = itt_on;
    }
    g_events->push_back(e);
    slot_ = g_events->size() - 1;
    start_ns_ = start;
    live_ = true;
}

OpRecord::OpRecord(const char* name, EventKind kind) {
    begin(name, nullptr, kind);
}

OpRecord::OpRecord(const std::string& name, EventKind kind) {
    begin(nullptr, &name, kind);
}

OpRecord::~OpRecord() {
    if (nvtx_open_) { nvtx_open_ = false; nvtx_span_end(); }
    if (itt_open_) { itt_open_ = false; itt_span_end(); }
    if (!live_) return;
    live_ = false;
    const uint64_t end = now_ns();
    // Slots are append-only and stable; our slot can only be missing after a
    // profiler_stop() cleared the buffer, in which case the session is over
    // and dropping the end-timestamp is fine.
    std::lock_guard<std::mutex> lock(g_mutex);
    if (slot_ < g_events->size()) {
        (*g_events)[slot_].end_ns = end;
    }
    if (counted_) {
        counted_ = false;
        release_inflight();
    }
}

void OpRecord::set_io_meta(ShapeVec&& shapes, DtypeVec&& dtypes) {
    if (!live_) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (slot_ < g_events->size()) {
        auto* slot = &(*g_events)[slot_];
        slot->shapes = std::make_shared<const ShapeVec>(std::move(shapes));
        slot->dtypes = std::make_shared<const DtypeVec>(std::move(dtypes));
    }
}

void OpRecord::set_output_bytes(int64_t nbytes) {
    if (!live_) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (slot_ < g_events->size()) (*g_events)[slot_].out_bytes = nbytes;
}

void OpRecord::set_shapes(ShapeVec&& shapes) {
    set_io_meta(std::move(shapes), DtypeVec{});
}

namespace {
// Live user-annotation slots of this thread (strict LIFO via the context
// manager contract).
thread_local std::vector<size_t> t_user_stack;
} // namespace

void user_span_begin(const std::string& name) {
    if (!g_active.load(std::memory_order_acquire)) return;
    const uint64_t start = now_ns();
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_active.load(std::memory_order_relaxed)) return;
    g_name_arena->push_back(name);
    Event e;
    e.name = g_name_arena->back().c_str();
    e.kind = EventKind::kUser;
    e.tid = this_thread_id();
    e.start_ns = start;
    e.end_ns = start;
    if (t_pending_site.valid) {
        e.site_id = t_pending_site.site_id;
        e.stack_id = t_pending_site.stack_id;
        t_pending_site.valid = false;
    }
    g_events->push_back(e);
    t_user_stack.push_back(g_events->size() - 1);
    tensorplay::prof::nvtx_span_begin(e.name);
}

void user_span_end() {
    const uint64_t end = now_ns();
    tensorplay::prof::nvtx_span_end();
    std::lock_guard<std::mutex> lock(g_mutex);
    if (t_user_stack.empty()) return;
    const size_t slot = t_user_stack.back();
    t_user_stack.pop_back();
    if (slot < g_events->size()) (*g_events)[slot].end_ns = end;
}

// Accessors used by the export path (binding layer) -- kept out of the
// header to avoid exposing the session internals.
TENSORPLAY_API uint32_t site_count() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_site_table ? static_cast<uint32_t>(g_site_table->size()) : 0;
}

TENSORPLAY_API std::string site_string(uint32_t id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_site_table || id >= g_site_table->size()) return "";
    return g_site_table->at(id).first;
}

// ---- Op-level FLOP estimation ----------------------------------------------
// Runs once per collected event at session stop (inside the binding's batch
// conversion), never on the dispatch hot path.  A multiply-accumulate pair
// counts as two operations.  Only operand shapes are available -- op
// attributes are not captured -- so:
//   * convolution assumes stride 1 / padding 0 / dilation 1;
//   * ops whose arithmetic depends on non-captured attributes (einsum's
//     equation) return 0;
//   * ops that decompose into counted primitives (linear -> addmm) return 0
//     so per-session totals never double count.
// Event names may carry an overload suffix ("mm.Tensor"); matching keys on
// the base spelling.

namespace {

int64_t flops_prod_from(const std::vector<int64_t>& dims, size_t begin) {
    int64_t out = 1;
    for (size_t i = begin; i < dims.size(); ++i) out *= dims[i];
    return out;
}

// Broadcast-compatible batch size of two shapes' leading dimensions
// (trailing dims aligned, missing dims broadcast to 1).
int64_t flops_batch_dims(const std::vector<int64_t>& a,
                         const std::vector<int64_t>& b) {
    int64_t batch = 1;
    size_t i = a.size() - 2;
    size_t j = b.size() - 2;
    while (i > 0 && j > 0) {
        batch *= std::max(a[i - 1], b[j - 1]);
        --i;
        --j;
    }
    while (i > 0) batch *= a[--i];
    while (j > 0) batch *= b[--j];
    return batch;
}

// Product of the leading dimensions dims[0, end) -- the batch prefix.
int64_t flops_batch_prefix(const std::vector<int64_t>& dims, size_t end) {
    int64_t out = 1;
    for (size_t i = 0; i < end; ++i) out *= dims[i];
    return out;
}

int64_t flops_matmul(const ShapeVec& shapes) {
    const auto& a = shapes[0];
    const auto& b = shapes[1];
    if (a.empty() || b.empty()) return 0;
    if (a.size() == 1 && b.size() == 1) return 2 * a[0] * b[0];
    if (a.size() == 1) {
        // Vector @ (batched) matrix: the vector broadcasts over b's batch
        // dims (everything before its [K, N] tail).
        return 2 * flops_batch_prefix(b, b.size() - 2) * a[0] *
               b[b.size() - 1];
    }
    if (b.size() == 1) {
        // (Batched) matrix @ vector: batch dims are everything before a's
        // [M, K] tail.
        return 2 * flops_batch_prefix(a, a.size() - 2) * a[a.size() - 2] *
               a[a.size() - 1];
    }
    return 2 * flops_batch_dims(a, b) * a[a.size() - 2] * a[a.size() - 1] *
           b[b.size() - 1];
}

int64_t flops_conv(const ShapeVec& shapes) {
    // Input [N, C, *D_in], weight [Cout, C/groups, *k]; output spatial size
    // approximated by the input's (stride 1 / padding 0 / dilation 1).
    const auto& inp = shapes[0];
    const auto& weight = shapes[1];
    if (inp.size() < 3 || weight.size() < 3) return 0;
    return 2 * inp[0] * weight[0] * flops_prod_from(inp, 2) * weight[1] *
           flops_prod_from(weight, 2);
}

} // namespace

TENSORPLAY_API int64_t estimate_flops(const char* name,
                                      const ShapeVec& shapes) {
    if (name == nullptr || shapes.size() < 2) return 0;
    const char* dot = std::strchr(name, '.');
    const std::string base(name, dot ? static_cast<size_t>(dot - name)
                                     : std::strlen(name));
    const auto& s0 = shapes[0];
    const auto& s1 = shapes[1];

    if (base == "mm") {
        if (s0.size() < 2 || s1.empty()) return 0;
        return 2 * s0[s0.size() - 2] * s0[s0.size() - 1] * s1[s1.size() - 1];
    }
    if (base == "addmm") {
        // addmm(input, mat1, mat2): the operands are arguments 1 and 2.
        if (shapes.size() < 3) return 0;
        const auto& a = shapes[1];
        const auto& b = shapes[2];
        if (a.size() < 2 || b.empty()) return 0;
        return 2 * a[a.size() - 2] * a[a.size() - 1] * b[b.size() - 1];
    }
    if (base == "bmm") {
        if (s0.size() < 3 || s1.size() < 3) return 0;
        return 2 * s0[0] * s0[s0.size() - 2] * s0[s0.size() - 1] *
               s1[s1.size() - 1];
    }
    if (base == "baddbmm") {
        if (shapes.size() < 3) return 0;
        const auto& a = shapes[1];
        const auto& b = shapes[2];
        if (a.size() < 3 || b.size() < 3) return 0;
        return 2 * a[0] * a[a.size() - 2] * a[a.size() - 1] *
               b[b.size() - 1];
    }
    if (base == "matmul") {
        return flops_matmul(shapes);
    }
    if (base == "conv1d" || base == "conv2d" || base == "conv3d") {
        return flops_conv(shapes);
    }
    return 0;
}

} // namespace prof
} // namespace tensorplay
