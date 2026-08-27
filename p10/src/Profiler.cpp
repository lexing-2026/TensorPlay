#include "Profiler.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
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
// Arenas keeping session-lifetime bytes alive (user names, deduped sites).
std::deque<std::string>* g_name_arena = nullptr;
std::vector<std::pair<std::string, int>>* g_site_table = nullptr;
std::unordered_map<std::string, uint32_t>* g_site_index = nullptr;

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
} t_pending_site;

void clear_session_locked() {
    if (!g_events) g_events = new std::vector<Event>();
    if (!g_name_arena) g_name_arena = new std::deque<std::string>();
    if (!g_site_table) g_site_table = new std::vector<std::pair<std::string, int>>();
    if (!g_site_index) g_site_index = new std::unordered_map<std::string, uint32_t>();
    g_events->clear();
    g_name_arena->clear();
}

} // namespace

TENSORPLAY_API std::atomic<bool> g_active{false};
TENSORPLAY_API std::atomic<bool> g_capture_shapes{false};
TENSORPLAY_API std::atomic<bool> g_capture_sites{false};

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
    // emit_nvtx parity: NVTX ranges fire even without a profiling session
    // (torch.autograd.profiler.emit_nvtx semantics).  Static/arena names are
    // durable; user-annotation bytes are only durable under a session, so
    // spans keep requiring one.
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

} // namespace prof
} // namespace tensorplay
