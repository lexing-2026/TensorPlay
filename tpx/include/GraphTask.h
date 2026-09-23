#pragma once
#include <memory>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include "Tensor.h"
#include "Edge.h"
#include "Node.h"
#include "InputBuffer.h"
#include "PythonDispatchModeTLS.h"

namespace tensorplay {
namespace tpx {

// Holds metadata for a single execution of backward()/grad().
// several worker threads may evaluate functions of one GraphTask
// concurrently, so the shared bookkeeping is guarded by mutex_.
struct GraphTask {
    struct ExecInfo {
        struct Capture {
            Capture(int input_idx, int output_idx)
                : input_idx_(input_idx), output_idx_(output_idx) {}
            int input_idx_;  // within Node inputs
            int output_idx_; // within the output vector of a GraphTask
        };
        bool should_execute() const { return needed_ || captures_; }
        bool needed_ = false;
        std::unique_ptr<std::vector<Capture>> captures_;
    };

    bool keep_graph_;
    bool grad_mode_;
    // Dispatch modes active when backward was requested; device workers
    // install them so backward operators reach the same modes.
    ::tensorplay::impl::DispatchModeState dispatch_modes_ =
        ::tensorplay::impl::DispatchModeTLS::get_state();

    // Monotonic id for TP_ENGINE_TRACE correlation across concurrent or
    // nested graphs. Zero cost when tracing is off.
    uint64_t trace_id_ = 0;

    // Per-node execution state.  Dependencies and the partial input buffer
    // live in one entry so a single hash lookup covers both.  `buffer` is
    // only populated once the node receives its first gradient.
    struct NodeState {
        int remaining_deps = 0;
        InputBuffer buffer;
    };

    // --- Shared state (guarded by mutex_ once execution starts) ---
    std::unordered_map<Node*, NodeState> not_ready_;
    std::unordered_set<Node*> nodes_in_graph_;
    // Empty -> execute everything (backward()). Non-empty -> only execute
    // nodes with should_execute() == true (grad()).
    std::unordered_map<Node*, ExecInfo> exec_info_;
    // Captured gradients returned to the caller of grad().
    std::vector<Tensor> captured_vars_;

    // Callbacks queued while this graph is executing. The callback may queue
    // another callback, so execution consumes the vector by index.
    std::vector<std::function<void()>> final_callbacks_;
    std::mutex final_callbacks_mutex_;

    // --- Completion tracking ---
    std::mutex mutex_;
    std::condition_variable cv_;
    // Number of NodeTasks enqueued but not yet fully evaluated.  Atomic so
    // enqueue/complete accounting never takes mutex_ -- the hot path executes
    // this twice per node.
    std::atomic<uint64_t> outstanding_tasks_{0};
    std::atomic<bool> completed_{false};
    // First error raised by any node; rethrown by the initiating thread.
    // Guarded by mutex_ (only written/read on error paths).
    std::exception_ptr exception_;

    explicit GraphTask(bool keep_graph, bool grad_mode)
        : keep_graph_(keep_graph), grad_mode_(grad_mode) {
        // A fresh GraphTask is built per backward()/grad() call; pre-sizing
        // the shared maps avoids repeated bucket rehashing while the few
        // per-node entries are inserted.
        not_ready_.reserve(16);
        nodes_in_graph_.reserve(16);
    }

    void init_to_execute(Node& graph_root, const edge_list& outputs, bool accumulate_grad, uint64_t min_topo_nr);

    // Enqueue accounting: called with mutex_ NOT held.
    void task_enqueued() {
        outstanding_tasks_.fetch_add(1, std::memory_order_release);
    }

    // Mark one dequeued task as fully evaluated; wakes the initiator when the
    // last task finishes. Returns true iff this call transitioned the graph to
    // completed, so the caller can wake the initiating thread's queue (which
    // blocks on its own CV, not on cv_).
    bool task_completed() {
        if (outstanding_tasks_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            completed_.store(true, std::memory_order_release);
            cv_.notify_all();
            return true;
        }
        return false;
    }

    // Record the first node error; the initiator rethrows it after the graph
    void record_exception(std::exception_ptr exc) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!exception_) exception_ = std::move(exc);
    }

    void add_final_callback(std::function<void()> callback) {
        std::lock_guard<std::mutex> lock(final_callbacks_mutex_);
        final_callbacks_.emplace_back(std::move(callback));
    }

    void run_final_callbacks() {
        size_t index = 0;
        for (;;) {
            std::function<void()> callback;
            {
                std::lock_guard<std::mutex> lock(final_callbacks_mutex_);
                if (index >= final_callbacks_.size()) return;
                callback = final_callbacks_[index++];
            }
            callback();
        }
    }

    bool is_completed() {
        return completed_.load(std::memory_order_acquire);
    }

    // Block until every enqueued task has been evaluated.
    void wait_for_completion() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return completed_.load(std::memory_order_acquire); });
    }

    // Wake every worker blocked on this task's queues (used on completion so
    // idle workers can exit their pop loop).
    void wake_all() { cv_.notify_all(); }
};

} // namespace tpx
} // namespace tensorplay
